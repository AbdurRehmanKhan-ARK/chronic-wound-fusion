#!/usr/bin/env python3
"""Ablation and fusion comparison table — no retraining, no GPU, no dataset needed.

Uses artifacts already produced by the clean pipeline:
  outputs/02_gating/gating_test_probs_v1.npy   shape (111, 18) = 3 backbones x 6 class probs
      (these are the fold-averaged test probabilities that were fed to the gate)
  outputs/02_gating/gating_test_preds_v1.csv   index, file_path, true_label, pred_label

Computes: per-backbone scores, simple average fusion, majority vote (defined tie rule),
and the gated MLP fusion, all on the same clean 111-image test set.

Usage (from repo root):
  python scripts/make_ablation_table.py \
      --probs outputs/02_gating/gating_test_probs_v1.npy \
      --preds outputs/02_gating/gating_test_preds_v1.csv \
      --models vgg19,densenet201,mobilenet_v2 \
      --save outputs/03_figures/ablation_table.txt

IMPORTANT: the --models order must exactly match the order used when training the gate
(default in README: vgg19,densenet201,mobilenet_v2). If the order is wrong the three
probability blocks will be attributed to the wrong backbone.
"""
import argparse
import csv
from math import sqrt
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

CLASS_NAMES = ["background", "diabetic", "normal-skin", "pressure", "surgical", "venous"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = (z / d) * sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, c - h), min(1.0, c + h))


def load_true_and_gated(preds_csv: Path):
    y_true, y_gated, paths = [], [], []
    with open(preds_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            y_true.append(int(row["true_label"]))
            y_gated.append(int(row["pred_label"]))
            paths.append(row.get("file_path", ""))
    return np.array(y_true), np.array(y_gated), paths


def score(name, y_true, pred):
    return {
        "model": name,
        "acc": accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro"),
        "weighted_f1": f1_score(y_true, pred, average="weighted"),
        "pred": pred,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probs", default="outputs/02_gating/gating_test_probs_v1.npy")
    ap.add_argument("--preds", default="outputs/02_gating/gating_test_preds_v1.csv")
    ap.add_argument("--models", default="vgg19,densenet201,mobilenet_v2",
                    help="Order must match gate training order")
    ap.add_argument("--save", default="outputs/03_figures/ablation_table.txt")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    P = np.load(args.probs)
    y_true, y_gated, paths = load_true_and_gated(Path(args.preds))

    n, total_dim = P.shape
    num_classes = 6
    if total_dim != len(models) * num_classes:
        raise SystemExit(f"probs has {total_dim} columns but {len(models)} models x {num_classes} = {len(models)*num_classes} expected")
    if len(y_true) != n:
        raise SystemExit(f"probs rows ({n}) != preds csv rows ({len(y_true)})")
    if y_true.min() < 0 or y_true.max() >= num_classes:
        raise SystemExit("labels out of range — wrong file?")
    for p in paths:
        if p and "/test/" not in p.replace("\\\\", "\\").replace("\\", "/"):
            print(f"[WARN] non-test path in preds csv: {p}")

    # sanity: each block must be a valid probability vector
    blocks = {m: P[:, i*num_classes:(i+1)*num_classes] for i, m in enumerate(models)}
    for m, b in blocks.items():
        s = b.sum(axis=1)
        if not np.allclose(s, 1.0, atol=0.02):
            print(f"[WARN] {m} block rows do not sum to 1 (min {s.min():.3f}, max {s.max():.3f})")

    rows = []
    for m, b in blocks.items():
        rows.append(score(f"{m} (single)", y_true, b.argmax(axis=1)))

    # simple average fusion
    avg = np.mean(list(blocks.values()), axis=0)
    rows.append(score("Simple average fusion", y_true, avg.argmax(axis=1)))

    # majority vote; tie (no class with 2+ votes) broken by highest summed probability
    votes = np.stack([b.argmax(axis=1) for b in blocks.values()])  # (3, n)
    counts = np.stack([(votes == c).sum(axis=0) for c in range(num_classes)])  # (6, n)
    mv = counts.argmax(axis=0)
    ties = counts.max(axis=0) < 2
    n_ties = int(ties.sum())
    mv = mv.copy()
    if n_ties:
        mv[ties] = avg[ties].argmax(axis=1)
    rows.append(score("Majority vote (tie->avg prob)", y_true, mv))

    # gated MLP fusion: predictions come from the saved CSV (produced by the real gate)
    rows.append(score("Gated MLP fusion (reported)", y_true, y_gated))

    # verify saved probs reproduce the gated predictions through argmax consistency check:
    # (the gate output is NOT in the npy; the CSV is authoritative)

    lines = []
    lines.append(f"Ablation / fusion comparison on clean test set (n={n})")
    lines.append(f"Source probs: {args.probs}")
    lines.append(f"Source preds: {args.preds}")
    lines.append(f"Model order: {', '.join(models)}")
    lines.append("")
    lines.append("| Model | Accuracy | Macro F1 | Weighted F1 |")
    lines.append("|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['model']} | {r['acc']*100:.2f}% | {r['macro_f1']*100:.2f} | {r['weighted_f1']*100:.2f} |")
    lines.append("")

    gate = rows[-1]
    singles = rows[:len(models)]
    best_single = max(singles, key=lambda r: r["macro_f1"])
    avg_row = rows[len(models)]
    mv_row = rows[len(models)+1]
    lines.append(f"Best single backbone: {best_single['model']} (macro F1 {best_single['macro_f1']*100:.2f})")
    lines.append(f"Gate vs best single: {(gate['macro_f1']-best_single['macro_f1'])*100:+.2f} macro F1 points")
    lines.append(f"Gate vs simple average: {(gate['macro_f1']-avg_row['macro_f1'])*100:+.2f} macro F1 points")
    lines.append(f"Gate vs majority vote: {(gate['macro_f1']-mv_row['macro_f1'])*100:+.2f} macro F1 points")
    lines.append(f"Majority-vote ties broken by summed probability: {n_ties} of {n} images")
    lines.append("")

    lo, hi = wilson(int((y_true == y_gated).sum()), n)
    lines.append(f"Wilson 95% CI for gated accuracy: [{lo*100:.1f}, {hi*100:.1f}]")
    lines.append("Differences smaller than ~5 accuracy points are within noise at this test size.")

    out = "\n".join(lines)
    print(out)
    save_path = Path(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(out, encoding="utf-8")
    print(f"\nSaved: {save_path}")


if __name__ == "__main__":
    main()
