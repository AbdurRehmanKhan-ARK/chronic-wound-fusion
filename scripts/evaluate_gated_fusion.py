#!/usr/bin/env python3
"""Evaluate the gated fusion model on the clean test-set predictions.

Reads gating_test_preds_v1.csv (produced by scripts/fusion/train_gating_mlp.py)
and computes final metrics: confusion matrix, per-class accuracy, macro F1,
weighted F1, and overall accuracy. Saves a text report and a confusion-matrix
figure alongside the CSV.

Usage:
  .venv\\Scripts\\python.exe scripts\\evaluate_gated_fusion.py --preds-csv outputs\\02_gating\\gating_test_preds_v1.csv --output outputs\\03_figures
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

CLASS_NAMES = ["background", "diabetic", "normal-skin", "pressure", "surgical", "venous"]


def load_preds(csv_path: Path):
    true_labels, pred_labels = [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            true_labels.append(int(row["true_label"]))
            pred_labels.append(int(row["pred_label"]))
    return np.array(true_labels), np.array(pred_labels)


def plot_confusion_matrix(cm, class_names, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Gated Fusion — Confusion Matrix (Clean Test Set)")

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate gated fusion predictions")
    parser.add_argument("--preds-csv", required=True, help="Path to gating_test_preds_v1.csv")
    parser.add_argument("--output", default="outputs/03_figures", help="Directory to save report + figure")
    parser.add_argument(
        "--class-names",
        default=",".join(CLASS_NAMES),
        help="Comma-separated class names in label-index order",
    )
    args = parser.parse_args()

    preds_csv = Path(args.preds_csv)
    if not preds_csv.exists():
        raise FileNotFoundError(f"Predictions CSV not found: {preds_csv}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    class_names = [c.strip() for c in args.class_names.split(",")]

    y_true, y_pred = load_preds(preds_csv)

    n_classes_seen = len(set(y_true.tolist()) | set(y_pred.tolist()))
    if len(class_names) < n_classes_seen:
        raise ValueError(
            f"--class-names has {len(class_names)} entries but predictions reference "
            f"{n_classes_seen} distinct label indices; supply matching class names."
        )

    overall_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    per_class_acc = cm.diagonal() / cm.sum(axis=1).clip(min=1)

    # --- Print summary to console ---
    print(f"\nEvaluated {len(y_true)} clean test-set samples\n")
    print(f"Overall accuracy : {overall_acc:.4f}")
    print(f"Macro F1         : {macro_f1:.4f}")
    print(f"Weighted F1      : {weighted_f1:.4f}\n")
    print("Per-class accuracy:")
    for name, acc in zip(class_names, per_class_acc):
        print(f"  {name:<15} {acc:.4f}")
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print(cm)
    print("\nFull classification report:")
    print(report)

    # --- Save text report ---
    report_path = out_dir / "gated_fusion_evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Evaluated {len(y_true)} clean test-set samples\n\n")
        f.write(f"Overall accuracy : {overall_acc:.4f}\n")
        f.write(f"Macro F1         : {macro_f1:.4f}\n")
        f.write(f"Weighted F1      : {weighted_f1:.4f}\n\n")
        f.write("Per-class accuracy:\n")
        for name, acc in zip(class_names, per_class_acc):
            f.write(f"  {name:<15} {acc:.4f}\n")
        f.write("\nConfusion matrix (rows = true, cols = predicted):\n")
        f.write(np.array2string(cm))
        f.write("\n\nFull classification report:\n")
        f.write(report)
    print(f"\nSaved text report to: {report_path}")

    # --- Save confusion matrix figure ---
    fig_path = out_dir / "gated_fusion_confusion_matrix.png"
    try:
        plot_confusion_matrix(cm, class_names, fig_path)
        print(f"Saved confusion matrix figure to: {fig_path}")
    except ImportError:
        print("matplotlib not available — skipped confusion matrix figure.")


if __name__ == "__main__":
    main()