#!/usr/bin/env python3
"""Train a gating MLP using OOF probability arrays and meta CSVs.

This script:
 - Verifies that the provided meta CSVs align (same ordering and file_path)
 - Loads the per-model OOF probability arrays
 - Trains a small MLP that outputs 3 weights (one per base model) for each image
 - Applies softmax over weights and produces a weighted fusion of the 3 model probs
 - Reports train/val accuracy and saves the gating model + fused OOF preds

Usage example:
  .\.venv\Scripts\python.exe scripts/train_gating.py --models vgg19,densenet201,mobilenet_v2 --oof-dir outputs --output outputs
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GatingNet(nn.Module):
    def __init__(self, input_dim: int, num_models: int = 3, hidden: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_models),
        )

    def forward(self, x):
        return self.fc(x)


def load_meta_paths(meta_paths):
    metas = [pd.read_csv(p, index_col=0) for p in meta_paths]
    # check alignment of file_path and labels
    base = metas[0]
    for m in metas[1:]:
        if not base['file_path'].equals(m['file_path']):
            raise ValueError('Meta file_path columns do not match across models')
        if not base['label'].equals(m['label']):
            raise ValueError('Meta label columns do not match across models')
    return base


def train(args):
    set_seed(args.seed)

    models_list = [m.strip() for m in args.models.split(',')]
    oof_dir = Path(args.oof_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # meta CSVs
    meta_paths = [oof_dir / f"{m}_oof_meta.csv" for m in models_list]
    for p in meta_paths:
        if not p.exists():
            raise FileNotFoundError(f"Missing meta CSV: {p}")

    meta = load_meta_paths(meta_paths)
    y = meta['label'].values.astype(int)

    # load oof arrays
    probs_list = []
    for m in models_list:
        p = oof_dir / f"{m}_oof_probs.npy"
        if not p.exists():
            raise FileNotFoundError(f"Missing OOF probs file: {p}")
        probs_list.append(np.load(p))

    N = probs_list[0].shape[0]
    C = probs_list[0].shape[1]
    # sanity shapes
    for arr in probs_list:
        if arr.shape[0] != N or arr.shape[1] != C:
            raise ValueError('All OOF arrays must have same (N, C) shape')

    # prepare X: concatenated probs (N, num_models*C)
    X = np.concatenate(probs_list, axis=1).astype(np.float32)
    num_models = len(probs_list)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # stratified split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    train_idx, val_idx = next(sss.split(X, y))

    X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
    y_train = torch.tensor(y[train_idx], dtype=torch.long).to(device)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
    y_val = torch.tensor(y[val_idx], dtype=torch.long).to(device)

    model = GatingNet(input_dim=X.shape[1], num_models=num_models, hidden=args.hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float('inf')
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(X_train.size(0), device=device)
        train_loss = 0.0
        for i in range(0, X_train.size(0), args.batch_size):
            idx = perm[i:i+args.batch_size]
            xb = X_train[idx]
            yb = y_train[idx]

            opt.zero_grad()
            logits = model(xb)  # (B, num_models)
            weights = F.softmax(logits, dim=1)  # (B, num_models)

            # reshape xb to (B, num_models, C)
            probs3 = xb.view(xb.size(0), num_models, C)
            weights_unsq = weights.unsqueeze(2)  # (B, num_models, 1)
            fused = (weights_unsq * probs3).sum(dim=1)  # (B, C)

            # loss: negative log-likelihood on fused probs
            eps = 1e-9
            logp = torch.log(fused + eps)
            loss = -logp[range(logp.size(0)), yb].mean()
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)

        train_loss /= X_train.size(0)

        # val
        model.eval()
        with torch.no_grad():
            logits = model(X_val)
            weights = F.softmax(logits, dim=1)
            probs3 = X_val.view(X_val.size(0), num_models, C)
            fused = (weights.unsqueeze(2) * probs3).sum(dim=1)
            preds = fused.argmax(dim=1).cpu().numpy()
            val_acc = (preds == y_val.cpu().numpy()).mean()
            eps = 1e-9
            val_loss = -torch.log(fused + eps)[range(fused.size(0)), y_val].mean().item()

        print(f"Epoch {epoch}/{args.epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # load best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Compute fused predictions for all samples (OOF fused)
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32).to(device)
        logits_all = model(Xt)
        weights_all = F.softmax(logits_all, dim=1)
        probs3_all = Xt.view(Xt.size(0), num_models, C)
        fused_all = (weights_all.unsqueeze(2) * probs3_all).sum(dim=1).cpu().numpy()
        preds_all = fused_all.argmax(axis=1)

    # save model and fused outputs
    torch.save(model.state_dict(), out_dir / 'gating_weights_mlp_v1.pt')
    np.save(out_dir / 'gating_fused_oof_probs_v1.npy', fused_all)

    # write CSV with index,file_path,true_label,pred_label,weights
    meta = meta.reset_index()
    weights_np = weights_all.cpu().numpy()
    out_csv = out_dir / 'gating_fused_oof_preds_v1.csv'
    import csv as _csv
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = _csv.writer(f)
        header = ['index', 'file_path', 'true_label', 'pred_label'] + [f'weight_{i}' for i in range(num_models)]
        w.writerow(header)
        for i, row in meta.iterrows():
            idx = int(row['index']) if 'index' in row else int(i)
            fp = row['file_path']
            lbl = int(row['label'])
            pred = int(preds_all[i])
            weights_row = weights_np[i].tolist()
            w.writerow([idx, fp, lbl, pred] + weights_row)

    print(f"Saved gating model and fused OOF preds to {out_dir}")


def cli():
    p = argparse.ArgumentParser(description='Train gating MLP from OOF arrays')
    p.add_argument('--models', default='vgg19,densenet201,mobilenet_v2')
    p.add_argument('--oof-dir', default='outputs')
    p.add_argument('--output', default='outputs')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--val-size', type=float, default=0.2)
    p.add_argument('--hidden', type=int, default=64)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    cli()
