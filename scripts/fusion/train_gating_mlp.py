#!/usr/bin/env python3
"""Train a gating MLP on OOF predictions and evaluate fused model on test set.

This is a clearer, formalized copy of the original gating trainer.
It accepts OOF arrays saved by `scripts/oof/gen_oof_predictions.py`.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import random
import torch #type: ignore
import torch.nn as nn #type: ignore
import torch.optim as optim #type: ignore
import torch.nn.functional as F #type: ignore
from torch.utils.data import DataLoader, Dataset #type: ignore
from torchvision import transforms, models  # type: ignore
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WoundTestDataset(Dataset):
    def __init__(self, meta_rows, transform=None):
        # meta_rows: list of (index, file_path, label)
        self.rows = meta_rows
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        _, fp, label = self.rows[idx]
        img = Image.open(fp).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, int(label)


val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model(model_name: str, num_classes: int):
    name = model_name.lower()
    if name == 'vgg19':
        m = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        numf = m.classifier[6].in_features
        m.classifier[6] = nn.Linear(numf, num_classes)
    elif name == 'densenet201':
        m = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif name in ('mobilenetv2', 'mobilenet_v2'):
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    else:
        raise ValueError('Unsupported model: ' + model_name)
    return m


class GatingMLP(nn.Module):
    def __init__(self, input_dim, num_classes, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def load_meta(meta_csv_path):
    rows = []
    with open(meta_csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((int(r['index']), r['file_path'], int(r['label'])))
    # sort by index
    rows.sort(key=lambda x: x[0])
    return rows


def resolve_model_dir(model_name: str, root_dir: Path) -> Path:
    model_name = model_name.strip()
    aliases = []
    aliases.extend([
        model_name,
        model_name.lower(),
        model_name.replace('_', ''),
        model_name.lower().replace('_', ''),
        f"{model_name}_clean",
        f"{model_name.lower()}_clean",
        f"{model_name.replace('_', '')}_clean",
        f"{model_name.lower().replace('_', '')}_clean",
    ])
    seen = set()
    for alias in aliases:
        if alias not in seen:
            seen.add(alias)
            cand = root_dir / alias
            if cand.is_dir():
                return cand
    matches = sorted(root_dir.glob(f"*{model_name}*"))
    for cand in matches:
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"Could not find model artifact directory for {model_name!r} under {root_dir}")


def resolve_checkpoint_dir(model_name: str, root_dir: Path) -> Path:
    model_dir = resolve_model_dir(model_name, root_dir)
    ckpt_dir = model_dir / 'checkpoints'
    if ckpt_dir.exists():
        return ckpt_dir

    matches = sorted(model_dir.glob('**/checkpoints'))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"No checkpoints directory found for model {model_name} under {root_dir}")


def average_test_probs_for_model(model_name, ckpt_dir, test_dataset, batch_size, device, num_classes):
    # Prefer the formal checkpoint naming produced by `gen_oof_preds.py`, but keep a legacy fallback for older folders.
    ckpts = sorted(ckpt_dir.glob(f"{model_name}_fold*_best.pt"))
    if len(ckpts) == 0:
        ckpts = sorted(ckpt_dir.glob(f"{model_name}_fold*_phase1_best.pt"))
    if len(ckpts) == 0:
        ckpts = sorted(ckpt_dir.glob(f"*_fold*_best.pt"))
    if len(ckpts) == 0:
        raise FileNotFoundError(f"No checkpoints found for model {model_name} in {ckpt_dir}")

    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probs_accum = None
    for ckpt in ckpts:
        model = build_model(model_name, num_classes=num_classes)
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)
        model = model.to(device)
        model.eval()
        allp = []
        with torch.no_grad():
            for images, _ in loader:
                images = images.to(device)
                p = F.softmax(model(images), dim=1).cpu().numpy()
                allp.append(p)
        allp = np.vstack(allp)
        if probs_accum is None:
            probs_accum = allp
        else:
            probs_accum += allp

    probs_accum /= float(len(ckpts))
    return probs_accum


def train_gating(X, y, input_dim, num_classes, device, epochs=50, batch_size=64, lr=1e-3, weight_decay=1e-4, patience=5):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(X, y))

    X_train = torch.tensor(X[train_idx], dtype=torch.float32).to(device)
    y_train = torch.tensor(y[train_idx], dtype=torch.long).to(device)
    X_val = torch.tensor(X[val_idx], dtype=torch.float32).to(device)
    y_val = torch.tensor(y[val_idx], dtype=torch.long).to(device)

    model = GatingMLP(input_dim, num_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    best_val = float('inf')
    epochs_no_improve = 0
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(X_train.size(0))
        running = 0.0
        for i in range(0, X_train.size(0), batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train[idx]
            yb = y_train[idx]
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            running += loss.item() * xb.size(0)
        train_loss = running / X_train.size(0)

        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_loss = crit(val_out, y_val).item()

        print(f"Gating epoch {epoch+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Gating early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def main():
    parser = argparse.ArgumentParser(description='Train gating MLP on OOF predictions')
    parser.add_argument('--models', required=True, help='Comma-separated base model names (vgg19,densenet201,mobilenet_v2)')
    parser.add_argument('--oof-dir', default='outputs', help='Directory containing *_oof_probs.npy and meta csv')
    parser.add_argument('--meta', default=None, help='Meta CSV path (if omitted, uses first model meta in oof-dir)')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--output', default='outputs')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    models_list = [m.strip() for m in args.models.split(',')]
    oof_dir = Path(args.oof_dir)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.meta:
        meta_path = Path(args.meta)
    else:
        meta_path = resolve_model_dir(models_list[0], oof_dir) / f"{models_list[0]}_oof_meta.csv"

    if not meta_path.exists():
        raise FileNotFoundError(f"Meta CSV not found: {meta_path}")

    meta_rows = load_meta(meta_path)
    y = np.array([r[2] for r in meta_rows])

    oof_list = []
    for m in models_list:
        model_dir = resolve_model_dir(m, oof_dir)
        p = model_dir / f"{m}_oof_probs.npy"
        if not p.exists():
            raise FileNotFoundError(f"OOF probs not found for model {m}: {p}")
        arr = np.load(p)
        oof_list.append(arr)

    X = np.concatenate(oof_list, axis=1)
    input_dim = X.shape[1]
    num_classes = oof_list[0].shape[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train gating model
    gating = train_gating(X, y, input_dim, num_classes, device, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_decay=args.weight_decay, patience=args.patience)

    # Save gating model (formal name)
    torch.save(gating.state_dict(), out_dir / 'gating_mlp_model_v1.pt')
    print(f"Saved gating model to: {out_dir / 'gating_mlp_model_v1.pt'}")

    # Build test rows directly from the clean processed test split.
    # Never reconstruct test paths from train metadata or copy files across splits.
    test_root = Path('data/processed/test')
    if not test_root.exists():
        raise FileNotFoundError(f"Clean processed test dir not found: {test_root}")

    class_to_idx = {cls.name: idx for idx, cls in enumerate(sorted(test_root.iterdir(), key=lambda p: p.name)) if cls.is_dir()}
    test_rows = []
    for class_dir in sorted(test_root.iterdir(), key=lambda p: p.name):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.iterdir(), key=lambda p: p.name):
            if img_path.is_file():
                test_rows.append((len(test_rows), str(img_path), class_to_idx[class_dir.name]))

    test_dataset = WoundTestDataset(test_rows, transform=val_transforms)

    # For each base model, average test probs across fold checkpoints
    test_probs_list = []
    for m in models_list:
        model_ckpt_dir = resolve_checkpoint_dir(m, oof_dir)
        probs = average_test_probs_for_model(m, model_ckpt_dir, test_dataset, batch_size=args.batch_size, device=device, num_classes=num_classes)
        test_probs_list.append(probs)

    X_test = np.concatenate(test_probs_list, axis=1)

    # Run gating on test
    gating.eval()
    with torch.no_grad():
        Xt = torch.tensor(X_test, dtype=torch.float32).to(device)
        out = gating(Xt)
        preds = torch.argmax(F.softmax(out, dim=1), dim=1).cpu().numpy()

    # Save test probs and predictions using actual clean test-set file paths only.
    np.save(out_dir / 'gating_test_probs_v1.npy', X_test)
    with open(out_dir / 'gating_test_preds_v1.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'file_path', 'true_label', 'pred_label'])
        for (idx, fp, lbl), p in zip(test_rows, preds):
            writer.writerow([idx, fp, lbl, int(p)])

    print(f"Saved gating test probs and preds to {out_dir}")


if __name__ == '__main__':
    main()
