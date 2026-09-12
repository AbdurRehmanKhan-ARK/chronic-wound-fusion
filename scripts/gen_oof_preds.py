#!/usr/bin/env python3
"""Generate OOF probability predictions with K-fold CV for base models.

Saves per-model OOF probability arrays and metadata (file paths + labels).

Usage example:
  .\.venv\Scripts\python.exe scripts\gen_oof_preds.py --model vgg19 --folds 5 \
    --phase1-epochs 8 --phase2-epochs 4 --batch-size 16 --lr 1e-4 --weight-decay 1e-4

This script implements staged transfer learning per-fold:
  Phase1: freeze backbone, train classifier head only.
  Phase2: unfreeze last feature block, fine-tune with LR/10.

OOF outputs:
  outputs/{model}_oof_probs.npy   # shape (N, C)
  outputs/{model}_oof_meta.csv    # index,file_path,label
  outputs/checkpoints/{model}_fold{fold}_best.pt
"""
import argparse
import random
import re
from pathlib import Path
import csv
import os

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

import torch 
import torch.nn as nn #type: ignore
import torch.optim as optim #type: ignore
import torch.nn.functional as F #type: ignore
from torch.utils.data import DataLoader, Subset #type: ignore
from torchvision import models, transforms #type: ignore
from PIL import Image


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) #type: ignore
    if torch.cuda.is_available(): #type: ignore
        torch.cuda.manual_seed_all(seed) #type: ignore


class WoundDataset(torch.utils.data.Dataset): #type: ignore
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for p in sorted(cls_dir.iterdir()):
                if p.is_file():
                    self.samples.append((p, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


class WoundDatasetView(torch.utils.data.Dataset): #type: ignore
    def __init__(self, samples, transform=None):
        self.samples = list(samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_group_id(filepath):
    """Map augmented train names back to their original source group.

    Examples from the current augmentation script:
        '10.jpg' -> '10'
        'aug_0000_10.jpg' -> '10'
        'aug_0001_10.png' -> '10'
    """
    path = Path(filepath)
    stem = path.stem
    stem = re.sub(r"^aug_\d+_", "", stem)
    stem = re.sub(r"(?:_aug\d+|_flip|_rot\d+)$", "", stem)
    return stem


train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def build_model(model_name: str, num_classes: int):
    name = model_name.lower()
    if name == 'vgg19':
        model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        # replace last classifier
        numf = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(numf, num_classes)
    elif name == 'densenet201':
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name in ('mobilenetv2', 'mobilenet_v2'):
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model


def freeze_backbone(model, model_name):
    name = model_name.lower()
    if name == 'vgg19':
        for p in model.features.parameters():
            p.requires_grad = False
    else:
        # densenet and mobilenet: common attribute names
        if hasattr(model, 'features'):
            for p in model.features.parameters():
                p.requires_grad = False


def unfreeze_last_block(model, model_name):
    name = model_name.lower()
    # Attempt to unfreeze the last child module of features
    if hasattr(model, 'features'):
        children = list(model.features.children())
        if len(children) > 0:
            last = children[-1]
            for p in last.parameters():
                p.requires_grad = True
            return
    # fallback: unfreeze all
    for p in model.parameters():
        p.requires_grad = True


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running += loss.item()
    return running / len(loader)


def eval_loss_and_probs(model, loader, criterion, device):
    model.eval()
    running = 0.0
    all_probs = []
    all_labels = []
    with torch.no_grad(): #type: ignore
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running += loss.item()
            probs = F.softmax(outputs, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    all_probs = np.vstack(all_probs) if len(all_probs) > 0 else np.zeros((0, model.classifier[-1].out_features if hasattr(model, 'classifier') else 0))
    return running / len(loader), all_probs, np.array(all_labels)


def main():
    parser = argparse.ArgumentParser(description="Generate OOF predictions via K-fold CV")
    parser.add_argument('--model', required=True, choices=['vgg19', 'densenet201', 'mobilenet_v2'])
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--phase1-epochs', type=int, default=8)
    parser.add_argument('--phase2-epochs', type=int, default=4)
    parser.add_argument('--phase2', action='store_true', help='If set, run Phase-2 (unfreeze last block). Default: False (skip Phase-2)')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--output', type=str, default='outputs')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    train_dir = Path('data/processed/train')
    if not train_dir.exists():
        raise FileNotFoundError('Augmented train folder not found at data/processed/train')

    dataset = WoundDataset(train_dir, transform=train_transforms)
    N = len(dataset)
    num_classes = len(dataset.classes)
    print(f"Dataset: {N} samples, {num_classes} classes")

    # Prepare outputs
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Meta: write index,file,label mapping
    meta_path = out_dir / f"{args.model}_oof_meta.csv"
    with open(meta_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'file_path', 'label'])
        for idx, (p, lbl) in enumerate(dataset.samples):
            writer.writerow([idx, str(p), lbl])

    # Prepare OOF storage
    oof_probs = np.zeros((N, num_classes), dtype=np.float32)

    # Group-aware stratified K-fold to avoid augmented siblings leaking between train/val.
    y = np.array([lbl for (_, lbl) in dataset.samples])
    groups = np.array([get_group_id(str(path)) for (path, _) in dataset.samples])
    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#type: ignore

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(np.zeros(N), y, groups)):
        print(f"\n=== Fold {fold + 1}/{args.folds} — train {len(train_idx)} val {len(val_idx)} ===")
        train_samples = [dataset.samples[i] for i in train_idx]
        val_samples = [dataset.samples[i] for i in val_idx]

        train_subset = WoundDatasetView(train_samples, transform=train_transforms)
        val_subset = WoundDatasetView(val_samples, transform=val_transforms)

        train_loader = DataLoader(train_subset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = build_model(args.model, num_classes)
        # Phase 1: freeze backbone
        freeze_backbone(model, args.model)

        model = model.to(device)

        criterion = nn.CrossEntropyLoss()

        # Phase 1 optimizer
        opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

        best_val_loss = float('inf')
        epochs_no_improve = 0
        best_state = None

            # Phase 1 training
        for epoch in range(args.phase1_epochs):
            tr_loss = train_one_epoch(model, train_loader, opt, criterion, device)
            val_loss, _, _ = eval_loss_and_probs(model, val_loader, criterion, device)
            print(f"[Fold {fold+1}] Phase1 Epoch {epoch+1}/{args.phase1_epochs}  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Phase1 early stopping at epoch {epoch+1}")
                break

        # Load best Phase1 weights
        if best_state is not None:
            model.load_state_dict(best_state)

        # Optionally run Phase 2 (unfreeze last block). Default: skip Phase-2 to save time
        final_state = None
        if args.phase2:
            print(f"Fold {fold+1}: running Phase-2 because --phase2 was set")
            unfreeze_last_block(model, args.model)
            phase2_lr = args.lr / 10.0
            opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=phase2_lr, weight_decay=args.weight_decay)

            # reset early stopping for Phase2
            best_val_loss_phase2 = float('inf')
            epochs_no_improve = 0
            best_state2 = None

            for epoch in range(args.phase2_epochs):
                tr_loss = train_one_epoch(model, train_loader, opt, criterion, device)
                val_loss, _, _ = eval_loss_and_probs(model, val_loader, criterion, device)
                print(f"[Fold {fold+1}] Phase2 Epoch {epoch+1}/{args.phase2_epochs}  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}")
                if val_loss < best_val_loss_phase2:
                    best_val_loss_phase2 = val_loss
                    epochs_no_improve = 0
                    best_state2 = {k: v.cpu() for k, v in model.state_dict().items()}
                else:
                    epochs_no_improve += 1
                if epochs_no_improve >= args.patience:
                    print(f"Phase2 early stopping at epoch {epoch+1}")
                    break

            final_state = best_state2 if best_state2 is not None else best_state
            if final_state is not None:
                model.load_state_dict(final_state)
        else:
            print(f"Fold {fold+1}: skipping Phase-2 (use --phase2 to enable)")
            final_state = best_state

        # Save fold checkpoint
        ckpt_path = ckpt_dir / f"{args.model}_fold{fold+1}_best.pt"
        torch.save(model.state_dict(), ckpt_path) #type: ignore
        print(f"Saved fold checkpoint: {ckpt_path}")

        # Predict probabilities on validation set and fill OOF
        model.eval()
        all_probs = []
        with torch.no_grad(): #type: ignore
            for images, _ in val_loader:
                images = images.to(device)
                probs = F.softmax(model(images), dim=1)
                all_probs.append(probs.cpu().numpy())
        if len(all_probs) > 0:
            all_probs = np.vstack(all_probs)
        else:
            all_probs = np.zeros((0, num_classes), dtype=np.float32)

        # assign into oof array
        oof_probs[val_idx, :] = all_probs

    # Save OOF arrays
    oof_path = out_dir / f"{args.model}_oof_probs.npy"
    np.save(oof_path, oof_probs)
    print(f"Saved OOF probabilities to: {oof_path}")


if __name__ == '__main__':
    main()
