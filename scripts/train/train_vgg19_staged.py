#!/usr/bin/env python3
# Formal copy of `scripts/train_vgg_small.py` renamed to a descriptive path.
import os
import random
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms  # type: ignore
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WoundDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for img_file in cls_dir.iterdir():
                if img_file.is_file():
                    self.samples.append((img_file, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def main():
    parser = argparse.ArgumentParser(description="Train VGG19 (staged transfer learning)")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    set_seed(42)

    train_dir = Path("data/processed/train")
    val_dir = Path("data/processed/val")

    if not train_dir.exists():
        raise FileNotFoundError("Training folder not found. Run preprocess.py first.")

    train_dataset = WoundDataset(train_dir, transform=train_transforms)
    val_dataset = WoundDataset(val_dir, transform=val_transforms)

    print("Classes found:", train_dataset.classes)
    print("Number of training images:", len(train_dataset))
    print("Number of validation images:", len(val_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    num_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_features, len(train_dataset.classes))

    if args.phase is None:
        TRAIN_PHASE = 2
    else:
        TRAIN_PHASE = args.phase

    print(f"Selected TRAIN_PHASE = {TRAIN_PHASE}")

    for param in model.features.parameters():
        param.requires_grad = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    if TRAIN_PHASE == 1:
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        print("Running Phase 1: backbone frozen, classifier head training only.")
    else:
        for param in model.features[-1].parameters():
            param.requires_grad = True
        phase_lr = args.lr / 10.0
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=phase_lr, weight_decay=args.weight_decay)
        print("Running Phase 2: last VGG block unfrozen, lower learning rate.")

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_ckpt = Path("checkpoints/vgg19_best.pt")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(all_labels, all_preds)

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss:   {val_loss:.4f}")
        print(f"  Val acc:    {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), best_ckpt)
            print(f"  Best model saved to: {best_ckpt}")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= args.patience:
            print(f"Early stopping: no improvement in {args.patience} epochs. Stopping training.")
            break

    final_ckpt = Path(f"checkpoints/vgg19_small_subset.pt")
    if best_ckpt.exists():
        best_ckpt.replace(final_ckpt)
        print(f"\nBest model moved to: {final_ckpt}")
    else:
        torch.save(model.state_dict(), final_ckpt)
        print(f"\nModel saved to: {final_ckpt}")


if __name__ == '__main__':
    main()
