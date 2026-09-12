"""Train a pretrained DenseNet201 or MobileNetV2 backbone on the small AZH subset.

Why this file exists:
- The previous VGG19 experiments showed that full fine-tuning is too aggressive
  for the tiny dataset.
- We now compare other pretrained backbones using the same small subset.
- We keep the same safety pattern: freeze backbone first, train only the head.

This is the right next step before deciding whether more data is needed.
"""

import argparse
import random
from pathlib import Path
import os

import numpy as np
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset # type: ignore
from torchvision import models, transforms  # type: ignore


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class WoundDataset(Dataset):
    """Simple dataset wrapper for class folders in data/processed/train/ or val/."""

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


def load_model(model_name: str, num_classes: int):
    """Load a pretrained backbone and replace the final classifier.

    We do NOT do full fine-tuning yet. The backbone is frozen at first so the
    model can learn a stable classifier on a tiny dataset.
    """
    model_name = model_name.lower()
    if model_name == "densenet201":
        model = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "mobilenetv2" or model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}. Use densenet201 or mobilenet_v2.")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DenseNet201 or MobileNetV2 on the small wound subset.")
    parser.add_argument("--model", choices=["densenet201", "mobilenet_v2"], required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Weight decay (L2) regularization for optimizer")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--phase", type=int, choices=[1, 2], default=1,
                        help="Training phase: 1=freeze backbone (train head), 2=unfreeze last block (fine-tune)")
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

    model = load_model(args.model, len(train_dataset.classes))

    # Phase handling: Phase 1 = freeze backbone and train only head.
    # Phase 2 = unfreeze only the last block of the backbone and fine-tune
    # with a lower learning rate.
    if args.phase == 1:
        # freeze all backbone parameters
        for param in model.features.parameters():
            param.requires_grad = False
        phase_lr = args.lr
        print("Phase 1: backbone frozen — training classifier head only.")
    else:
        # Start by freezing everything, then unfreeze the last feature block
        # to allow controlled fine-tuning.
        for param in model.features.parameters():
            param.requires_grad = False
        # get children of features and unfreeze the last child module
        feat_children = list(model.features.children())
        if len(feat_children) > 0:
            last_block = feat_children[-1]
            for param in last_block.parameters():
                param.requires_grad = True
            print("Phase 2: last feature block unfrozen — fine-tuning last block + head.")
        else:
            # fallback: unfreeze all if structure unexpected
            for param in model.features.parameters():
                param.requires_grad = True
            print("Phase 2: could not identify last block — unfreezing full backbone.")
        # use a smaller LR for fine-tuning
        phase_lr = args.lr / 10.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}")
    print(f"Trainable parameters after phase setup: {trainable_params}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=phase_lr,
                           weight_decay=args.weight_decay)

    train_losses = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_ckpt = Path(f"checkpoints/{args.model}_best.pt")

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
        train_losses.append(train_loss)

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
        val_losses.append(val_loss)

        val_acc = accuracy_score(all_labels, all_preds)
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"  Train loss: {train_loss:.4f}")
        print(f"  Val loss:   {val_loss:.4f}")
        print(f"  Val acc:    {val_acc:.4f}")

        # Early stopping / save best
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

    # If best checkpoint exists, keep it as the primary saved model, otherwise save final
    final_ckpt = Path(f"checkpoints/{args.model}_small_subset.pt")
    if best_ckpt.exists():
        best_ckpt.replace(final_ckpt)
        print(f"\nBest model moved to: {final_ckpt}")
    else:
        torch.save(model.state_dict(), final_ckpt)
        print(f"\nModel saved to: {final_ckpt}")

    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    final_acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    print("\nFinal validation results:")
    print(f"Accuracy: {final_acc}")
    print("Confusion Matrix:")
    print(cm)
    print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))
