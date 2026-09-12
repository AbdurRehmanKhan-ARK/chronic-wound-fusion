"""
This is the first real training script for the wound project.

What this script does:
1. Loads the small subset dataset created by preprocess.py
2. Uses a pre-trained VGG19 model
3. Trains it on the 6 wound classes
4. Prints training and validation loss/accuracy
5. Saves the trained model checkpoint

Why this is the right first step:
- The paper uses VGG19, DenseNet201, MobileNetV2
- We do one model first so we understand the pipeline
- After this works, we add the other models and then fusion

IMPORTANT:
- Run this using Python 3.11
- Use the virtual environment we created
- This is for the small subset first, not full dataset
"""

import os
import random
from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from torch.utils.data import DataLoader, Dataset # type: ignore
from torchvision import models, transforms  # type: ignore
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


   
# 1. Set seeds so results are reproducible
   
def set_seed(seed=42):
    """Fix random seeds so training results are consistent.

    This helps because deep learning can be unpredictable if random values
    change every run.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(42)

# ---- CLI args (allow running Phase 1 or Phase 2 without editing file) ----
parser = argparse.ArgumentParser(description="Train VGG19 on small wound subset")
parser.add_argument("--phase", type=int, choices=[1, 2], default=None,
                    help="Training phase: 1=frozen backbone, 2=unfreeze last block")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to run")
parser.add_argument("--batch-size", type=int, default=8, help="Batch size for training")
parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate for Phase 1")
parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay (L2)")
parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
args = parser.parse_args()



   
# 2. Define the dataset class
   
class WoundDataset(Dataset):
    """
    Simple dataset class for image folders.

    Structure expected:
        data/processed/train/background/...
        data/processed/train/diabetic/...
        data/processed/train/normal-skin/...
        ...
    """

    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []

        # Loop through each class folder and collect all images
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


   
# 3. Training and validation transforms
   
train_transforms = transforms.Compose([
    # Resize image to 224 x 224 to match the paper
    transforms.Resize((224, 224)),

    # Random augmentation to help model generalize
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.2),
    transforms.RandomRotation(15),

    # Convert PIL image to tensor
    transforms.ToTensor(),

    # Normalize with ImageNet stats
    # This is standard and helps pretrained model work better
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

   
# 4. Paths to data
   
# These folders were created by preprocess.py
train_dir = Path("data/processed/train")
val_dir = Path("data/processed/val")
test_dir = Path("data/processed/test")

if not train_dir.exists():
    raise FileNotFoundError(
        "Training folder not found. First run preprocess.py to create train/val/test folders."
    )

   
# 5. Build datasets
   
train_dataset = WoundDataset(train_dir, transform=train_transforms)
val_dataset = WoundDataset(val_dir, transform=val_transforms)

# Check class mapping for understanding
print("Classes found:", train_dataset.classes)
print("Number of training images:", len(train_dataset))
print("Number of validation images:", len(val_dataset))

   
# 6. Data loaders
   
batch_size = args.batch_size  # small batch because CPU and small dataset

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

   
# 7. Load pre-trained VGG19
   
# The paper uses VGG19 as one of the base models.
# We use pretrained ImageNet weights because transfer learning helps
# a lot when the medical dataset is small.
model = models.vgg19(weights="IMAGENET1K_V1")

# ---------------------------------------------------------------------
# TRAINING STRATEGY: staged transfer learning
# ---------------------------------------------------------------------
# We will run the model in two phases:
#   Phase 1: freeze backbone, train only classifier head
#   Phase 2: unfreeze the last VGG block and continue with a lower LR
#
# Why this is necessary:
#   - Our dataset is tiny (84 train, 18 val)
#   - Full VGG19 fine-tuning collapsed to one class in the earlier run
#   - The safe method is to first stabilize the final decision layer,
#     then allow the last feature extractor to adapt gently.
#
# Phase 1 already succeeded: validation accuracy reached 0.50 and the model
# stopped collapsing to one class. Now we move to Phase 2.
#
# Set phase = 1 for the initial run, and phase = 2 only after Phase 1 shows
# meaningful learning.
#
# We are now doing a controlled fine-tuning step:
#   - unfreeze only the last VGG block
#   - keep the rest of the backbone frozen
#   - lower the learning rate so the pretrained features are adapted gently
if args.phase is None:
    TRAIN_PHASE = 2
else:
    TRAIN_PHASE = args.phase

print(f"Selected TRAIN_PHASE = {TRAIN_PHASE}")

# Backbone vs Head:
#   - Backbone = model.features (feature extractor)
#   - Head = model.classifier (final decision layers)
#
# For Phase 1, we keep all backbone weights frozen. This means the pretrained
# features remain stable while the new 6-class classifier learns to use them.
# For Phase 2, only the last conv block is unfrozen, and we reduce the LR to
# avoid destroying the useful pretrained representation too quickly.

# Freeze all backbone layers before replacing the classifier head.
for param in model.features.parameters():
    param.requires_grad = False

# Replace the last fully connected layer with 6 output classes
# because our task is 6-class classification.
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, len(train_dataset.classes))

# Move model to CPU for now because no GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Print number of trainable parameters to confirm the freeze worked.
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters after freezing backbone: {trainable_params}")

# ---------------------------------------------------------------------
# Phase 1 details
# ---------------------------------------------------------------------
# In Phase 1, the backbone is frozen and only the final classifier is trained.
# This is the first run to check if the model can learn the wound classes
# without collapsing to one class.
#
# If Phase 1 reaches a good validation score, then move to Phase 2.
#
# Phase 2 details
# ---------------------------------------------------------------------
# In Phase 2, we unfreeze only the last convolutional block and reduce the
# learning rate to a smaller value. This lets the model adapt to the wound
# dataset without overriding the good ImageNet features too aggressively.
#
# This is the step after the classifier learns a stable mapping.
#
# To switch phases:
#   TRAIN_PHASE = 2
#   then unfreeze the last block before creating the optimizer.

# 8. Loss function and optimizer
   
criterion = nn.CrossEntropyLoss()

# Phase 1: optimizer updates only trainable params.
# Phase 2: after unfreezing, optimizer will be recreated with the new set of
# trainable parameters and a smaller learning rate.
if TRAIN_PHASE == 1:
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    print("Running Phase 1: backbone frozen, classifier head training only.")
else:
    # Phase 2: unfreeze only the last convolutional block.
    # This is intentionally limited because the dataset is still small.
    # We do NOT unfreeze the whole VGG19 backbone at once.
    #
    # Reasoning:
    #   - Phase 1 proved the classifier can learn from pretrained features.
    #   - Now we allow the last feature block to adapt to wound images.
    #   - The smaller LR prevents the model from overwriting useful ImageNet 
    #     knowledge too aggressively.
    for param in model.features[-1].parameters():
        param.requires_grad = True

    phase_lr = args.lr / 10.0
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=phase_lr,
        weight_decay=args.weight_decay
    )
    print("Running Phase 2: last VGG block unfrozen, lower learning rate.")

   
# 9. Training loop
   
# We keep this modest for now because the goal is a stable first phase.
# The experiment is meant to prove the classifier can learn before we unfreeze
# the backbone and risk destabilizing the model again.
num_epochs = args.epochs

# Early stopping setup
best_val_loss = float('inf')
epochs_no_improve = 0
best_ckpt = Path("checkpoints/vgg19_best.pt")

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
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

    # --------------------------------------------------------
    # Validation step
    # --------------------------------------------------------
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

    print(f"Epoch {epoch + 1}/{num_epochs}")
    print(f"  Train loss: {train_loss:.4f}")
    print(f"  Val loss:   {val_loss:.4f}")
    print(f"  Val acc:    {val_acc:.4f}")

    # Early stopping and save best
    os.makedirs("checkpoints", exist_ok=True)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), best_ckpt)
        print(f"  Best model saved to: {best_ckpt}")
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= args.patience:
        print(f"Early stopping: no improvement in {args.patience} epochs. Stopping.")
        break

   
# 10. Save model checkpoint
   
os.makedirs("checkpoints", exist_ok=True)
final_ckpt = Path("checkpoints/vgg19_small_subset.pt")
if best_ckpt.exists():
    best_ckpt.replace(final_ckpt)
    print(f"\nBest model moved to: {final_ckpt}")
else:
    torch.save(model.state_dict(), final_ckpt)
    print(f"\nModel saved to: {final_ckpt}")

   
# 11. Final validation evaluation
   
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

acc = accuracy_score(all_labels, all_preds)
cm = confusion_matrix(all_labels, all_preds)

print("\nFinal validation results:")
print("Accuracy:", acc)
print("Confusion Matrix:")
print(cm)
print(classification_report(all_labels, all_preds, target_names=train_dataset.classes))

   
# 12. Plot training curves
   
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train loss")
plt.plot(val_losses, label="Val loss")
plt.title("Loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label="Val accuracy")
plt.title("Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
os.makedirs("results/figures", exist_ok=True)
save_path = "results/figures/vgg19_small_subset_training.png"
plt.savefig(save_path)
plt.close()

print(f"\nTraining curves saved to: {save_path}")