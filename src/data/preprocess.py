"""Preprocessing utilities for the chronic wound classification project.

This file does three things:
1. Verifies that the dataset folder structure is correct.
2. Creates a small subset for fast experimentation.
3. Builds PyTorch transforms for training and validation/test images.

The preprocessing follows the paper closely:
- resize all images to 224 x 224
- normalize images to [0, 1] or ImageNet normalization
- apply augmentation on training data: rotation, flipping, zoom, brightness

This script is intentionally beginner-friendly and has clear comments.
"""

from __future__ import annotations

import random
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch # type: ignore
from PIL import Image
from torchvision import transforms  # type: ignore


 
# Global configuration
 
SEED = 42
IMAGE_SIZE = (224, 224)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


 
# Utility functions
 
def set_seed(seed: int = SEED) -> None:
    """Set all random seeds for reproducibility.

    This is important because image augmentation and train/val split order
    should be deterministic when you rerun experiments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_class_dirs(root_dir: str | Path) -> List[Path]:
    """Return sorted class folders found inside the dataset root.

    Example:
        root_dir = data/raw/azh/wound_classification-main/data/ROI
        class dirs = [background, diabetic, normal-skin, pressure, surgical, venous]
    """
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    class_dirs = sorted([p for p in root.iterdir() if p.is_dir()])
    if not class_dirs:
        raise ValueError(f"No class directories were found inside {root}")

    return class_dirs


def get_class_name_from_path(path: Path) -> str:
    """Return the folder name such as 'diabetic' or 'venous'."""
    return path.name


def split_files(files: List[Path], val_ratio: float = VAL_RATIO, test_ratio: float = TEST_RATIO) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split a list of files into train / val / test.

    This function keeps the split simple and understandable for beginners.
    It is not a stratified sklearn split, but it is enough for this project.
    """
    if len(files) < 3:
        raise ValueError("Each class needs at least 3 images to create train/val/test sets.")

    random.shuffle(files)

    total = len(files)
    val_count = max(1, round(total * val_ratio))
    test_count = max(1, round(total * test_ratio))

    # Keep room for train.
    remaining = total - val_count - test_count
    if remaining <= 0:
        remaining = max(1, total - val_count - test_count)

    train_files = files[:remaining]
    val_files = files[remaining:remaining + val_count]
    test_files = files[remaining + val_count:]

    return train_files, val_files, test_files


def create_small_subset_dataset(
    source_root: str | Path,
    output_root: str | Path,
    images_per_class: int = 20,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
) -> None:
    """Create a small subset dataset for fast experimentation.

    Example:
        source_root = data/raw/azh/wound_classification-main/data/ROI
        output_root = data/processed

    This function copies only a limited number of images per class into:
        data/processed/train/<class>/
        data/processed/val/<class>/
        data/processed/test/<class>/

    The goal is to allow model testing before full training.
    """
    source_root = Path(source_root)
    output_root = Path(output_root)

    class_dirs = list_class_dirs(source_root)

    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = sorted(class_dir.iterdir())
        image_files = [p for p in image_files if p.is_file()]

        # If the class has more images than the limit, keep a subset.
        if images_per_class is not None and len(image_files) > images_per_class:
            random.shuffle(image_files)
            image_files = image_files[:images_per_class]

        train_files, val_files, test_files = split_files(image_files, val_ratio=val_ratio, test_ratio=test_ratio)

        for split_name, files in {
            "train": train_files,
            "val": val_files,
            "test": test_files,
        }.items():
            dest_dir = output_root / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for image_path in files:
                # Use copy2 so metadata is preserved as much as possible.
                shutil.copy2(image_path, dest_dir / image_path.name)

    print(f"Small subset dataset created successfully at: {output_root}")


 
# Data transforms
 
def get_train_transforms() -> transforms.Compose:
    """Return augmentation pipeline used for training images.

    This matches the paper's preprocessing idea closely:
    - resize to 224x224
    - random rotation
    - horizontal and vertical flips
    - brightness adjustment
    - convert to tensor
    - normalize with ImageNet statistics
    """
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.RandomRotation(15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_eval_transforms() -> transforms.Compose:
    """Return preprocessing pipeline for validation and test data.

    We do not apply random augmentations here because validation/test data must
    be stable and reproducible.
    """
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def verify_dataset_structure(root_dir: str | Path) -> None:
    """Print class counts to confirm the dataset structure is correct.

    This is useful before training because it confirms the target classes exist:
    background, diabetic, normal-skin, pressure, surgical, venous
    """
    class_dirs = list_class_dirs(root_dir)
    for class_dir in class_dirs:
        file_count = len([p for p in class_dir.iterdir() if p.is_file()])
        print(f"{class_dir.name}: {file_count} images")


 
# Entry point
 
if __name__ == "__main__":
    set_seed(SEED)

    # Default dataset path for the extended AZH folder.
    source_dir = Path("data/raw/azh/wound_classification-main/data/ROI")
    output_dir = Path("data/processed")

    # We are moving to a larger balanced subset because the earlier tiny subset
    # was too small to reliably compare models and confirm stable learning.
    # For the next stage, 50 images per class gives a more meaningful validation
    # signal without yet jumping to the full dataset.
    create_small_subset_dataset(
        source_root=source_dir,
        output_root=output_dir,
        images_per_class=50,
        val_ratio=0.15,
        test_ratio=0.15,
    )

    # Print class counts so you know the data is correct.
    print("\nDataset verification:")
    verify_dataset_structure(source_dir)

    print("\nSmall subset dataset summary:")
    for split_name in ["train", "val", "test"]:
        split_dir = output_dir / split_name
        if split_dir.exists():
            print(f"--- {split_name} ---")
            for class_dir in sorted(split_dir.iterdir()):
                if class_dir.is_dir():
                    print(f"{class_dir.name}: {len(list(class_dir.iterdir()))} images")

    print("\nImportant: we can increase images_per_class when ready for full training")
    print("Example: images_per_class=50 or 100 for a larger experiment.")