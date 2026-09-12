#!/usr/bin/env python3
"""Offline augmentation: generate augmented images per class to reach target counts.

Usage examples:
  python scripts/augment_and_save.py --input data/processed/train --output data/processed_aug/train --targets 350 --quality 90

This script only augments the TRAIN split. Validation and test must remain unchanged.
"""
from pathlib import Path
import argparse
import random
from PIL import Image, ImageEnhance, ImageFilter
import shutil


def random_resized_crop(img: Image.Image, size=(224,224), scale=(0.8,1.0)):
    w, h = img.size
    scale_factor = random.uniform(scale[0], scale[1])
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    if new_w >= w or new_h >= h:
        return img.resize(size, Image.BILINEAR)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    crop = img.crop((left, top, left + new_w, top + new_h))
    return crop.resize(size, Image.BILINEAR)


def color_jitter(img: Image.Image, brightness=0.2, contrast=0.2, saturation=0.15):
    if brightness > 0:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(1.0 + random.uniform(-brightness, brightness))
    if contrast > 0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.0 + random.uniform(-contrast, contrast))
    if saturation > 0:
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(1.0 + random.uniform(-saturation, saturation))
    return img


def random_augment(img: Image.Image, size=(224,224)):
    # Apply a sequence of light, medically-plausible augmentations
    img = random_resized_crop(img, size=size, scale=(0.85, 1.0))
    if random.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < 0.15:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    if random.random() < 0.7:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, resample=Image.BILINEAR)
    img = color_jitter(img, brightness=0.12, contrast=0.12, saturation=0.08)
    if random.random() < 0.15:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
    return img


def ensure_output_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description="Offline augmentation to reach target images per class")
    parser.add_argument("--input", required=True, help="Input train folder (data/processed/train)")
    parser.add_argument("--output", required=True, help="Output train folder for augmented data")
    parser.add_argument("--targets", type=int, default=350, help="Target images per class after augmentation")
    parser.add_argument("--quality", type=int, default=90, help="JPEG quality for saved images (1-95)")
    parser.add_argument("--size", type=int, default=224, help="Output image size (square)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    inp = Path(args.input)
    out = Path(args.output)

    if not inp.exists():
        raise FileNotFoundError(f"Input folder not found: {inp}")

    ensure_output_dir(out)

    class_dirs = sorted([d for d in inp.iterdir() if d.is_dir()])
    summary = {}
    for class_dir in class_dirs:
        cls = class_dir.name
        src_files = sorted([p for p in class_dir.iterdir() if p.is_file()])
        n_src = len(src_files)
        target = args.targets

        dest_dir = out / cls
        ensure_output_dir(dest_dir)

        # Copy originals first
        for p in src_files:
            shutil.copy2(p, dest_dir / p.name)

        # Determine how many augmented images to create
        if n_src >= target:
            created = 0
        else:
            created = target - n_src

        print(f"Class {cls}: source={n_src}, target={target}, to_create={created}")

        # Create augmented images by sampling originals randomly
        for i in range(created):
            src = random.choice(src_files)
            try:
                img = Image.open(src).convert('RGB')
            except Exception as e:
                print(f"Warning: could not open {src}: {e}")
                continue
            aug = random_augment(img, size=(args.size, args.size))
            out_name = f"aug_{i:04d}_{src.stem}.jpg"
            out_path = dest_dir / out_name
            aug.save(out_path, format='JPEG', quality=args.quality)

        summary[cls] = (n_src, target)

    print("\nAugmentation complete. Summary:")
    total_before = sum(v[0] for v in summary.values())
    total_after = sum(v[1] for v in summary.values())
    for cls, (before, after) in summary.items():
        print(f"  {cls}: {before} -> {after}")
    print(f"Total train images: {total_before} -> {total_after}")


if __name__ == '__main__':
    main()
