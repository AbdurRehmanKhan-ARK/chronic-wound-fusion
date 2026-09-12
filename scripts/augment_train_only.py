from pathlib import Path
from PIL import Image, ImageOps, ImageEnhance
import random
import os

ROOT = Path("data/processed/train")
TARGET_PER_CLASS = {
    "background": 350,
    "diabetic": 350,
    "normal-skin": 350,
    "pressure": 350,
    "surgical": 350,
    "venous": 350,
}

random.seed(42)

def augment_image(img: Image.Image):
    # Random horizontal/vertical flip
    if random.random() < 0.5:
        img = ImageOps.mirror(img)
    if random.random() < 0.5:
        img = ImageOps.flip(img)

    # Random rotation
    angle = random.uniform(-20, 20)
    img = img.rotate(angle, resample=Image.BILINEAR, expand=False)

    # Random brightness/contrast
    brightness = random.uniform(0.8, 1.2)
    contrast = random.uniform(0.8, 1.2)
    img = ImageEnhance.Brightness(img).enhance(brightness)
    img = ImageEnhance.Contrast(img).enhance(contrast)

    # Random crop-like zoom effect by resizing then center-cropping
    width, height = img.size
    new_w = int(width * random.uniform(0.92, 1.0))
    new_h = int(height * random.uniform(0.92, 1.0))
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    left = max(0, (new_w - width) // 2)
    top = max(0, (new_h - height) // 2)
    cropped = resized.crop((left, top, left + width, top + height))
    return cropped

def safe_save(source: Path, class_dir: Path, idx: int):
    stem = source.stem
    ext = source.suffix
    out_name = f"aug_{idx:04d}_{stem}{ext}"
    out_path = class_dir / out_name
    if out_path.exists():
        return
    img = Image.open(source).convert("RGB")
    aug = augment_image(img)
    aug.save(out_path)

def main():
    if not ROOT.exists():
        raise FileNotFoundError(f"Train root not found: {ROOT}")

    for class_dir in sorted(ROOT.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        originals = sorted(
            p for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        )

        current_count = len(originals)
        target = TARGET_PER_CLASS.get(class_name, current_count)

        if current_count >= target:
            print(f"[SKIP] {class_name}: already at {current_count}, target {target}")
            continue

        generated = 0
        idx = 0

        while current_count + generated < target:
            for src in originals:
                if current_count + generated >= target:
                    break
                safe_save(src, class_dir, idx)
                generated += 1
                idx += 1

        print(f"[OK] {class_name}: originals={current_count}, augmented={generated}, total={current_count + generated}")

    print("Augmentation finished for TRAIN only. VAL and TEST were not modified.")

if __name__ == "__main__":
    main()