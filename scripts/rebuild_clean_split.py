from pathlib import Path
import shutil
import random

# ----------------------------
# CONFIG
# ----------------------------
RAW_ROOT = Path("data/raw/azh/wound_classification-main/data/ROI")
OUT_ROOT = Path("data/processed")
SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def list_class_files(class_dir: Path):
    return sorted(
        [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}]
    )

def copy_files(files, dest_dir: Path):
    ensure_dir(dest_dir)
    for src in files:
        dst = dest_dir / src.name
        shutil.copy2(src, dst)

# ----------------------------
# Main
# ----------------------------
def main():
    random.seed(SEED)

    if not RAW_ROOT.exists():
        raise FileNotFoundError(f"RAW_ROOT not found: {RAW_ROOT}")

    # clear any old processed folders before creating fresh split
    for split_name in ["train", "val", "test"]:
        split_dir = OUT_ROOT / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)

    ensure_dir(OUT_ROOT)

    class_dirs = sorted([p for p in RAW_ROOT.iterdir() if p.is_dir()])
    if not class_dirs:
        raise FileNotFoundError(f"No class folders found under {RAW_ROOT}")

    for class_dir in class_dirs:
        cls_name = class_dir.name
        files = list_class_files(class_dir)
        if not files:
            print(f"[WARN] No images in class: {cls_name}")
            continue

        random.shuffle(files)

        n_total = len(files)
        n_train = int(round(n_total * TRAIN_RATIO))
        n_val = int(round(n_total * VAL_RATIO))
        n_test = n_total - n_train - n_val

        # safety: make sure no split is empty
        if n_train <= 0:
            n_train = max(1, n_total // 2)
        if n_val <= 0:
            n_val = 1
        if n_test <= 0:
            n_test = max(1, n_total - n_train - n_val)

        # enforce exact total
        if n_train + n_val + n_test != n_total:
            # fix last element
            diff = n_total - (n_train + n_val + n_test)
            n_test += diff

        train_files = files[:n_train]
        val_files = files[n_train:n_train + n_val]
        test_files = files[n_train + n_val:n_train + n_val + n_test]

        copy_files(train_files, OUT_ROOT / "train" / cls_name)
        copy_files(val_files, OUT_ROOT / "val" / cls_name)
        copy_files(test_files, OUT_ROOT / "test" / cls_name)

        print(f"{cls_name}: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

    print("\nClean split created successfully.")
    print(f"Train folder: {OUT_ROOT / 'train'}")
    print(f"Val folder:   {OUT_ROOT / 'val'}")
    print(f"Test folder:  {OUT_ROOT / 'test'}")

if __name__ == "__main__":
    main()