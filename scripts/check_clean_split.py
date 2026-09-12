from pathlib import Path
import hashlib

ROOT = Path("data/processed")
SPLITS = ["train", "val", "test"]

def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    exact = 0
    byte_identical = 0

    for cls_dir in sorted((ROOT / "train").iterdir()):
        if not cls_dir.is_dir():
            continue

        train_files = sorted((ROOT / "train" / cls_dir.name).iterdir())
        val_files = sorted((ROOT / "val" / cls_dir.name).iterdir())
        test_files = sorted((ROOT / "test" / cls_dir.name).iterdir())

        train_names = {p.name for p in train_files}
        val_names = {p.name for p in val_files}
        test_names = {p.name for p in test_files}

        overlaps = (train_names & val_names) | (train_names & test_names) | (val_names & test_names)
        if overlaps:
            exact += len(overlaps)
            print(f"[OVERLAP] {cls_dir.name}: {overlaps}")

        # byte-identical check across sets
        for split_name, files in [("val", val_files), ("test", test_files)]:
            for f in files:
                if f.name in train_names:
                    t = ROOT / "train" / cls_dir.name / f.name
                    if t.exists() and md5(t) == md5(f):
                        byte_identical += 1
                        print(f"[LEAK] exact same bytes in train and {split_name}: {cls_dir.name}/{f.name}")

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Exact overlaps across splits: {exact}")
    print(f"Byte-identical duplicates across splits: {byte_identical}")

    if byte_identical > 0:
        print("DANGER: train/val/test overlap detected.")
    else:
        print("GOOD: no exact byte-identical overlap detected.")

if __name__ == "__main__":
    main()