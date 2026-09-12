"""
Leakage audit for the chronic-wound gated-fusion project.
    python leakage_audit.py

What it checks:
  1. Exact filename collisions between data/processed/train and
     data/processed/test (and val) per class -- these are the files
     train_gating.py could have copied across, or that were already
     ambiguous.
  2. Byte-identical file content between any train file and any
     test/val file with the same class -- this proves an actual copy
     happened (not just a coincidental filename).
  3. Group-id collisions: original image "10.jpg" vs its augmented
     siblings "aug_0000_10.jpg" appearing across train and test/val
     (a lighter form of leakage from the original random split, less
     severe but worth knowing).

No files are modified. This only reads and reports.
"""
import hashlib
import os
import re
import sys
from pathlib import Path

ROOT = Path("data/processed")
SPLITS = ["train", "val", "test"]
AUG_PREFIX_RE = re.compile(r"^aug_\d+_(?P<orig_stem>.+)$")


def md5sum(path, block_size=65536):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            h.update(chunk)
    return h.hexdigest()


def orig_stem(filename: str) -> str:
    """Map 'aug_0003_10.jpg' -> '10.jpg' ; '10.jpg' -> '10.jpg'"""
    stem, ext = os.path.splitext(filename)
    m = AUG_PREFIX_RE.match(stem)
    base = m.group("orig_stem") if m else stem
    return base + ext


def index_split(split_dir: Path):
    """
    Returns dict: class_name -> list of (filename, full_path)
    """
    out = {}
    if not split_dir.exists():
        print(f"  [WARN] split not found: {split_dir}")
        return out
    for cls_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        files = [
            (f.name, f) for f in sorted(cls_dir.iterdir()) if f.is_file()
        ]
        out[cls_dir.name] = files
    return out


def main():
    if not ROOT.exists():
        print(f"ERROR: {ROOT} not found. Run this from your repo root.")
        sys.exit(1)

    print(f"Auditing {ROOT} ...\n")
    idx = {s: index_split(ROOT / s) for s in SPLITS}

    all_classes = sorted(set().union(*[idx[s].keys() for s in SPLITS if idx[s]]))

    total_exact_name_collisions = 0
    total_byte_identical = 0
    total_group_collisions = 0

    for cls in all_classes:
        train_files = idx.get("train", {}).get(cls, [])
        val_files = idx.get("val", {}).get(cls, [])
        test_files = idx.get("test", {}).get(cls, [])

        train_by_name = {name: path for name, path in train_files}

        # --- 1. exact filename collisions: train vs val/test ---
        for split_name, files in [("val", val_files), ("test", test_files)]:
            for name, path in files:
                if name in train_by_name:
                    total_exact_name_collisions += 1
                    train_path = train_by_name[name]
                    same_bytes = md5sum(train_path) == md5sum(path)
                    if same_bytes:
                        total_byte_identical += 1
                        print(
                            f"[LEAK-CONFIRMED] class={cls} filename={name} "
                            f"-> IDENTICAL bytes in train and {split_name} "
                            f"(train: {train_path}  {split_name}: {path})"
                        )
                    else:
                        print(
                            f"[NAME-COLLISION-ONLY] class={cls} filename={name} "
                            f"exists in both train and {split_name} but content differs "
                            f"(likely coincidental numbering, not a copy)"
                        )

        # --- 2. group-id collisions (original vs augmented siblings) ---
        train_groups = {}
        for name, path in train_files:
            g = orig_stem(name)
            train_groups.setdefault(g, []).append(name)

        for split_name, files in [("val", val_files), ("test", test_files)]:
            for name, path in files:
                g = orig_stem(name)
                if g in train_groups:
                    total_group_collisions += 1
                    print(
                        f"[GROUP-COLLISION] class={cls} {split_name}/{name} shares "
                        f"original-image group '{g}' with train file(s): "
                        f"{train_groups[g]}"
                    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Exact filename collisions (train vs val/test): {total_exact_name_collisions}")
    print(f"  of which BYTE-IDENTICAL (confirmed leak):     {total_byte_identical}")
    print(f"Group-id collisions (original/aug siblings split across sets): {total_group_collisions}")

    if total_byte_identical > 0:
        print(
            "\n>>> ACTION NEEDED: byte-identical files found in train AND test/val.\n"
            ">>> Your test/val metrics are likely inflated. Do not report them as-is.\n"
            ">>> Next step: rebuild val/test from the original raw AZH data (never touched\n"
            ">>> by augmentation or the buggy copy-fallback), keyed off group id, and\n"
            ">>> rerun evaluation."
        )
    elif total_group_collisions > 0:
        print(
            "\n>>> Milder issue: some original/augmented sibling images are split across\n"
            ">>> train and val/test. This can slightly inflate metrics. Consider a\n"
            ">>> group-aware re-split before final numbers, if time allows."
        )
    else:
        print("\n>>> No leakage detected. Your val/test numbers look trustworthy.")


if __name__ == "__main__":
    main()