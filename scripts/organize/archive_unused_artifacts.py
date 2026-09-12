#!/usr/bin/env python3
"""Archive stale/leak-prone project artifacts into a dedicated cleanup folder.

This keeps the project in a clean state while preserving all verified work.

Keeps active:
  - data/raw
  - data/processed
  - scripts/
  - src/
  - models/
  - docs/
  - outputs/01_oof/vgg19_clean
  - outputs/01_oof/densenet201_clean
  - outputs/01_oof/mobilenetv2_clean

Archives stale items such as:
  - archive/leaked_run_2025_09_08
  - archive/processed_subset_backup
  - outputs/01_oof/vgg19
  - outputs/01_oof/vgg19_fixed
  - outputs/01_oof/densenet201
  - venv
  - repomix-output.xml
  - desktop.ini

Run from repo root:
  .\.venv\Scripts\python.exe scripts\organize\archive_unused_artifacts.py
"""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_ROOT = ROOT / "archive" / "cleanup_2026_09_12"

ACTIVE_OUTPUT_DIRS = {
    "outputs/01_oof/vgg19_clean",
    "outputs/01_oof/densenet201_clean",
    "outputs/01_oof/mobilenetv2_clean",
}

STALE_ITEMS = [
    "archive/leaked_run_2025_09_08",
    "archive/processed_subset_backup",
    "outputs/01_oof/vgg19",
    "outputs/01_oof/vgg19_fixed",
    "outputs/01_oof/densenet201",
    "venv",
    "repomix-output.xml",
    "desktop.ini",
]


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def move_if_exists(src: Path, dst_parent: Path):
    if not src.exists():
        return False

    dst = dst_parent / src.name
    if dst.exists():
        counter = 1
        while True:
            cand = dst_parent / f"{src.name}_{counter}"
            if not cand.exists():
                dst = cand
                break
            counter += 1

    shutil.move(str(src), str(dst))
    return True


def main():
    ensure_dir(ARCHIVE_ROOT)

    moved = []
    for rel in STALE_ITEMS:
        src = ROOT / rel
        if src.exists():
            moved.append(str(src))
            move_if_exists(src, ARCHIVE_ROOT)

    # Also archive any OOF dirs that are not part of the clean final structure.
    oof_root = ROOT / "outputs" / "01_oof"
    if oof_root.exists():
        for child in sorted(oof_root.iterdir()):
            if not child.is_dir():
                continue
            rel = child.relative_to(ROOT).as_posix()
            if rel in ACTIVE_OUTPUT_DIRS:
                continue
            if child.name.startswith("."):
                continue
            moved.append(str(child))
            move_if_exists(child, ARCHIVE_ROOT)

    print("Archived stale artifacts to:")
    print(f"  {ARCHIVE_ROOT}")
    if moved:
        for item in moved:
            print(f"  - {item}")
    else:
        print("  (nothing to archive)")

    print("\nActive final structure is kept as:")
    for rel in sorted(ACTIVE_OUTPUT_DIRS):
        p = ROOT / rel
        print(f"  - {p}")

    print("\nCurrent MobileNet target output directory:")
    print("  - outputs/01_oof/mobilenetv2_clean")


if __name__ == "__main__":
    main()
