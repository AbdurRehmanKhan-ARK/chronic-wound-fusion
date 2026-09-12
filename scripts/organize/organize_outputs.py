#!/usr/bin/env python3
"""Organize the `outputs/` folder into chronological, technical subfolders.

This script moves existing output files into a logical structure without modifying contents:

- outputs/
  - 01_oof/
    - probs/         # *_oof_probs.npy
    - meta/          # *_oof_meta.csv
    - checkpoints/   # outputs/checkpoints/* -> moved here
  - 02_gating/
    - models/        # gating model files (pt)
    - results/       # fused probs / preds (npy/csv)
    - weights/       # any weight files
  - 03_figures/      # figures
  - 99_misc/         # anything else

Run from repository root:
  .\.venv\Scripts\python.exe scripts\organize\organize_outputs.py
"""
from pathlib import Path
import shutil
import os


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    base = dest.stem
    suff = dest.suffix
    parent = dest.parent
    i = 1
    while True:
        new = parent / f"{base}_{i}{suff}"
        if not new.exists():
            return new
        i += 1


def move_file(src: Path, dst_dir: Path):
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    dst = unique_dest(dst)
    shutil.move(str(src), str(dst))
    return dst


def main():
    repo_root = Path.cwd()
    out = repo_root / 'outputs'
    if not out.exists():
        print('No outputs/ directory found.')
        return

    # target structure
    od_01 = out / '01_oof'
    od_01_probs = od_01 / 'probs'
    od_01_meta = od_01 / 'meta'
    od_01_ckpt = od_01 / 'checkpoints'

    od_02 = out / '02_gating'
    od_02_models = od_02 / 'models'
    od_02_weights = od_02 / 'weights'
    od_02_results = od_02 / 'results'

    od_03_fig = out / '03_figures'
    od_99 = out / '99_misc'

    ensure_dir(od_01_probs)
    ensure_dir(od_01_meta)
    ensure_dir(od_01_ckpt)
    ensure_dir(od_02_models)
    ensure_dir(od_02_weights)
    ensure_dir(od_02_results)
    ensure_dir(od_03_fig)
    ensure_dir(od_99)

    moved = []

    # Move top-level files into targets based on patterns
    for p in list(out.iterdir()):
        if p.name == 'checkpoints' and p.is_dir():
            # move entire checkpoints directory under 01_oof/checkpoints
            for ck in p.iterdir():
                if ck.is_file():
                    dst = move_file(ck, od_01_ckpt)
                    moved.append((ck, dst))
            # remove empty original folder
            try:
                p.rmdir()
            except Exception:
                pass
            continue

        if p.is_dir():
            # skip newly created organization dirs
            if p.name in ('01_oof', '02_gating', '03_figures', '99_misc'):
                continue
            # move unexpected directories into 99_misc
            dst = move_file(p, od_99)
            moved.append((p, dst))
            continue

        name = p.name.lower()
        try:
            if name.endswith('_oof_probs.npy'):
                dst = move_file(p, od_01_probs)
            elif name.endswith('_oof_meta.csv'):
                dst = move_file(p, od_01_meta)
            elif name.startswith('gating') and name.endswith('.pt'):
                # gating model files
                dst = move_file(p, od_02_models)
            elif 'weight' in name and (name.endswith('.pt') or name.endswith('.pth')):
                dst = move_file(p, od_02_weights)
            elif name.startswith('gating') and (name.endswith('.npy') or name.endswith('.csv')):
                dst = move_file(p, od_02_results)
            elif name.endswith('.png') or name.endswith('.jpg') or name.endswith('.svg'):
                dst = move_file(p, od_03_fig)
            else:
                # default: misc
                dst = move_file(p, od_99)
        except Exception as e:
            print(f'Failed to move {p}: {e}')
            continue
        moved.append((p, dst))

    print('Reorganization complete. Summary:')
    for s, d in moved:
        print(f'  {s} -> {d}')


if __name__ == '__main__':
    main()
