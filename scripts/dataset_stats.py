from pathlib import Path

def count_images(folder: Path):
    counts = {}
    for cls_dir in sorted(folder.iterdir()):
        if cls_dir.is_dir():
            counts[cls_dir.name] = len([p for p in cls_dir.iterdir() if p.is_file()])
    return counts

if __name__ == '__main__':
    base = Path('data/processed')
    for split in ['train','val','test']:
        folder = base / split
        if not folder.exists():
            print(f"{split}: missing")
            continue
        counts = count_images(folder)
        total = sum(counts.values())
        print(f"{split.capitalize()} — total: {total}")
        for cls, c in counts.items():
            print(f"  {cls}: {c}")
        print('')
