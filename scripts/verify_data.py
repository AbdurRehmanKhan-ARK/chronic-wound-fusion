"""Simple script to verify dataset folder structure and image counts."""
import os
import argparse

def count_images(path):
    counts = {}
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        return counts
    for entry in sorted(os.listdir(path)):
        p = os.path.join(path, entry)
        if os.path.isdir(p):
            files = [f for f in os.listdir(p) if os.path.isfile(os.path.join(p, f))]
            counts[entry] = len(files)
    return counts

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='Path to ROI folder containing class subfolders')
    args = parser.parse_args()
    counts = count_images(args.data_dir)
    for k,v in counts.items():
        print(f"{k}: {v} images")
