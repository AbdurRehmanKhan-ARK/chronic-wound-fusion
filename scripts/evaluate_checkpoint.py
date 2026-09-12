import argparse
from pathlib import Path
import torch #type: ignore
import torch.nn as nn #type: ignore
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader #type: ignore

# import helpers from train_backbone_small
from train_backbone_small import load_model, WoundDataset, val_transforms


def evaluate(model_name, checkpoint_path, split='test', batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    # dataset
    data_dir = Path('data/processed') / split
    dataset = WoundDataset(data_dir, transform=val_transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = load_model(model_name, num_classes=len(dataset.classes))
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=dataset.classes)

    print(f"Evaluation on {split} set ({len(dataset)} samples)")
    print(f"Accuracy: {acc}")
    print("Confusion Matrix:")
    print(cm)
    print(report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=['densenet201','mobilenet_v2'], help='Model name')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--split', default='test', choices=['train','val','test'], help='Split to evaluate')
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    evaluate(args.model, args.checkpoint, split=args.split, batch_size=args.batch_size)
