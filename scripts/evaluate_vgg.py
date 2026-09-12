import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class WoundDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for img_file in cls_dir.iterdir():
                if img_file.is_file():
                    self.samples.append((img_file, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def evaluate(checkpoint_path: str, split: str = 'test', batch_size: int = 8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = Path(checkpoint_path)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    data_dir = Path('data/processed') / split
    dataset = WoundDataset(data_dir, transform=val_transforms)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    # replace classifier final layer
    if isinstance(model.classifier, nn.Sequential):
        in_feats = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_feats, len(dataset.classes))
    else:
        model.classifier = nn.Linear(model.classifier.in_features, len(dataset.classes))

    state = torch.load(ckpt, map_location=device)
    model.load_state_dict(state)
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
    parser.add_argument('--checkpoint', required=True, help='Path to VGG checkpoint')
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()

    evaluate(args.checkpoint, split=args.split, batch_size=args.batch_size)
