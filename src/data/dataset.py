"""PyTorch Dataset skeleton for wound ROI images."""
from PIL import Image
import os
from torch.utils.data import Dataset

class WoundROIDataset(Dataset):
    def __init__(self, root_dir, transform=None, classes=None, small_subset=False, subset_per_class=50):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = classes if classes is not None else sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir,d))])
        for idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir) if os.path.isfile(os.path.join(cls_dir,f))]
            if small_subset:
                files = files[:subset_per_class]
            for f in files:
                self.samples.append((f, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
