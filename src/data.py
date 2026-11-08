"""Turn a split CSV into a PyTorch DataLoader."""
import cv2
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils import processed_dir, splits_dir

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def read_split(cfg, name):
    """name: 'test', 'train_<exp>' or 'val_<exp>'."""
    path = splits_dir(cfg) / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.prepare_data` first.")
    return pd.read_csv(path)


class BlurDataset(Dataset):
    def __init__(self, cfg, df, augment=False):
        root = processed_dir(cfg)
        classes = cfg["data"]["classes"]
        self.paths = [root / p for p in df["filepath"]]
        self.labels = [classes.index(l) for l in df["label"]]
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = cv2.imread(str(self.paths[i]))
        if img is None:
            raise IOError(f"Unreadable image: {self.paths[i]}")
        x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0
        if self.augment and torch.rand(1).item() < 0.5:
            x = x.flip(-1)  # horizontal flip keeps the blur unchanged
        return (x - IMAGENET_MEAN) / IMAGENET_STD, self.labels[i]


def make_loader(cfg, df, shuffle=False, augment=False, seed=None):
    g = torch.Generator().manual_seed(cfg["seed"] if seed is None else seed)
    return DataLoader(
        BlurDataset(cfg, df, augment=augment),
        batch_size=cfg["train"]["batch_size"],
        shuffle=shuffle,
        generator=g,
        num_workers=cfg["train"].get("num_workers", 0),
    )
