"""
Shared data loading utilities for jet images.

Supports two modes:
1. Preprocessed .npy (fast, recommended): run preprocess.py first
2. Raw HDF5 (slow, fallback): reads and normalizes on the fly
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit


class JetImageDataset(Dataset):
    """Dataset backed by a preprocessed .npy file (memory-mapped)."""

    def __init__(self, npy_path):
        self.images = np.load(npy_path, mmap_mode='r')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.from_numpy(self.images[idx].copy())


def get_data_loaders(npy_path, train_split=0.8, batch_size=512, seed=42):
    """Create train/val loaders using stratified split consistent with Task 2/3.

    First half of data is quark (label=0), second half is gluon (label=1).
    80% train, 20% val. The train indices match those used in Task 3.
    """
    dataset = JetImageDataset(npy_path)
    n = len(dataset)
    labels = np.zeros(n, dtype=np.int64)
    labels[n // 2:] = 1

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_split, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(n), labels))

    train_ds = Subset(dataset, train_idx)
    val_ds = Subset(dataset, val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, n
