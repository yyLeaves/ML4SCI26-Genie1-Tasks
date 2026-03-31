"""
Physics-motivated augmentations for jet image contrastive learning.

Following arXiv:2602.00141 — NO rotations (not physical in eta-phi plane).
"""

import torch
import torch.nn as nn
import torchvision.transforms as T


class JetAugmentation(nn.Module):
    """Compose jet-specific augmentations for contrastive pairs."""

    def __init__(self, img_size=128, translate_frac=0.1, noise_std=0.02,
                 intensity_range=(0.8, 1.2), erase_prob=0.3, erase_scale=(0.02, 0.1)):
        super().__init__()
        self.translate_px = int(img_size * translate_frac)
        self.noise_std = noise_std
        self.intensity_range = intensity_range
        self.erase_prob = erase_prob
        self.erase_scale = erase_scale

    def forward(self, x):
        """Apply random augmentations to a jet image tensor (3, H, W)."""
        # 1. Random translation in eta-phi
        dx = torch.randint(-self.translate_px, self.translate_px + 1, (1,)).item()
        dy = torch.randint(-self.translate_px, self.translate_px + 1, (1,)).item()
        x = torch.roll(x, shifts=(dx, dy), dims=(1, 2))

        # 2. Gaussian noise (simulates detector smearing)
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = torch.clamp(x + noise, min=0)

        # 3. Intensity scaling (simulates energy calibration uncertainty)
        scale = torch.empty(1).uniform_(*self.intensity_range).item()
        x = x * scale

        # 4. Random erasing (simulates detector inefficiency)
        if torch.rand(1).item() < self.erase_prob:
            x = T.RandomErasing(
                p=1.0, scale=self.erase_scale, ratio=(0.5, 2.0), value=0
            )(x)

        return x


class ContrastivePairDataset(torch.utils.data.Dataset):
    """Wraps a jet image dataset to produce augmented pairs for contrastive learning."""

    def __init__(self, images_mmap, indices, labels, augmentation):
        """
        Args:
            images_mmap: numpy mmap array (N_total, 3, H, W)
            indices: array of indices into images_mmap for this split
            labels: array of labels for each index
            augmentation: JetAugmentation instance
        """
        self.images = images_mmap
        self.indices = indices
        self.labels = labels
        self.aug = augmentation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        global_idx = self.indices[idx]
        img = torch.from_numpy(self.images[global_idx].copy())
        label = self.labels[idx]
        view1 = self.aug(img)
        view2 = self.aug(img)
        return view1, view2, label
