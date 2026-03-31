"""
Swin-T + SupCon (Supervised Contrastive) + linear probe.

Usage:
    python task3_contrastive/swin_supcon.py --config task3_contrastive/configs/supcon_swin.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.logger import get_logger
from task3_contrastive.model import SwinEncoder, CustomSwinEncoder
from task3_contrastive.augmentations import JetAugmentation


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2, labels):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        N = z1.size(0)
        z = torch.cat([z1, z2], dim=0)
        labels_2n = torch.cat([labels, labels], dim=0)
        sim = torch.mm(z, z.t()) / self.temperature
        self_mask = torch.eye(2 * N, device=z.device).bool()
        sim.masked_fill_(self_mask, -1e9)
        pos_mask = (labels_2n.unsqueeze(0) == labels_2n.unsqueeze(1)) & ~self_mask
        log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
        pos_log_prob = (pos_mask.float() * log_prob).sum(dim=1)
        num_pos = pos_mask.float().sum(dim=1).clamp(min=1)
        return -(pos_log_prob / num_pos).mean()


class SupConPairDataset(torch.utils.data.Dataset):
    def __init__(self, images_mmap, indices, labels, augmentation):
        self.images = images_mmap
        self.indices = indices
        self.labels = labels
        self.aug = augmentation

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[self.indices[idx]].copy())
        return self.aug(img), self.aug(img), self.labels[idx]


def train_swin_supcon(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("outputs/task3") / cfg["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"task3.{cfg['experiment_name']}", log_dir=str(output_dir), log_file="train.log")
    logger.info(f"Experiment: {cfg['experiment_name']}")
    logger.info(f"Config: {cfg}")

    # Load data
    images = np.load(cfg["data_path"], mmap_mode='r')
    n_total = len(images)
    labels = np.zeros(n_total, dtype=np.int64)
    labels[n_total // 2:] = 1
    logger.info(f"Data: {n_total} images")

    # Stratified split 80/10/10
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg["seed"])
    train_idx, temp_idx = next(splitter.split(np.zeros(n_total), labels))
    temp_labels = labels[temp_idx]
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=cfg["seed"])
    val_rel, test_rel = next(splitter2.split(np.zeros(len(temp_labels)), temp_labels))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]
    logger.info(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    np.save(output_dir / "test_indices.npy", test_idx)

    # Augmentation
    aug = JetAugmentation(
        img_size=128,
        translate_frac=cfg.get("translate_frac", 0.1),
        noise_std=cfg.get("noise_std", 0.02),
        intensity_range=tuple(cfg.get("intensity_range", [0.8, 1.2])),
        erase_prob=cfg.get("erase_prob", 0.3),
    )
    train_dataset = SupConPairDataset(images, train_idx, labels[train_idx], aug)
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    logger.info(f"DataLoader: {len(train_loader)} batches")

    # Encoder
    encoder_type = cfg.get("encoder", "swin")
    embed_dim = cfg.get("embed_dim", 256)
    if encoder_type == "custom_swin":
        encoder = CustomSwinEncoder(embed_dim=embed_dim)
    else:
        encoder = SwinEncoder(embed_dim=embed_dim, pretrained=cfg.get("pretrained", False))
    encoder = encoder.to(device)

    total_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    logger.info(f"Encoder: {encoder_type}, params: {total_params:,}")

    criterion = SupConLoss(temperature=cfg.get("temperature", 0.1))
    optimizer = AdamW(encoder.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["pretrain_epochs"], eta_min=cfg["min_lr"])

    # Phase 1: SupCon
    logger.info("=== Phase 1: SupCon Pretraining ===")
    best_loss = float("inf")

    for epoch in range(cfg["pretrain_epochs"]):
        encoder.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        for batch_i, (view1, view2, batch_labels) in enumerate(train_loader):
            view1, view2 = view1.to(device), view2.to(device)
            batch_labels = batch_labels.to(device)
            _, z1 = encoder(view1)
            _, z2 = encoder(view2)
            loss = criterion(z1, z2, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * view1.size(0)

            b = batch_i + 1
            if b in (1, 10, 100) or b % 200 == 0:
                logger.info(f"  Epoch {epoch+1} batch {b}/{num_batches} | loss: {loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / len(train_dataset)
        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"SupCon Epoch {epoch+1:3d}/{cfg['pretrain_epochs']} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "encoder_state_dict": encoder.state_dict(),
                "loss": avg_loss,
                "config": cfg,
            }, output_dir / "pretrained.pt")
            logger.info(f"  -> Best model saved (loss={avg_loss:.4f})")

    logger.info(f"SupCon done. Best loss: {best_loss:.4f}")

    # Phase 2: Linear probe
    logger.info("=== Phase 2: Linear Evaluation ===")
    encoder.eval()

    def extract_features(indices):
        feats = []
        with torch.no_grad():
            for i in range(0, len(indices), cfg["batch_size"]):
                batch_idx = indices[i:i + cfg["batch_size"]]
                batch = torch.from_numpy(images[batch_idx].copy()).to(device)
                h, _ = encoder(batch)
                feats.append(h.cpu())
        return torch.cat(feats)

    logger.info("Extracting features...")
    train_feats = extract_features(train_idx)
    val_feats = extract_features(val_idx)
    test_feats = extract_features(test_idx)
    feat_dim = train_feats.shape[1]
    logger.info(f"Feature dim: {feat_dim}")

    torch.manual_seed(cfg["seed"])
    linear = nn.Linear(feat_dim, 2).to(device)
    linear_epochs = cfg.get("linear_epochs", 50)
    lin_lr = cfg.get("linear_lr", 1e-3)
    lin_optimizer = AdamW(linear.parameters(), lr=lin_lr, weight_decay=1e-4)
    lin_scheduler = CosineAnnealingLR(lin_optimizer, T_max=linear_epochs, eta_min=lin_lr * 0.01)
    lin_criterion = nn.CrossEntropyLoss()

    train_loader_lin = DataLoader(TensorDataset(train_feats, torch.tensor(labels[train_idx])), batch_size=512, shuffle=True)
    val_loader_lin = DataLoader(TensorDataset(val_feats, torch.tensor(labels[val_idx])), batch_size=512)

    best_val_auc = 0.0
    for epoch in range(linear_epochs):
        linear.train()
        for f, l in train_loader_lin:
            f, l = f.to(device), l.to(device)
            loss = lin_criterion(linear(f), l)
            lin_optimizer.zero_grad()
            loss.backward()
            lin_optimizer.step()

        linear.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for f, l in val_loader_lin:
                f, l = f.to(device), l.to(device)
                val_probs.extend(torch.softmax(linear(f), dim=1)[:, 1].cpu().numpy())
                val_true.extend(l.cpu().numpy())

        lin_scheduler.step()
        val_auc = roc_auc_score(val_true, val_probs)
        val_acc = ((np.array(val_probs) > 0.5).astype(int) == np.array(val_true)).mean()

        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            torch.save({"linear_state_dict": linear.state_dict(), "val_auc": val_auc}, output_dir / "linear_best.pt")
        marker = " *" if is_best else ""
        if (epoch + 1) % 10 == 0 or is_best:
            logger.info(f"Linear {epoch+1:3d}/{linear_epochs} | Val AUC: {val_auc:.4f} Acc: {val_acc:.4f}{marker}")

    logger.info(f"Linear probe done. Best val AUC: {best_val_auc:.4f}")

    # Phase 3: Test
    logger.info("=== Phase 3: Test Evaluation ===")
    linear.load_state_dict(torch.load(output_dir / "linear_best.pt", weights_only=False)["linear_state_dict"])
    linear.eval()
    test_probs, test_true = [], []
    with torch.no_grad():
        for f, l in DataLoader(TensorDataset(test_feats, torch.tensor(labels[test_idx])), batch_size=512):
            probs = torch.softmax(linear(f.to(device)), dim=1)[:, 1]
            test_probs.extend(probs.cpu().numpy())
            test_true.extend(l.numpy())
    test_auc = roc_auc_score(test_true, test_probs)
    test_acc = ((np.array(test_probs) > 0.5).astype(int) == np.array(test_true)).mean()
    logger.info(f"Test AUC: {test_auc:.4f}")
    logger.info(f"Test Acc: {test_acc:.4f}")
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_swin_supcon(cfg)
