"""
Fine-tune a pretrained encoder for quark/gluon classification.

Loads a MoCo/SimCLR pretrained encoder, unfreezes last N layers,
trains with CrossEntropy (+ optional SupCon) loss.

Usage:
    python task3_contrastive/finetune.py --config task3_contrastive/configs/finetune_swin.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.logger import get_logger
from task3_contrastive.model import SwinEncoder, CustomSwinEncoder, ResNetEncoder, MoCo


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


class FineTuneDataset(torch.utils.data.Dataset):
    """Jet image dataset with labels, reads from mmap."""

    def __init__(self, images_mmap, indices, labels):
        self.images = images_mmap
        self.indices = indices
        self.labels = labels

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[self.indices[idx]].copy())
        label = self.labels[idx]
        return img, label


def finetune(cfg):
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

    train_dataset = FineTuneDataset(images, train_idx, labels[train_idx])
    val_dataset = FineTuneDataset(images, val_idx, labels[val_idx])
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    logger.info(f"DataLoader ready: {len(train_loader)} train batches")

    # Build encoder
    encoder_type = cfg.get("encoder", "swin")
    embed_dim = cfg.get("embed_dim", 256)
    if encoder_type == "custom_swin":
        encoder = CustomSwinEncoder(embed_dim=embed_dim)
    elif encoder_type == "swin":
        encoder = SwinEncoder(embed_dim=embed_dim, pretrained=False)
    else:
        encoder = ResNetEncoder(embed_dim=embed_dim)

    # Load pretrained weights
    pretrain_path = cfg.get("pretrain_checkpoint")
    if pretrain_path:
        logger.info(f"Loading pretrained weights from {pretrain_path}...")
        ckpt = torch.load(pretrain_path, map_location="cpu", weights_only=False)
        # Support MoCo (model_state_dict with encoder_q.) and SupCon (encoder_state_dict)
        if "encoder_state_dict" in ckpt:
            encoder.load_state_dict(ckpt["encoder_state_dict"])
            logger.info(f"Loaded SupCon encoder weights ({len(ckpt['encoder_state_dict'])} keys)")
        else:
            state = ckpt["model_state_dict"]
            encoder_state = {}
            for k, v in state.items():
                if k.startswith("encoder_q."):
                    encoder_state[k.replace("encoder_q.", "")] = v
            if encoder_state:
                encoder.load_state_dict(encoder_state)
                logger.info(f"Loaded MoCo encoder_q weights ({len(encoder_state)} keys)")
            else:
                encoder.load_state_dict(state, strict=False)
                logger.info("Loaded encoder weights directly")

    # Classifier head
    if encoder_type == "custom_swin":
        feat_dim = encoder.out_dim  # 384
    elif encoder_type == "swin":
        feat_dim = 768
    else:
        feat_dim = 512
    classifier = nn.Sequential(
        nn.Linear(feat_dim, 256),
        nn.ReLU(),
        nn.Dropout(cfg.get("dropout", 0.1)),
        nn.Linear(256, 2),
    )

    # Freeze / unfreeze strategy
    if cfg.get("unfreeze_all", False):
        # Unfreeze everything
        for param in encoder.parameters():
            param.requires_grad = True
    else:
        unfreeze_layers = cfg.get("unfreeze_layers", 2)
        for param in encoder.parameters():
            param.requires_grad = False
        if encoder_type == "swin":
            stages = list(encoder.swin_features)
            for stage in stages[-unfreeze_layers:]:
                for param in stage.parameters():
                    param.requires_grad = True
            for param in encoder.swin_norm.parameters():
                param.requires_grad = True
        elif encoder_type == "custom_swin":
            for stage in encoder.stages[-unfreeze_layers:]:
                for param in stage.parameters():
                    param.requires_grad = True
            for param in encoder.norm.parameters():
                param.requires_grad = True
        for param in encoder.projector.parameters():
            param.requires_grad = True

    model = nn.ModuleDict({"encoder": encoder, "classifier": classifier}).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")

    # Differential learning rate
    criterion = nn.CrossEntropyLoss()
    if "learning_rate_encoder" in cfg:
        param_groups = [
            {"params": model["encoder"].parameters(), "lr": cfg["learning_rate_encoder"]},
            {"params": model["classifier"].parameters(), "lr": cfg["learning_rate_head"]},
        ]
        optimizer = AdamW(param_groups, weight_decay=cfg["weight_decay"])
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"], eta_min=cfg.get("min_lr_encoder", 1e-8))
        logger.info(f"Differential LR: encoder={cfg['learning_rate_encoder']}, head={cfg['learning_rate_head']}")
    else:
        optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"], eta_min=cfg.get("min_lr", 1e-7))

    # Training loop
    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(cfg["num_epochs"]):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        num_batches = len(train_loader)

        for batch_i, (imgs, labs) in enumerate(train_loader):
            imgs, labs = imgs.to(device), labs.to(device)

            h, _ = model["encoder"](imgs)
            out = model["classifier"](h)
            loss = criterion(out, labs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labs).sum().item()
            total += imgs.size(0)

            b = batch_i + 1
            if b in (1, 10, 100) or b % 200 == 0:
                logger.info(f"  Epoch {epoch+1} batch {b}/{num_batches} | loss: {loss.item():.4f}")

        train_loss = total_loss / total
        train_acc = correct / total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_probs, val_true = [], []

        with torch.no_grad():
            for imgs, labs in val_loader:
                imgs, labs = imgs.to(device), labs.to(device)
                h, _ = model["encoder"](imgs)
                out = model["classifier"](h)
                loss = criterion(out, labs)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (out.argmax(1) == labs).sum().item()
                val_total += imgs.size(0)
                probs = torch.softmax(out, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_true.extend(labs.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(val_true, val_probs)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch+1:3d}/{cfg['num_epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f} | "
            f"LR: {lr:.2e}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": {k: v for k, v in model.state_dict().items()},
                "val_auc": val_auc,
                "val_acc": val_acc,
                "config": cfg,
            }, output_dir / "best.pt")
            logger.info(f"  -> Best model saved (AUC={val_auc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg.get("patience", 10):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    logger.info(f"Done. Best val AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    finetune(cfg)
