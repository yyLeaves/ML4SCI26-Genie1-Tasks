"""
Training script for Specific Task 1: Contrastive Learning for Q/G Classification.

Two phases:
  1. Contrastive pretraining (SimCLR or MoCo)
  2. Linear evaluation (frozen encoder + linear classifier)

Usage:
    python task3_contrastive/train.py --config task3_contrastive/configs/simclr_swin.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.logger import get_logger
from task3_contrastive.augmentations import JetAugmentation, ContrastivePairDataset
from task3_contrastive.model import SwinEncoder, CustomSwinEncoder, ResNetEncoder, NTXentLoss, MoCo


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_labels(data_path, n_total):
    """First half quark (0), second half gluon (1)."""
    labels = np.zeros(n_total, dtype=np.int64)
    labels[n_total // 2:] = 1
    return labels


def train_contrastive(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("outputs/task3") / cfg["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"task3.{cfg['experiment_name']}", log_dir=str(output_dir), log_file="train.log")
    logger.info(f"Experiment: {cfg['experiment_name']}")
    logger.info(f"Config: {cfg}")

    # Load data
    images = np.load(cfg["data_path"], mmap_mode='r')
    n_total = len(images)
    labels = get_labels(cfg["data_path"], n_total)
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
    np.save(output_dir / "val_indices.npy", val_idx)
    np.save(output_dir / "train_indices.npy", train_idx)

    # Augmentation
    aug = JetAugmentation(
        img_size=128,
        translate_frac=cfg.get("translate_frac", 0.1),
        noise_std=cfg.get("noise_std", 0.02),
        intensity_range=tuple(cfg.get("intensity_range", [0.8, 1.2])),
        erase_prob=cfg.get("erase_prob", 0.3),
    )

    train_dataset = ContrastivePairDataset(images, train_idx, labels[train_idx], aug)
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=True)
    logger.info(f"DataLoader ready: {len(train_loader)} batches, batch_size={cfg['batch_size']}")

    # Encoder
    encoder_type = cfg.get("encoder", "swin")
    embed_dim = cfg.get("embed_dim", 256)
    if encoder_type == "custom_swin":
        encoder = CustomSwinEncoder(embed_dim=embed_dim)
    elif encoder_type == "swin":
        encoder = SwinEncoder(embed_dim=embed_dim, pretrained=cfg.get("pretrained", False))
    else:
        encoder = ResNetEncoder(embed_dim=embed_dim)

    # Training method
    method = cfg.get("method", "simclr")
    if method == "moco":
        model = MoCo(encoder, embed_dim=embed_dim,
                      queue_size=cfg.get("queue_size", 4096),
                      momentum=cfg.get("moco_momentum", 0.999),
                      temperature=cfg.get("temperature", 0.1)).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        model = encoder.to(device)
        criterion = NTXentLoss(temperature=cfg.get("temperature", 0.1))

    logger.info(f"Method: {method}, Encoder: {encoder_type}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {total_params:,}")

    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["pretrain_epochs"], eta_min=cfg["min_lr"])

    # Phase 1: Contrastive pretraining
    logger.info("=== Phase 1: Contrastive Pretraining ===")
    best_loss = float("inf")

    for epoch in range(cfg["pretrain_epochs"]):
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        for batch_i, (view1, view2, _) in enumerate(train_loader):
            view1, view2 = view1.to(device), view2.to(device)

            if method == "moco":
                logits, moco_labels = model(view1, view2)
                loss = criterion(logits, moco_labels)
            else:
                _, z1 = model(view1)
                _, z2 = model(view2)
                loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * view1.size(0)

            # Log at batch 1, 10, 100, 1000, then every 200
            b = batch_i + 1
            if b in (1, 10, 100, 1000) or b % 200 == 0:
                avg_so_far = total_loss / (b * view1.size(0))
                logger.info(f"  Epoch {epoch+1} batch {b}/{num_batches} | loss: {loss.item():.4f} (avg: {avg_so_far:.4f})")

        scheduler.step()
        avg_loss = total_loss / len(train_dataset)
        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Pretrain Epoch {epoch+1:3d}/{cfg['pretrain_epochs']} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": avg_loss,
                "config": cfg,
            }, output_dir / "pretrained.pt")
            logger.info(f"  -> Best pretrained model saved (loss={avg_loss:.4f})")

    logger.info(f"Pretraining done. Best loss: {best_loss:.4f}")

    # Phase 2: Linear evaluation
    logger.info("=== Phase 2: Linear Evaluation ===")

    # Extract encoder from MoCo wrapper if needed
    if method == "moco":
        eval_encoder = model.encoder_q
    else:
        eval_encoder = model

    eval_encoder.eval()

    # Extract features for train and val
    def extract_features(indices):
        feats, labs = [], []
        with torch.no_grad():
            for i in range(0, len(indices), cfg["batch_size"]):
                batch_idx = indices[i:i + cfg["batch_size"]]
                batch = torch.from_numpy(images[batch_idx].copy()).to(device)
                h, _ = eval_encoder(batch)
                feats.append(h.cpu())
                labs.extend(labels[batch_idx])
        return torch.cat(feats), torch.tensor(labs)

    logger.info("Extracting features...")
    train_feats, train_labels = extract_features(train_idx)
    val_feats, val_labels = extract_features(val_idx)

    # Determine feature dim
    feat_dim = train_feats.shape[1]
    logger.info(f"Feature dim: {feat_dim}")

    # Linear classifier
    linear = nn.Linear(feat_dim, 2).to(device)
    lin_optimizer = AdamW(linear.parameters(), lr=1e-3, weight_decay=1e-4)
    lin_criterion = nn.CrossEntropyLoss()

    train_feat_ds = torch.utils.data.TensorDataset(train_feats, train_labels)
    val_feat_ds = torch.utils.data.TensorDataset(val_feats, val_labels)
    train_feat_loader = DataLoader(train_feat_ds, batch_size=512, shuffle=True)
    val_feat_loader = DataLoader(val_feat_ds, batch_size=512)

    best_val_auc = 0.0
    for epoch in range(cfg.get("linear_epochs", 50)):
        linear.train()
        for feats_batch, labels_batch in train_feat_loader:
            feats_batch, labels_batch = feats_batch.to(device), labels_batch.to(device)
            out = linear(feats_batch)
            loss = lin_criterion(out, labels_batch)
            lin_optimizer.zero_grad()
            loss.backward()
            lin_optimizer.step()

        # Validate
        linear.eval()
        val_probs, val_true = [], []
        with torch.no_grad():
            for feats_batch, labels_batch in val_feat_loader:
                feats_batch = feats_batch.to(device)
                out = linear(feats_batch)
                probs = torch.softmax(out, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_true.extend(labels_batch.numpy())

        val_auc = roc_auc_score(val_true, val_probs)
        val_acc = (np.array(val_probs) > 0.5).astype(int)
        val_acc = (val_acc == np.array(val_true)).mean()

        if (epoch + 1) % 5 == 0 or val_auc > best_val_auc:
            logger.info(f"Linear Epoch {epoch+1:3d} | Val AUC: {val_auc:.4f} Acc: {val_acc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "linear_state_dict": linear.state_dict(),
                "val_auc": val_auc,
            }, output_dir / "linear_best.pt")

    logger.info(f"Linear eval done. Best val AUC: {best_val_auc:.4f}")
    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train_contrastive(cfg)
