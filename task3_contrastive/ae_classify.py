"""
Classify quark/gluon using Task 1 autoencoder's bottleneck representation.

Phase 1: Extract features from frozen SegNet encoder
Phase 2: Train linear classifier on extracted features
Phase 3: (Optional) Fine-tune encoder + classifier with CE loss

Usage:
    python task3_contrastive/ae_classify.py --config task3_contrastive/configs/ae_linear.yaml
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

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.logger import get_logger
from task1_autoencoder.model import SegNetAutoencoder


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def extract_features(model, images, indices, batch_size, device, logger):
    """Extract bottleneck features from frozen encoder."""
    model.eval()
    feats, labs = [], []
    n = len(indices)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_idx = indices[i:i + batch_size]
            batch = torch.from_numpy(images[batch_idx].copy()).to(device)
            z = model.encode(batch)
            feats.append(z.cpu())
            if (i // batch_size + 1) % 50 == 0:
                logger.info(f"  Extracted {min(i + batch_size, n)}/{n}")
    return torch.cat(feats)


def train_ae_classify(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("outputs/task3") / cfg["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"task3.{cfg['experiment_name']}", log_dir=str(output_dir), log_file="train.log")
    logger.info(f"Experiment: {cfg['experiment_name']}")
    logger.info(f"Config: {cfg}")

    # Load SegNet encoder
    ckpt = torch.load(cfg["ae_checkpoint"], map_location="cpu", weights_only=False)
    ae_cfg = ckpt["config"]
    model = SegNetAutoencoder(
        latent_dim=ae_cfg.get("latent_dim", 512),
        channels=ae_cfg["encoder_channels"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded SegNet encoder from {ae_cfg['experiment_name']} (epoch {ckpt['epoch']})")

    # Load data
    images = np.load(ae_cfg["data_path"], mmap_mode='r')
    n = len(images)
    labels = np.zeros(n, dtype=np.int64)
    labels[n // 2:] = 1
    logger.info(f"Data: {n} images")

    # Stratified split 80/10/10
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg["seed"])
    train_idx, temp_idx = next(splitter.split(np.zeros(n), labels))
    temp_labels = labels[temp_idx]
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=cfg["seed"])
    val_rel, test_rel = next(splitter2.split(np.zeros(len(temp_labels)), temp_labels))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    logger.info(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    np.save(output_dir / "test_indices.npy", test_idx)

    # Phase 1: Extract features
    logger.info("=== Phase 1: Feature Extraction ===")
    batch_size = cfg.get("batch_size", 256)
    logger.info("Extracting train features...")
    train_feats = extract_features(model, images, train_idx, batch_size, device, logger)
    logger.info("Extracting val features...")
    val_feats = extract_features(model, images, val_idx, batch_size, device, logger)
    logger.info("Extracting test features...")
    test_feats = extract_features(model, images, test_idx, batch_size, device, logger)

    train_labels = torch.tensor(labels[train_idx])
    val_labels = torch.tensor(labels[val_idx])
    test_labels = torch.tensor(labels[test_idx])

    feat_dim = train_feats.shape[1]
    logger.info(f"Feature dim: {feat_dim}")

    # Phase 2: Linear classifier
    logger.info("=== Phase 2: Linear Classification ===")
    linear_epochs = cfg.get("linear_epochs", 50)
    linear_lr = cfg.get("linear_lr", 1e-3)
    linear = nn.Linear(feat_dim, 2).to(device)
    weight_decay = cfg.get("weight_decay", 1e-4)
    min_lr = cfg.get("min_lr", linear_lr * 0.01)
    opt_name = cfg.get("optimizer", "adamw")
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(linear.parameters(), lr=linear_lr, momentum=0.9, weight_decay=weight_decay)
    elif opt_name == "adam":
        optimizer = torch.optim.Adam(linear.parameters(), lr=linear_lr, weight_decay=weight_decay)
    else:
        optimizer = AdamW(linear.parameters(), lr=linear_lr, weight_decay=weight_decay)
    logger.info(f"Optimizer: {opt_name}, lr={linear_lr}, wd={weight_decay}, min_lr={min_lr}")
    scheduler = CosineAnnealingLR(optimizer, T_max=linear_epochs, eta_min=min_lr)

    train_loader = DataLoader(TensorDataset(train_feats, train_labels), batch_size=512, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_feats, val_labels), batch_size=512)
    criterion = nn.CrossEntropyLoss()

    # Set random seed for reproducibility
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    best_val_auc = 0.0

    for epoch in range(linear_epochs):
        # Train
        linear.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for f, l in train_loader:
            f, l = f.to(device), l.to(device)
            out = linear(f)
            loss = criterion(out, l)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * f.size(0)
            train_correct += (out.argmax(1) == l).sum().item()
            train_total += f.size(0)
        train_loss /= train_total
        train_acc = train_correct / train_total
        train_probs_all, train_true_all = [], []
        linear.eval()
        with torch.no_grad():
            for f, l in train_loader:
                probs = torch.softmax(linear(f.to(device)), dim=1)[:, 1]
                train_probs_all.extend(probs.cpu().numpy())
                train_true_all.extend(l.numpy())
        train_auc = roc_auc_score(train_true_all, train_probs_all)

        # Validate
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_probs, val_true = [], []
        with torch.no_grad():
            for f, l in val_loader:
                f, l = f.to(device), l.to(device)
                out = linear(f)
                loss = criterion(out, l)
                val_loss += loss.item() * f.size(0)
                val_correct += (out.argmax(1) == l).sum().item()
                val_total += f.size(0)
                probs = torch.softmax(out, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_true.extend(l.cpu().numpy())
        val_loss /= val_total
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(val_true, val_probs)

        scheduler.step()

        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            torch.save({"linear_state_dict": linear.state_dict(), "val_auc": val_auc}, output_dir / "linear_best.pt")
        marker = " *" if is_best else ""
        logger.info(
            f"Epoch {epoch+1:3d}/{linear_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} AUC: {train_auc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}{marker}"
        )

    logger.info(f"Linear probe done. Best val AUC: {best_val_auc:.4f}")

    # Phase 3: Test evaluation
    logger.info("=== Phase 3: Test Evaluation ===")
    linear.load_state_dict(torch.load(output_dir / "linear_best.pt", weights_only=False)["linear_state_dict"])
    linear.eval()
    test_probs, test_true = [], []
    with torch.no_grad():
        test_loader = DataLoader(TensorDataset(test_feats, test_labels), batch_size=512)
        for f, l in test_loader:
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
    train_ae_classify(cfg)
