"""
AE encoder → MoCo contrastive fine-tune → linear probe.

Loads Task 1 SegNet encoder, wraps it for MoCo, fine-tunes with
contrastive loss, then evaluates with linear probe.

Usage:
    python task3_contrastive/ae_moco.py --config task3_contrastive/configs/ae_moco.yaml
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
from task3_contrastive.model import SegNetEncoderWrapper, MoCo, NTXentLoss
from task3_contrastive.augmentations import JetAugmentation, ContrastivePairDataset


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_ae_moco(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("outputs/task3") / cfg["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"task3.{cfg['experiment_name']}", log_dir=str(output_dir), log_file="train.log")
    logger.info(f"Experiment: {cfg['experiment_name']}")
    logger.info(f"Config: {cfg}")

    # Load SegNet encoder
    ckpt = torch.load(cfg["ae_checkpoint"], map_location="cpu", weights_only=False)
    ae_cfg = ckpt["config"]
    segnet = SegNetAutoencoder(
        latent_dim=ae_cfg.get("latent_dim", 512),
        channels=ae_cfg["encoder_channels"],
    )
    segnet.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded SegNet from {ae_cfg['experiment_name']} (epoch {ckpt['epoch']})")

    # Wrap encoder for MoCo
    embed_dim = cfg.get("embed_dim", 256)
    encoder = SegNetEncoderWrapper(segnet, embed_dim=embed_dim)
    encoder_params = sum(p.numel() for p in encoder.parameters())
    logger.info(f"SegNet encoder wrapper: {encoder_params:,} params, latent_dim={encoder.latent_dim}")

    # Load data
    images = np.load(ae_cfg["data_path"], mmap_mode='r')
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

    # Augmentation + dataloader
    aug = JetAugmentation(
        img_size=128,
        translate_frac=cfg.get("translate_frac", 0.1),
        noise_std=cfg.get("noise_std", 0.02),
        intensity_range=tuple(cfg.get("intensity_range", [0.8, 1.2])),
        erase_prob=cfg.get("erase_prob", 0.3),
    )
    train_dataset = ContrastivePairDataset(images, train_idx, labels[train_idx], aug)
    train_loader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    logger.info(f"DataLoader: {len(train_loader)} batches, batch_size={cfg['batch_size']}")

    # MoCo
    model = MoCo(encoder, embed_dim=embed_dim,
                  queue_size=cfg.get("queue_size", 4096),
                  momentum=cfg.get("moco_momentum", 0.999),
                  temperature=cfg.get("temperature", 0.1)).to(device)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"MoCo trainable params: {total_params:,}")

    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["pretrain_epochs"], eta_min=cfg["min_lr"])

    # Phase 1: MoCo contrastive fine-tune
    logger.info("=== Phase 1: MoCo Contrastive Fine-tune ===")
    best_loss = float("inf")

    for epoch in range(cfg["pretrain_epochs"]):
        model.train()
        total_loss = 0.0
        num_batches = len(train_loader)
        for batch_i, (view1, view2, _) in enumerate(train_loader):
            view1, view2 = view1.to(device), view2.to(device)
            logits, moco_labels = model(view1, view2)
            loss = criterion(logits, moco_labels)
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
        logger.info(f"Pretrain Epoch {epoch+1:3d}/{cfg['pretrain_epochs']} | Loss: {avg_loss:.4f} | LR: {lr:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "loss": avg_loss,
                "config": cfg,
            }, output_dir / "pretrained.pt")
            logger.info(f"  -> Best model saved (loss={avg_loss:.4f})")

    logger.info(f"Pretraining done. Best loss: {best_loss:.4f}")

    # Phase 2: Linear probe
    logger.info("=== Phase 2: Linear Evaluation ===")
    eval_encoder = model.encoder_q
    eval_encoder.eval()

    def extract_features(indices):
        feats = []
        with torch.no_grad():
            for i in range(0, len(indices), cfg["batch_size"]):
                batch_idx = indices[i:i + cfg["batch_size"]]
                batch = torch.from_numpy(images[batch_idx].copy()).to(device)
                h, _ = eval_encoder(batch)
                feats.append(h.cpu())
        return torch.cat(feats)

    logger.info("Extracting features...")
    train_feats = extract_features(train_idx)
    val_feats = extract_features(val_idx)
    test_feats = extract_features(test_idx)
    feat_dim = train_feats.shape[1]
    logger.info(f"Feature dim: {feat_dim}")

    # Linear classifier
    torch.manual_seed(cfg["seed"])
    linear = nn.Linear(feat_dim, 2).to(device)
    lin_lr = cfg.get("linear_lr", 1e-3)
    linear_epochs = cfg.get("linear_epochs", 50)
    lin_optimizer = AdamW(linear.parameters(), lr=lin_lr, weight_decay=1e-4)
    lin_scheduler = CosineAnnealingLR(lin_optimizer, T_max=linear_epochs, eta_min=lin_lr * 0.01)
    lin_criterion = nn.CrossEntropyLoss()

    train_loader_lin = DataLoader(TensorDataset(train_feats, torch.tensor(labels[train_idx])), batch_size=512, shuffle=True)
    val_loader_lin = DataLoader(TensorDataset(val_feats, torch.tensor(labels[val_idx])), batch_size=512)

    best_val_auc = 0.0
    for epoch in range(linear_epochs):
        linear.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for f, l in train_loader_lin:
            f, l = f.to(device), l.to(device)
            out = linear(f)
            loss = lin_criterion(out, l)
            lin_optimizer.zero_grad()
            loss.backward()
            lin_optimizer.step()
            train_loss += loss.item() * f.size(0)
            train_correct += (out.argmax(1) == l).sum().item()
            train_total += f.size(0)

        linear.eval()
        val_probs, val_true = [], []
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for f, l in val_loader_lin:
                f, l = f.to(device), l.to(device)
                out = linear(f)
                val_loss += lin_criterion(out, l).item() * f.size(0)
                val_correct += (out.argmax(1) == l).sum().item()
                val_total += f.size(0)
                val_probs.extend(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
                val_true.extend(l.cpu().numpy())

        lin_scheduler.step()
        val_auc = roc_auc_score(val_true, val_probs)
        val_acc = val_correct / val_total
        train_acc = train_correct / train_total

        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            torch.save({"linear_state_dict": linear.state_dict(), "val_auc": val_auc}, output_dir / "linear_best.pt")
        marker = " *" if is_best else ""
        logger.info(
            f"Linear {epoch+1:3d}/{linear_epochs} | "
            f"Train Loss: {train_loss/train_total:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss/val_total:.4f} Acc: {val_acc:.4f} AUC: {val_auc:.4f}{marker}"
        )

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
    train_ae_moco(cfg)
