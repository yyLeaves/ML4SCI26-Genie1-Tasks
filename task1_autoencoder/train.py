"""
Training script for Task 1: Convolutional Autoencoder

Usage:
    python preprocess.py                   # run once
    python task1_autoencoder/train.py --config task1_autoencoder/configs/baseline.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import get_data_loaders
from common.logger import get_logger
from task1_autoencoder.model import ConvAutoencoder, ShallowAutoencoder, SegNetAutoencoder


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train(cfg):
    exp_name = cfg["experiment_name"]
    output_dir = Path("outputs") / "task1" / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger(f"task1.{exp_name}", log_dir=str(output_dir))
    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Config: {cfg}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # Data
    data_path = cfg["data_path"]
    if not Path(data_path).exists():
        logger.error(f"{data_path} not found. Run: python preprocess.py")
        return

    train_loader, val_loader, total = get_data_loaders(
        data_path,
        train_split=cfg["train_split"],
        batch_size=cfg["batch_size"],
        seed=cfg["seed"],
    )
    logger.info(f"Data: {total} samples")

    # Model
    model_type = cfg.get("model", "conv")
    if model_type == "segnet":
        model = SegNetAutoencoder(
            latent_dim=cfg.get("latent_dim", 512),
            channels=cfg["encoder_channels"],
            output_act=cfg.get("output_act", "relu"),
        ).to(device)
    elif model_type == "shallow":
        model = ShallowAutoencoder(
            bottleneck_ch=cfg.get("bottleneck_ch", 4),
            channels=cfg["encoder_channels"],
        ).to(device)
    else:
        model = ConvAutoencoder(
            latent_dim=cfg.get("latent_dim", 128),
            channels=cfg["encoder_channels"],
            output_act=cfg.get("output_act", "relu"),
        ).to(device)
    logger.info(f"Parameters: {model.count_parameters():,}")

    # Resume from checkpoint
    start_epoch = 0
    if cfg.get("_resume"):
        ckpt = torch.load(cfg["_resume"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        logger.info(f"Resumed from epoch {start_epoch} (val_loss={ckpt['val_loss']:.6f})")

    # Loss function
    loss_type = cfg.get("loss", "mse")
    if loss_type == "weighted_mse":
        nonzero_weight = cfg.get("nonzero_weight", 10.0)
        logger.info(f"Loss: weighted MSE (nonzero_weight={nonzero_weight})")
        def criterion(pred, target):
            mse = (pred - target) ** 2
            weight = torch.where(target > 0, nonzero_weight, 1.0)
            return (mse * weight).mean()
    else:
        logger.info("Loss: MSE")
        criterion = nn.MSELoss()

    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"], eta_min=cfg["min_lr"])

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(start_epoch, cfg["num_epochs"]):
        model.train()
        train_loss = 0.0
        for images in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            images = images.to(device)
            loss = criterion(model(images), images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images in val_loader:
                images = images.to(device)
                val_loss += criterion(model(images), images).item()
        val_loss /= len(val_loader)

        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        logger.info(f"Epoch {epoch+1:3d}/{cfg['num_epochs']} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "config": cfg,
            }, output_dir / "best.pt")
            logger.info(f"  -> Best model saved ({val_loss:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg["patience"]:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    logger.info(f"Done. Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["_resume"] = args.resume
    train(cfg)
