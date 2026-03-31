"""
Training script for Task 2: GNN Quark/Gluon Classifier.

Usage:
    python task2_gnn/train.py --config task2_gnn/configs/gcn_baseline.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.loader import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.logger import get_logger
from task2_gnn.dataset import JetGraphDataset
from task2_gnn.model import GCNClassifier, ParticleNet


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path("outputs/task2") / cfg["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(
        f"task2.{cfg['experiment_name']}",
        log_dir=str(output_dir),
        log_file="train.log",
    )
    logger.info(f"Experiment: {cfg['experiment_name']}")
    logger.info(f"Config: {cfg}")

    # Load dataset
    k = cfg.get("k", 16)
    graph_dir = Path(cfg["data_root"]) / "processed" / f"graphs_k{k}"
    dataset = JetGraphDataset(str(graph_dir))
    logger.info(f"Dataset: {len(dataset)} graphs")

    # Stratified split
    labels = np.array([dataset[i].y.item() for i in range(len(dataset))])
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=cfg["seed"])
    train_idx, temp_idx = next(splitter.split(np.zeros(len(labels)), labels))

    # Split temp into val + test (50/50 of remaining 20% = 10% each)
    temp_labels = labels[temp_idx]
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=cfg["seed"])
    val_rel, test_rel = next(splitter2.split(np.zeros(len(temp_labels)), temp_labels))
    val_idx = temp_idx[val_rel]
    test_idx = temp_idx[test_rel]

    logger.info(f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    logger.info(f"  Train quark/gluon: {(labels[train_idx]==0).sum()}/{(labels[train_idx]==1).sum()}")
    logger.info(f"  Val   quark/gluon: {(labels[val_idx]==0).sum()}/{(labels[val_idx]==1).sum()}")
    logger.info(f"  Test  quark/gluon: {(labels[test_idx]==0).sum()}/{(labels[test_idx]==1).sum()}")

    train_subset = torch.utils.data.Subset(dataset, train_idx)
    val_subset = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_subset, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=cfg["batch_size"], shuffle=False)

    # Save test indices for evaluation
    np.save(output_dir / "test_indices.npy", test_idx)

    # Model
    model_type = cfg.get("model", "gcn")
    if model_type == "particlenet":
        model = ParticleNet(
            in_channels=cfg.get("in_channels", 3),
            k=cfg.get("k", 16),
            num_classes=2,
        ).to(device)
    else:
        model = GCNClassifier(
            in_channels=cfg.get("in_channels", 3),
            hidden=cfg.get("hidden", 128),
            num_classes=2,
        ).to(device)

    logger.info(f"Model: {model_type}, Parameters: {model.count_parameters():,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["num_epochs"], eta_min=cfg["min_lr"])

    # Training loop
    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(cfg["num_epochs"]):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs
            pred = out.argmax(dim=1)
            train_correct += (pred == batch.y).sum().item()
            train_total += batch.num_graphs
        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_probs = []
        val_labels = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch)
                loss = criterion(out, batch.y)
                val_loss += loss.item() * batch.num_graphs
                pred = out.argmax(dim=1)
                val_correct += (pred == batch.y).sum().item()
                val_total += batch.num_graphs
                probs = torch.softmax(out, dim=1)[:, 1]
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(batch.y.cpu().numpy())

        val_loss /= val_total
        val_acc = val_correct / val_total
        val_auc = roc_auc_score(val_labels, val_probs)

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
                "model_state_dict": model.state_dict(),
                "val_auc": val_auc,
                "val_acc": val_acc,
                "config": cfg,
            }, output_dir / "best.pt")
            logger.info(f"  -> Best model saved (AUC={val_auc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= cfg["patience"]:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    logger.info(f"Done. Best val AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
