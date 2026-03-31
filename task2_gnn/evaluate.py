"""
Evaluation for Task 2: GNN Quark/Gluon Classification.

Usage:
    python task2_gnn/evaluate.py --checkpoint outputs/task2/gcn_baseline/best.pt
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from torch_geometric.loader import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.logger import get_logger
from task2_gnn.dataset import JetGraphDataset
from task2_gnn.model import GCNClassifier, ParticleNet


def evaluate_model(model, loader, device):
    """Run model on loader, return predictions and labels."""
    model.eval()
    all_probs = []
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            probs = torch.softmax(out, dim=1)[:, 1]
            preds = out.argmax(dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def plot_roc_curve(labels, probs, save_path, model_name="Model"):
    """Plot ROC curve with AUC."""
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, label=f"{model_name} (AUC={auc:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Quark vs Gluon", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(labels, preds, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Quark", "Gluon"], fontsize=11)
    ax.set_yticklabels(["Quark", "Gluon"], fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    fontsize=14, color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_score_distribution(labels, probs, save_path):
    """Plot gluon probability distribution for quark and gluon jets."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(probs[labels == 0], bins=50, alpha=0.6, label="Quark", color="blue", density=True)
    ax.hist(probs[labels == 1], bins=50, alpha=0.6, label="Gluon", color="red", density=True)
    ax.set_xlabel("Gluon Probability", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Classification Score Distribution", fontsize=14)
    ax.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = Path(args.checkpoint).parent
    logger = get_logger(f"task2.eval.{cfg['experiment_name']}", log_dir=str(save_dir), log_file="eval.log")
    logger.info(f"Evaluating: {cfg['experiment_name']}")

    # Load dataset
    k = cfg.get("k", 16)
    graph_dir = Path(cfg["data_root"]) / "processed" / f"graphs_k{k}"
    dataset = JetGraphDataset(str(graph_dir))

    # Load test indices
    test_idx = np.load(save_dir / "test_indices.npy")
    test_data = [dataset[i] for i in test_idx]
    test_loader = DataLoader(test_data, batch_size=cfg.get("batch_size", 128), shuffle=False)
    logger.info(f"Test set: {len(test_idx)} samples")

    # Build model
    model_type = cfg.get("model", "gcn")
    if model_type == "particlenet":
        model = ParticleNet(in_channels=cfg.get("in_channels", 3), k=cfg.get("k", 16)).to(device)
    else:
        model = GCNClassifier(in_channels=cfg.get("in_channels", 3), hidden=cfg.get("hidden", 128)).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded model from epoch {ckpt['epoch']+1}, val_auc={ckpt['val_auc']:.4f}")

    # Evaluate
    labels, preds, probs = evaluate_model(model, test_loader, device)
    auc = roc_auc_score(labels, probs)
    acc = (preds == labels).mean()

    logger.info(f"--- Test Metrics ---")
    logger.info(f"  AUC:      {auc:.4f}")
    logger.info(f"  Accuracy: {acc:.4f}")
    logger.info(f"\n{classification_report(labels, preds, target_names=['Quark', 'Gluon'])}")

    # Plots
    model_name = cfg["experiment_name"]
    plot_roc_curve(labels, probs, save_dir / "roc_curve.png", model_name)
    logger.info("Saved ROC curve")
    plot_confusion_matrix(labels, preds, save_dir / "confusion_matrix.png")
    logger.info("Saved confusion matrix")
    plot_score_distribution(labels, probs, save_dir / "score_distribution.png")
    logger.info("Saved score distribution")

    logger.info("Done!")


if __name__ == "__main__":
    main()
