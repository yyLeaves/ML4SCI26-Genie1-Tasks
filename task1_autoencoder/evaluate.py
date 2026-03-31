"""
Evaluation for Task 1: reconstruction metrics + visual comparison.

Usage:
    python task1_autoencoder/evaluate.py --checkpoint outputs/task1/baseline/best.pt
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from pathlib import Path
from torch.utils.data import DataLoader

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.data import JetImageDataset, get_data_loaders
from common.logger import get_logger
from task1_autoencoder.model import ConvAutoencoder, ShallowAutoencoder, SegNetAutoencoder

CHANNEL_NAMES = ["Tracks", "ECAL", "HCAL"]
LOG_EPS = 1e-6
RAW_SIZE = 125


def inverse_preprocess(x_np, total_pt):
    """Invert per-channel pT norm + log scaling back to original energy space.

    x_np: (3, 128, 128) preprocessed image
    total_pt: (3,) per-channel total pT from raw data
    Returns: (3, 125, 125) in original energy units
    """
    # Crop padding: 128 -> 125
    x = x_np[:, 1:RAW_SIZE+1, 1:RAW_SIZE+1]
    # Invert log: log(1 + v/eps) -> v
    x = (np.exp(x) - 1) * LOG_EPS
    # Invert per-channel pT norm: v * total_pt_per_channel
    for ch in range(3):
        x[ch] = x[ch] * total_pt[ch]
    return x


def load_raw_total_pt(data_dir="data"):
    """Load per-channel total pT for each sample from raw HDF5 files."""
    import h5py
    data_dir = Path(data_dir)
    all_pt = []
    for fname in ["quark_jets.h5", "gluon_jets.h5"]:
        fp = data_dir / fname
        with h5py.File(fp, 'r') as f:
            n = f['images'].shape[0]
            pt = np.zeros((n, 3), dtype=np.float32)
            chunk = 2000
            for start in range(0, n, chunk):
                end = min(start + chunk, n)
                imgs = np.array(f['images'][start:end], dtype=np.float32)
                imgs = np.clip(imgs, 0, None)
                pt[start:end] = imgs.sum(axis=(2, 3))
            all_pt.append(pt)
    return np.concatenate(all_pt, axis=0)  # (N, 3)


def compute_metrics(model, loader, device):
    """Compute MSE, MAE, and per-channel MSE on the given loader."""
    model.eval()
    mse_sum, mae_sum, count = 0.0, 0.0, 0
    channel_mse = np.zeros(3)

    with torch.no_grad():
        for images in loader:
            images = images.to(device)
            recon = model(images)
            diff = recon - images
            mse_sum += (diff ** 2).mean().item() * images.size(0)
            mae_sum += diff.abs().mean().item() * images.size(0)
            for ch in range(3):
                channel_mse[ch] += ((diff[:, ch]) ** 2).mean().item() * images.size(0)
            count += images.size(0)

    return {
        "mse": mse_sum / count,
        "mae": mae_sum / count,
        "channel_mse": channel_mse / count,
    }


def plot_jet_events(model, dataset, save_dir, num_samples=4, seed=42, num_quark=69653):
    """Plot original vs reconstructed in paper style: Combined + 3 channels, log-scale colorbar.
    Samples half from quark, half from gluon."""
    rng = np.random.RandomState(seed)
    n_per_class = max(num_samples // 2, 1)
    quark_idx = rng.choice(num_quark, n_per_class, replace=False)
    gluon_idx = rng.choice(range(num_quark, len(dataset)), n_per_class, replace=False)
    indices = np.concatenate([quark_idx, gluon_idx])

    model.eval()
    device = next(model.parameters()).device

    col_titles = ["Combined", "Tracks", "ECAL", "HCAL"]
    save_dir = Path(save_dir)

    for i, idx in enumerate(indices):
        img = dataset[idx].unsqueeze(0).to(device)
        with torch.no_grad():
            recon = model(img)

        img_np = img[0].cpu().numpy()
        recon_np = recon[0].cpu().numpy()

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        cls_label = "quark" if idx < num_quark else "gluon"
        fig.suptitle(f"Event idx={int(idx)} ({cls_label})", fontsize=15, y=1.02)

        for row, (data, label) in enumerate([(img_np, "Original"), (recon_np, "Reconstructed")]):
            combined = data.sum(axis=0)
            panels = [combined, data[0], data[1], data[2]]

            for col, panel in enumerate(panels):
                ax = axes[row, col]
                vmin = 1e-3
                vmax = max(panel.max(), vmin * 10)
                # Mask zeros for log scale
                masked = np.ma.masked_where(panel <= 0, panel)
                im = ax.pcolormesh(masked, norm=LogNorm(vmin=vmin, vmax=vmax), cmap="viridis")
                ax.set_aspect("equal")
                ax.invert_yaxis()
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

                if row == 0:
                    ax.set_title(col_titles[col], fontsize=12)
                if col == 0:
                    ax.set_ylabel(label, fontsize=12)

                ax.set_xticks([0, 32, 64, 96, 128])
                ax.set_yticks([0, 32, 64, 96, 128])

        plt.tight_layout()
        fname = save_dir / f"event_{int(idx)}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")


def plot_scatter_comparison(model, dataset, save_dir, num_samples=4, seed=42, num_quark=69653):
    """Scatter plot style: nonzero pixels as dots, size/color = energy, log scale.
    Samples half from quark, half from gluon."""
    rng = np.random.RandomState(seed)
    n_per_class = max(num_samples // 2, 1)
    quark_idx = rng.choice(num_quark, n_per_class, replace=False)
    gluon_idx = rng.choice(range(num_quark, len(dataset)), n_per_class, replace=False)
    indices = np.concatenate([quark_idx, gluon_idx])

    model.eval()
    device = next(model.parameters()).device
    save_dir = Path(save_dir)

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 6 * num_samples))
    if num_samples == 1:
        axes = axes[np.newaxis, :]

    axes[0, 0].set_title("Original Image", fontsize=14, pad=15)
    axes[0, 1].set_title("Reconstructed", fontsize=14, pad=15)

    for i, idx in enumerate(indices):
        img = dataset[idx].unsqueeze(0).to(device)
        with torch.no_grad():
            recon = model(img)

        img_np = img[0].cpu().numpy().sum(axis=0)    # (128, 128)
        recon_np = recon[0].cpu().numpy().sum(axis=0)

        # Shared colorbar range for original and reconstructed
        all_vals = np.concatenate([img_np[img_np > 0], recon_np[recon_np > 0]]) if (img_np > 0).any() else np.array([1e-3])
        shared_vmin = max(all_vals.min(), 1e-3)
        shared_vmax = all_vals.max()

        for col, (data, label) in enumerate([(img_np, "Original"), (recon_np, "Reconstructed")]):
            ax = axes[i, col]

            # Extract nonzero pixels
            ys, xs = np.where(data > 0)
            vals = data[ys, xs]

            if len(vals) > 0:
                sizes = 5 + 80 * (vals - shared_vmin) / (shared_vmax - shared_vmin + 1e-8)

                sc = ax.scatter(xs, ys, c=vals, s=sizes, cmap="viridis",
                                norm=LogNorm(vmin=shared_vmin, vmax=max(shared_vmax, shared_vmin * 10)),
                                edgecolors="none", alpha=0.8)
                cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label("Energy (a.u.)", fontsize=9)

            ax.set_xlim(0, 128)
            ax.set_ylim(128, 0)
            ax.set_xlabel("iφ", fontsize=10)
            ax.set_ylabel("iη", fontsize=10)
            ax.set_aspect("equal")
            if col == 0:
                ax.text(-0.15, 0.5, f"idx={int(idx)}", transform=ax.transAxes,
                        fontsize=11, va="center", rotation=90)

    plt.tight_layout()
    fname = save_dir / "reconstruction_scatter.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {fname}")


def _plot_reconstruction_impl(samples, save_dir, suffix=""):
    """Internal: plot original/reconstructed/difference for each channel."""
    save_dir = Path(save_dir)
    num_samples = len(samples)

    for ch in range(3):
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
        if num_samples == 1:
            axes = axes[np.newaxis, :]

        tag = f" (raw)" if suffix else ""
        fig.suptitle(f"{CHANNEL_NAMES[ch]} Channel{tag}", fontsize=15, y=1.01)
        axes[0, 0].set_title("Original", fontsize=13)
        axes[0, 1].set_title("Reconstructed", fontsize=13)
        axes[0, 2].set_title("|Difference|", fontsize=13)

        for i, (idx, img_np, recon_np) in enumerate(samples):
            orig = img_np[ch]
            rec = recon_np[ch]
            diff = np.abs(orig - rec)
            vmax = max(orig.max(), rec.max(), 1e-6)

            axes[i, 0].imshow(orig, cmap="hot", vmin=0, vmax=vmax)
            axes[i, 1].imshow(rec, cmap="hot", vmin=0, vmax=vmax)
            axes[i, 2].imshow(diff, cmap="hot", vmin=0, vmax=vmax)

            for col in range(3):
                axes[i, col].set_xticks([])
                axes[i, col].set_yticks([])
            axes[i, 0].set_ylabel(f"idx={idx}", fontsize=11)

        plt.tight_layout()
        fname = save_dir / f"reconstruction_{CHANNEL_NAMES[ch].lower()}{suffix}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {fname}")


def plot_reconstruction(model, dataset, save_dir, num_samples=4, seed=42, total_pt=None, num_quark=69653):
    """Plot original / reconstructed / difference for each channel separately.
    Samples half from quark, half from gluon.
    Generates both normalized-space and raw-space versions if total_pt provided."""
    rng = np.random.RandomState(seed)
    n_per_class = max(num_samples // 2, 1)
    quark_idx = rng.choice(num_quark, n_per_class, replace=False)
    gluon_idx = rng.choice(range(num_quark, len(dataset)), n_per_class, replace=False)
    indices = np.concatenate([quark_idx, gluon_idx])

    model.eval()
    device = next(model.parameters()).device

    samples_norm = []
    samples_raw = []
    for idx in indices:
        img = dataset[idx].unsqueeze(0).to(device)
        with torch.no_grad():
            recon = model(img)
        img_np = img[0].cpu().numpy()
        recon_np = recon[0].cpu().numpy()
        samples_norm.append((int(idx), img_np, recon_np))

        if total_pt is not None:
            pt = total_pt[idx]
            img_raw = inverse_preprocess(img_np, pt)
            recon_raw = inverse_preprocess(recon_np, pt)
            samples_raw.append((int(idx), img_raw, recon_raw))

    # Normalized space
    _plot_reconstruction_impl(samples_norm, save_dir)

    # Raw space
    if samples_raw:
        _plot_reconstruction_impl(samples_raw, save_dir, suffix="_raw")


def _plot_avg_impl(avg_orig, avg_recon, save_path, tag=""):
    """Internal: plot average original vs reconstructed."""
    title = "Average Original vs Average Reconstructed"
    if tag:
        title += f" ({tag})"
    fig, axes = plt.subplots(3, 2, figsize=(6, 9))
    fig.suptitle(title, fontsize=14, y=1.01)

    for ch in range(3):
        vmax = max(avg_orig[ch].max(), avg_recon[ch].max(), 1e-6)

        axes[ch, 0].imshow(avg_orig[ch], cmap="inferno", vmin=0, vmax=vmax)
        axes[ch, 0].set_ylabel(CHANNEL_NAMES[ch], fontsize=11)
        axes[ch, 0].set_xticks([])
        axes[ch, 0].set_yticks([])
        if ch == 0:
            axes[ch, 0].set_title("Original")

        axes[ch, 1].imshow(avg_recon[ch], cmap="inferno", vmin=0, vmax=vmax)
        axes[ch, 1].set_xticks([])
        axes[ch, 1].set_yticks([])
        if ch == 0:
            axes[ch, 1].set_title("Reconstructed")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def plot_average_reconstruction(model, loader, device, save_path, total_pt=None, val_indices=None):
    """Plot average image vs average reconstruction per channel.
    Generates both normalized and raw-space versions if total_pt provided."""
    model.eval()
    avg_orig = None
    avg_recon = None
    avg_orig_raw = None
    avg_recon_raw = None
    count = 0
    sample_idx = 0

    with torch.no_grad():
        for images in loader:
            batch_size = images.size(0)
            images = images.to(device)
            recon = model(images)

            if avg_orig is None:
                avg_orig = images.sum(dim=0).cpu().numpy()
                avg_recon = recon.sum(dim=0).cpu().numpy()
            else:
                avg_orig += images.sum(dim=0).cpu().numpy()
                avg_recon += recon.sum(dim=0).cpu().numpy()

            if total_pt is not None and val_indices is not None:
                for j in range(batch_size):
                    global_idx = val_indices[sample_idx + j]
                    pt = total_pt[global_idx]
                    o_raw = inverse_preprocess(images[j].cpu().numpy(), pt)
                    r_raw = inverse_preprocess(recon[j].cpu().numpy(), pt)
                    if avg_orig_raw is None:
                        avg_orig_raw = o_raw
                        avg_recon_raw = r_raw
                    else:
                        avg_orig_raw += o_raw
                        avg_recon_raw += r_raw

            count += batch_size
            sample_idx += batch_size

    avg_orig /= count
    avg_recon /= count

    # Normalized space
    _plot_avg_impl(avg_orig, avg_recon, save_path)

    # Raw space
    if avg_orig_raw is not None:
        avg_orig_raw /= count
        avg_recon_raw /= count
        raw_path = str(save_path).replace(".png", "_raw.png")
        _plot_avg_impl(avg_orig_raw, avg_recon_raw, raw_path, tag="raw")


def plot_avg_by_class(model, dataset, device, save_dir, train_indices, val_indices, num_quark=69653):
    """Plot average reconstruction separately for quark/gluon and train/val splits."""
    model.eval()
    save_dir = Path(save_dir)

    train_set = set(train_indices)
    val_set = set(val_indices)

    groups = {}
    for cls_name, cls_indices in [("quark", range(num_quark)), ("gluon", range(num_quark, len(dataset)))]:
        for split_name, split_set in [("train", train_set), ("val", val_set)]:
            key = f"{cls_name}_{split_name}"
            groups[key] = [i for i in cls_indices if i in split_set]

    for group_name, indices in groups.items():
        if not indices:
            continue
        avg_orig = None
        avg_recon = None
        count = 0

        with torch.no_grad():
            batch_size = 256
            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start:start + batch_size]
                imgs = torch.stack([dataset[i] for i in batch_idx]).to(device)
                recon = model(imgs)

                if avg_orig is None:
                    avg_orig = imgs.sum(dim=0).cpu().numpy()
                    avg_recon = recon.sum(dim=0).cpu().numpy()
                else:
                    avg_orig += imgs.sum(dim=0).cpu().numpy()
                    avg_recon += recon.sum(dim=0).cpu().numpy()
                count += len(batch_idx)

        avg_orig /= count
        avg_recon /= count

        save_path = save_dir / f"avg_reconstruction_{group_name}.png"
        _plot_avg_impl(avg_orig, avg_recon, save_path, tag=group_name.replace("_", " "))


def plot_loss_distribution(model, dataset, val_indices, device, save_path, num_quark=69653):
    """Plot per-sample MSE distribution, split by quark/gluon."""
    model.eval()
    quark_losses = []
    gluon_losses = []

    with torch.no_grad():
        batch_size = 256
        for start in range(0, len(val_indices), batch_size):
            batch_idx = val_indices[start:start + batch_size]
            imgs = torch.stack([dataset[i] for i in batch_idx]).to(device)
            recon = model(imgs)
            per_sample = ((recon - imgs) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
            for j, idx in enumerate(batch_idx):
                if idx < num_quark:
                    quark_losses.append(per_sample[j])
                else:
                    gluon_losses.append(per_sample[j])

    quark_losses = np.array(quark_losses)
    gluon_losses = np.array(gluon_losses)
    all_losses = np.concatenate([quark_losses, gluon_losses])

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(np.log10(all_losses.min() + 1e-10), np.log10(all_losses.max() + 1e-10), 80)
    ax.hist(np.log10(quark_losses + 1e-10), bins=bins, alpha=0.6, label=f"Quark (mean={quark_losses.mean():.2e})", color="blue")
    ax.hist(np.log10(gluon_losses + 1e-10), bins=bins, alpha=0.6, label=f"Gluon (mean={gluon_losses.mean():.2e})", color="red")
    ax.set_xlabel("log10(Per-sample MSE)")
    ax.set_ylabel("Count")
    ax.set_title("Reconstruction Loss Distribution (Validation)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_samples", type=int, default=6)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    save_dir = Path(args.checkpoint).parent
    logger = get_logger("task1.eval", log_dir=str(save_dir), log_file="eval.log")

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
    model.load_state_dict(ckpt["model_state_dict"])
    logger.info(f"Loaded model from epoch {ckpt['epoch']+1}, val_loss={ckpt['val_loss']:.6f}")

    # Data loaders (same split as training)
    data_path = cfg["data_path"]
    train_loader, val_loader, total = get_data_loaders(
        data_path,
        train_split=cfg["train_split"],
        batch_size=cfg["batch_size"],
        seed=cfg["seed"],
    )
    logger.info(f"Data: {total} samples (train: {len(train_loader.dataset)}, val: {len(val_loader.dataset)})")

    # Metrics
    logger.info("--- Validation Metrics ---")
    val_metrics = compute_metrics(model, val_loader, device)
    logger.info(f"  MSE:  {val_metrics['mse']:.6f}")
    logger.info(f"  MAE:  {val_metrics['mae']:.6f}")
    for ch, name in enumerate(CHANNEL_NAMES):
        logger.info(f"  MSE ({name:6s}): {val_metrics['channel_mse'][ch]:.6f}")

    logger.info("--- Train Metrics ---")
    train_metrics = compute_metrics(model, train_loader, device)
    logger.info(f"  MSE:  {train_metrics['mse']:.6f}")
    logger.info(f"  MAE:  {train_metrics['mae']:.6f}")

    # Load raw total_pt for inverse transform
    logger.info("Loading raw total pT for inverse transform...")
    total_pt_all = load_raw_total_pt()
    logger.info(f"Loaded total_pt: {total_pt_all.shape}")

    # Get val indices for inverse transform in average plot
    generator = torch.Generator().manual_seed(cfg["seed"])
    num_train = int(len(JetImageDataset(data_path)) * cfg["train_split"])
    num_val = len(JetImageDataset(data_path)) - num_train
    all_indices = torch.randperm(len(JetImageDataset(data_path)), generator=generator).tolist()
    val_indices = all_indices[num_train:]

    # Plots
    full_dataset = JetImageDataset(data_path)
    plot_reconstruction(model, full_dataset, save_dir, args.num_samples, total_pt=total_pt_all)
    logger.info("Saved reconstruction plots (normalized + raw)")
    plot_jet_events(model, full_dataset, save_dir, args.num_samples)
    logger.info("Saved jet event plots")
    plot_scatter_comparison(model, full_dataset, save_dir, args.num_samples)
    logger.info("Saved scatter comparison plot")
    plot_average_reconstruction(model, val_loader, device, save_dir / "avg_reconstruction.png",
                                total_pt=total_pt_all, val_indices=val_indices)
    logger.info(f"Saved: {save_dir / 'avg_reconstruction.png'} (normalized + raw)")
    plot_loss_distribution(model, full_dataset, val_indices, device, save_dir / "loss_distribution.png")
    logger.info(f"Saved: {save_dir / 'loss_distribution.png'}")
    train_indices = all_indices[:num_train]
    plot_avg_by_class(model, full_dataset, device, save_dir, train_indices, val_indices)
    logger.info("Saved quark/gluon x train/val avg reconstruction plots")

    logger.info("Done!")


if __name__ == "__main__":
    main()
