"""
Preprocess jet images: normalize, scale, zero-pad, and save as .npy.

Supports multiple preprocessing methods, each saved to a separate folder.

Usage:
    python preprocess.py --method perchannel_log    (default, existing)
    python preprocess.py --method perchannel_sqrt

Creates:
    data/processed/{method}/all_images.npy   (float32, memory-mapped)
"""

import argparse
import h5py
import numpy as np
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.logger import get_logger

DATA_DIR = Path("data")
QUARK_FILE = DATA_DIR / "quark_jets.h5"
GLUON_FILE = DATA_DIR / "gluon_jets.h5"
OUTPUT_BASE = DATA_DIR / "processed"
PAD_SIZE = 128
RAW_SIZE = 125
LOG_EPS = 1e-6


def count_samples(file_paths):
    total = 0
    for fp in file_paths:
        with h5py.File(fp, 'r') as f:
            total += f['images'].shape[0]
    return total


def compute_clip_vals(file_paths, percentile=99.9, transform=None):
    """Compute per-channel clip values from nonzero pixels."""
    nonzero_vals = [[], [], []]
    for fp in file_paths:
        with h5py.File(fp, 'r') as f:
            n = f['images'].shape[0]
            chunk_size = 5000
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                images = np.array(f['images'][start:end], dtype=np.float32)
                images = np.clip(images, 0, None)

                if transform == "pt_log":
                    total_pt = images.sum(axis=(2, 3), keepdims=True)
                    total_pt = np.maximum(total_pt, 1e-8)
                    images = images / total_pt
                    images = np.log(1 + images / LOG_EPS)

                for ch in range(3):
                    ch_data = images[:, ch].ravel()
                    nonzero_vals[ch].append(ch_data[ch_data > 0])

    clip_vals = np.zeros(3, dtype=np.float32)
    for ch in range(3):
        nz = np.concatenate(nonzero_vals[ch])
        clip_vals[ch] = np.percentile(nz, percentile)
    return np.maximum(clip_vals, 1e-8)


def apply_scaling(images, method, clip_vals=None):
    """Apply normalization + scaling."""
    images = np.clip(images, 0, None)

    if method == "percentile":
        for ch in range(3):
            images[:, ch] = np.clip(images[:, ch], 0, clip_vals[ch]) / clip_vals[ch]
    elif method == "pt_log_percentile":
        # pT norm -> log -> percentile clip to [0,1]
        total_pt = images.sum(axis=(2, 3), keepdims=True)
        total_pt = np.maximum(total_pt, 1e-8)
        images = images / total_pt
        images = np.log(1 + images / LOG_EPS)
        for ch in range(3):
            images[:, ch] = np.clip(images[:, ch], 0, clip_vals[ch]) / clip_vals[ch]
    else:
        # Per-channel pT normalization: each channel sums to 1
        total_pt = images.sum(axis=(2, 3), keepdims=True)
        total_pt = np.maximum(total_pt, 1e-8)
        images = images / total_pt

        if method == "perchannel_log":
            images = np.log(1 + images / LOG_EPS)
        elif method == "perchannel_sqrt":
            images = np.sqrt(images)
        # perchannel_raw: no further scaling

    return images


def process_and_write(file_paths, out_path, method, logger, clip_vals=None):
    """Process HDF5 files with specified method, write to mmap."""
    total = count_samples(file_paths)
    mmap = np.lib.format.open_memmap(
        str(out_path), mode='w+', dtype=np.float32,
        shape=(total, 3, PAD_SIZE, PAD_SIZE),
    )

    offset = 0
    for fp in file_paths:
        logger.info(f"Processing {fp}...")
        with h5py.File(fp, 'r') as f:
            n = f['images'].shape[0]
            chunk_size = 5000
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                images = np.array(f['images'][start:end], dtype=np.float32)
                images = apply_scaling(images, method, clip_vals=clip_vals)
                mmap[offset:offset + (end - start), :, 1:RAW_SIZE+1, 1:RAW_SIZE+1] = images
                offset += end - start

        logger.info(f"Written {n} samples (total so far: {offset})")

    mmap.flush()
    size_gb = offset * 3 * PAD_SIZE * PAD_SIZE * 4 / 1e9
    logger.info(f"Total: {offset} samples, {out_path} ({size_gb:.1f} GB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="perchannel_log",
                        choices=["perchannel_log", "perchannel_sqrt", "perchannel_raw", "percentile", "pt_log_percentile"],
                        help="Preprocessing method")
    args = parser.parse_args()

    out_dir = OUTPUT_BASE / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger(f"preprocess.{args.method}", log_dir=str(out_dir), log_file="preprocess.log")
    file_paths = [str(QUARK_FILE), str(GLUON_FILE)]

    clip_vals = None
    if args.method == "percentile":
        logger.info("Computing 99.9 percentile clip values (raw)...")
        clip_vals = compute_clip_vals(file_paths)
        for ch, name in enumerate(["Tracks", "ECAL", "HCAL"]):
            logger.info(f"  {name}: clip at {clip_vals[ch]:.6f}")
    elif args.method == "pt_log_percentile":
        logger.info("Computing 99.9 percentile clip values (pT+log space)...")
        clip_vals = compute_clip_vals(file_paths, transform="pt_log")
        for ch, name in enumerate(["Tracks", "ECAL", "HCAL"]):
            logger.info(f"  {name}: clip at {clip_vals[ch]:.6f}")

    logger.info(f"Preprocessing method: {args.method}")
    out_path = out_dir / "all_images.npy"
    process_and_write(file_paths, out_path, args.method, logger, clip_vals=clip_vals)
    logger.info("Done!")


if __name__ == "__main__":
    main()
