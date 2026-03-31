"""
Convert jet images to point clouds for GNN classification.

Step 1: preprocess_pointclouds() — extract nonzero pixels, save as .pt (no edges)
Step 2: JetGraphDataset loads point clouds, builds kNN edges on-the-fly via transform

Each nonzero pixel becomes a node with:
  - pos: (eta, phi) physical coordinates
  - x: [Tracks, ECAL, HCAL] energy values at that pixel
  - y: 0 (quark) or 1 (gluon)
"""

import argparse
import time
import h5py
import numpy as np
import torch
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import KNNGraph

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common.logger import get_logger


PIXEL_SIZE = 0.025  # radians per pixel
CENTER = 62  # center pixel index (125x125 image)


def image_to_pointcloud(image, label, max_nodes=None):
    """Convert a 3-channel 125x125 jet image to a PyG Data (no edges).

    Args:
        image: (3, 125, 125) numpy array [Tracks, ECAL, HCAL]
        label: 0 (quark) or 1 (gluon)
        max_nodes: if set, keep only top-N nodes by total energy
    Returns:
        torch_geometric.data.Data with pos, x, y
    """
    mask = image.sum(axis=0) > 0
    rows, cols = np.where(mask)

    if len(rows) == 0:
        return Data(
            x=torch.zeros(1, 3, dtype=torch.float),
            pos=torch.zeros(1, 2, dtype=torch.float),
            y=torch.tensor([label], dtype=torch.long),
        )

    # Top-N filtering by total energy
    if max_nodes and len(rows) > max_nodes:
        total_energy = image.sum(axis=0)[rows, cols]
        top_idx = np.argsort(total_energy)[-max_nodes:]
        rows = rows[top_idx]
        cols = cols[top_idx]

    eta = (rows - CENTER) * PIXEL_SIZE
    phi = (cols - CENTER) * PIXEL_SIZE
    pos = np.stack([eta, phi], axis=1)
    features = image[:, rows, cols].T  # (N, 3)

    return Data(
        x=torch.tensor(features, dtype=torch.float),
        pos=torch.tensor(pos, dtype=torch.float),
        y=torch.tensor([label], dtype=torch.long),
    )


def _process_chunk(args):
    """Worker function: convert a chunk of HDF5 images to point clouds."""
    images, label, start_idx, max_nodes = args
    results = []
    for j in range(images.shape[0]):
        results.append(image_to_pointcloud(images[j], label, max_nodes=max_nodes))
    return start_idx, results


def _process_chunk_npy(args):
    """Worker function: convert a chunk of npy images to point clouds."""
    images, batch_labels, start_idx, max_nodes = args
    results = []
    for j in range(images.shape[0]):
        results.append(image_to_pointcloud(images[j], int(batch_labels[j]), max_nodes=max_nodes))
    return start_idx, results


def preprocess_pointclouds(data_dir="data", output_path=None, num_workers=4, max_nodes=None, npy_path=None):
    """Convert jet images to point clouds and save as .pt file.

    Can read from HDF5 (raw) or preprocessed .npy (percentile normalized).
    Uses multiprocessing to speed up conversion.
    """
    data_dir = Path(data_dir)
    if output_path is None:
        output_path = data_dir / "processed" / "jet_pointclouds.pt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger = get_logger("preprocess_pc", log_dir=str(output_path.parent), log_file="pointcloud_preprocess.log")

    all_data = []
    total_nodes = 0

    if npy_path:
        # Read from preprocessed .npy (percentile normalized, 128x128 padded)
        logger.info(f"Reading from {npy_path}...")
        images = np.load(npy_path, mmap_mode='r')
        n = len(images)
        n_quark = n // 2  # first half quark, second half gluon
        logger.info(f"Total: {n} images ({n_quark} quark + {n - n_quark} gluon)")

        chunk_size = 2000
        chunks = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            batch = np.array(images[start:end])
            # Crop padding 128->125: images are padded at [1:126, 1:126]
            batch = batch[:, :, 1:126, 1:126]
            batch_labels = np.array([0 if (start + j) < n_quark else 1 for j in range(end - start)])
            chunks.append((batch, batch_labels, start, max_nodes))

        logger.info(f"Converting {n} images with {num_workers} workers (max_nodes={max_nodes})...")
        chunk_results = [None] * len(chunks)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_chunk_npy, c): i for i, c in enumerate(chunks)}
            done_count = 0
            for future in as_completed(futures):
                idx = futures[future]
                start_idx, results = future.result()
                chunk_results[idx] = results
                done_count += 1
                if done_count % 5 == 0 or done_count == len(chunks):
                    processed = sum(len(r) for r in chunk_results if r is not None)
                    logger.info(f"  {processed}/{n} converted")

        for results in chunk_results:
            for d in results:
                total_nodes += d.x.shape[0]
                all_data.append(d)
    else:
        # Read from raw HDF5
        files = [
            (data_dir / "quark_jets.h5", 0),
            (data_dir / "gluon_jets.h5", 1),
        ]

        for fp, label in files:
            cls_name = "quark" if label == 0 else "gluon"
            logger.info(f"Processing {fp} (label={label}, class={cls_name})...")
            t0 = time.time()

            with h5py.File(fp, "r") as f:
                n = f["images"].shape[0]
                chunk_size = 2000
                chunks = []

                for start in range(0, n, chunk_size):
                    end = min(start + chunk_size, n)
                    batch = np.array(f["images"][start:end], dtype=np.float32)
                    batch = np.clip(batch, 0, None)
                    chunks.append((batch, label, start, max_nodes))

            logger.info(f"  Converting {n} images with {num_workers} workers (max_nodes={max_nodes})...")
        chunk_results = [None] * len(chunks)

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(_process_chunk, c): i for i, c in enumerate(chunks)}
            done_count = 0
            for future in as_completed(futures):
                idx = futures[future]
                start_idx, results = future.result()
                chunk_results[idx] = results
                done_count += 1
                if done_count % 5 == 0 or done_count == len(chunks):
                    processed = sum(len(r) for r in chunk_results if r is not None)
                    logger.info(f"  {processed}/{n} converted")

        # Flatten in order
        for results in chunk_results:
            for d in results:
                total_nodes += d.x.shape[0]
                all_data.append(d)

        elapsed = time.time() - t0
        logger.info(f"  {cls_name}: {n} images -> {n} point clouds in {elapsed:.1f}s")

    logger.info(f"Total: {len(all_data)} point clouds, avg nodes: {total_nodes / len(all_data):.1f}")
    logger.info(f"Saving to {output_path}...")
    torch.save(all_data, output_path)
    logger.info(f"Done! File size: {output_path.stat().st_size / 1e6:.1f} MB")
    return all_data


def build_edges_and_shard(pc_path, k=16, output_dir=None, shard_size=16384):
    """Add kNN edges and save as sharded files for fast loading.

    Saves:
      output_dir/shard_0000.pt, shard_0001.pt, ... (each has shard_size graphs)
      output_dir/metadata.pt  (labels, num_nodes, shard info)
    """
    pc_path = Path(pc_path)
    if output_dir is None:
        output_dir = pc_path.parent / f"graphs_k{k}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("build_edges", log_dir=str(output_dir), log_file="build_edges.log")
    logger.info(f"Loading point clouds from {pc_path}...")
    data_list = torch.load(pc_path, weights_only=False)
    n = len(data_list)
    logger.info(f"Loaded {n} point clouds")

    logger.info(f"Building kNN edges (k={k}), shard_size={shard_size}...")
    t0 = time.time()
    labels = []
    num_nodes = []
    shard = []
    shard_idx = 0

    for i, data in enumerate(data_list):
        if data.x.shape[0] > 1:
            k_actual = min(k, data.x.shape[0] - 1)
            transform = KNNGraph(k=k_actual)
            data = transform(data)

        labels.append(data.y.item())
        num_nodes.append(data.x.shape[0])
        shard.append(data)

        if len(shard) == shard_size:
            torch.save(shard, output_dir / f"shard_{shard_idx:04d}.pt")
            shard_idx += 1
            shard = []

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            logger.info(f"  {i+1}/{n} ({elapsed:.1f}s)")

    # Save remaining
    if shard:
        torch.save(shard, output_dir / f"shard_{shard_idx:04d}.pt")
        shard_idx += 1

    metadata = {
        "n": n,
        "labels": np.array(labels, dtype=np.int64),
        "num_nodes": np.array(num_nodes, dtype=np.int64),
        "k": k,
        "shard_size": shard_size,
        "num_shards": shard_idx,
    }
    torch.save(metadata, output_dir / "metadata.pt")

    elapsed = time.time() - t0
    logger.info(f"Done in {elapsed:.1f}s. {shard_idx} shards saved to {output_dir}")
    logger.info(f"Avg nodes: {np.mean(num_nodes):.1f}, median: {np.median(num_nodes):.0f}")


class JetGraphDataset(torch.utils.data.Dataset):
    """Dataset of jet graphs stored as sharded files.

    Loads shards on demand and caches them in memory.
    """

    def __init__(self, graph_dir):
        self.graph_dir = Path(graph_dir)
        meta = torch.load(self.graph_dir / "metadata.pt", weights_only=False)
        self.n = meta["n"]
        self.labels = meta["labels"]
        self.shard_size = meta["shard_size"]
        self.num_shards = meta["num_shards"]
        self._cache = {}

    def __len__(self):
        return self.n

    def _load_shard(self, shard_idx):
        if shard_idx not in self._cache:
            self._cache[shard_idx] = torch.load(
                self.graph_dir / f"shard_{shard_idx:04d}.pt", weights_only=False
            )
        return self._cache[shard_idx]

    def __getitem__(self, idx):
        shard_idx = idx // self.shard_size
        local_idx = idx % self.shard_size
        shard = self._load_shard(shard_idx)
        return shard[local_idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", default="pointclouds", choices=["pointclouds", "edges"])
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--output", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--max_nodes", type=int, default=None, help="Keep top-N nodes by energy")
    parser.add_argument("--npy_path", default=None, help="Path to preprocessed .npy (e.g. percentile)")
    args = parser.parse_args()

    if args.action == "pointclouds":
        preprocess_pointclouds(args.data_dir, args.output, args.workers, max_nodes=args.max_nodes, npy_path=args.npy_path)
    elif args.action == "edges":
        pc_path = args.output or str(Path(args.data_dir) / "processed" / "jet_pointclouds.pt")
        build_edges_and_shard(pc_path, k=args.k, output_dir=None)
