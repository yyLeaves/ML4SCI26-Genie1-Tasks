"""
Data preparation script for Task 1

Converts the downloaded quark-gluon dataset into organized HDF5 files
for training and evaluation.

Usage:
    python prepare_data.py
"""

import h5py
import numpy as np
from pathlib import Path
import sys


def prepare_data(source_file="quark-gluon_data-set_n139306.hdf5"):
    """
    Read the downloaded dataset and split into quark/gluon HDF5 files.
    
    Actual source file structure:
    - X_jets: shape (N, 125, 125, 3) in (H, W, C) format
    - y: shape (N,) with 0=quark, 1=gluon
    - m0, pt: additional jet properties
    
    Creates:
    - data/quark_jets.h5 (for training)
    - data/gluon_jets.h5 (for evaluation)
    """

    # Create output directory
    output_dir = Path("data")
    output_dir.mkdir(exist_ok=True)

    print(f"Reading source file: {source_file}")

    # Open source file and read data
    try:
        with h5py.File(source_file, 'r') as f:
            print(f"Opened {source_file}")
            print(f"  Available keys: {list(f.keys())}")

            # Inspect structure
            for key in f.keys():
                print(f"    {key}: shape {f[key].shape}, dtype {f[key].dtype}")

            # Read X_jets and convert from (N, H, W, C) to (N, C, H, W)
            if 'X_jets' not in f:
                print("Error: 'X_jets' not found in HDF5 file")
                return False

            images_hwc = np.array(f['X_jets'])  # (N, 125, 125, 3)
            images = np.transpose(images_hwc,
                                  (0, 3, 1, 2))  # Convert to (N, 3, 125, 125)
            print(
                f"\nLoaded images: shape {images.shape} (converted from {images_hwc.shape})"
            )

            # Read labels
            if 'y' not in f:
                print("Error: 'y' not found in HDF5 file")
                return False

            labels = np.array(f['y'])
            print(f"Loaded labels: shape {labels.shape}")

    except FileNotFoundError:
        print(f"Error: File not found: {source_file}")
        return False
    except Exception as e:
        print(f"Error reading file: {e}")
        return False

    # Split by label
    quark_mask = labels == 0
    gluon_mask = labels == 1

    quark_images = images[quark_mask]
    gluon_images = images[gluon_mask]

    print(f"\nLabel breakdown:")
    print(f"  Quark jets (label 0): {quark_images.shape[0]:,}")
    print(f"  Gluon jets (label 1): {gluon_images.shape[0]:,}")

    # Save to separate HDF5 files
    quark_file = output_dir / "quark_jets.h5"
    gluon_file = output_dir / "gluon_jets.h5"

    print(f"\nSaving to HDF5 files...")

    # Quark file
    with h5py.File(quark_file, 'w') as f:
        f.create_dataset('images',
                         data=quark_images,
                         compression='gzip',
                         compression_opts=4)
    print(f"Saved {quark_file}: {quark_images.shape}")

    # Gluon file
    with h5py.File(gluon_file, 'w') as f:
        f.create_dataset('images',
                         data=gluon_images,
                         compression='gzip',
                         compression_opts=4)
    print(f"Saved {gluon_file}: {gluon_images.shape}")

    print(f"\nData preparation complete!")
    print(f"Ready to train with: python train.py")

    return True


if __name__ == "__main__":
    success = prepare_data()
    sys.exit(0 if success else 1)
