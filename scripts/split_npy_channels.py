#!/usr/bin/env python3
"""Split a (D, H, W, C) .npy volume into C independent (D, H, W) .npy files."""

import argparse
from pathlib import Path

import numpy as np


def split_channels(npy_path: Path) -> None:
    volume = np.load(npy_path)
    if volume.ndim != 4:
        raise ValueError(f"Expected 4D array (D, H, W, C), got shape {volume.shape}")

    D, H, W, num_channels = volume.shape
    print(f"Loaded {npy_path.name}: shape {volume.shape}")

    out_dir = npy_path.parent / npy_path.stem
    out_dir.mkdir(exist_ok=True)

    stem = npy_path.stem
    for i in range(num_channels):
        out_path = out_dir / f"{stem}_channel_{i}.npy"
        np.save(out_path, volume[..., i])

    print(f"Saved {num_channels} channel files to {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npy_path", type=Path, help="Path to the (D, H, W, C) .npy file")
    args = parser.parse_args()

    if not args.npy_path.exists():
        raise FileNotFoundError(f"File not found: {args.npy_path}")

    split_channels(args.npy_path)


if __name__ == "__main__":
    main()
