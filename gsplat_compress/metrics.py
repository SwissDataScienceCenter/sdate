"""
Quality and compression metrics for Gaussian-splatting video compression.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from skimage.metrics import structural_similarity as _ssim

from gsplat_compress.compression import (
    PARAMS_PER_GAUSSIAN_EFF,
    D_EFF,
    CompressedFrame,
)


# ═══════════════════════════════════════════════════════════════════════════
# Image quality metrics
# ═══════════════════════════════════════════════════════════════════════════

def psnr(prediction: np.ndarray, target: np.ndarray) -> float:
    """Peak signal-to-noise ratio (dB)."""
    data_range = float(target.max() - target.min())
    mse = float(np.mean((np.clip(prediction, target.min(), target.max()) - target) ** 2))
    return 10 * math.log10(data_range ** 2 / (mse + 1e-10))


def ssim(prediction: np.ndarray, target: np.ndarray, data_range: float = None) -> float:
    """Structural similarity index."""
    if data_range is None:
        data_range = float(target.max() - target.min())
    return float(_ssim(target, np.clip(prediction, 0, 1), data_range=data_range))


def mse(prediction: np.ndarray, target: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((np.clip(prediction, 0, 1) - target) ** 2))


# ═══════════════════════════════════════════════════════════════════════════
# Compression accounting
# ═══════════════════════════════════════════════════════════════════════════

def keyframe_bytes(n_gaussians: int, use_fp16: bool = True) -> int:
    """Storage for a keyframe in bytes (non-redundant parameters only).

    Uses ``PARAMS_PER_GAUSSIAN_EFF = 9`` non-redundant parameters per Gaussian
    (means_xy(2) + mean_z(1) + log_scales_xy(2) + angle(2) + rgb_scalar(1) + opacity(1)).
    """
    bpf = 2 if use_fp16 else 4
    return n_gaussians * PARAMS_PER_GAUSSIAN_EFF * bpf


def delta_frame_bytes(compressed: CompressedFrame, use_fp16: bool = True) -> dict[str, int]:
    """Storage breakdown for one delta frame."""
    return compressed.storage_bytes(use_fp16)


def raw_frame_bytes(H: int, W: int, bytes_per_pixel: int = 2) -> int:
    """Uncompressed frame size (default int16)."""
    return H * W * bytes_per_pixel


def compression_ratio(
    compressed_bytes: int,
    H: int,
    W: int,
    bytes_per_pixel: int = 2,
) -> float:
    """Ratio  raw_bytes / compressed_bytes."""
    raw = raw_frame_bytes(H, W, bytes_per_pixel)
    return raw / compressed_bytes if compressed_bytes > 0 else float("inf")


def sequence_storage_summary(
    n_gaussians: int,
    compressed_frames: list[CompressedFrame],
    H: int,
    W: int,
    use_fp16: bool = True,
    bytes_per_pixel: int = 2,
) -> dict:
    """Full storage summary for a compressed video sequence.

    Returns a dictionary with per-frame and total statistics.
    """
    kf_bytes = keyframe_bytes(n_gaussians, use_fp16)
    raw_single = raw_frame_bytes(H, W, bytes_per_pixel)
    n_frames = 1 + len(compressed_frames)  # keyframe + delta frames

    frame_details = [{"frame": 0, "type": "keyframe", "bytes": kf_bytes}]
    total = kf_bytes

    for i, cf in enumerate(compressed_frames, 1):
        info = cf.storage_bytes(use_fp16)
        frame_details.append({
            "frame": i,
            "type": "delta",
            "bytes": info["total_bytes"],
            "codebook_bytes": info["codebook_bytes"],
            "label_bytes": info["label_bytes"],
        })
        total += info["total_bytes"]

    raw_total = n_frames * raw_single
    return {
        "n_frames": n_frames,
        "total_compressed_bytes": total,
        "total_raw_bytes": raw_total,
        "overall_compression_ratio": raw_total / total if total > 0 else float("inf"),
        "keyframe_bytes": kf_bytes,
        "per_frame": frame_details,
    }
