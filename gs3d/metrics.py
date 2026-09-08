"""
Evaluation metrics for volumetric Gaussian Splatting.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean squared error between two tensors."""
    return F.mse_loss(pred.float(), target.float()).item()


def psnr(mse_val: float, max_val: float = 1.0) -> float:
    """Peak signal-to-noise ratio in dB."""
    if mse_val <= 0:
        return float("inf")
    return 10.0 * math.log10(max_val ** 2 / mse_val)


def ssim_3d(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 7,
) -> float:
    """Structural similarity index for 3-D volumes (slice-averaged).

    Computes 2-D SSIM on each slice along dim-0 and averages.

    Args:
        pred, target: (D, H, W)  tensors on CPU or GPU.
        window_size:  Gaussian window size.

    Returns:
        SSIM value in [0, 1].
    """
    pred = pred.float()
    target = target.float()
    D = pred.shape[0]
    ssim_sum = 0.0
    for i in range(D):
        ssim_sum += _ssim_2d(pred[i], target[i], window_size)
    return ssim_sum / D


def _ssim_2d(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 7,
) -> float:
    """Compute SSIM between two 2-D images (H, W)."""
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Add batch and channel dims
    x = img1.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    y = img2.unsqueeze(0).unsqueeze(0)

    # Gaussian kernel
    k = _gaussian_kernel_2d(window_size, 1.5, x.device)  # (1, 1, k, k)

    mu_x = F.conv2d(x, k, padding=window_size // 2)
    mu_y = F.conv2d(y, k, padding=window_size // 2)

    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x * x, k, padding=window_size // 2) - mu_x2
    sigma_y2 = F.conv2d(y * y, k, padding=window_size // 2) - mu_y2
    sigma_xy = F.conv2d(x * y, k, padding=window_size // 2) - mu_xy

    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)

    return (num / den).mean().item()


def _gaussian_kernel_2d(
    size: int,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    """Create a 2-D Gaussian kernel (1, 1, size, size)."""
    coords = torch.arange(size, device=device).float() - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel = g.outer(g)
    kernel = kernel / kernel.sum()
    return kernel.unsqueeze(0).unsqueeze(0)


def compression_ratio(
    original_shape: tuple,
    model_size_bytes: int,
    dtype_bits: int = 32,
) -> float:
    """Compute compression ratio = original_size / model_size.

    Args:
        original_shape: e.g. (D, H, W) of the target volume.
        model_size_bytes: total parameter bytes of the model.
        dtype_bits: bits per voxel in the original (default 32 for float32).
    """
    total_voxels = 1
    for s in original_shape:
        total_voxels *= s
    original_bytes = total_voxels * dtype_bits / 8
    if model_size_bytes <= 0:
        return float("inf")
    return original_bytes / model_size_bytes
