"""Fast GPU anomaly-score metrics.

MSE and MAE only (no SSIM -- too expensive to compute at streaming rates on
GPU per the project's own experience). Both trivially GPU-native: the
reconstructed volumes already live on-device coming out of ``reconstruct()``,
so this is a couple of masked elementwise ops, no extra transfer.
"""
from __future__ import annotations

import torch


def masked_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    return float(torch.mean((a[..., mask] - b[..., mask]) ** 2))


def masked_mae(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> float:
    return float(torch.mean((a[..., mask] - b[..., mask]).abs()))
