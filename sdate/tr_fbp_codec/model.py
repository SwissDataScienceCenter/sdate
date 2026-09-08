"""3D ResNet predictor: k context frames (+1 optional FBP-prior tap) -> per-pixel
12-bit class logits for the target frame.

This is our own architecture choice (see README.md) -- the paper's 32x32
block-tiled TCN+Octave-Conv network is treated as a loose rule of thumb, not
reproduced literally. ``CodecConfig.use_fbp_prior`` (config.py) toggles
whether the extra prior tap is part of ``depth_in``, so the exact same
model class produces both the "paper reproduction" baseline and "ours".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    depth_in: int  # k context frames, +1 if the FBP-prior tap is included
    n_classes: int = 4096
    base_channels: int = 32
    n_blocks: int = 6


class ResBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class FramePredictorResNet3D(nn.Module):
    """(B, 1, D, H, W) -> (B, n_classes, H, W), D == cfg.depth_in."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        c = cfg.base_channels
        self.stem = nn.Sequential(
            nn.Conv3d(1, c, kernel_size=3, padding=1),
            nn.BatchNorm3d(c),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResBlock3D(c) for _ in range(cfg.n_blocks)])
        # Collapse the context/depth axis to 1 with a full-depth conv (rather
        # than e.g. mean-pooling) so the network can learn to weight context
        # frames vs. the FBP-prior tap differently.
        self.collapse = nn.Conv3d(c, c, kernel_size=(cfg.depth_in, 1, 1))
        self.head = nn.Conv3d(c, cfg.n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] != self.cfg.depth_in:
            raise ValueError(f"expected depth={self.cfg.depth_in}, got {x.shape[2]}")
        out = self.stem(x)
        out = self.blocks(out)
        out = self.collapse(out)  # (B, c, 1, H, W)
        out = self.head(out)  # (B, n_classes, 1, H, W)
        return out.squeeze(2)


def stack_inputs(
    context: torch.Tensor, prior: Optional[torch.Tensor], n_levels: int = 4096
) -> torch.Tensor:
    """Build the (B, 1, D, H, W) input stack from raw-count tensors.

    ``context``: (B, k, H, W) raw counts, oldest-first.
    ``prior``: (B, H, W) FBP-reprojected raw-count-domain tap, or None to
    reproduce the paper's no-prior baseline (must then match
    ``CodecConfig.use_fbp_prior=False`` / ``ModelConfig.depth_in=k``).
    """
    scale = n_levels - 1
    ctx = (context.float() / scale).unsqueeze(1)  # (B, 1, k, H, W)
    if prior is None:
        return ctx
    p = (prior.float() / scale).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, H, W)
    return torch.cat([ctx, p], dim=2)  # (B, 1, k+1, H, W)
