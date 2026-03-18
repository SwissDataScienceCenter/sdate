"""
Gaussian parameter initialisation for 2D image fitting.

Two modes:
- ``uniform``: regular grid with subpixel jitter — even coverage
- ``intensity``: positions sampled proportional to image intensity
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

import torch


@dataclass
class GaussianParams:
    """Container for the five Gaussian parameter tensors.

    All tensors live on the same device and have ``requires_grad`` set
    according to the caller's needs.
    """

    means: torch.Tensor        # [N, 3]
    log_scales: torch.Tensor   # [N, 3]
    quats: torch.Tensor        # [N, 4]
    rgbs: torch.Tensor         # [N, 3]
    opacities: torch.Tensor    # [N]

    # ── convenience ──────────────────────────────────────────────────
    @property
    def n_gaussians(self) -> int:
        return self.means.shape[0]

    @property
    def device(self) -> torch.device:
        return self.means.device

    def detach_clone(self) -> "GaussianParams":
        """Return a detached copy (no grad, new storage)."""
        return GaussianParams(
            means=self.means.detach().clone(),
            log_scales=self.log_scales.detach().clone(),
            quats=self.quats.detach().clone(),
            rgbs=self.rgbs.detach().clone(),
            opacities=self.opacities.detach().clone(),
        )

    def require_grad_(self) -> "GaussianParams":
        """Enable gradients on all parameters (in-place)."""
        self.means.requires_grad_(True)
        self.log_scales.requires_grad_(True)
        self.quats.requires_grad_(True)
        self.rgbs.requires_grad_(True)
        self.opacities.requires_grad_(True)
        return self

    def param_count(self) -> int:
        """Total number of scalar parameters."""
        return (
            self.means.numel()
            + self.log_scales.numel()
            + self.quats.numel()
            + self.rgbs.numel()
            + self.opacities.numel()
        )

    def tensors(self) -> list[torch.Tensor]:
        """Return a flat list of all parameter tensors."""
        return [self.means, self.log_scales, self.quats, self.rgbs, self.opacities]


def init_gaussians(
    image: torch.Tensor,
    n_points: int,
    device: torch.device,
    mode: Literal["uniform", "intensity"] = "intensity",
    init_scale_px: float = 3.0,
    init_opacity: float = 0.3,
    seed: int = 42,
) -> GaussianParams:
    """Initialise N 3-D Gaussians for 2-D orthographic image fitting.

    Parameters
    ----------
    image : Tensor [H, W]
        Grayscale target image normalised to [0, 1].
    n_points : int
        Number of Gaussians to create.
    device : torch.device
        Target device.
    mode : ``'uniform'`` | ``'intensity'``
        Placement strategy.
    init_scale_px : float
        Initial Gaussian size in pixels (half the width at 1σ).
    init_opacity : float
        Initial sigmoid-space opacity value.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    GaussianParams
        Initialised parameters with ``requires_grad=True``.
    """
    H, W = image.shape
    torch.manual_seed(seed)

    # ── Means ─────────────────────────────────────────────────────────
    if mode == "uniform":
        n_side = int(math.ceil(math.sqrt(n_points)))
        n_points = n_side * n_side
        xs = torch.linspace(0, W, n_side, device=device)
        ys = torch.linspace(0, H, n_side, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        means_xy = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

    elif mode == "intensity":
        img_flat = image.to(device).reshape(-1).float()
        eps = 1e-3
        weights = img_flat + eps
        weights = weights / weights.sum()
        pixel_indices = torch.multinomial(weights, n_points, replacement=True)
        px = (pixel_indices % W).float() + torch.rand(n_points, device=device) - 0.5
        py = (pixel_indices // W).float() + torch.rand(n_points, device=device) - 0.5
        px.clamp_(0, W - 1e-3)
        py.clamp_(0, H - 1e-3)
        means_xy = torch.stack([px, py], dim=-1)
    else:
        raise ValueError(f"Unknown init mode '{mode}'. Choose 'uniform' or 'intensity'.")

    means_z = 1.0 + 0.1 * torch.randn(n_points, 1, device=device)
    means = torch.cat([means_xy, means_z], dim=-1)                    # [N, 3]

    # ── Scales ────────────────────────────────────────────────────────
    log_scales = torch.full((n_points, 3), math.log(init_scale_px), device=device)
    log_scales[:, 2] = math.log(0.5)  # z-scale small (invisible in ortho)

    # ── Quaternions: identity rotation ────────────────────────────────
    quats = torch.zeros(n_points, 4, device=device)
    quats[:, 0] = 1.0

    # ── Colors: sample from target image ──────────────────────────────
    px_idx = means[:, 0].long().clamp(0, W - 1)
    py_idx = means[:, 1].long().clamp(0, H - 1)
    sampled = image.to(device)[py_idx, px_idx]
    rgbs = torch.logit(sampled.clamp(0.01, 0.99)).unsqueeze(-1).repeat(1, 3)

    # ── Opacities ─────────────────────────────────────────────────────
    opacities = torch.logit(torch.full((n_points,), init_opacity, device=device))

    params = GaussianParams(means, log_scales, quats, rgbs, opacities)
    params.require_grad_()
    return params
