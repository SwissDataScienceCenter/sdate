"""
Gaussian parameter initialisation for 2D image fitting.

Available modes (``init_gaussians`` ``mode`` argument)
------------------------------------------------------
``'uniform'``
    Regular grid with sub-pixel jitter — even coverage.
``'intensity'``
    Positions sampled proportional to image intensity.
``'multiresolution_residual'``
    Coarse-to-fine greedy matching pursuit — places more Gaussians where
    the residual error and image gradient are largest.  Delegates to
    :func:`gsplat_compress.initializations.init_2d.multiresolution_residual_2d`.

The ``gsplat_compress.initializations`` sub-package contains the canonical
implementations of each strategy and is the recommended import path for
new code.  The convenience wrapper :func:`init_gaussians` kept here is
retained for backward compatibility.
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
    mode: Literal["uniform", "intensity", "multiresolution_residual"] = "intensity",
    init_scale_px: float = 3.0,
    init_opacity: float = 0.3,
    seed: int = 42,
    # extra kwargs forwarded to multiresolution_residual_2d
    **mr_kwargs,
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
    mode : ``'uniform'`` | ``'intensity'`` | ``'multiresolution_residual'``
        Placement strategy.
    init_scale_px : float
        Initial Gaussian size in pixels (half the width at 1σ).
    init_opacity : float
        Initial sigmoid-space opacity value.
    seed : int
        Random seed for reproducibility.
    **mr_kwargs
        Extra keyword arguments forwarded to
        :func:`~gsplat_compress.initializations.init_2d.multiresolution_residual_2d`
        when ``mode='multiresolution_residual'``
        (e.g. ``n_stages``, ``eta``, ``amplitude_iters``).

    Returns
    -------
    GaussianParams
        Initialised parameters with ``requires_grad=True``.
    """
    # Delegate to the canonical sub-package implementations
    if mode == "uniform":
        from gsplat_compress.initializations.init_2d import uniform_2d
        return uniform_2d(image, n_points, device,
                          init_scale_px=init_scale_px,
                          init_opacity=init_opacity, seed=seed)

    elif mode == "intensity":
        from gsplat_compress.initializations.init_2d import intensity_2d
        return intensity_2d(image, n_points, device,
                            init_scale_px=init_scale_px * 0.5,
                            init_opacity=init_opacity, seed=seed)

    elif mode == "multiresolution_residual":
        from gsplat_compress.initializations.init_2d import multiresolution_residual_2d
        return multiresolution_residual_2d(
            image, n_points, device,
            init_scale_px=init_scale_px,
            init_opacity=init_opacity,
            seed=seed,
            **mr_kwargs,
        )

    else:
        raise ValueError(
            f"Unknown init mode '{mode}'. "
            "Choose 'uniform', 'intensity', or 'multiresolution_residual'."
        )


# ── legacy inline helpers kept for backward compatibility ─────────────────
def _uniform_grid_params(image, n_points, device, init_scale_px, init_opacity):
    """Internal helper used by old callers. Delegates to uniform_2d."""
    from gsplat_compress.initializations.init_2d import uniform_2d
    return uniform_2d(image, n_points, device,
                      init_scale_px=init_scale_px, init_opacity=init_opacity)
