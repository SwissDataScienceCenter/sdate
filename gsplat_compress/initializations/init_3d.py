"""
3-D Gaussian initialisation strategies for perspective / volumetric scenes.

Currently only a uniform-grid placeholder is provided.  Extend with
point-cloud-driven, SfM-based, or multi-resolution residual strategies
as needed for specific 3-D use cases.
"""

from __future__ import annotations

import math

import torch

from gsplat_compress.initialize import GaussianParams


def uniform_3d(
    scene_bounds: tuple[float, float, float, float, float, float],
    n_points: int,
    device: torch.device,
    init_scale: float = 0.05,
    init_opacity: float = 0.1,
    init_color: float = 0.5,
    seed: int = 42,
) -> GaussianParams:
    """Uniform random initialisation inside an axis-aligned bounding box.

    Places ``n_points`` Gaussians with uniformly random 3-D means inside the
    given scene bounds.  All other parameters (rotation, scale, colour,
    opacity) are constant at initialisation — suitable as a warm-start
    baseline before full per-parameter optimisation.

    Parameters
    ----------
    scene_bounds : (x_min, x_max, y_min, y_max, z_min, z_max)
        Axis-aligned bounding box of the scene in world units.
    n_points : int
        Number of Gaussians to create.
    device : torch.device
    init_scale : float
        Initial isotropic σ (world units).
    init_opacity : float
        Initial sigmoid-space opacity.
    init_color : float
        Initial sigmoid-space colour (same value for all channels).
    seed : int

    Returns
    -------
    GaussianParams with ``requires_grad = True``.

    Notes
    -----
    This is a placeholder.  For real 3-D scenes consider:

    - SfM point cloud initialisation (NeRF/3DGS standard).
    - Multi-resolution residual allocation adapted for 3-D
      (render → residual in 2-D views → place new Gaussians).
    - Depth-map / voxel-grid driven initialisation.
    """
    torch.manual_seed(seed)

    x_min, x_max, y_min, y_max, z_min, z_max = scene_bounds

    # Uniform random 3-D positions
    means = torch.stack(
        [
            torch.rand(n_points, device=device) * (x_max - x_min) + x_min,
            torch.rand(n_points, device=device) * (y_max - y_min) + y_min,
            torch.rand(n_points, device=device) * (z_max - z_min) + z_min,
        ],
        dim=-1,
    )  # [N, 3]

    log_scales = torch.full((n_points, 3), math.log(init_scale), device=device)

    quats = torch.zeros(n_points, 4, device=device)
    quats[:, 0] = 1.0  # identity rotation

    rgbs = torch.full((n_points, 3), math.log(init_color / (1.0 - init_color)), device=device)

    opacities = torch.full(
        (n_points,), math.log(init_opacity / (1.0 - init_opacity)), device=device
    )

    params = GaussianParams(means, log_scales, quats, rgbs, opacities)
    params.require_grad_()
    return params
