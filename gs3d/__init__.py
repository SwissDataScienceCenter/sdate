"""
gs3d — 3D Gaussian Splatting for volumetric scene representation.

Each Gaussian is parameterised by:
    μ_i  ∈ ℝ³          — mean position
    q_i  ∈ ℝ⁴          — unit quaternion  → rotation matrix R_i
    s_i  ∈ ℝ³ (>0)     — log-scale        → diagonal scaling S_i = diag(exp(s_i))
    α_i  ∈ ℝ  (>0)     — intensity / opacity (sigmoid-activated raw param)

Covariance:  Σ_i = R_i S_i S_i^T R_i^T

The library is pure-PyTorch and targets millions of Gaussians on GPU.
"""

from gs3d.model import GaussianModel3D
from gs3d.renderer import VolumeRenderer
from gs3d.optimizer import GaussianOptimizer, OptimConfig
from gs3d.metrics import psnr, mse, ssim_3d, compression_ratio
from gs3d.spatial import SpatialGrid, world_frame_margin

__all__ = [
    "GaussianModel3D",
    "VolumeRenderer",
    "GaussianOptimizer",
    "OptimConfig",
    "SpatialGrid",
    "world_frame_margin",
    "psnr",
    "mse",
    "ssim_3d",
    "compression_ratio",
]
