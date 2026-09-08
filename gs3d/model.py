"""
Core 3D Gaussian Splatting model.

Stores N Gaussians with learnable parameters on GPU.
Supports batched evaluation, covariance computation, and serialisation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Quaternion / rotation helpers  (fully batched, no loops)
# ---------------------------------------------------------------------------

def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions to 3×3 rotation matrices (batch).

    Args:
        q: (N, 4)  quaternions [w, x, y, z], **not necessarily unit**.
           They are normalised internally.

    Returns:
        R: (N, 3, 3) rotation matrices.
    """
    q = F.normalize(q, p=2, dim=-1)
    w, x, y, z = q.unbind(-1)

    R = torch.stack([
        1 - 2*(y*y + z*z),   2*(x*y - w*z),     2*(x*z + w*y),
        2*(x*y + w*z),       1 - 2*(x*x + z*z), 2*(y*z - w*x),
        2*(x*z - w*y),       2*(y*z + w*x),     1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)
    return R


def random_quaternions(n: int, device: torch.device) -> torch.Tensor:
    """Sample *n* uniformly-random unit quaternions (Shoemake's method)."""
    u = torch.rand(n, 3, device=device)
    q = torch.stack([
        torch.sqrt(1 - u[:, 0]) * torch.sin(2 * math.pi * u[:, 1]),
        torch.sqrt(1 - u[:, 0]) * torch.cos(2 * math.pi * u[:, 1]),
        torch.sqrt(u[:, 0])     * torch.sin(2 * math.pi * u[:, 2]),
        torch.sqrt(u[:, 0])     * torch.cos(2 * math.pi * u[:, 2]),
    ], dim=-1)
    return F.normalize(q, p=2, dim=-1)


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

@dataclass
class GaussianModelConfig:
    """Configuration for a ``GaussianModel3D``."""
    n_init: int = 100_000
    """Initial number of Gaussians."""
    init_scale_log: float = -4.0
    """Initial log-scale (log of standard deviation in each axis)."""
    init_intensity_logit: float = 0.0
    """Initial intensity logit (pre-sigmoid)."""
    volume_extent: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Spatial extent of the scene volume (used for initialisation)."""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class GaussianModel3D(nn.Module):
    """A set of *N* 3D Gaussians with learnable parameters.

    Parameters (all shapes have leading dim *N*):
        _means        (N, 3)   raw positions
        _quats        (N, 4)   quaternion rotations  [w, x, y, z]
        _log_scales   (N, 3)   log of per-axis standard deviation
        _intensity_logit (N,)  pre-sigmoid intensity
    """

    def __init__(self, config: GaussianModelConfig | None = None, **kwargs):
        super().__init__()
        if config is None:
            config = GaussianModelConfig(**kwargs)
        self.config = config
        self._init_parameters(config.n_init)

    # ------------------------------------------------------------------ init
    def _init_parameters(self, n: int) -> None:
        """Create *n* Gaussians with random initialisation."""
        dev = torch.device("cpu")  # will be moved to GPU by .to()
        ex, ey, ez = self.config.volume_extent

        # positions: uniform inside volume
        means = torch.rand(n, 3, device=dev) * torch.tensor([ex, ey, ez], device=dev)

        # rotations: identity (w=1, x=y=z=0)
        quats = torch.zeros(n, 4, device=dev)
        quats[:, 0] = 1.0

        # scales: small isotropic
        log_scales = torch.full((n, 3), self.config.init_scale_log, device=dev)

        # intensity
        intensity_logit = torch.full((n,), self.config.init_intensity_logit, device=dev)

        self._means = nn.Parameter(means)
        self._quats = nn.Parameter(quats)
        self._log_scales = nn.Parameter(log_scales)
        self._intensity_logit = nn.Parameter(intensity_logit)

    # -------------------------------------------------------------- properties
    @property
    def n_gaussians(self) -> int:
        return self._means.shape[0]

    @property
    def device(self) -> torch.device:
        return self._means.device

    @property
    def means(self) -> torch.Tensor:
        """(N, 3) centres."""
        return self._means

    @property
    def rotations(self) -> torch.Tensor:
        """(N, 3, 3) rotation matrices from quaternions."""
        return quaternion_to_rotation_matrix(self._quats)

    @property
    def scales(self) -> torch.Tensor:
        """(N, 3) positive per-axis standard deviations."""
        return torch.exp(self._log_scales)

    @property
    def intensities(self) -> torch.Tensor:
        """(N,) intensities in (0, 1)."""
        return torch.sigmoid(self._intensity_logit)

    # ------------------------------------------------------------ covariance
    def covariances(self) -> torch.Tensor:
        """Compute (N, 3, 3) covariance matrices  Σ = R S S^T R^T."""
        R = self.rotations                   # (N, 3, 3)
        s = self.scales                      # (N, 3)
        # RS  where S = diag(s)
        RS = R * s.unsqueeze(-2)             # (N, 3, 3) broadcast multiply cols
        return RS @ RS.transpose(-1, -2)     # (N, 3, 3)

    def precisions(self) -> torch.Tensor:
        """Compute (N, 3, 3) precision (inverse covariance) matrices."""
        R = self.rotations
        inv_s = 1.0 / (self.scales + 1e-8)
        RinvS = R * inv_s.unsqueeze(-2)
        return RinvS @ RinvS.transpose(-1, -2)

    # --------------------------------------------------- evaluate at points
    @torch.no_grad()
    def evaluate_at(
        self,
        coords: torch.Tensor,
        batch_size: int = 65536,
    ) -> torch.Tensor:
        """Evaluate the Gaussian mixture at arbitrary 3-D coordinates.

        Args:
            coords: (M, 3)  query points.
            batch_size:  chunk size over query points.

        Returns:
            values: (M,)  sum of Gaussian contributions at each point.
        """
        M = coords.shape[0]
        values = torch.zeros(M, device=coords.device)
        intensities = self.intensities        # (N,)
        prec = self.precisions()              # (N, 3, 3)
        means = self.means                    # (N, 3)

        for start in range(0, M, batch_size):
            end = min(start + batch_size, M)
            pts = coords[start:end]           # (B, 3)
            # diff: (B, N, 3)
            diff = pts.unsqueeze(1) - means.unsqueeze(0)
            # Mahalanobis:  diff^T P diff  → (B, N)
            mahal = (diff @ prec.unsqueeze(0)).mul(diff).sum(-1)
            gauss = torch.exp(-0.5 * mahal)   # (B, N)
            values[start:end] = (gauss * intensities.unsqueeze(0)).sum(-1)

        return values

    # ------------------------------------- differentiable evaluate (for optim)
    def forward_at(
        self,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable evaluation of the mixture at coords.

        Args:
            coords: (B, 3)   — must fit in GPU memory together with N Gaussians.

        Returns:
            (B,)  sum-of-Gaussians values at each point.
        """
        diff = coords.unsqueeze(1) - self._means.unsqueeze(0)   # (B, N, 3)
        prec = self.precisions()                                 # (N, 3, 3)
        # einsum is efficient for batched bilinear form
        mahal = torch.einsum("bni,nij,bnj->bn", diff, prec, diff)
        gauss = torch.exp(-0.5 * mahal)                         # (B, N)
        intensities = torch.sigmoid(self._intensity_logit)       # (N,)
        return (gauss * intensities.unsqueeze(0)).sum(-1)        # (B,)

    # -------------------------------------------------- rasterise full volume
    @torch.no_grad()
    def render_volume(
        self,
        shape: Tuple[int, int, int],
        batch_size: int = 262144,
    ) -> torch.Tensor:
        """Render the Gaussians onto a regular 3-D grid.

        Args:
            shape: (D, H, W) output voxel grid.
            batch_size: number of voxels per chunk.

        Returns:
            volume: (D, H, W)  on CPU.
        """
        D, H, W = shape
        total = D * H * W

        # Build coordinate grid  — normalised to volume_extent
        ez, ey, ex = self.config.volume_extent
        zz = torch.linspace(0, ez, D, device=self.device)
        yy = torch.linspace(0, ey, H, device=self.device)
        xx = torch.linspace(0, ex, W, device=self.device)
        grid = torch.stack(torch.meshgrid(zz, yy, xx, indexing="ij"), dim=-1)  # (D,H,W,3)
        coords = grid.reshape(-1, 3)  # (D*H*W, 3)

        values = self.evaluate_at(coords, batch_size=batch_size)
        return values.reshape(D, H, W).cpu()

    # --------------------------------------------------------- differentiable render (tiled)
    def forward_volume_sampled(
        self,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable evaluation on a batch of coords.

        Uses a tiled approach to keep memory in check for large N.

        Args:
            coords: (B, 3)  arbitrary points in volume-extent coords.

        Returns:
            (B,)  values.
        """
        N = self.n_gaussians
        B = coords.shape[0]
        TILE = max(1, min(N, 2_000_000 // max(B, 1)))  # heuristic tile size

        result = torch.zeros(B, device=coords.device)
        intensities = torch.sigmoid(self._intensity_logit)

        for g_start in range(0, N, TILE):
            g_end = min(g_start + TILE, N)
            means_t = self._means[g_start:g_end]          # (T, 3)
            quats_t = self._quats[g_start:g_end]          # (T, 4)
            ls_t    = self._log_scales[g_start:g_end]     # (T, 3)
            int_t   = intensities[g_start:g_end]          # (T,)

            R = quaternion_to_rotation_matrix(quats_t)     # (T, 3, 3)
            inv_s = torch.exp(-ls_t)                       # (T, 3)
            RiS = R * inv_s.unsqueeze(-2)                  # (T, 3, 3)
            prec = RiS @ RiS.transpose(-1, -2)            # (T, 3, 3)

            diff = coords.unsqueeze(1) - means_t.unsqueeze(0)  # (B, T, 3)
            mahal = torch.einsum("bti,tij,btj->bt", diff, prec, diff)
            gauss = torch.exp(-0.5 * mahal)
            result += (gauss * int_t.unsqueeze(0)).sum(-1)

        return result

    # ------------------------------------------------------------ serialise
    def state_size_bytes(self) -> int:
        """Number of bytes for all parameters (float32)."""
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def get_info(self) -> str:
        n = self.n_gaussians
        mb = self.state_size_bytes() / 1e6
        return (
            f"GaussianModel3D  |  {n:,} Gaussians  |  "
            f"{mb:.2f} MB  |  "
            f"params: means({n},3) quats({n},4) log_scales({n},3) intensity({n})"
        )

    def save(self, path: str) -> None:
        payload = {
            "config": self.config.to_dict(),
            "means": self._means.data.cpu(),
            "quats": self._quats.data.cpu(),
            "log_scales": self._log_scales.data.cpu(),
            "intensity_logit": self._intensity_logit.data.cpu(),
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "GaussianModel3D":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        cfg = GaussianModelConfig(**payload["config"])
        model = cls(config=cfg)
        model._means.data = payload["means"]
        model._quats.data = payload["quats"]
        model._log_scales.data = payload["log_scales"]
        model._intensity_logit.data = payload["intensity_logit"]
        if device:
            model = model.to(device)
        return model

    # ----------------------------------------------------- adaptive control helpers
    def _replace_parameters(
        self,
        means: torch.Tensor,
        quats: torch.Tensor,
        log_scales: torch.Tensor,
        intensity_logit: torch.Tensor,
    ) -> None:
        """Replace all Gaussian parameters (used by pruning / densification)."""
        assert means.shape[0] == quats.shape[0] == log_scales.shape[0] == intensity_logit.shape[0]
        self._means = nn.Parameter(means)
        self._quats = nn.Parameter(quats)
        self._log_scales = nn.Parameter(log_scales)
        self._intensity_logit = nn.Parameter(intensity_logit)
