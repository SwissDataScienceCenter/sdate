"""Time-resolved neural implicit field (``TimeResolvedNafField``).

A coordinate network mapping each **spatial** voxel ``(x, y, z)`` to ``K``
temporal-basis coefficients.  Contracted against a fixed temporal basis
(:class:`~sdate.tr_naf.temporal_basis.BSplineTemporalBasis`) it yields the
attenuation of that voxel at any scan time:

    mu(x, y, z, t) = softplus( sum_k c_k(x, y, z) * phi_k(t) )

This is the direct analog of the SAXS-NAF field (``smartt.saxs_naf``), which
emits spherical-harmonic coefficients contracted against an *angular* basis;
here the coefficients describe a smooth *temporal* curve per voxel.  The spatial
machinery — multiresolution hash encoding + MLP trunk, coarse-to-fine spatial
annealing via ``level_weights`` — is reused unchanged.

Non-negativity (attenuation >= 0) is enforced with a softplus on the *combined*
value, so the model is linear in the coefficients up to that final nonlinearity.

Cold start: the head is zero-initialised, so every coefficient equals the head
bias (0) and the field is spatially uniform **and static** in time — the
well-conditioned starting point the reconstruction loop relaxes away from.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .hash_encoding import MultiResolutionHashEncoding


class TimeResolvedNafField(nn.Module):
    """Coordinate MLP emitting per-voxel temporal-basis coefficients.

    Parameters
    ----------
    volume_shape : ``(X, Y, Z)`` voxel grid.
    K : Number of temporal coefficients per voxel (must match the temporal basis).
    n_levels, n_features_per_level, base_resolution, max_resolution, table_size :
        Hash-encoding hyper-parameters.  ``max_resolution`` defaults to
        ``max(volume_shape)`` and ``table_size`` to keep every level dense.
    hidden_dim, n_hidden_layers : MLP trunk.
    """

    def __init__(
        self,
        volume_shape: Tuple[int, int, int],
        K: int = 6,
        n_levels: int = 8,
        n_features_per_level: int = 4,
        base_resolution: int = 8,
        max_resolution: Optional[int] = None,
        table_size: Optional[int] = None,
        hidden_dim: int = 128,
        n_hidden_layers: int = 3,
        cold_start_value: Optional[float] = None,
    ):
        super().__init__()
        self.volume_shape = tuple(int(s) for s in volume_shape)
        self.K = int(K)
        self.cold_start_value = cold_start_value

        if max_resolution is None:
            max_resolution = max(self.volume_shape)

        self.encoding = MultiResolutionHashEncoding(
            n_dims=3,
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            base_resolution=base_resolution,
            max_resolution=max_resolution,
            table_size=table_size,
            include_input=True,
        )

        layers = []
        in_dim = self.encoding.output_dim
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(inplace=True)]
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(in_dim, self.K)

        # Cold start: zero head weight -> uniform, static field. With zero bias the
        # field starts at mu = softplus(0) ~ 0.69; set ``cold_start_value`` (e.g. ~0
        # for the count/Poisson regime, where we want p_hat ~ 0 -> lambda ~ I0) to
        # shift the bias so mu(init) = softplus(bias) = cold_start_value everywhere.
        nn.init.zeros_(self.head.weight)
        if cold_start_value is None:
            nn.init.zeros_(self.head.bias)
        else:
            y = max(float(cold_start_value), 1e-6)
            bias = float(np.log(np.expm1(y))) if y < 20 else y  # softplus^{-1}
            nn.init.constant_(self.head.bias, bias)

        self.register_buffer("grid_coords", self._build_grid_coords())

    def _build_grid_coords(self) -> torch.Tensor:
        """``(X*Y*Z, 3)`` coords in ``[0, 1]``, isotropic (cubic voxels)."""
        X, Y, Z = self.volume_shape
        axes = [torch.arange(n, dtype=torch.float32) for n in (X, Y, Z)]
        gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
        idx = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
        centres = torch.tensor(
            [(n - 1) / 2.0 for n in self.volume_shape], dtype=torch.float32
        )
        extent = float(max(self.volume_shape) - 1) if max(self.volume_shape) > 1 else 1.0
        return (idx - centres) / extent + 0.5

    def coeffs(self, level_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample the field on the full grid -> ``(X, Y, Z, K)`` coefficients."""
        feats = self.encoding(self.grid_coords, level_weights=level_weights)
        c = self.head(self.trunk(feats))
        return c.reshape(*self.volume_shape, self.K)

    @staticmethod
    def volume_at(coeffs: torch.Tensor, phi_row: torch.Tensor) -> torch.Tensor:
        """Attenuation volume at one time. ``coeffs (X,Y,Z,K)``, ``phi_row (K,)``.

        Returns ``(X, Y, Z)`` with ``mu >= 0``.
        """
        combined = torch.einsum("xyzk,k->xyz", coeffs, phi_row)
        return F.softplus(combined)

    def temporal_roughness(self, coeffs: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """Mean-per-voxel ``c^T R c`` (curvature energy of the time curves).

        Normalised by the voxel count so the regulariser weight is independent of
        volume size and directly comparable to the (mean) MSE data term.
        """
        n_vox = coeffs[..., 0].numel()
        return torch.einsum("xyzk,kl,xyzl->", coeffs, R, coeffs) / n_vox

    @staticmethod
    def temporal_tv(coeffs: torch.Tensor, phi_time: torch.Tensor) -> torch.Tensor:
        """Mean L1 total variation of each voxel's attenuation curve over time.

        Edge-preserving temporal prior: promotes piecewise-constant curves with
        sharp jumps (e.g. a combustion front arriving), unlike the L2 curvature
        penalty which smears steps.  ``phi_time`` is the ``(T, K)`` basis matrix
        at the (time-ordered) sample times; the curve is the softplus'd
        reconstruction ``mu(x,y,z,t)``.
        """
        mu = F.softplus(torch.einsum("xyzk,tk->xyzt", coeffs, phi_time))  # (X,Y,Z,T)
        return (mu[..., 1:] - mu[..., :-1]).abs().mean()

    @staticmethod
    def tv_regularization(coeffs: torch.Tensor) -> torch.Tensor:
        """Mean squared spatial total variation of the coefficient field (all axes/channels)."""
        tv = (coeffs[1:] - coeffs[:-1]).pow(2).mean()
        tv = tv + (coeffs[:, 1:] - coeffs[:, :-1]).pow(2).mean()
        tv = tv + (coeffs[:, :, 1:] - coeffs[:, :, :-1]).pow(2).mean()
        return tv
