"""Multi-resolution hash / dense feature-grid encoding for SAXS NAF.

Ported from the ``sdate`` NAF demo and adapted for SAXS tensor tomography:

* **Hybrid dense / hash levels.**  A level whose dense grid fits in the table
  (``resolution ** n_dims <= table_size``) is stored *densely* with a direct
  flat index — collision-free.  Only finer levels fall back to the
  Instant-NGP XOR-prime spatial hash.  For the small volumes used here
  (≤ 100³) every level is typically dense, so the encoding behaves as a
  collision-free multiresolution feature pyramid; it degrades gracefully to
  hashing if a much larger volume is supplied.

* **Per-level visibility weights.**  :meth:`forward` accepts ``level_weights``
  (one scalar in ``[0, 1]`` per level) so the coarse-to-fine spatial annealing
  schedule can fade finer levels in over training (FreeNeRF-style).

Coordinates are expected normalised to ``[0, 1]`` along every axis.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


# Large primes for the spatial hash (one per dimension, up to 4-D).
_HASH_PRIMES = [1, 2654435761, 805459861, 3674653429]


def _spatial_hash(coords: torch.Tensor, table_size: int) -> torch.Tensor:
    """XOR-prime spatial hash of integer vertex coordinates ``(..., D)``."""
    primes = torch.tensor(
        _HASH_PRIMES[: coords.shape[-1]], dtype=torch.int64, device=coords.device
    )
    coords_long = coords.long()
    h = torch.zeros(coords.shape[:-1], dtype=torch.int64, device=coords.device)
    for d in range(coords.shape[-1]):
        h = h ^ (coords_long[..., d] * primes[d])
    return (h % table_size).long()


class GridLevel(nn.Module):
    """A single resolution level — dense flat index or hashed.

    Parameters
    ----------
    n_dims : Number of input dimensions (3 here).
    n_features : Features stored per vertex.
    resolution : Grid resolution ``N`` for this level.
    table_size : Maximum table size ``T``.  The level is dense iff
        ``resolution ** n_dims <= table_size``.
    """

    def __init__(self, n_dims: int, n_features: int, resolution: int, table_size: int):
        super().__init__()
        self.n_dims = n_dims
        self.n_features = n_features
        self.resolution = resolution

        dense_entries = resolution ** n_dims
        self.dense = dense_entries <= table_size
        n_entries = dense_entries if self.dense else table_size
        self.table_size = n_entries

        # Instant-NGP initialisation: small random features.
        self.table = nn.Parameter(torch.randn(n_entries, n_features) * 1e-4)

        # Hypercube corner offsets, shape (2**n_dims, n_dims).
        n_vertices = 2 ** n_dims
        offsets = torch.zeros(n_vertices, n_dims, dtype=torch.long)
        for i in range(n_vertices):
            for d in range(n_dims):
                offsets[i, d] = (i >> d) & 1
        self.register_buffer("vertex_offsets", offsets)

        if self.dense:
            # Row-major strides for flat indexing of an N**n_dims grid.
            strides = [resolution ** (n_dims - 1 - d) for d in range(n_dims)]
            self.register_buffer(
                "strides", torch.tensor(strides, dtype=torch.long)
            )

    def _index(self, vertices: torch.Tensor) -> torch.Tensor:
        """Map integer vertices ``(..., n_dims)`` to table indices ``(...)``."""
        if self.dense:
            return (vertices * self.strides).sum(dim=-1)
        return _spatial_hash(vertices, self.table_size)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Multilinear-interpolated features for ``coords`` in ``[0, 1]``.

        Returns a tensor of shape ``(..., n_features)``.
        """
        scaled = coords * (self.resolution - 1)
        floor = torch.floor(scaled).long()
        frac = scaled - floor.float()
        floor = torch.clamp(floor, 0, self.resolution - 2)

        batch_shape = coords.shape[:-1]
        n_vertices = 2 ** self.n_dims

        vertices = floor.unsqueeze(-2) + self.vertex_offsets  # (..., V, n_dims)
        idx = self._index(vertices)                            # (..., V)
        feats = self.table[idx]                                # (..., V, F)

        # Multilinear weights: prod_d (frac_d if bit_d else 1-frac_d).
        w = torch.ones(*batch_shape, n_vertices, device=coords.device)
        for d in range(self.n_dims):
            bit = self.vertex_offsets[:, d]                    # (V,)
            f_d = frac[..., d : d + 1]                          # (..., 1)
            w = w * torch.where(
                bit.expand(*batch_shape, -1) == 1,
                f_d.expand(*batch_shape, n_vertices),
                (1.0 - f_d).expand(*batch_shape, n_vertices),
            )
        return (w.unsqueeze(-1) * feats).sum(dim=-2)


class MultiResolutionHashEncoding(nn.Module):
    """Geometric stack of :class:`GridLevel`s (dense where possible)."""

    def __init__(
        self,
        n_dims: int = 3,
        n_levels: int = 8,
        n_features_per_level: int = 4,
        base_resolution: int = 8,
        max_resolution: int = 96,
        table_size: Optional[int] = None,
        include_input: bool = True,
    ):
        super().__init__()
        self.n_dims = n_dims
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.include_input = include_input
        # Default table large enough to keep every level dense up to max_resolution.
        if table_size is None:
            table_size = max_resolution ** n_dims

        if n_levels > 1:
            b = np.exp(np.log(max_resolution / base_resolution) / (n_levels - 1))
        else:
            b = 1.0

        self.resolutions: List[int] = []
        self.levels = nn.ModuleList()
        for level in range(n_levels):
            res = int(np.floor(base_resolution * (b ** level)))
            res = max(res, 2)
            self.resolutions.append(res)
            self.levels.append(
                GridLevel(n_dims, n_features_per_level, res, table_size)
            )

        self.output_dim = n_levels * n_features_per_level
        if include_input:
            self.output_dim += n_dims

    def forward(
        self, coords: torch.Tensor, level_weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Concatenate per-level features.

        Parameters
        ----------
        coords : ``(..., n_dims)`` coordinates in ``[0, 1]``.
        level_weights : optional ``(n_levels,)`` tensor in ``[0, 1]`` scaling
            each level's features (coarse-to-fine spatial annealing).  The raw
            ``include_input`` channels are never masked.
        """
        feats = []
        if self.include_input:
            feats.append(coords)
        for i, level in enumerate(self.levels):
            f = level(coords)
            if level_weights is not None:
                f = f * level_weights[i]
            feats.append(f)
        return torch.cat(feats, dim=-1)

    @property
    def n_params(self) -> int:
        return sum(l.table.numel() for l in self.levels)

    def describe(self) -> str:
        kinds = ["dense" if l.dense else "hash" for l in self.levels]
        pairs = [f"{r}({k})" for r, k in zip(self.resolutions, kinds)]
        return f"levels={self.n_levels} resolutions=[{', '.join(pairs)}]"
