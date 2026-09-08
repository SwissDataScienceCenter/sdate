"""Fully-joint multi-resolution hash-grid encoding (spec 4 fallback).

The pairwise HexPlane factorisation (:mod:`sdate.sino_hexplane.model`) is a
low-rank product of 2-D planes; empirically it plateaus ~15 dB below a fully
joint encoding on clean sinograms and is fragile to optimise (adding channels
can *reduce* accuracy).  This module provides the alternative the spec flags as
the feasible fallback: a single Instant-NGP-style hash grid over the *joint*
coordinate, which is fully expressive and trains faster.

Periodic ``theta`` is handled with the spec's **helix embedding** — theta enters
as ``(cos theta, sin theta)`` (mapped to ``[0, 1]``), so the 0 / 2*pi seam is
seamless without any wraparound bookkeeping.  Full coordinate:
``(cos theta, sin theta, s, v, t)`` (5-D) — or ``(theta, s, v, t)`` (4-D) when
``periodic_theta=False``.

Coordinates arrive normalised to ``[0, 1]`` (``theta`` in ``[0, 1)`` = angle /
2*pi); ``s, v, t`` already in ``[0, 1]``.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn


# Enough large primes for the spatial hash up to 6-D (Instant-NGP style).
_PRIMES = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437]


def _hash(vertices: torch.Tensor, table_size: int) -> torch.Tensor:
    """XOR-prime hash of integer vertex coords ``(..., D)`` -> index ``(...)``."""
    primes = torch.tensor(_PRIMES[: vertices.shape[-1]], dtype=torch.int64,
                          device=vertices.device)
    h = torch.zeros(vertices.shape[:-1], dtype=torch.int64, device=vertices.device)
    v = vertices.long()
    for d in range(vertices.shape[-1]):
        h = h ^ (v[..., d] * primes[d])
    return (h % table_size).long()


class _JointLevel(nn.Module):
    """One resolution level of an N-D grid with **per-axis** resolution.

    ``res_vec`` gives the grid resolution along each axis, so a slow axis (e.g.
    time) can be kept coarse while spatial/angular axes stay fine — an
    anisotropic smoothness prior.
    """

    def __init__(self, n_dims: int, n_features: int, res_vec: Sequence[int],
                 table_size: int):
        super().__init__()
        self.n_dims = n_dims
        res = [int(r) for r in res_vec]
        dense_entries = int(np.prod(res))
        self.dense = dense_entries <= table_size
        n_entries = dense_entries if self.dense else table_size
        self.table_size = n_entries
        self.table = nn.Parameter(torch.randn(n_entries, n_features) * 1e-4)
        self.register_buffer("res", torch.tensor(res, dtype=torch.long))  # (D,)

        n_vertices = 2 ** n_dims
        offsets = torch.zeros(n_vertices, n_dims, dtype=torch.long)
        for i in range(n_vertices):
            for d in range(n_dims):
                offsets[i, d] = (i >> d) & 1
        self.register_buffer("vertex_offsets", offsets)
        if self.dense:
            strides = [int(np.prod(res[d + 1:])) for d in range(n_dims)]
            self.register_buffer("strides", torch.tensor(strides, dtype=torch.long))

    def _index(self, vertices: torch.Tensor) -> torch.Tensor:
        if self.dense:
            return (vertices * self.strides).sum(dim=-1)
        return _hash(vertices, self.table_size)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        scaled = coords * (self.res.float() - 1.0)                # per-axis
        floor = torch.floor(scaled).long()
        frac = scaled - floor.float()
        floor = torch.maximum(floor, torch.zeros_like(floor))
        floor = torch.minimum(floor, self.res - 2)                # per-axis clamp

        batch = coords.shape[:-1]
        nv = 2 ** self.n_dims
        vertices = floor.unsqueeze(-2) + self.vertex_offsets      # (..., V, D)
        feats = self.table[self._index(vertices)]                 # (..., V, F)
        w = torch.ones(*batch, nv, device=coords.device)
        for d in range(self.n_dims):
            bit = self.vertex_offsets[:, d]
            f_d = frac[..., d:d + 1]
            w = w * torch.where(bit.expand(*batch, -1) == 1,
                                f_d.expand(*batch, nv), (1.0 - f_d).expand(*batch, nv))
        return (w.unsqueeze(-1) * feats).sum(dim=-2)


class JointHashEncoding(nn.Module):
    """Instant-NGP multi-resolution hash grid over the joint (theta,s,v,t) coord.

    Parameters
    ----------
    max_resolution : finest grid resolution for the spatial/angular axes.
    n_levels, n_features : NGP defaults 16 x 2.
    base_resolution, table_size : coarsest level / hash-table cap.
    periodic_theta : embed theta as ``(cos, sin)`` (5-D) vs a raw axis (4-D).
    t_max_resolution : finest resolution of the **time** axis.  Capping it below
        ``max_resolution`` is an anisotropic temporal-smoothness prior: the model
        cannot represent independent per-frame detail (i.e. per-frame noise) and
        must share structure across time — the project's temporal-redundancy
        assumption made explicit.  Default ``None`` = ``max_resolution`` (isotropic).
    theta_max_resolution : finest resolution of the angular axis/axes (default
        ``max_resolution``).
    include_input : append the raw coordinate to the features.
    """

    def __init__(self, max_resolution: int = 256, n_levels: int = 16,
                 n_features: int = 2, base_resolution: int = 8,
                 table_size: int = 2 ** 19, periodic_theta: bool = True,
                 t_max_resolution: Optional[int] = None,
                 theta_max_resolution: Optional[int] = None,
                 include_input: bool = True):
        super().__init__()
        self.periodic_theta = bool(periodic_theta)
        self.include_input = bool(include_input)
        n_dims = 5 if self.periodic_theta else 4
        self.n_dims = n_dims

        th_max = theta_max_resolution or max_resolution
        t_max = t_max_resolution or max_resolution
        # Per-axis finest resolution, in coordinate order.
        if self.periodic_theta:   # (cos, sin, s, v, t)
            axis_max = [th_max, th_max, max_resolution, max_resolution, t_max]
        else:                      # (theta, s, v, t)
            axis_max = [th_max, max_resolution, max_resolution, t_max]

        # Geometric per-axis schedule from a per-axis base (<= that axis's max).
        def _sched(mx: int) -> List[int]:
            base = min(base_resolution, mx)
            g = math.exp(math.log(mx / base) / (n_levels - 1)) if n_levels > 1 else 1.0
            return [min(max(int(math.floor(base * (g ** l))), 2), mx) for l in range(n_levels)]

        per_axis = [_sched(mx) for mx in axis_max]   # (D, n_levels)
        self.levels = nn.ModuleList()
        self.res_vectors: List[List[int]] = []
        for lvl in range(n_levels):
            res_vec = [per_axis[d][lvl] for d in range(n_dims)]
            self.res_vectors.append(res_vec)
            self.levels.append(_JointLevel(n_dims, n_features, res_vec, table_size))
        self.output_dim = n_levels * n_features + (n_dims if include_input else 0)

    def _coords(self, theta, s, v, t) -> torch.Tensor:
        if self.periodic_theta:
            ang = theta * (2.0 * math.pi)
            c = (torch.cos(ang) + 1.0) * 0.5
            sn = (torch.sin(ang) + 1.0) * 0.5
            return torch.stack([c, sn, s, v, t], dim=-1)
        return torch.stack([theta, s, v, t], dim=-1)

    def forward(self, theta, s, v, t) -> torch.Tensor:
        x = self._coords(theta, s, v, t)
        feats = [x] if self.include_input else []
        for lvl in self.levels:
            feats.append(lvl(x))
        return torch.cat(feats, dim=-1)

    @property
    def n_params(self) -> int:
        return sum(l.table.numel() for l in self.levels)

    def describe(self) -> str:
        kinds = ["dense" if l.dense else "hash" for l in self.levels]
        finest = "x".join(str(r) for r in self.res_vectors[-1])
        order = "cos,sin,s,v,t" if self.periodic_theta else "theta,s,v,t"
        return (f"JointHash {self.n_dims}D ({'periodic' if self.periodic_theta else 'raw'} theta) "
                f"levels={len(self.levels)} finest[{order}]={finest} "
                f"({kinds.count('dense')} dense/{kinds.count('hash')} hash)")
