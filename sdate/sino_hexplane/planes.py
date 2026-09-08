"""2-D multi-resolution feature planes for the sinogram-domain HexPlane INR.

A :class:`Plane2D` is a stack of 2-D feature grids (a geometric multi-resolution
pyramid) over one coordinate *pair* — e.g. ``(theta, s)`` or ``(s, t)``.  At a
query point it bilinearly interpolates each level and concatenates the per-level
features.  Six such planes (one per pairwise combination of the 4 coordinates)
are combined by an elementwise (Hadamard) product in
:mod:`~sdate.sino_hexplane.model`.

Two things distinguish this from the object-space
:class:`~sdate.tr_naf.hash_encoding.MultiResolutionHashEncoding`:

* **Per-axis periodicity.**  ``theta`` is the rotation angle and is genuinely
  2*pi-periodic (real continuous rotation; the synthetic advancing-wedge data is
  built to span the full turn).  A periodic axis uses wraparound cell indexing so
  the grid has no seam at the 0 / 2*pi identification.  ``s`` and ``v`` are
  ordinary Euclidean detector axes and are *never* periodic.
* **Multiplicative-fusion initialisation.**  Because plane features are
  Hadamard-*multiplied* across the six planes, near-zero init (Instant-NGP's
  1e-4) would make the product underflow.  Following K-Planes, features are
  initialised near 1 so the six-way product starts O(1) with a well-scaled
  gradient.

Backends per level, chosen automatically (overridable):

* **dense** — a full ``res_a * res_b`` table with a direct flat index
  (collision-free).  Used when it fits in ``table_size``.
* **hashed** — Instant-NGP XOR-prime spatial hash, for fine levels of the
  ``(s, t)`` / ``(v, t)`` planes where ``s``/``v`` (~detector resolution) and
  ``t`` (~thousands of frames) would make a dense grid too large.

All coordinates are expected normalised to ``[0, 1]`` along both axes; for a
periodic axis ``[0, 1)`` maps onto the full period with 1 == 0.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn


# Two large primes for the 2-D spatial hash (Instant-NGP style).
_HASH_PRIMES = (1, 2654435761)


def _hash2d(vi: torch.Tensor, vj: torch.Tensor, table_size: int) -> torch.Tensor:
    """XOR-prime hash of integer vertex coords ``(...,)`` -> table index ``(...,)``."""
    h = (vi.long() * _HASH_PRIMES[0]) ^ (vj.long() * _HASH_PRIMES[1])
    return (h % table_size).long()


class _GridLevel2D(nn.Module):
    """One resolution level of a 2-D feature plane.

    Parameters
    ----------
    res_a, res_b : grid resolution along axis a / b.
    n_features : features stored per vertex.
    table_size : dense iff ``res_a * res_b <= table_size``; else hashed.
    periodic_a, periodic_b : wraparound indexing along that axis.
    init_mean, init_std : feature init ~ ``N(init_mean, init_std)`` (near 1 for
        multiplicative fusion).
    force_backend : ``None`` (auto), ``"dense"`` or ``"hashed"``.
    """

    def __init__(self, res_a: int, res_b: int, n_features: int, table_size: int,
                 periodic_a: bool = False, periodic_b: bool = False,
                 init_mean: float = 1.0, init_std: float = 0.1,
                 force_backend: Optional[str] = None):
        super().__init__()
        self.res_a = int(res_a)
        self.res_b = int(res_b)
        self.n_features = int(n_features)
        self.periodic_a = bool(periodic_a)
        self.periodic_b = bool(periodic_b)

        dense_entries = self.res_a * self.res_b
        if force_backend == "dense":
            self.dense = True
        elif force_backend == "hashed":
            self.dense = False
        else:
            self.dense = dense_entries <= table_size
        n_entries = dense_entries if self.dense else int(table_size)
        self.table_size = n_entries

        self.table = nn.Parameter(torch.randn(n_entries, n_features) * init_std + init_mean)

    def _cell(self, coord: torch.Tensor, res: int, periodic: bool):
        """Return ``(i0, i1, frac)`` cell indices + fractional position for one axis."""
        if periodic:
            # [0,1) spans `res` cells that wrap; cell i sits between vertex i and i+1 (mod res).
            scaled = (coord % 1.0) * res
            i0 = torch.floor(scaled).long() % res
            i1 = (i0 + 1) % res
            frac = scaled - torch.floor(scaled)
        else:
            scaled = coord.clamp(0.0, 1.0) * (res - 1)
            i0 = torch.floor(scaled).long().clamp(0, res - 2)
            i1 = i0 + 1
            frac = scaled - i0.float()
        return i0, i1, frac

    def _gather(self, ia: torch.Tensor, ib: torch.Tensor) -> torch.Tensor:
        """Features at integer vertex ``(ia, ib)`` -> ``(..., n_features)``."""
        if self.dense:
            idx = ia * self.res_b + ib
        else:
            idx = _hash2d(ia, ib, self.table_size)
        return self.table[idx]

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Bilinearly interpolated features at ``(a, b)`` in ``[0, 1]``.

        ``a``, ``b`` are ``(...,)``; returns ``(..., n_features)``.
        """
        ia0, ia1, fa = self._cell(a, self.res_a, self.periodic_a)
        ib0, ib1, fb = self._cell(b, self.res_b, self.periodic_b)
        fa = fa.unsqueeze(-1)
        fb = fb.unsqueeze(-1)

        f00 = self._gather(ia0, ib0)
        f01 = self._gather(ia0, ib1)
        f10 = self._gather(ia1, ib0)
        f11 = self._gather(ia1, ib1)

        return (f00 * (1 - fa) * (1 - fb)
                + f01 * (1 - fa) * fb
                + f10 * fa * (1 - fb)
                + f11 * fa * fb)


class Plane2D(nn.Module):
    """Multi-resolution 2-D feature plane over one coordinate pair.

    Parameters
    ----------
    max_res : ``int`` or ``(max_res_a, max_res_b)`` finest resolution per axis.
    base_res : ``int`` or ``(base_a, base_b)`` coarsest resolution per axis.
    n_levels : number of geometrically-spaced resolution levels.
    n_features : features per vertex per level.  Output dim = ``n_levels * n_features``.
    periodic_a, periodic_b : wraparound axes (``theta`` -> True).
    table_size : dense-vs-hashed cutoff per level.  Defaults to keeping every
        level dense up to ``max_res`` (product of the two max resolutions).
    backend : ``"auto"`` (default), ``"dense"`` or ``"hashed"`` forced on all levels.
    """

    def __init__(self, max_res: Union[int, Tuple[int, int]] = 128,
                 base_res: Union[int, Tuple[int, int]] = 8,
                 n_levels: int = 4, n_features: int = 4,
                 periodic_a: bool = False, periodic_b: bool = False,
                 table_size: Optional[int] = None, backend: str = "auto",
                 init_mean: float = 1.0, init_std: float = 0.1):
        super().__init__()
        max_a, max_b = (max_res, max_res) if isinstance(max_res, int) else max_res
        base_a, base_b = (base_res, base_res) if isinstance(base_res, int) else base_res
        self.n_levels = int(n_levels)
        self.n_features = int(n_features)
        self.periodic_a = bool(periodic_a)
        self.periodic_b = bool(periodic_b)

        if table_size is None:
            table_size = 2 ** int(np.ceil(np.log2(max(max_a * max_b, 2))))
        force_backend = None if backend == "auto" else backend

        def _geom(base, mx, i):
            if self.n_levels > 1:
                g = np.exp(np.log(max(mx, 2) / max(base, 2)) / (self.n_levels - 1))
            else:
                g = 1.0
            return max(int(np.floor(base * (g ** i))), 2)

        self.res_a: List[int] = []
        self.res_b: List[int] = []
        self.levels = nn.ModuleList()
        for i in range(self.n_levels):
            ra = _geom(base_a, max_a, i)
            rb = _geom(base_b, max_b, i)
            self.res_a.append(ra)
            self.res_b.append(rb)
            self.levels.append(_GridLevel2D(
                ra, rb, n_features, table_size,
                periodic_a=periodic_a, periodic_b=periodic_b,
                init_mean=init_mean, init_std=init_std, force_backend=force_backend))

        self.output_dim = self.n_levels * self.n_features

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Concatenated per-level features -> ``(..., n_levels * n_features)``."""
        return torch.cat([lvl(a, b) for lvl in self.levels], dim=-1)

    @property
    def n_params(self) -> int:
        return sum(l.table.numel() for l in self.levels)

    def describe(self) -> str:
        kinds = ["dense" if l.dense else "hash" for l in self.levels]
        pairs = [f"{a}x{b}({k})" for a, b, k in zip(self.res_a, self.res_b, kinds)]
        per = "periodic" if (self.periodic_a or self.periodic_b) else "euclid"
        return f"[{per}] levels={self.n_levels} " + " ".join(pairs)
