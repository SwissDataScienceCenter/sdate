"""
Spatial grid index for fast Gaussian-to-window overlap queries.

Implements a **uniform 3-D cell grid** that assigns each Gaussian to the
cell containing its centre and records the maximum world-frame support radius
per cell.  Window queries use a two-stage approach:

    1. **Cell-level** — identify cells whose expanded region (cell bounds
       ± per-cell ``max_radius``) overlaps the query window.  Cost: O(G³),
       where G is the grid resolution (typically 16 → 4 096 cells).
    2. **Gaussian-level** — exact **rotation-aware** axis-aligned bounding-
       box check (μ ± c · σ_world) on the candidate Gaussians gathered
       from the active cells.  Cost: O(candidates).

Build cost is O(N), fully vectorised on GPU.

Why rotation-aware
------------------
The local-frame scales ``(s_z, s_y, s_x)`` define the Gaussian extent in
the **body** frame.  After rotation *R*, the world-frame per-axis half-
extent at ``c`` standard deviations is:

    h_j = c · √Σ_{jj}  =  c · √(Σ_k R_{jk}² s_k²)

Using the local-frame scales directly (ignoring *R*) can drastically
**underestimate** the bounding box of rotated anisotropic Gaussians,
leading to false negatives in the culling step.
"""

from __future__ import annotations

from typing import Tuple

import torch

from gs3d.model import quaternion_to_rotation_matrix


# ---------------------------------------------------------------------------
# Utility: rotation-aware per-axis margin
# ---------------------------------------------------------------------------


def world_frame_margin(
    quats: torch.Tensor,        # (N, 4)
    log_scales: torch.Tensor,   # (N, 3)
    cutoff_sigma: float,
) -> torch.Tensor:
    r"""Per-axis half-extent of the axis-aligned bounding box.

    For a Gaussian with rotation *R* and diagonal scales *s*,

    .. math::

        h_j \;=\; c \sqrt{\Sigma_{jj}}
            \;=\; c \sqrt{\sum_k R_{jk}^2 \, s_k^2}

    Parameters
    ----------
    quats : (N, 4)
        Quaternion rotations (w, x, y, z).
    log_scales : (N, 3)
        Log of the per-axis scales.
    cutoff_sigma : float
        Number of standard deviations for the support cutoff.

    Returns
    -------
    margin : (N, 3)
        World-frame per-axis half-extents.
    """
    R = quaternion_to_rotation_matrix(quats)              # (N, 3, 3)
    s2 = torch.exp(2.0 * log_scales)                      # (N, 3)
    # Σ_{jj} = Σ_k R_{jk}² s_k²  →  (R² @ s²)
    diag_cov = (R.pow(2) @ s2.unsqueeze(-1)).squeeze(-1)  # (N, 3)
    return cutoff_sigma * diag_cov.sqrt()


# ---------------------------------------------------------------------------
# SpatialGrid
# ---------------------------------------------------------------------------


class SpatialGrid:
    """Uniform 3-D grid for fast spatial queries on Gaussians.

    Parameters
    ----------
    volume_extent : (ez, ey, ex)
        Physical size of the volume (matching the model's ``volume_extent``).
    grid_resolution : (Gz, Gy, Gx)
        Number of cells per axis.  16³ is a good default for extents ~1.
    device : torch.device
    """

    def __init__(
        self,
        volume_extent: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        grid_resolution: Tuple[int, int, int] = (16, 16, 16),
        device: torch.device = torch.device("cuda"),
    ):
        self.extent = volume_extent
        self.grid_res = grid_resolution
        self.device = device
        Gz, Gy, Gx = grid_resolution
        ez, ey, ex = volume_extent

        self.cell_size = torch.tensor(
            [ez / Gz, ey / Gy, ex / Gx],
            device=device, dtype=torch.float32,
        )
        self._n_cells = Gz * Gy * Gx

        # Pre-compute cell physical boundaries — (n_cells, 3)
        zz = torch.arange(Gz, device=device, dtype=torch.float32)
        yy = torch.arange(Gy, device=device, dtype=torch.float32)
        xx = torch.arange(Gx, device=device, dtype=torch.float32)
        gz, gy, gx = torch.meshgrid(zz, yy, xx, indexing="ij")
        idx_flat = torch.stack(
            [gz.reshape(-1), gy.reshape(-1), gx.reshape(-1)], dim=-1,
        )
        self._cell_min = idx_flat * self.cell_size.unsqueeze(0)            # (C, 3)
        self._cell_max = self._cell_min + self.cell_size.unsqueeze(0)      # (C, 3)

        # Populated by build()
        self._sorted_gids: torch.Tensor | None = None      # Gaussian IDs sorted by cell
        self._cell_start: torch.Tensor | None = None        # (n_cells,) start offset
        self._cell_count: torch.Tensor | None = None        # (n_cells,) count per cell
        self._cell_max_radius: torch.Tensor | None = None   # (n_cells,) max iso radius
        self._built = False

    def _ensure_device(self, device: torch.device) -> None:
        """Move cached tensors if the working device has changed."""
        if self.device != device:
            self.device = device
            self.cell_size = self.cell_size.to(device)
            self._cell_min = self._cell_min.to(device)
            self._cell_max = self._cell_max.to(device)

    # ---------------------------------------------------------------- build

    @torch.no_grad()
    def build(
        self,
        means: torch.Tensor,       # (N, 3)  detached
        quats: torch.Tensor,       # (N, 4)  detached
        log_scales: torch.Tensor,  # (N, 3)  detached
        cutoff_sigma: float = 3.0,
    ) -> None:
        """(Re)build the spatial index from current Gaussian parameters.

        Call once after initialisation and again after every structural
        change (prune / densify).  Periodic rebuilds during normal training
        keep the cell assignment fresh as means drift.
        """
        N = means.shape[0]
        Gz, Gy, Gx = self.grid_res
        self._ensure_device(means.device)

        # Rotation-aware per-axis margin and max isotropic radius
        margin = world_frame_margin(quats, log_scales, cutoff_sigma)   # (N, 3)
        max_radius = margin.max(dim=-1).values                          # (N,)

        # Centre → cell index
        cidx = (means / self.cell_size).long()                          # (N, 3)
        cidx[:, 0].clamp_(0, Gz - 1)
        cidx[:, 1].clamp_(0, Gy - 1)
        cidx[:, 2].clamp_(0, Gx - 1)
        flat_cell = (
            cidx[:, 0] * (Gy * Gx) + cidx[:, 1] * Gx + cidx[:, 2]
        )  # (N,)

        # Sort Gaussian IDs by cell so each cell's members are contiguous
        _, order = flat_cell.sort()
        self._sorted_gids = order

        # Per-cell count / start (bincount + cumsum)
        self._cell_count = torch.bincount(flat_cell, minlength=self._n_cells)
        self._cell_start = torch.zeros(
            self._n_cells, device=self.device, dtype=torch.long,
        )
        if self._n_cells > 1:
            self._cell_start[1:] = self._cell_count.cumsum(0)[:-1]

        # Per-cell maximum isotropic radius (for conservative cell overlap)
        self._cell_max_radius = torch.zeros(self._n_cells, device=self.device)
        self._cell_max_radius.scatter_reduce_(
            0, flat_cell, max_radius, reduce="amax", include_self=True,
        )
        self._built = True

    # ---------------------------------------------------------------- query

    @torch.no_grad()
    def query_window(
        self,
        box_min: torch.Tensor,      # (3,)  physical coords
        box_max: torch.Tensor,      # (3,)  physical coords
        means: torch.Tensor,        # (N, 3) — current (may differ from build)
        quats: torch.Tensor,        # (N, 4) — current
        log_scales: torch.Tensor,   # (N, 3) — current
        cutoff_sigma: float = 3.0,
    ) -> torch.Tensor:
        """Return a boolean ``(N,)`` mask of overlapping Gaussians.

        **Stage 1** — cell-level conservative filter: for each of the G³
        cells, check whether the cell expanded by its ``max_radius``
        overlaps the query box.  Cost: O(G³).

        **Stage 2** — exact rotation-aware axis-aligned bounding-box check
        on the candidate Gaussians gathered from the active cells.
        Cost: O(candidates).

        Parameters
        ----------
        box_min, box_max : (3,)
            Physical-coordinate corners of the query window.
        means, quats, log_scales
            **Current** (possibly detached) Gaussian parameters.
            These are used for the *exact* Stage-2 check, even if the grid
            was built from an earlier snapshot.
        cutoff_sigma : float
            Support cutoff in σ units (default 3).

        Returns
        -------
        mask : (N,) bool
        """
        N = means.shape[0]
        if not self._built:
            raise RuntimeError("SpatialGrid.build() has not been called yet.")
        self._ensure_device(means.device)

        # ── Stage 1: cell-level overlap ──────────────────────────────
        # A cell *might* contribute when its region expanded by
        # its max_radius overlaps the query box (per axis).
        mr = self._cell_max_radius.unsqueeze(-1)                   # (C, 1)
        cell_overlap = (
            ((self._cell_min - mr) < box_max.unsqueeze(0))
            & ((self._cell_max + mr) > box_min.unsqueeze(0))
        ).all(dim=-1) & (self._cell_count > 0)                    # (C,)

        active_cells = torch.where(cell_overlap)[0]
        if active_cells.numel() == 0:
            return torch.zeros(N, dtype=torch.bool, device=self.device)

        # ── Gather candidate Gaussian IDs from active cells ──────────
        starts = self._cell_start[active_cells]                    # (K,)
        counts = self._cell_count[active_cells]                    # (K,)
        total = int(counts.sum().item())
        if total == 0:
            return torch.zeros(N, dtype=torch.bool, device=self.device)

        # Vectorised segment-expand: for each active cell k, emit
        # indices  start[k], start[k]+1, …, start[k]+count[k]-1.
        starts_rep = torch.repeat_interleave(starts, counts)       # (total,)
        cum = counts.cumsum(0)
        seg_origins = torch.zeros_like(cum)
        if cum.numel() > 1:
            seg_origins[1:] = cum[:-1]
        seg_origins_rep = torch.repeat_interleave(seg_origins, counts)
        local = torch.arange(total, device=self.device) - seg_origins_rep
        gather_idx = starts_rep + local                             # (total,)

        candidate_ids = self._sorted_gids[gather_idx].unique()

        # ── Stage 2: exact rotation-aware bbox on candidates ─────────
        margin = world_frame_margin(
            quats[candidate_ids],
            log_scales[candidate_ids],
            cutoff_sigma,
        )                                                          # (C', 3)
        c_means = means[candidate_ids]
        lower = c_means - margin
        upper = c_means + margin
        exact = (
            (lower < box_max.unsqueeze(0))
            & (upper > box_min.unsqueeze(0))
        ).all(dim=-1)

        mask = torch.zeros(N, dtype=torch.bool, device=self.device)
        mask[candidate_ids[exact]] = True
        return mask
