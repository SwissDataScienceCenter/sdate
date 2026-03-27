"""Limited-angle tomography dataset for time-resolved reconstruction.

This module constructs synthetic limited-angle tomography datasets from a full
set of projections, simulating a continuously rotating acquisition where each
time-slice (height row) only has access to a small contiguous set of angles.

Public API
----------
LimitedAngleConfig:
    Dataclass holding all configuration parameters.

build_limited_angle_dataset(volume, config) -> LimitedAngleDataset:
    Construct the limited-angle dataset from a full projection volume.

LimitedAngleDataset:
    Container for all the data needed by downstream reconstruction.

BaseLimitedAngleReconstructions:
    torch.utils.data.Dataset of 2-D FBP reconstructions obtained by sliding a
    window of length ``total_projections`` over the temporally-assembled sinogram
    (total_sino / total_angles).  Designed as the training dataset for the ladiff
    diffusion model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from astra_torch.lamino import (
    build_lamino_projector,
    fbp_reconstruction_masked,
    gd_reconstruction_masked,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LimitedAngleConfig:
    """Configuration for the limited-angle tomography dataset.

    Parameters
    ----------
    n_slices : int
        Number of time-slices (height rows) to extract.
    k_angles : int
        Number of contiguous angles per slice.
    total_projections : int
        Total number of projections in the full dataset (e.g. 1501).
    angular_range_deg : tuple[float, float]
        (start, stop) of the angular range in degrees.
    det_spacing_mm : float
        Detector pixel spacing in mm (for ASTRA geometry).
    voxel_size_mm : float
        Voxel size in mm.
    start_angle_offset : int
        Offset for the first slice's starting angle index (default 0).
    height_skip : int
        Number of height rows to skip **between** consecutive slices.
        - ``0`` (default) → consecutive rows: indices 0, 1, 2, ..., n_slices-1
        - ``s > 0``       → indices 0, 1+s, 2*(1+s), ..., (n_slices-1)*(1+s)
        Setting ``s = H // n_slices - 1`` recovers approximately the old
        uniform-spread behaviour (one slice every H/n_slices rows).
    height_indices : list[int] | None
        Explicit height row indices to use as slices.  If None, rows are
        chosen using ``height_skip`` starting from row 0.
    """
    n_slices: int = 15
    k_angles: int = 100
    total_projections: int = 1501
    angular_range_deg: Tuple[float, float] = (0.0, 180.0)
    det_spacing_mm: float = 1.0
    voxel_size_mm: float = 1.0
    start_angle_offset: int = 0
    height_skip: int = 0
    height_indices: Optional[list] = None


# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

@dataclass
class LimitedAngleDataset:
    """Container for a limited-angle tomography dataset.

    Attributes
    ----------
    config : LimitedAngleConfig
        The configuration used to build this dataset.
    all_angles_deg : np.ndarray, shape (total_projections,)
        Full array of projection angles in degrees.
    height_indices : np.ndarray, shape (n_slices,)
        Which height rows in the projection volume are used as slices.
    slice_angle_indices : list[np.ndarray]
        Per-slice list of angle indices into ``all_angles_deg``.
    full_sinograms : torch.Tensor, shape (n_slices, total_projections, W)
        Full sinogram for each slice (all angles).
    limited_sinograms : list[torch.Tensor]
        Per-slice limited sinograms, each of shape (k_i, W) where k_i is
        the number of available angles for that slice.
    slice_angles_deg : list[np.ndarray]
        Per-slice array of available angles in degrees.
    vol_shape : tuple[int, int]
        Reconstruction volume shape (ny, nx) — square, matching detector width.
    W : int
        Detector width (number of columns).
    """
    config: LimitedAngleConfig
    all_angles_deg: np.ndarray
    height_indices: np.ndarray
    slice_angle_indices: list
    full_sinograms: torch.Tensor
    limited_sinograms: list
    slice_angles_deg: list
    vol_shape: Tuple[int, int]
    W: int

    @property
    def n_slices(self) -> int:
        return len(self.height_indices)

    def angle_coverage_summary(self) -> str:
        """Return a human-readable summary of angle coverage per slice."""
        lines = [
            f"Limited-angle dataset: {self.n_slices} slices, "
            f"k={self.config.k_angles} angles/slice, "
            f"total={self.config.total_projections} projections",
            f"{'Slice':>6}  {'Height':>7}  {'Angle start':>12}  {'Angle end':>10}  {'#Angles':>8}",
            "-" * 55,
        ]
        for i, (h_idx, ang_idx) in enumerate(
            zip(self.height_indices, self.slice_angle_indices)
        ):
            angles = self.all_angles_deg[ang_idx]
            lines.append(
                f"{i:>6d}  {int(h_idx):>7d}  "
                f"{angles[0]:>12.2f}°  {angles[-1]:>10.2f}°  {len(ang_idx):>8d}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_limited_angle_dataset(
    volume: torch.Tensor,
    config: LimitedAngleConfig,
) -> LimitedAngleDataset:
    """Build a limited-angle tomography dataset from a full projection volume.

    Parameters
    ----------
    volume : torch.Tensor, shape (H, W, N_projections)
        Full projection volume from ``ProjectionSliceDataset.get_full_volume()``.
        Dim-0 is height (time), dim-1 is detector width, dim-2 is projection angle.
    config : LimitedAngleConfig
        Dataset configuration.

    Returns
    -------
    LimitedAngleDataset

    Notes
    -----
    The angle assignment wraps around: slice *i* gets angles
    ``[offset + i*k, offset + (i+1)*k) mod total_projections``.
    This simulates a continuously rotating gantry where each time-step
    (height row) only captures ``k`` contiguous projections.
    """
    H, W, N_proj = volume.shape
    assert N_proj >= config.total_projections, (
        f"Volume has {N_proj} projections but config expects {config.total_projections}"
    )

    # ── Compute all angles ------------------------------------------------
    all_angles_deg = np.linspace(
        config.angular_range_deg[0],
        config.angular_range_deg[1],
        config.total_projections,
        endpoint=False,
    )

    # ── Select height rows -------------------------------------------------
    if config.height_indices is not None:
        height_indices = np.array(config.height_indices, dtype=int)
    else:
        # stride = 1 + height_skip rows per step
        # skip=0 → consecutive: 0, 1, 2, ...
        # skip=s → 0, 1+s, 2*(1+s), ...
        stride = 1 + config.height_skip
        height_indices = np.arange(config.n_slices, dtype=int) * stride
    assert len(height_indices) == config.n_slices

    # ── Compute per-slice angle indices ------------------------------------
    slice_angle_indices = []
    for i in range(config.n_slices):
        start = (config.start_angle_offset + i * config.k_angles) % config.total_projections
        indices = np.arange(start, start + config.k_angles) % config.total_projections
        slice_angle_indices.append(indices)

    # ── Extract sinograms --------------------------------------------------
    # full_sinograms[i] = volume[h_i, :, :total_projections]  -> (N_proj, W)
    full_sinograms = torch.stack(
        [volume[int(h), :, :config.total_projections].T for h in height_indices],
        dim=0,
    )  # (n_slices, total_projections, W)

    limited_sinograms = []
    slice_angles_list = []
    for i, idx in enumerate(slice_angle_indices):
        sino_i = full_sinograms[i][idx]  # (k, W)
        limited_sinograms.append(sino_i)
        slice_angles_list.append(all_angles_deg[idx])

    vol_shape = (W, W)  # square volume matching detector width

    return LimitedAngleDataset(
        config=config,
        all_angles_deg=all_angles_deg,
        height_indices=height_indices,
        slice_angle_indices=slice_angle_indices,
        full_sinograms=full_sinograms,
        limited_sinograms=limited_sinograms,
        slice_angles_deg=slice_angles_list,
        vol_shape=vol_shape,
        W=W,
    )


# ---------------------------------------------------------------------------
# Convenience: normalise sinogram
# ---------------------------------------------------------------------------

def normalize_sinogram(sino: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    """Normalise a sinogram to [0, 1] and return (normalised, min, max)."""
    vmin = float(sino.min())
    vmax = float(sino.max())
    return (sino - vmin) / (vmax - vmin + 1e-8), vmin, vmax


def normalize_image(img: torch.Tensor) -> Tuple[torch.Tensor, float, float]:
    """Normalise a 2D image to [0, 1] and return (normalised, min, max)."""
    vmin = float(img.min())
    vmax = float(img.max())
    return (img - vmin) / (vmax - vmin + 1e-8), vmin, vmax


# ---------------------------------------------------------------------------
# BaseLimitedAngleReconstructions – training dataset for ladiff
# ---------------------------------------------------------------------------

class BaseLimitedAngleReconstructions(Dataset):
    """FBP reconstructions obtained by sliding a window over the full sinogram.

    The dataset stacks all per-slice limited-angle sinograms in temporal order
    to form ``total_sino`` (shape ``(K_ANGLES * N_SLICES, W)``).  For each
    index *i* the item returned is the 2-D FBP reconstruction of the window
    ``total_sino[i : i + window_size]`` using the corresponding
    ``total_angles[i : i + window_size]`` as projection angles.  The window
    size equals ``total_projections`` (= ``len(la_data.all_angles_deg)``), so
    each reconstruction pools contributions from several neighbouring time
    slices – analogous to the SW-FBP baseline.

    Parameters
    ----------
    data_path : str | Path
        Folder with the extracted tomography projection files.
    num_projections : int
        Number of projections available in the dataset (e.g. 1501).
    target_size : tuple[int, int]
        ``(height, width)`` to which each projection is resized before
        building the 3-D volume.
    n_slices : int
        Number of time slices to extract from the volume.
    k_angles : int
        Number of contiguous projections assigned to each time slice.
    angular_range_deg : tuple[float, float], optional
        ``(start, stop)`` of the angular sweep in degrees. Default (0, 180).
    height_skip : int, optional
        Rows skipped between consecutive slices (see ``LimitedAngleConfig``).
    height_indices : list[int] | None, optional
        Explicit height-row indices for the slices.
    det_spacing_mm : float, optional
        Detector pixel spacing in mm passed to ``fbp_reconstruction_masked``.
    filter_type : str, optional
        Ramp-filter variant (``'hann'`` by default).
    device : torch.device | None, optional
        Device used for FBP.  Defaults to CUDA if available.
    transform : callable | None, optional
        Optional transform applied to the reconstructed (ny, nx) tensor before
        it is returned by ``__getitem__``.

    Notes
    -----
    FBP is performed lazily in ``__getitem__`` via
    ``astra_torch.lamino.fbp_reconstruction_masked``.  When using a
    PyTorch ``DataLoader`` keep ``num_workers=0`` to avoid forked ASTRA
    processes competing for GPU resources.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        num_projections: int,
        target_size: Tuple[int, int],
        n_slices: int,
        k_angles: int,
        angular_range_deg: Tuple[float, float] = (0.0, 180.0),
        height_skip: int = 0,
        height_indices: Optional[List[int]] = None,
        det_spacing_mm: float = 1.0,
        filter_type: str = "hann",
        device: Optional[torch.device] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.det_spacing_mm = det_spacing_mm
        self.filter_type = filter_type
        self.transform = transform
        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        # ── Build LimitedAngleDataset ──────────────────────────────────────
        from inct.dataset_slices import ProjectionSliceDataset

        dataset = ProjectionSliceDataset(
            folder_path=data_path,
            num_projections=num_projections,
            target_size=target_size,
            verbose=True,
            use_attenuation=True,
        )

        la_config = LimitedAngleConfig(
            n_slices=n_slices,
            k_angles=k_angles,
            total_projections=num_projections,
            angular_range_deg=angular_range_deg,
            height_skip=height_skip,
            height_indices=height_indices,
        )

        la_data: LimitedAngleDataset = build_limited_angle_dataset(
            dataset.volume, la_config
        )

        self.la_data = la_data
        self.k_angles = k_angles
        self.n_slices = n_slices

        # ── Assemble total_sino / total_idx / total_angles ─────────────────
        # full_sinograms: (N_SLICES, N_PROJ, W)
        # meas_full:      (N_PROJ,  N_SLICES, W)  — angles-first view
        meas_full = la_data.full_sinograms.permute(1, 0, 2).contiguous()  # (N_PROJ, N_SLICES, W)
        W = la_data.W

        total_len = k_angles * n_slices
        total_sino = torch.zeros(total_len, W)
        total_idx = torch.zeros(total_len, dtype=torch.long)
        total_angles = torch.zeros(total_len)

        for t in range(n_slices):
            idx = la_data.slice_angle_indices[t]  # shape (k_angles,)
            start = t * k_angles
            total_sino[start : start + k_angles] = meas_full[idx, t]
            total_idx[start : start + k_angles] = torch.from_numpy(idx.astype(np.int64))
            total_angles[start : start + k_angles] = torch.from_numpy(
                la_data.all_angles_deg[idx].astype(np.float32)
            )

        self.total_sino: torch.Tensor = total_sino      # (total_len, W)
        self.total_idx: torch.Tensor = total_idx        # (total_len,)
        self.total_angles: torch.Tensor = total_angles  # (total_len,)

        # ── Sliding window parameters ──────────────────────────────────────
        # Window size = full number of projection angles (len(all_angles_deg))
        self._window_size: int = len(la_data.all_angles_deg)
        self._total_len: int = total_len
        self._n_samples: int = max(0, total_len - self._window_size + 1)
        self.vol_shape: Tuple[int, int] = la_data.vol_shape  # (W, W)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._n_samples

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return the FBP reconstruction for window position *idx*.

        Parameters
        ----------
        idx : int
            Window start index in ``[0, len(self))``.

        Returns
        -------
        recon : (ny, nx) float32 tensor
            2-D FBP reconstruction, optionally transformed.
        """

        w = slice(idx, idx + self._window_size)
        sino = self.total_sino[w].to(self.device)           # (window_size, W)
        angles = self.total_angles[w].cpu().numpy().astype(np.float64)  # (window_size,)

        recon = fbp_reconstruction_masked(
            projs_vrc=sino.unsqueeze(1),  # (window_size, 1, W)
            angles_deg=angles,
            vol_shape=(1, *self.vol_shape),
            det_spacing_mm=self.det_spacing_mm,
            lamino_angle_deg=0.0,
            filter_type=self.filter_type,
            device=self.device,
        )  # (ny, nx) float32 tensor

        if self.transform is not None:
            recon = self.transform(recon)

        return recon

    # ------------------------------------------------------------------
    def get_window_center(self, idx: int) -> int:
        """Return the center position (row in total_sino) for window *idx*."""
        return idx + self._window_size // 2

    def which_slice(self, idx: int) -> int:
        """Return the time-slice that contains the window center for *idx*."""
        center = self.get_window_center(idx)
        return center // self.k_angles
