"""NpyVolumeSliceDataset — 2D slice dataset from pre-computed FBP .npy volumes.

Each .npy file contains a 3D array of shape ``(num_slices, H, W)``, as produced
by the Large_LA_Dataset notebook (``la_fbp``).  The dataset concatenates slices
across all loaded files and exposes individual 2D float32 tensors.

Usage
-----
Single file::

    ds = NpyVolumeSliceDataset(
        data_path='/data/reconstruction/la_fbp_10.npy',
    )

Directory (loads all .npy files found in it)::

    ds = NpyVolumeSliceDataset(
        data_path='/data/reconstruction/',
        augment=True,
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset


def _find_sidecar_norm(
    data_path: Path, files: List[Path]
) -> Optional[Tuple[float, float]]:
    """Look for a ``_norm.json`` sidecar and return ``(norm_min, norm_max)`` or ``None``.

    For a single ``.npy`` file, the sidecar is expected at
    ``<parent>/<stem>_norm.json``.  For a directory, it is expected at
    ``<directory>/norm.json``.
    """
    if data_path.is_file():
        candidate = data_path.parent / (data_path.stem + "_norm.json")
    else:
        candidate = data_path / "norm.json"

    if not candidate.exists():
        return None

    with open(candidate) as f:
        d = json.load(f)

    try:
        return float(d["norm_min"]), float(d["norm_max"])
    except KeyError as exc:
        raise KeyError(
            f"Sidecar {candidate} must contain 'norm_min' and 'norm_max'. "
            f"Missing key: {exc}"
        )


def _resolve_npy_files(data_path: Path) -> List[Path]:
    """Return sorted list of .npy files from a file or directory path."""
    if data_path.is_file():
        if data_path.suffix.lower() != ".npy":
            raise ValueError(f"Expected a .npy file, got: {data_path}")
        return [data_path]
    if data_path.is_dir():
        files = sorted(data_path.glob("*.npy"))
        if not files:
            raise FileNotFoundError(f"No .npy files found in directory: {data_path}")
        return files
    raise FileNotFoundError(f"Path does not exist: {data_path}")


class NpyVolumeSliceDataset(Dataset):
    """PyTorch dataset of 2D slices from pre-computed limited-angle FBP volumes.

    Each .npy file is a 3D array of shape ``(num_slices, H, W)``.  The dataset
    linearises all slices across all loaded files so that ``dataset[i]`` returns
    the *i*-th 2D slice as a float32 tensor of shape ``(H, W)``.

    Parameters
    ----------
    data_path:
        Path to a single ``.npy`` file **or** a directory.  When a directory is
        given, all ``.npy`` files inside it are loaded and concatenated.
    normalize_range:
        ``(vmin, vmax)`` pair used for linear normalisation to ``[0, 1]``.
        When ``None``, the dataset first looks for a ``<stem>_norm.json`` sidecar
        file next to each ``.npy`` (single-file) or a ``norm.json`` inside the
        directory (multi-file).  If no sidecar is found, the 1st and 99th
        percentiles are computed from all loaded data and used as the range.
    augment:
        Enable random data augmentation at ``__getitem__`` time.
    scale_range:
        ``(min_scale, max_scale)`` for random isotropic scaling.
        Default ``(0.9, 1.1)`` (±10 %).
    rotation_deg:
        Half-width of the uniform rotation range in degrees.
        Default ``5.0`` (i.e. rotations drawn from ``[−5°, +5°]``).
    shift_fraction:
        Maximum translation as a fraction of the image dimension.
        Default ``0.05`` (5 %).
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        normalize_range: Optional[Tuple[float, float]] = None,
        augment: bool = True,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        rotation_deg: float = 5.0,
        shift_fraction: float = 0.05,
    ) -> None:
        data_path = Path(data_path)
        self.files: List[Path] = _resolve_npy_files(data_path)
        self.augment = augment
        self.scale_range = scale_range
        self.rotation_deg = rotation_deg
        self.shift_fraction = shift_fraction

        # Load all volumes into memory (deferred – cached on first access).
        self._volumes: List[Optional[np.ndarray]] = [None] * len(self.files)
        self._slices_per_file: List[int] = []

        for i, f in enumerate(self.files):
            vol = self._load_volume(i)
            if vol.ndim != 3:
                raise ValueError(
                    f"Expected a 3-D array (num_slices, H, W) in {f}, "
                    f"got shape {vol.shape}"
                )
            self._slices_per_file.append(vol.shape[0])

        self._cumulative: np.ndarray = np.concatenate(
            [[0], np.cumsum(self._slices_per_file)]
        ).astype(np.int64)

        # Normalisation parameters.
        if normalize_range is None:
            sidecar = _find_sidecar_norm(data_path, self.files)
            if sidecar is not None:
                self.norm_min, self.norm_max = sidecar
                print(f"NpyVolumeSliceDataset: loaded norm from sidecar  "
                      f"norm_min={self.norm_min:.4g}, norm_max={self.norm_max:.4g}")
            else:
                # Compute 1st-99th percentile across all loaded data.
                all_data = np.concatenate(
                    [self._load_volume(i).ravel() for i in range(len(self.files))]
                )
                self.norm_min = float(np.percentile(all_data, 1))
                self.norm_max = float(np.percentile(all_data, 99))
                print(f"NpyVolumeSliceDataset: computed percentile norm  "
                      f"norm_min={self.norm_min:.4g}, norm_max={self.norm_max:.4g}")
        else:
            self.norm_min = float(normalize_range[0])
            self.norm_max = float(normalize_range[1])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_volume(self, file_idx: int) -> np.ndarray:
        """Load (and cache) file *file_idx* as a float32 numpy array."""
        if self._volumes[file_idx] is None:
            self._volumes[file_idx] = np.load(
                str(self.files[file_idx])
            ).astype(np.float32)
        return self._volumes[file_idx]  # type: ignore[return-value]

    def _augment(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply random geometric augmentation to a ``(H, W)`` float32 tensor.

        Augmentation consists of:
        - Random isotropic scaling within *scale_range*.
        - Random rotation within ``[−rotation_deg, +rotation_deg]``.
        - Random translation up to ``shift_fraction`` of image size.
        """
        h, w = tensor.shape[-2], tensor.shape[-1]

        scale = float(torch.empty(1).uniform_(self.scale_range[0], self.scale_range[1]))
        angle = float(torch.empty(1).uniform_(-self.rotation_deg, self.rotation_deg))
        translate = [
            float(torch.empty(1).uniform_(-self.shift_fraction * w, self.shift_fraction * w)),
            float(torch.empty(1).uniform_(-self.shift_fraction * h, self.shift_fraction * h)),
        ]

        tensor = TF.affine(
            tensor.unsqueeze(0),   # (1, H, W) — affine needs a channel dim
            angle=angle,
            translate=translate,
            scale=scale,
            shear=0.0,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        ).squeeze(0)

        return tensor

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        file_idx = int(np.searchsorted(self._cumulative[1:], idx, side="right"))
        slice_idx = idx - int(self._cumulative[file_idx])

        vol = self._load_volume(file_idx)
        img = vol[slice_idx]  # (H, W), float32

        denom = self.norm_max - self.norm_min
        if denom > 0.0:
            img = (img - self.norm_min) / denom
        else:
            img = img - self.norm_min

        tensor = torch.from_numpy(img.copy())  # (H, W)

        if self.augment:
            tensor = self._augment(tensor)

        return tensor

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def num_files(self) -> int:
        return len(self.files)

    @property
    def slices_per_file(self) -> List[int]:
        return list(self._slices_per_file)

    @property
    def image_size(self) -> Tuple[int, int]:
        vol = self._load_volume(0)
        return (vol.shape[1], vol.shape[2])

    def __repr__(self) -> str:
        return (
            f"NpyVolumeSliceDataset("
            f"files={self.num_files}, "
            f"total_slices={len(self)}, "
            f"slices_per_file={self.slices_per_file}, "
            f"image_size={self.image_size}, "
            f"norm=[{self.norm_min:.4g}, {self.norm_max:.4g}], "
            f"augment={self.augment})"
        )
