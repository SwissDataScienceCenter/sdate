"""Shared base class and helpers for 3D volume datasets."""

from __future__ import annotations

import json
import math
from abc import abstractmethod
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

VolumeSize = Union[int, Tuple[int, int, int]]


def _resolve_npy_files(data_path: Path) -> List[Path]:
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


def _find_sidecar_norm(
    data_path: Path,
    files: List[Path],
) -> Optional[Tuple[float, float]]:
    if data_path.is_file():
        candidates = [data_path.parent / f"{data_path.stem}_norm.json"]
    else:
        candidates = [data_path / "norm.json"]

    for candidate in candidates:
        if candidate.exists():
            with open(candidate) as f:
                cfg = json.load(f)
            return float(cfg["norm_min"]), float(cfg["norm_max"])
    return None


def _normalize_volume(volume: np.ndarray, norm_min: float, norm_max: float) -> torch.Tensor:
    denom = norm_max - norm_min
    if denom > 0:
        volume = (volume - norm_min) / denom
    else:
        volume = volume - norm_min
    return torch.from_numpy(np.ascontiguousarray(volume.astype(np.float32)))



class BaseVolumeDataset(Dataset):
    """Base class for 3D volume datasets that sample patches and carve Fourier wedges.

    ``target_size`` may be a single int for cubic patches or a ``(D, H, W)`` tuple
    for slab-shaped patches where ``D <= H == W``.  The pipeline always first
    center-crops a ``H^3`` cube (after optional rotation), then takes a random
    depth crop of size ``D`` from that cube.

    When ``target_path`` is provided the dataset loads a second frozen set of
    volumes (v1) alongside the main v0 volumes.  ``__getitem__`` then returns
    ``(v0_patch, v1_patch)`` where both patches are sampled from the exact same
    spatial location.  The wedge is carved only from v0 by ``VolumeAugmentor``;
    v1 is passed through as the stable supervision target.  When ``target_path``
    is ``None`` the behaviour is identical to the original single-source dataset.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        cone_width_deg: float,
        patch_size: int = 112,
        target_size: VolumeSize = 64,
        normalize_range: Optional[Tuple[float, float]] = None,
        samples_per_volume: Optional[int] = None,
        tilt_axis: int = 0,
        rotate: bool = True,
        target_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.files = _resolve_npy_files(self.data_path)
        self.cone_width_deg = float(cone_width_deg)
        self.patch_size = int(patch_size)
        self.samples_per_volume = samples_per_volume
        self.tilt_axis = int(tilt_axis)
        self.rotate = bool(rotate)

        if isinstance(target_size, (list, tuple)):
            target_size = tuple(int(v) for v in target_size)
            d, h, w = target_size
            if h != w:
                raise ValueError(f"target_size H and W must be equal, got {target_size}")
            if d > h:
                raise ValueError(f"target_size D must be <= H=W, got {target_size}")
            self.target_size: VolumeSize = target_size
            self._cube_size = h
            self._is_slab = True
        else:
            self.target_size = int(target_size)
            self._cube_size = int(target_size)
            self._is_slab = False

        if self.rotate and self.patch_size < math.ceil(self._cube_size * math.sqrt(3.0)):
            raise ValueError(
                "patch_size must be at least ceil(cube_size * sqrt(3)) so the "
                "post-rotation crop lies inside the inscribed sphere. For "
                f"cube_size={self._cube_size}, use patch_size >= "
                f"{math.ceil(self._cube_size * math.sqrt(3.0))}."
            )
        elif not self.rotate and self.patch_size < self._cube_size:
            raise ValueError(f"patch_size ({self.patch_size}) must be >= cube_size ({self._cube_size}) when not rotating.")

        self._volumes: List[Optional[np.ndarray]] = [None] * len(self.files)
        self._shapes: List[Tuple[int, int, int]] = []
        for i in range(len(self.files)):
            vol = self._load_volume(i)
            if vol.ndim != 3:
                raise ValueError(f"Expected 3D .npy volume in {self.files[i]}, got {vol.shape}")
            if min(vol.shape) < self.patch_size:
                raise ValueError(
                    f"Volume {self.files[i]} shape {vol.shape} is smaller than "
                    f"patch_size={self.patch_size}"
                )
            self._shapes.append(tuple(int(v) for v in vol.shape))

        if normalize_range is None:
            sidecar = _find_sidecar_norm(self.data_path, self.files)
            if sidecar is not None:
                self.norm_min, self.norm_max = sidecar
                print(
                    f"{self.__class__.__name__}: loaded norm from sidecar "
                    f"norm_min={self.norm_min:.4g}, norm_max={self.norm_max:.4g}"
                )
            else:
                all_data = np.concatenate(
                    [self._load_volume(i).ravel() for i in range(len(self.files))]
                )
                self.norm_min = float(np.percentile(all_data, 1))
                self.norm_max = float(np.percentile(all_data, 99))
                print(
                    f"{self.__class__.__name__}: computed percentile norm "
                    f"norm_min={self.norm_min:.4g}, norm_max={self.norm_max:.4g}"
                )
        else:
            self.norm_min = float(normalize_range[0])
            self.norm_max = float(normalize_range[1])

        if samples_per_volume is None:
            counts = [shape[0] for shape in self._shapes]
        else:
            counts = [int(samples_per_volume)] * len(self.files)
        self._samples_per_file = counts
        self._cumulative = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

        self._target_files: Optional[List[Path]] = None
        self._target_volumes: Optional[List[Optional[np.ndarray]]] = None

        if target_path is not None:
            target_path = Path(target_path)
            self._target_files = _resolve_npy_files(target_path)
            if len(self._target_files) != len(self.files):
                raise ValueError(
                    f"target_path has {len(self._target_files)} .npy file(s) but "
                    f"data_path has {len(self.files)}; counts must match"
                )
            self._target_volumes = [None] * len(self._target_files)
            for i, tf in enumerate(self._target_files):
                tvol = np.load(str(tf)).astype(np.float32)
                if tvol.shape != self._shapes[i]:
                    raise ValueError(
                        f"Target volume {tf} shape {tvol.shape} does not match "
                        f"source volume {self.files[i]} shape {self._shapes[i]}"
                    )
                self._target_volumes[i] = tvol
            print(
                f"{self.__class__.__name__}: loaded {len(self._target_files)} frozen "
                f"target volume(s) from {target_path}"
            )

    def _load_volume(self, file_idx: int) -> np.ndarray:
        if self._volumes[file_idx] is None:
            self._volumes[file_idx] = np.load(str(self.files[file_idx])).astype(np.float32)
        return self._volumes[file_idx]  # type: ignore[return-value]

    def _load_target_volume(self, file_idx: int) -> np.ndarray:
        return self._target_volumes[file_idx]  # type: ignore[index,return-value]

    def _sample_patch_coords(self, volume: np.ndarray) -> Tuple[int, int, int]:
        starts = []
        for dim in volume.shape:
            max_start = dim - self.patch_size
            starts.append(int(torch.randint(0, max_start + 1, (1,)).item()))
        return tuple(starts)  # type: ignore[return-value]

    def _sample_patch_at(self, volume: np.ndarray, coords: Tuple[int, int, int]) -> np.ndarray:
        z0, y0, x0 = coords
        return volume[
            z0 : z0 + self.patch_size,
            y0 : y0 + self.patch_size,
            x0 : x0 + self.patch_size,
        ]

    def _sample_patch(self, volume: np.ndarray) -> np.ndarray:
        return self._sample_patch_at(volume, self._sample_patch_coords(volume))

    @abstractmethod
    def _carve_wedge(self, volume: torch.Tensor) -> torch.Tensor:
        """Apply the missing-wedge mask and return the carved volume."""

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")
        file_idx = int(np.searchsorted(self._cumulative[1:], idx, side="right"))
        v0_vol = self._load_volume(file_idx)
        coords = self._sample_patch_coords(v0_vol)
        v0_patch = _normalize_volume(self._sample_patch_at(v0_vol, coords), self.norm_min, self.norm_max)
        if self._target_files is not None:
            v1_patch = _normalize_volume(
                self._sample_patch_at(self._load_target_volume(file_idx), coords),
                self.norm_min,
                self.norm_max,
            )
            return v0_patch, v1_patch
        return v0_patch

    @property
    def num_files(self) -> int:
        return len(self.files)

    @property
    def volume_shapes(self) -> List[Tuple[int, int, int]]:
        return list(self._shapes)
