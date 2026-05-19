"""3D self-supervised missing-wedge training pairs.

``MissingConeVolumes`` loads one or more 3D ``.npy`` volumes, samples a random
``patch_size`` cube, applies an arbitrary 3D rotation, carves an additional fixed
missing wedge, then center-crops to ``target_size`` to remove rotation boundary
artifacts.  Each item is ``(carved_x, x)`` where both tensors are normalized
``float32`` volumes with shape ``(target_size, target_size, target_size)``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from isodiffusion.fourier_wedge import apply_missing_wedge


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


def _random_rotation_matrix() -> torch.Tensor:
    """Uniform random SO(3) rotation matrix from a random unit quaternion."""
    u1, u2, u3 = torch.rand(3)
    q1 = torch.sqrt(1.0 - u1) * torch.sin(2.0 * math.pi * u2)
    q2 = torch.sqrt(1.0 - u1) * torch.cos(2.0 * math.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2.0 * math.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2.0 * math.pi * u3)
    x, y, z, w = q1, q2, q3, q4

    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float32,
    )


def _rotate_volume(volume: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    """Rotate a ``(D, H, W)`` tensor around its center using trilinear sampling."""
    if volume.ndim != 3:
        raise ValueError(f"Expected a 3D tensor, got shape {tuple(volume.shape)}")
    theta = torch.zeros((1, 3, 4), dtype=volume.dtype, device=volume.device)
    theta[0, :, :3] = rotation.to(device=volume.device, dtype=volume.dtype).T
    grid = F.affine_grid(
        theta,
        size=(1, 1, *volume.shape),
        align_corners=True,
    )
    rotated = F.grid_sample(
        volume.unsqueeze(0).unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return rotated.squeeze(0).squeeze(0)


def _center_crop_cube(volume: torch.Tensor, size: int) -> torch.Tensor:
    d, h, w = volume.shape
    if min(d, h, w) < size:
        raise ValueError(f"Cannot crop size {size} from volume shape {tuple(volume.shape)}")
    z0 = (d - size) // 2
    y0 = (h - size) // 2
    x0 = (w - size) // 2
    return volume[z0 : z0 + size, y0 : y0 + size, x0 : x0 + size].contiguous()


class MissingConeVolumes(Dataset):
    """Dataset of ``(carved_x, x)`` 3D conditional diffusion training pairs."""

    def __init__(
        self,
        data_path: Union[str, Path],
        cone_width_deg: float,
        patch_size: int = 112,
        target_size: int = 64,
        normalize_range: Optional[Tuple[float, float]] = None,
        samples_per_volume: Optional[int] = None,
        carve_center_angle_deg: float = 0.0,
        tilt_axis: int = 0,
        rotate: bool = True,
    ) -> None:
        self.data_path = Path(data_path)
        self.files = _resolve_npy_files(self.data_path)
        self.cone_width_deg = float(cone_width_deg)
        self.patch_size = int(patch_size)
        self.target_size = int(target_size)
        self.samples_per_volume = samples_per_volume
        self.carve_center_angle_deg = float(carve_center_angle_deg)
        self.tilt_axis = int(tilt_axis)
        self.rotate = bool(rotate)

        if self.patch_size < math.ceil(self.target_size * math.sqrt(3.0)):
            raise ValueError(
                "patch_size must be at least ceil(target_size * sqrt(3)) so the "
                "post-rotation crop lies inside the inscribed sphere. For "
                f"target_size={self.target_size}, use patch_size >= "
                f"{math.ceil(self.target_size * math.sqrt(3.0))}."
            )

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
                    "MissingConeVolumes: loaded norm from sidecar "
                    f"norm_min={self.norm_min:.4g}, norm_max={self.norm_max:.4g}"
                )
            else:
                all_data = np.concatenate(
                    [self._load_volume(i).ravel() for i in range(len(self.files))]
                )
                self.norm_min = float(np.percentile(all_data, 1))
                self.norm_max = float(np.percentile(all_data, 99))
                print(
                    "MissingConeVolumes: computed percentile norm "
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

    def _load_volume(self, file_idx: int) -> np.ndarray:
        if self._volumes[file_idx] is None:
            self._volumes[file_idx] = np.load(str(self.files[file_idx])).astype(np.float32)
        return self._volumes[file_idx]  # type: ignore[return-value]

    def _sample_patch(self, volume: np.ndarray) -> np.ndarray:
        starts = []
        for dim in volume.shape:
            max_start = dim - self.patch_size
            starts.append(int(torch.randint(0, max_start + 1, (1,)).item()))
        z0, y0, x0 = starts
        return volume[
            z0 : z0 + self.patch_size,
            y0 : y0 + self.patch_size,
            x0 : x0 + self.patch_size,
        ]

    def _carve_fixed_wedge(self, volume: torch.Tensor) -> torch.Tensor:
        kept_range = 180.0 - self.cone_width_deg
        start_angle = (self.carve_center_angle_deg + self.cone_width_deg / 2.0) % 180.0
        return apply_missing_wedge(
            volume,
            angular_range_deg=kept_range,
            start_angle_deg=start_angle,
            tilt_axis=self.tilt_axis,
        )

    def __len__(self) -> int:
        return int(self._cumulative[-1])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of length {len(self)}")

        file_idx = int(np.searchsorted(self._cumulative[1:], idx, side="right"))
        raw_patch = self._sample_patch(self._load_volume(file_idx))
        patch = _normalize_volume(raw_patch, self.norm_min, self.norm_max)

        if self.rotate:
            patch = _rotate_volume(patch, _random_rotation_matrix())

        x = _center_crop_cube(patch, self.target_size)
        # carved_x = _center_crop_cube(self._carve_fixed_wedge(patch), self.target_size)
        carved_x = self._carve_fixed_wedge(x)
        return carved_x.contiguous(), x.contiguous()

    @property
    def num_files(self) -> int:
        return len(self.files)

    @property
    def volume_shapes(self) -> List[Tuple[int, int, int]]:
        return list(self._shapes)

    def __repr__(self) -> str:
        return (
            "MissingConeVolumes("
            f"files={self.num_files}, total_samples={len(self)}, "
            f"patch_size={self.patch_size}, target_size={self.target_size}, "
            f"cone_width_deg={self.cone_width_deg:.1f}, tilt_axis={self.tilt_axis}, "
            f"norm=[{self.norm_min:.4g}, {self.norm_max:.4g}], rotate={self.rotate})"
        )
