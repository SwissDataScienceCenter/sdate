"""GPU-accelerated batch augmentation for 3D volume patches.

Public API:
    VolumeAugmentor  — rotate, crop, and carve a batch of raw patches
    rotate_patches   — batched SO(3) rotation via F.affine_grid + F.grid_sample
    center_crop_cube — centre-crop (B, D, H, W) to (B, size, size, size)
    random_depth_crop — random depth crop (B, D, H, W) → (B, target_d, H, W)
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset


def _random_rotation_matrices(
    batch_size: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Return ``(B, 3, 3)`` uniform SO(3) rotation matrices."""
    u = torch.rand(batch_size, 3, device=device)
    u1, u2, u3 = u[:, 0], u[:, 1], u[:, 2]
    q1 = torch.sqrt(1.0 - u1) * torch.sin(2.0 * math.pi * u2)
    q2 = torch.sqrt(1.0 - u1) * torch.cos(2.0 * math.pi * u2)
    q3 = torch.sqrt(u1) * torch.sin(2.0 * math.pi * u3)
    q4 = torch.sqrt(u1) * torch.cos(2.0 * math.pi * u3)
    x, y, z, w = q1, q2, q3, q4
    rows = [
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y),
    ]
    return torch.stack(rows, dim=1).reshape(batch_size, 3, 3).to(dtype=dtype)


def rotate_patches(patches: torch.Tensor) -> torch.Tensor:
    """Apply independent uniform SO(3) rotations to each patch in a batch.

    Args:
        patches: ``(B, D, H, W)`` float tensor on any device.

    Returns:
        ``(B, D, H, W)`` rotated tensor, same device and dtype.
    """
    B, D, H, W = patches.shape
    R = _random_rotation_matrices(B, patches.device, patches.dtype)
    theta = torch.zeros(B, 3, 4, device=patches.device, dtype=patches.dtype)
    theta[:, :, :3] = R.transpose(1, 2)  # affine_grid expects R^T for rotation
    grid = F.affine_grid(theta, size=(B, 1, D, H, W), align_corners=True)
    rotated = F.grid_sample(
        patches.unsqueeze(1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    return rotated.squeeze(1)


def center_crop_cube(patches: torch.Tensor, size: int) -> torch.Tensor:
    """Centre-crop ``(B, D, H, W)`` to ``(B, size, size, size)``."""
    _, D, H, W = patches.shape
    z0 = (D - size) // 2
    y0 = (H - size) // 2
    x0 = (W - size) // 2
    return patches[:, z0:z0 + size, y0:y0 + size, x0:x0 + size].contiguous()


def random_depth_crop(patches: torch.Tensor, target_d: int) -> torch.Tensor:
    """Random depth crop ``(B, D, H, W)`` → ``(B, target_d, H, W)``."""
    D = patches.shape[1]
    z0 = int(torch.randint(0, D - target_d + 1, (1,)).item()) if D > target_d else 0
    return patches[:, z0:z0 + target_d].contiguous()


def _unwrap_base(dataset: Dataset):
    """Peel Subset wrappers to reach the underlying BaseVolumeDataset."""
    while isinstance(dataset, Subset):
        dataset = dataset.dataset
    return dataset


class VolumeAugmentor:
    """Rotate (GPU-batched), crop, and carve a batch of raw volume patches.

    Constructed from a ``BaseVolumeDataset`` (or ``Subset`` thereof).  Reads
    the dataset's geometry and borrows ``_carve_wedge`` so no dataset-specific
    logic needs to appear in training scripts.

    All ops run on whatever device the input tensor lives on, so the same
    augmentor object handles both CPU preview passes and GPU training passes.
    """

    def __init__(self, dataset: Dataset) -> None:
        from isodiffusion.datasets._base import BaseVolumeDataset

        base = _unwrap_base(dataset)
        if not isinstance(base, BaseVolumeDataset):
            raise TypeError(
                f"Expected a BaseVolumeDataset (or Subset thereof), got {type(base).__name__}"
            )
        self._carve_wedge = base._carve_wedge
        self._cube_size: int = base._cube_size
        self._is_slab: bool = base._is_slab
        self._target_d: int = base.target_size[0] if base._is_slab else 0
        self.rotate: bool = base.rotate

    def __call__(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rotate, crop, and carve a batch of raw patches.

        Args:
            patches: ``(B, patch_size, patch_size, patch_size)`` on any device.

        Returns:
            ``(carved_x, x)`` — both ``(B, *target_size)`` on the same device.
        """
        if self.rotate:
            patches = rotate_patches(patches)
        x = center_crop_cube(patches, self._cube_size)
        if self._is_slab:
            x = random_depth_crop(x, self._target_d)
        carved_x = torch.stack([self._carve_wedge(x[i]) for i in range(x.shape[0])])
        return carved_x.contiguous(), x.contiguous()
