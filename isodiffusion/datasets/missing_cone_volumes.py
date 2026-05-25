"""3D self-supervised missing-wedge training pairs.

``MissingConeVolumes`` loads one or more 3D ``.npy`` volumes, samples a random
``patch_size`` cube, applies an arbitrary 3D rotation, carves an additional fixed
missing wedge, then center-crops to ``target_size`` to remove rotation boundary
artifacts.  Each item is a normalized ``float32`` patch; see ``BaseVolumeDataset``
for the optional ``target_path`` dual-source mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch

from isodiffusion.fourier_wedge import apply_missing_wedge

from ._base import BaseVolumeDataset, VolumeSize


class MissingConeVolumes(BaseVolumeDataset):
    """Dataset of raw volume patches for missing-wedge training.

    A fixed missing wedge of ``cone_width_deg`` is carved from every patch in
    the same orientation, controlled by ``carve_center_angle_deg``.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        cone_width_deg: float,
        patch_size: int = 112,
        target_size: VolumeSize = 64,
        normalize_range: Optional[Tuple[float, float]] = None,
        samples_per_volume: Optional[int] = None,
        carve_center_angle_deg: float = 0.0,
        tilt_axis: int = 0,
        rotate: bool = True,
        target_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.carve_center_angle_deg = float(carve_center_angle_deg)
        super().__init__(
            data_path=data_path,
            cone_width_deg=cone_width_deg,
            patch_size=patch_size,
            target_size=target_size,
            normalize_range=normalize_range,
            samples_per_volume=samples_per_volume,
            tilt_axis=tilt_axis,
            rotate=rotate,
            target_path=target_path,
        )

    def _carve_wedge(self, volume: torch.Tensor) -> torch.Tensor:
        kept_range = 180.0 - self.cone_width_deg
        start_angle = (self.carve_center_angle_deg + self.cone_width_deg / 2.0) % 180.0
        return apply_missing_wedge(
            volume,
            angular_range_deg=kept_range,
            start_angle_deg=start_angle,
            tilt_axis=self.tilt_axis,
        )

    def __repr__(self) -> str:
        return (
            "MissingConeVolumes("
            f"files={self.num_files}, total_samples={len(self)}, "
            f"patch_size={self.patch_size}, target_size={self.target_size}, "
            f"cone_width_deg={self.cone_width_deg:.1f}, tilt_axis={self.tilt_axis}, "
            f"norm=[{self.norm_min:.4g}, {self.norm_max:.4g}], rotate={self.rotate}, "
            f"frozen_target={self._target_files is not None})"
        )
