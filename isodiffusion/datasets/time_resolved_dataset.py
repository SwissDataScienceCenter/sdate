"""3D self-supervised time-resolved missing-wedge training pairs.

``TimeResolvedVolumes`` loads one or more 3D ``.npy`` volumes and produces
``(carved_x, x)`` pairs in the same way as ``MissingConeVolumes``, but with a
per-axial-slice missing wedge: each z-slice (along ``tilt_axis``) has the same
cone width yet a different orientation, rotating by ``angular_range_deg`` per
slice.  This mimics a time-resolved acquisition where each frame covers a
different angular range of the tilt series.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch

from isodiffusion.fourier_wedge import _build_mask_2d_torch

from ._base import BaseVolumeDataset

 
class TimeResolvedVolumes(BaseVolumeDataset):
    """Dataset of ``(carved_x, x)`` pairs with a per-slice rotating missing wedge.

    The missing cone for axial slice ``i`` (along ``tilt_axis``) starts at
    ``start_angle_deg + i * angular_range_deg`` (mod 180), where
    ``angular_range_deg = 180 - cone_width_deg``.  Each slice is processed
    independently via a 2-D FFT so that z-information is never mixed during
    masking.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        cone_width_deg: float = 150.0,
        patch_size: int = 112,
        target_size: int = 64,
        normalize_range: Optional[Tuple[float, float]] = None,
        samples_per_volume: Optional[int] = None,
        start_angle_deg: float = 0.0,
        tilt_axis: int = 0,
        rotate: bool = True,
    ) -> None:
        self.start_angle_deg = float(start_angle_deg)
        super().__init__(
            data_path=data_path,
            cone_width_deg=cone_width_deg,
            patch_size=patch_size,
            target_size=target_size,
            normalize_range=normalize_range,
            samples_per_volume=samples_per_volume,
            tilt_axis=tilt_axis,
            rotate=rotate,
        )

    def _carve_wedge(self, volume: torch.Tensor) -> torch.Tensor:
        kept_range = 180.0 - self.cone_width_deg
        plane_axes = [ax for ax in range(3) if ax != self.tilt_axis]
        pa0, pa1 = plane_axes

        vol = volume.float()
        pa0_size = vol.shape[pa0]
        pa1_size = vol.shape[pa1]
        n_slices = vol.shape[self.tilt_axis]

        carved_slices = []
        for i in range(n_slices):
            slice_2d = vol.select(self.tilt_axis, i)
            start_angle_i = (self.start_angle_deg + i * kept_range) % 180.0
            mask_2d = _build_mask_2d_torch(pa0_size, pa1_size, kept_range, start_angle_i, vol.device)
            fft = torch.fft.fftshift(torch.fft.fftn(slice_2d))
            fft = fft * mask_2d
            carved_slices.append(torch.fft.ifftn(torch.fft.ifftshift(fft)).real.float())

        return torch.stack(carved_slices, dim=self.tilt_axis).contiguous()

    def __repr__(self) -> str:
        angular_range = 180.0 - self.cone_width_deg
        return (
            "TimeResolvedVolumes("
            f"files={self.num_files}, total_samples={len(self)}, "
            f"patch_size={self.patch_size}, target_size={self.target_size}, "
            f"cone_width_deg={self.cone_width_deg:.1f}, "
            f"start_angle_deg={self.start_angle_deg:.1f}, "
            f"angular_range_deg={angular_range:.1f}, "
            f"tilt_axis={self.tilt_axis}, "
            f"norm=[{self.norm_min:.4g}, {self.norm_max:.4g}], rotate={self.rotate})"
        )
