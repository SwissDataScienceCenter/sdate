"""MissingConeDataset – conditional diffusion training pairs for missing-cone recovery.

Each item is a triple ``(carved_x, x, beta_int)`` where:

* ``x``         – a 2D slice from the LA reconstruction, rotated by a uniformly
                  sampled angle ``alpha`` so the existing missing Fourier cone
                  now points at ``alpha``.
* ``carved_x``  – ``x`` with an additional Fourier cone of ``cone_width_deg``
                  removed around angle ``beta``.
* ``beta_int``  – integer angle in ``[0, 179]`` used as class conditioning so the
                  model knows *where* the carved cone is.

The diffusion model is trained to predict the noise added to ``x`` given
``carved_x`` concatenated channel-wise with the noisy ``x_t``, and ``beta_int``
as a class label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from ladiff.datasets.npy_volume import NpyVolumeSliceDataset
from ladiff.fourier_wedge import apply_missing_wedge, apply_circle_mask


class MissingConeDataset(NpyVolumeSliceDataset):
    """Dataset of ``(carved_x, x, beta_int)`` triples for unsupervised missing-cone recovery.

    Extends :class:`NpyVolumeSliceDataset` by adding two extra steps in
    ``__getitem__``:

    1. **Random cone rotation** – the slice ``x`` is rotated by a uniformly
       sampled angle ``alpha ∈ [0°, 180°)`` in image space.  By the Fourier
       rotation property, this moves the existing missing Fourier cone from 0°
       to ``alpha``.

    2. **Fourier-cone carving** – a second angle ``beta`` is sampled at least
       ``cone_width_deg`` away from ``alpha`` (circular distance mod 180°).
       The 2-D Fourier frequencies in a cone of width ``cone_width_deg`` centred
       at ``beta`` are zeroed out via iFFT → producing ``carved_x``.

    The resulting ``(carved_x, x)`` pair provides supervised signal: the model
    must learn to recover the frequencies at ``beta`` that were intentionally
    removed from ``x`` to create ``carved_x``.

    Parameters
    ----------
    data_path:
        Path to a single ``.npy`` file or a directory of ``.npy`` files
        (same as :class:`NpyVolumeSliceDataset`).
    cone_width_deg:
        Width in degrees of the Fourier cone that is carved out (and of the
        existing missing cone in the LA reconstruction).  Equal to
        ``int((1 - angular_range_frac) * 180)`` for a standard limited-angle
        reconstruction, or ``int(angular_range_frac * 180)`` depending on the
        convention used when producing the data.
    normalize_range:
        ``(vmin, vmax)`` pair for linear normalisation to ``[0, 1]``.  When
        ``None``, the value is resolved from a sidecar ``.json`` file or
        computed from percentiles (same as parent class).
    augment:
        Enable scale and translation augmentation.  Rotation augmentation is
        replaced by the full-range ``alpha`` rotation so no additional random
        rotation is applied.
    scale_range:
        ``(min_scale, max_scale)`` for random isotropic scaling when
        ``augment=True``.  Default ``(0.9, 1.1)`` (±10 %).
    shift_fraction:
        Maximum translation as a fraction of the image dimension when
        ``augment=True``.  Default ``0.05`` (5 %).
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        cone_width_deg: float,
        normalize_range: Optional[Tuple[float, float]] = None,
        augment: bool = True,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        shift_fraction: float = 0.05,
    ) -> None:
        # Disable parent augmentation; we implement scale/translate here and
        # replace rotation with the full-range alpha rotation.
        super().__init__(
            data_path=data_path,
            normalize_range=normalize_range,
            augment=False,
            scale_range=scale_range,
            shift_fraction=shift_fraction,
        )
        self.cone_width_deg = float(cone_width_deg)
        self.do_augment = augment
        self._scale_range = scale_range
        self._shift_fraction = shift_fraction

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _augment_scale_translate(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply random scale + translation (no rotation) to a ``(H, W)`` tensor."""
        h, w = tensor.shape[-2], tensor.shape[-1]
        scale = float(
            torch.empty(1).uniform_(self._scale_range[0], self._scale_range[1])
        )
        translate = [
            float(
                torch.empty(1).uniform_(
                    -self._shift_fraction * w, self._shift_fraction * w
                )
            ),
            float(
                torch.empty(1).uniform_(
                    -self._shift_fraction * h, self._shift_fraction * h
                )
            ),
        ]
        return TF.affine(
            tensor.unsqueeze(0),  # (1, H, W)
            angle=0.0,
            translate=translate,
            scale=scale,
            shear=0.0,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        ).squeeze(0)

    def _sample_beta(self, alpha: float) -> float:
        """Sample ``beta`` uniformly from angles at least ``cone_width_deg`` from ``alpha``.

        All angles are in ``[0°, 180°)`` with Friedel-symmetric circular distance
        (mod 180°).

        If the cones are too wide to guarantee non-overlap (``cone_width_deg >= 90°``),
        the most opposite angle (``alpha + 90°``) is returned.
        """
        valid_range = 180.0 - 2.0 * self.cone_width_deg
        if valid_range <= 0.0:
            # Cones too wide to guarantee separation; use the most opposite angle.
            return (alpha + 90.0) % 180.0
        # Sample offset uniformly from [cone_width_deg, 180 - cone_width_deg]
        offset = float(
            torch.empty(1).uniform_(self.cone_width_deg, 180.0 - self.cone_width_deg)
        )
        return (alpha + offset) % 180.0

    def _carve_cone_2d(self, img: torch.Tensor, center_angle: float) -> torch.Tensor:
        """Zero out a Fourier cone of ``cone_width_deg`` degrees centred at ``center_angle``.

        The 2D image is treated as a single-slice 3D volume ``(1, H, W)`` so that
        the existing :func:`~ladiff.fourier_wedge.apply_missing_wedge` can be
        reused without modification.

        Parameters
        ----------
        img:
            2D float32 tensor of shape ``(H, W)`` on any device.
        center_angle:
            Centre of the cone to carve, in degrees ``[0°, 180°)``.

        Returns
        -------
        carved : torch.Tensor, shape ``(H, W)``, same device as ``img``
        """
        vol = img.unsqueeze(0)  # (1, H, W)
        # The kept region is the complement of the carved cone.
        kept_range = 180.0 - self.cone_width_deg
        # Start just after the cone ends (right edge of the carved region).
        start_angle = (center_angle + self.cone_width_deg / 2.0) % 180.0
        carved = apply_missing_wedge(vol, kept_range, start_angle, tilt_axis=0)
        return carved.squeeze(0)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Return ``(carved_x, x, beta_int)`` for the slice at position ``idx``.

        Returns
        -------
        carved_x : torch.Tensor, shape ``(H, W)``
            Normalised float32 slice with the Fourier cone at ``beta`` removed.
        x : torch.Tensor, shape ``(H, W)``
            Normalised float32 slice rotated so the existing missing cone points
            at ``alpha`` (this is the prediction target).
        beta_int : int
            Integer angle in ``[0, 179]`` used as the class conditioning label.
        """
        # --- Step 0: load and normalise the raw slice (no parent augmentation) ---
        x = super().__getitem__(idx)  # (H, W), float32, ~[0, 1]

        # --- Step 0b (optional): scale + translation augmentation ---
        if self.do_augment:
            x = self._augment_scale_translate(x)

        # --- Step 1: rotate by a uniformly sampled alpha ∈ [0°, 180°) ---
        # This moves the existing missing Fourier cone from 0° to alpha.
        alpha = float(torch.empty(1).uniform_(0.0, 180.0))
        x_rotated = TF.rotate(
            x.unsqueeze(0),
            angle=alpha,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        ).squeeze(0)

        # --- Step 2: sample beta at least cone_width_deg away from alpha ---
        beta = self._sample_beta(alpha)

        # --- Step 3: carve a Fourier cone of cone_width_deg centred at beta ---
        carved_x = self._carve_cone_2d(x_rotated, beta)

        carved_x = TF.rotate(
            carved_x.unsqueeze(0),
            angle=-beta, # rotate back so the carved cone is at beta in the output space
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        ).squeeze(0)

        x_rotated = TF.rotate(
            x_rotated.unsqueeze(0),
            angle=-beta, # rotate back so the carved cone is at beta in the output space
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        ).squeeze(0)

        alpha_int = int(round(alpha) - round(beta)) % 180
        beta_int = 0.
        return apply_circle_mask(carved_x), apply_circle_mask(x_rotated), beta_int, alpha_int

    def __repr__(self) -> str:
        return (
            f"MissingConeDataset("
            f"files={self.num_files}, "
            f"total_slices={len(self)}, "
            f"image_size={self.image_size}, "
            f"cone_width_deg={self.cone_width_deg:.1f}, "
            f"norm=[{self.norm_min:.4g}, {self.norm_max:.4g}], "
            f"augment={self.do_augment})"
        )
