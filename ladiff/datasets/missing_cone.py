"""MissingConeDataset – conditional diffusion training pairs for missing-cone recovery.

Each item is a 5-tuple ``(carved_x, x, beta_int, alpha_int, slice_type)`` where:

* ``x``         – a 2D slice from the LA reconstruction, rotated by a uniformly
                  sampled angle ``alpha`` so the existing missing Fourier cone
                  now points at ``alpha``.
* ``carved_x``  – ``x`` with an additional Fourier cone of width ``cone_width_deg``
                  removed around angle ``beta``.
* ``beta_int``  – always ``0.`` (placeholder for class conditioning).
* ``alpha_int`` – integer angle ``(round(alpha) - round(beta)) % 180`` encoding
                  the relative cone offset.
* ``slice_type`` – one of ``'axis0'``, ``'axis1'``, ``'axis2'`` or ``'rand'``
                  indicating the source slice type.

Slices are drawn from three complementary sources (when enabled):

1. **Axis-0 slices** – the canonical missing-cone slices ``vol[i, :, :]``.
2. **Orthogonal-axis slices** – ``vol[:, j, :]`` (axis 1) and ``vol[:, :, k]``
   (axis 2).  These have no intrinsic missing cone but the same carving pipeline
   is applied, effectively doubling the training set size.
3. **Random-plane slices** – arbitrary planes through the centre of a randomly
   chosen volume, sampled with trilinear interpolation.  Defaults to the same
   count as axis-0 slices, bringing the total to 4× the original size.

The diffusion model is trained to predict the noise added to ``x`` given
``carved_x`` concatenated channel-wise with the noisy ``x_t``, and ``beta_int``
as a class label.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode

from ladiff.datasets.npy_volume import NpyVolumeSliceDataset
from ladiff.fourier_wedge import apply_missing_wedge, apply_circle_mask


class MissingConeDataset(NpyVolumeSliceDataset):
    """Dataset of ``(carved_x, x, beta_int, alpha_int, slice_type)`` tuples for missing-cone recovery.

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

    In addition to the canonical axis-0 slices (``vol[i, :, :]``), the dataset
    can include:

    * **Orthogonal-axis slices** – slices along axes 1 (``vol[:, j, :]``) and 2
      (``vol[:, :, k]``).  These have no intrinsic missing cone, but the same
      random-rotation + carving pipeline is applied when ``"axis1"`` and
      ``"axis2"`` are included in ``slice_types``.  Defaults to included when
      ``slice_types=None``.
    * **Random-plane slices** (``n_random_slices``) – arbitrary planes through
      the centre of a randomly chosen volume, sampled with trilinear
      interpolation.  Defaults to the number of axis-0 slices so the combined
      dataset is 4× the original size when ``slice_types=None``.

    Parameters
    ----------
    data_path:
        Path to a single ``.npy`` file **or** a directory.  When a directory is
        given, all ``.npy`` files inside it are loaded and concatenated.
    cone_width_deg:
        Width in degrees of the Fourier cone that is carved out.
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
    apply_circle_masking:
        Apply a circular mask to the output slices.  Default ``True``.
    gamma:
        Orientation correction in degrees.  Applied only to axis-0 slices
        (which carry the real missing cone).  Each such slice is rotated by
        ``-gamma`` so the missing cone is brought to 0° before the random
        ``alpha`` rotation is applied.  Default ``0.0`` (no correction).
    slice_types:
        Optional subset of slice source types to include.  Valid values are
        ``("axis0", "axis1", "axis2", "rand")``.  When ``None`` (default),
        all source types are included: axis-0, axis-1, axis-2, and random-plane
        slices.
    n_random_slices:
        Number of random-plane slices added per epoch.  Each call to
        ``__getitem__`` for these indices samples a fresh random plane from a
        randomly chosen volume (stochastic, good for training).  If ``None``
        (default), equals the number of axis-0 slices so the total is 4×.
        Pass ``0`` to disable random-plane slices entirely.
    target_size:
        Target ``(height, width)`` (or a single integer for square output) used
        when cropping orthogonal-axis and random-plane slices to a fixed size.
        If ``None`` (default), uses ``image_size`` from the parent dataset
        (i.e. ``(vol.shape[1], vol.shape[2])``).
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        cone_width_deg: float,
        normalize_range: Optional[Tuple[float, float]] = None,
        augment: bool = True,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        shift_fraction: float = 0.05,
        apply_circle_masking: bool = True,
        gamma: float = 0.0,
        slice_types: Optional[Tuple[str, ...]] = None,
        n_random_slices: Optional[int] = None,
        target_size: Optional[Union[int, Tuple[int, int]]] = None,
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
        self._apply_circle_masking = apply_circle_masking
        self.gamma = float(gamma)
        self.slice_types = self._normalize_slice_types(slice_types, n_random_slices)

        # Number of axis-0 slices (the canonical missing-cone slices).
        self._n0: int = int(self._cumulative[-1]) if "axis0" in self.slice_types else 0

        # Target image size for orthogonal / random-plane slices.
        if target_size is None:
            h, w = self.image_size
            self._target_size: Tuple[int, int] = (h, w)
        elif isinstance(target_size, int):
            self._target_size = (target_size, target_size)
        else:
            self._target_size = (int(target_size[0]), int(target_size[1]))

        # Build cumulative index tables for orthogonal axes.
        if "axis1" in self.slice_types or "axis2" in self.slice_types:
            axis1_counts = [self._load_volume(i).shape[1] for i in range(len(self.files))]
            axis2_counts = [self._load_volume(i).shape[2] for i in range(len(self.files))]
            self._n1: int = int(sum(axis1_counts)) if "axis1" in self.slice_types else 0
            self._n2: int = int(sum(axis2_counts)) if "axis2" in self.slice_types else 0
            if self._n1 > 0:
                self._cumulative_axis1: np.ndarray = np.concatenate(
                    [[0], np.cumsum(axis1_counts)]
                ).astype(np.int64)
            else:
                self._cumulative_axis1 = np.array([0], dtype=np.int64)
            if self._n2 > 0:
                self._cumulative_axis2: np.ndarray = np.concatenate(
                    [[0], np.cumsum(axis2_counts)]
                ).astype(np.int64)
            else:
                self._cumulative_axis2 = np.array([0], dtype=np.int64)
        else:
            self._n1 = 0
            self._n2 = 0
            self._cumulative_axis1 = np.array([0], dtype=np.int64)
            self._cumulative_axis2 = np.array([0], dtype=np.int64)

        # Number of random-plane slices; defaults to axis-0 count when axis0 is present.
        self._n_rand: int = 0
        if "rand" in self.slice_types:
            self._n_rand = self._n0 if n_random_slices is None else int(n_random_slices)

    # ------------------------------------------------------------------
    # Slice-extraction helpers
    # ------------------------------------------------------------------

    def _normalize_slice_types(
        self,
        slice_types: Optional[Tuple[str, ...]],
        n_random_slices: Optional[int],
    ) -> set[str]:
        """Normalize and validate requested slice source types."""
        canonical = {
            "axis0": "axis0",
            "axial": "axis0",
            "0": "axis0",
            "axis1": "axis1",
            "coronal": "axis1",
            "1": "axis1",
            "axis2": "axis2",
            "sagittal": "axis2",
            "2": "axis2",
            "rand": "rand",
            "random": "rand",
            "randomplane": "rand",
            "random_plane": "rand",
        }

        if slice_types is None:
            slice_types = ["axis0", "axis1", "axis2"]
            if n_random_slices is None or n_random_slices != 0:
                slice_types.append("rand")

        if isinstance(slice_types, str):
            slice_types = (slice_types,)

        normalized = set()
        for value in slice_types:
            key = str(value).lower().replace("-", "").replace(" ", "")
            if key not in canonical:
                raise ValueError(
                    f"Invalid slice type '{value}'. "
                    f"Valid values are {sorted(set(canonical.values()))}."
                )
            normalized.add(canonical[key])

        return normalized

    def _normalize_raw(self, img: np.ndarray) -> torch.Tensor:
        """Normalise a raw float32 ndarray and return a ``(H, W)`` float32 tensor."""
        denom = self.norm_max - self.norm_min
        if denom > 0.0:
            img = (img - self.norm_min) / denom
        else:
            img = img - self.norm_min
        return torch.from_numpy(np.ascontiguousarray(img.astype(np.float32)))

    def _center_crop(self, tensor: torch.Tensor) -> torch.Tensor:
        """Center-crop a ``(H, W)`` tensor to ``self._target_size`` if needed."""
        th, tw = self._target_size
        if tensor.shape[-2] == th and tensor.shape[-1] == tw:
            return tensor
        return TF.center_crop(tensor.unsqueeze(0), [th, tw]).squeeze(0)

    def _get_axis1_slice(self, local_idx: int) -> torch.Tensor:
        """Return a normalised, cropped slice ``vol[:, j, :]`` (axis 1)."""
        file_idx = int(np.searchsorted(self._cumulative_axis1[1:], local_idx, side="right"))
        j = local_idx - int(self._cumulative_axis1[file_idx])
        vol = self._load_volume(file_idx)
        return self._center_crop(self._normalize_raw(vol[:, j, :]))

    def _get_axis2_slice(self, local_idx: int) -> torch.Tensor:
        """Return a normalised, cropped slice ``vol[:, :, k]`` (axis 2)."""
        file_idx = int(np.searchsorted(self._cumulative_axis2[1:], local_idx, side="right"))
        k = local_idx - int(self._cumulative_axis2[file_idx])
        vol = self._load_volume(file_idx)
        return self._center_crop(self._normalize_raw(vol[:, :, k]))

    def _get_random_plane_slice(self) -> torch.Tensor:
        """Sample a random plane through the centre of a randomly chosen volume.

        A random unit normal ``n`` defines the plane orientation.  Two orthonormal
        vectors ``e1``, ``e2`` in the plane are computed via Gram-Schmidt.  A
        regular pixel grid in the ``(u, v)`` plane is mapped to 3-D voxel
        coordinates and the volume is sampled with trilinear interpolation
        (``torch.nn.functional.grid_sample``).  The output is cropped to
        ``self._target_size``.
        """
        file_idx = int(torch.randint(len(self.files), (1,)).item())
        vol = self._load_volume(file_idx)
        D, H, W = vol.shape
        target_h, target_w = self._target_size

        # Normalise volume → (1, 1, D, H, W) float32 tensor.
        vol_t = torch.from_numpy(vol)  # already float32 from _load_volume
        denom = self.norm_max - self.norm_min
        if denom > 0.0:
            vol_t = (vol_t - self.norm_min) / denom
        else:
            vol_t = vol_t - self.norm_min
        vol_t = vol_t.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

        # Random unit normal (spherical coordinates).
        theta = float(torch.empty(1).uniform_(0.0, 2.0 * math.pi))
        phi = float(torch.empty(1).uniform_(0.0, math.pi))
        n = torch.tensor(
            [
                math.sin(phi) * math.cos(theta),
                math.sin(phi) * math.sin(theta),
                math.cos(phi),
            ],
            dtype=torch.float32,
        )

        # Orthonormal basis in the plane via Gram-Schmidt.
        ref = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        if abs(float(torch.dot(n, ref))) > 0.9:
            ref = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        e1 = ref - torch.dot(ref, n) * n
        e1 = e1 / e1.norm()
        e2 = torch.linalg.cross(n, e1)

        # 2-D pixel grid centred at the volume centre (in voxel units).
        # One step = one voxel.
        u = torch.arange(target_w, dtype=torch.float32) - (target_w - 1) / 2.0
        v = torch.arange(target_h, dtype=torch.float32) - (target_h - 1) / 2.0
        uu, vv = torch.meshgrid(u, v, indexing="xy")  # each: (target_h, target_w)

        # 3-D displacements from centre in voxel units.
        # e1 / e2 components map to (D, H, W) axes respectively.
        p = uu.unsqueeze(-1) * e1 + vv.unsqueeze(-1) * e2  # (target_h, target_w, 3)

        # Convert to grid_sample normalised coords [-1, 1] per axis.
        # grid_sample convention for 5-D input (N,C,D,H,W): grid[..., 0]=x→W,
        # grid[..., 1]=y→H, grid[..., 2]=z→D.
        x_norm = p[..., 2] * (2.0 / max(W - 1, 1))  # W component
        y_norm = p[..., 1] * (2.0 / max(H - 1, 1))  # H component
        z_norm = p[..., 0] * (2.0 / max(D - 1, 1))  # D component
        grid = torch.stack([x_norm, y_norm, z_norm], dim=-1)  # (target_h, target_w, 3)
        grid = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, target_h, target_w, 3)

        sampled = F.grid_sample(
            vol_t,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )  # (1, 1, 1, target_h, target_w)
        return sampled.squeeze()  # (target_h, target_w)

    def _get_raw_slice(self, idx: int) -> torch.Tensor:
        """Return a normalised ``(H, W)`` tensor for the given dataset index.

        Indices are partitioned as:
        ``[0, _n0)`` → axis-0 slices,
        ``[_n0, _n0+_n1)`` → axis-1 slices,
        ``[_n0+_n1, _n0+_n1+_n2)`` → axis-2 slices,
        ``[_n0+_n1+_n2, ...)`` → random-plane slices.
        """
        if idx < self._n0:
            return super().__getitem__(idx)
        elif idx < self._n0 + self._n1:
            return self._get_axis1_slice(idx - self._n0)
        elif idx < self._n0 + self._n1 + self._n2:
            return self._get_axis2_slice(idx - self._n0 - self._n1)
        else:
            return self._get_random_plane_slice()

    def _get_slice_type(self, idx: int) -> str:
        """Return the canonical slice type for the given dataset index."""
        if idx < self._n0:
            return "axis0"
        elif idx < self._n0 + self._n1:
            return "axis1"
        elif idx < self._n0 + self._n1 + self._n2:
            return "axis2"
        return "rand"

    # ------------------------------------------------------------------
    # Augmentation / cone helpers
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

    def __len__(self) -> int:
        return self._n0 + self._n1 + self._n2 + self._n_rand

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, float, int, str]:
        """Return ``(carved_x, x, beta_int, alpha_int, slice_type)`` for the slice at ``idx``.

        Returns
        -------
        carved_x : torch.Tensor, shape ``(H, W)``
            Normalised float32 slice with the Fourier cone at ``beta`` removed.
        x : torch.Tensor, shape ``(H, W)``
            Normalised float32 slice (prediction target).
        beta_int : float
            Placeholder (always ``0.``); retained for API compatibility.
        alpha_int : int
            ``(round(alpha) - round(beta)) % 180``, encoding the relative cone
            offset used as the class conditioning label.
        slice_type : str
            One of ``'axis0'``, ``'axis1'``, ``'axis2'`` or ``'rand'`` indicating
            which source slice type produced this sample.
        """
        # --- Step 0: load and normalise the raw slice ---
        x = self._get_raw_slice(idx)

        # --- Step 0b (optional): scale + translation augmentation ---
        if self.do_augment:
            x = self._augment_scale_translate(x)

        # --- Step 0c: correct for dataset-specific cone orientation ---
        # Only axis-0 slices carry the real missing cone; apply gamma only there.
        if self.gamma != 0.0 and idx < self._n0:
            x = TF.rotate(
                x.unsqueeze(0),
                angle=-self.gamma,
                interpolation=InterpolationMode.BILINEAR,
                fill=0.0,
            ).squeeze(0)

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
            angle=-beta,  # rotate back so the carved cone is at 0° in output space
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        ).squeeze(0)

        x_rotated = TF.rotate(
            x_rotated.unsqueeze(0),
            angle=-beta,  # rotate back so the carved cone is at 0° in output space
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        ).squeeze(0)

        alpha_int = int(round(alpha) - round(beta)) % 180
        beta_int = 0.
        slice_type = self._get_slice_type(idx)
        if self._apply_circle_masking:
            return (
                apply_circle_mask(carved_x),
                apply_circle_mask(x_rotated),
                beta_int,
                alpha_int,
                slice_type,
            )
        else:
            return carved_x, x_rotated, beta_int, alpha_int, slice_type

    def __repr__(self) -> str:
        n_sources = (
            f"axis0={self._n0}"
            + (f", axis1={self._n1}" if self._n1 > 0 else "")
            + (f", axis2={self._n2}" if self._n2 > 0 else "")
            + (f", rand={self._n_rand}" if self._n_rand > 0 else "")
        )
        return (
            f"MissingConeDataset("
            f"files={self.num_files}, "
            f"total_slices={len(self)} ({n_sources}), "
            f"image_size={self.image_size}, "
            f"cone_width_deg={self.cone_width_deg:.1f}, "
            f"gamma={self.gamma:.1f}, "
            f"slice_types={sorted(self.slice_types)}, "
            f"norm=[{self.norm_min:.4g}, {self.norm_max:.4g}], "
            f"augment={self.do_augment})"
        )
