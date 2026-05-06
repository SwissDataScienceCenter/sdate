"""fourier_wedge.py – Missing-wedge simulation and inpainting in 3-D Fourier space.

The missing wedge is the region of Fourier space that is unsampled in limited-angle
electron tomography.  The sample is tilted around one axis (the *tilt axis*), so
projections are collected only within a restricted angular range.  In Fourier space
this corresponds to a wedge-shaped gap in the plane perpendicular to the tilt axis.

All functions accept and return **PyTorch tensors** and work on any device
(CPU or CUDA).  The output tensor lives on the same device as the input.

Public API
----------
build_missing_wedge_mask(vol_shape, angular_range_deg, start_angle_deg, tilt_axis, device)
    Build a boolean 3-D mask that is True where data are *kept* (i.e. outside the
    missing wedge) and False inside the missing wedge.

apply_missing_wedge(volume, angular_range_deg, start_angle_deg, tilt_axis)
    Forward-simulate the missing-wedge artefact: FFT → zero out missing region → iFFT.
    Input and output are float32 tensors on the same device.

inpaint_fourier_wedge(target_volume, source_volume, angular_range_deg,
                      start_angle_deg, tilt_axis)
    Fill the missing-wedge region of *target_volume* in Fourier space with the
    corresponding Fourier content taken from *source_volume*.
    Input and output are float32 tensors on the same device.
"""

from __future__ import annotations

import math
from typing import Literal, Optional, Tuple

import torch

__all__ = [
    "build_missing_wedge_mask",
    "apply_missing_wedge",
    "inpaint_fourier_wedge",
    "extract_central_slice_at_angle",
]

# Type alias for the three spatial axes
Axis = Literal[0, 1, 2]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_mask_2d_torch(
    shape_pa0: int,
    shape_pa1: int,
    angular_range_deg: float,
    start_angle_deg: float,
    device: torch.device,
) -> torch.Tensor:
    """Build the 2-D boolean keep-mask in the wedge plane on *device*."""
    c0 = torch.arange(shape_pa0, dtype=torch.float32, device=device) - shape_pa0 // 2
    c1 = torch.arange(shape_pa1, dtype=torch.float32, device=device) - shape_pa1 // 2
    K0, K1 = torch.meshgrid(c0, c1, indexing="ij")

    # atan2(K1, K0) % 180 enforces Friedel symmetry of real-valued volumes
    angle_eff = torch.atan2(K1, K0).mul_(180.0 / math.pi).remainder_(180.0)

    a_start = start_angle_deg % 180.0
    a_end = (start_angle_deg + angular_range_deg) % 180.0

    if a_start < a_end:
        mask = (angle_eff >= a_start) & (angle_eff < a_end)
    else:  # range wraps around 180°
        mask = (angle_eff >= a_start) | (angle_eff < a_end)

    # Always keep the DC component
    mask[shape_pa0 // 2, shape_pa1 // 2] = True
    return mask


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_circle_mask(img):
    W, H = img.shape[-2:]
    cp = torch.cartesian_prod(torch.arange(W, device=img.device), torch.arange(H, device=img.device))
    circle_mask = (cp[:, 0] - W / 2) ** 2 + (cp[:, 1] - W / 2) ** 2 <= (W / 2) ** 2
    return img * circle_mask.reshape(img.shape[-2:])

def build_missing_wedge_mask(
    vol_shape: Tuple[int, int, int],
    angular_range_deg: float,
    start_angle_deg: float = 0.0,
    tilt_axis: Axis = 0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Build a 3-D boolean keep-mask for a missing-wedge geometry.

    The volume is assumed to have shape ``(nz, ny, nx)``.  The mask is ``True``
    where Fourier data are *retained* (i.e. outside the missing wedge) and
    ``False`` inside the missing wedge.

    The missing wedge is defined in the plane perpendicular to *tilt_axis*.
    All slices along the tilt axis share the same 2-D pattern; the 2-D mask is
    broadcast via a singleton dimension.

    Parameters
    ----------
    vol_shape : (int, int, int)
        Shape of the volume ``(nz, ny, nx)``.
    angular_range_deg : int or float
        Total angular span that is *kept* (degrees).  Use
        ``int(angular_range_frac * 180)`` to convert from a fraction.
    start_angle_deg : float, optional
        Starting angle of the kept range in degrees.  Default ``0.0``.
    tilt_axis : {0, 1, 2}, optional
        Axis around which the sample rotates:
        * ``0`` → kz  (default)
        * ``1`` → ky
        * ``2`` → kx
        The missing wedge lives in the plane of the *other two* axes.
    device : torch.device or None, optional
        Target device.  Defaults to CPU.

    Returns
    -------
    mask : torch.Tensor, dtype bool, shape ``vol_shape``
    """
    assert tilt_axis in (0, 1, 2), "tilt_axis must be 0, 1, or 2"
    if device is None:
        device = torch.device("cpu")

    plane_axes = [a for a in range(3) if a != tilt_axis]
    pa0, pa1 = plane_axes

    mask_2d = _build_mask_2d_torch(
        vol_shape[pa0], vol_shape[pa1], angular_range_deg, start_angle_deg, device
    )

    # Insert singleton along tilt axis → broadcasts over all tilt-axis slices
    mask_3d = mask_2d.unsqueeze(tilt_axis)
    return mask_3d.expand(vol_shape).contiguous()


def apply_missing_wedge(
    volume: torch.Tensor,
    angular_range_deg: float,
    start_angle_deg: float = 0.0,
    tilt_axis: Axis = 0,
) -> torch.Tensor:
    """Forward-simulate the missing-wedge artefact on *volume*.

    Computes the 3-D FFT, zeroes out the missing-wedge region, then applies the
    inverse FFT and returns the real part as a float32 tensor on the same device.

    Parameters
    ----------
    volume : torch.Tensor, shape (nz, ny, nx)
        Input volume.  Must be a float32 tensor (CPU or CUDA).
    angular_range_deg : int or float
        Total angular span that is *kept* (degrees).
    start_angle_deg : float, optional
        Starting angle of the kept range in degrees.  Default ``0.0``.
    tilt_axis : {0, 1, 2}, optional
        Tilt axis (see :func:`build_missing_wedge_mask`).  Default ``0``.

    Returns
    -------
    result : torch.Tensor, dtype float32, same shape and device as *volume*
        Volume after missing-wedge zeroing in Fourier space.
    """
    if not isinstance(volume, torch.Tensor):
        raise TypeError(f"volume must be a torch.Tensor, got {type(volume)}")

    vol = volume.float()
    mask = build_missing_wedge_mask(
        tuple(vol.shape), angular_range_deg, start_angle_deg, tilt_axis, device=vol.device
    )

    fft_shifted = torch.fft.fftshift(torch.fft.fftn(vol))
    fft_shifted = fft_shifted * mask
    result = torch.fft.ifftn(torch.fft.ifftshift(fft_shifted)).real.float()
    return result


def inpaint_fourier_wedge(
    target_volume: torch.Tensor,
    source_volume: torch.Tensor,
    angular_range_deg: float,
    start_angle_deg: float = 0.0,
    tilt_axis: Axis = 0,
) -> torch.Tensor:
    """Fill the missing-wedge region of *target_volume* with Fourier content from *source_volume*.

    In the kept region the Fourier content of *target_volume* is preserved; in the
    missing-wedge region it is replaced with the corresponding Fourier content of
    *source_volume*.

    Parameters
    ----------
    target_volume : torch.Tensor, shape (nz, ny, nx)
        Volume whose missing-wedge region will be filled.  Must be float32.
    source_volume : torch.Tensor, shape (nz, ny, nx)
        Volume that provides the Fourier content inside the missing wedge.
        Must have the same shape as *target_volume*.  Moved to the same device
        as *target_volume* automatically if needed.
    angular_range_deg : int or float
        Total angular span that is *kept* in *target_volume* (degrees).
    start_angle_deg : float, optional
        Starting angle of the kept range in degrees.  Default ``0.0``.
    tilt_axis : {0, 1, 2}, optional
        Tilt axis (see :func:`build_missing_wedge_mask`).  Default ``0``.

    Returns
    -------
    result : torch.Tensor, dtype float32, same shape and device as *target_volume*
        Inpainted volume in the spatial domain.
    """
    if not isinstance(target_volume, torch.Tensor):
        raise TypeError(f"target_volume must be a torch.Tensor, got {type(target_volume)}")
    if not isinstance(source_volume, torch.Tensor):
        raise TypeError(f"source_volume must be a torch.Tensor, got {type(source_volume)}")

    tgt = target_volume.float()
    src = source_volume.float().to(tgt.device)

    if tgt.shape != src.shape:
        raise ValueError(
            f"target_volume shape {tuple(tgt.shape)} != source_volume shape {tuple(src.shape)}"
        )

    keep_mask = build_missing_wedge_mask(
        tuple(tgt.shape), angular_range_deg, start_angle_deg, tilt_axis, device=tgt.device
    )

    fft_tgt = torch.fft.fftshift(torch.fft.fftn(tgt))
    fft_src = torch.fft.fftshift(torch.fft.fftn(src))

    # Composite: keep target outside the wedge, fill with source inside the wedge
    fft_inpainted = torch.where(keep_mask, fft_tgt, fft_src)

    result = torch.fft.ifftn(torch.fft.ifftshift(fft_inpainted)).real.float()
    return result


def extract_central_slice_at_angle(
    volume: torch.Tensor,
    angle_deg: float,
    tilt_axis: Axis = 0,
    n_samples: Optional[int] = None,
) -> torch.Tensor:
    """Extract a 2-D vertical slice through the centre of *volume* at azimuthal angle *angle_deg*.

    The slice plane **contains** the tilt axis and is oriented at *angle_deg* in the
    plane perpendicular to it, using the same angle convention as the missing-wedge mask::

        angle = atan2(K_pa1, K_pa0) % 180    (Friedel-symmetric)

    where ``pa0`` and ``pa1`` are the two axes that are *not* the tilt axis.

    This means:

    * angle = 0° → the slice lies along the *pa0* direction
    * angle = 90° → the slice lies along the *pa1* direction

    Parameters
    ----------
    volume : torch.Tensor, shape (d0, d1, d2)
        Input volume (float32, any device).
    angle_deg : float
        Azimuthal angle in degrees.
    tilt_axis : {0, 1, 2}, optional
        Which axis is the rotation / tilt axis.  Default ``0``.
    n_samples : int, optional
        Number of samples along the radial direction.
        Defaults to ``max(n_pa0, n_pa1)``.

    Returns
    -------
    slice_2d : torch.Tensor, dtype float32, shape (n_tilt, n_samples)
        Rows correspond to positions along the tilt axis;
        columns correspond to radial positions along the slice direction.
        Bilinear interpolation is used; out-of-bounds pixels are zero.
    """
    if not isinstance(volume, torch.Tensor):
        raise TypeError(f"volume must be a torch.Tensor, got {type(volume)}")

    import torch.nn.functional as F

    vol    = volume.float()
    device = vol.device

    plane_axes = [a for a in range(3) if a != tilt_axis]
    pa0, pa1 = plane_axes
    n_tilt = vol.shape[tilt_axis]
    n_pa0  = vol.shape[pa0]
    n_pa1  = vol.shape[pa1]

    if n_samples is None:
        n_samples = max(n_pa0, n_pa1)

    angle_rad = math.radians(angle_deg)
    # Slice direction in (pa0, pa1): (cos θ, sin θ)  — matches atan2(K_pa1, K_pa0)
    cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

    # Parametric line through the centre of the perpendicular plane
    half = max(n_pa0, n_pa1) / 2.0
    t    = torch.linspace(-half, half, n_samples, device=device)

    coord_pa0 = n_pa0 / 2.0 + t * cos_a  # (n_samples,)
    coord_pa1 = n_pa1 / 2.0 + t * sin_a  # (n_samples,)

    # Normalise to [-1, 1] for grid_sample (align_corners=True)
    coord_pa0_n = 2.0 * coord_pa0 / (n_pa0 - 1) - 1.0
    coord_pa1_n = 2.0 * coord_pa1 / (n_pa1 - 1) - 1.0

    # Permute so layout is (n_tilt, n_pa0, n_pa1), then add channel dim
    perm   = [tilt_axis] + plane_axes
    vol_gs = vol.permute(*perm).contiguous().unsqueeze(1)  # (n_tilt, 1, n_pa0, n_pa1)

    # Build sampling grid: (n_tilt, 1, n_samples, 2)
    # grid_sample convention: grid[..., 0] = x (W = pa1), grid[..., 1] = y (H = pa0)
    gx   = coord_pa1_n.view(1, 1, n_samples).expand(n_tilt, 1, n_samples)
    gy   = coord_pa0_n.view(1, 1, n_samples).expand(n_tilt, 1, n_samples)
    grid = torch.stack([gx, gy], dim=-1)  # (n_tilt, 1, n_samples, 2)

    sampled = F.grid_sample(
        vol_gs, grid,
        mode="bilinear", padding_mode="zeros", align_corners=True,
    )
    # sampled: (n_tilt, 1, 1, n_samples) → (n_tilt, n_samples)
    return sampled.squeeze(1).squeeze(1)
