"""Missing-wedge Fourier operators for independent 3D iso-diffusion code."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import torch

Axis = Literal[0, 1, 2]


def _build_mask_2d_torch(
    shape_pa0: int,
    shape_pa1: int,
    angular_range_deg: float,
    start_angle_deg: float,
    device: torch.device,
) -> torch.Tensor:
    c0 = torch.arange(shape_pa0, dtype=torch.float32, device=device) - shape_pa0 // 2
    c1 = torch.arange(shape_pa1, dtype=torch.float32, device=device) - shape_pa1 // 2
    k0, k1 = torch.meshgrid(c0, c1, indexing="ij")

    angle_eff = torch.atan2(k1, k0).mul_(180.0 / math.pi).remainder_(180.0)
    a_start = start_angle_deg % 180.0
    a_end = (start_angle_deg + angular_range_deg) % 180.0

    if a_start < a_end:
        mask = (angle_eff >= a_start) & (angle_eff < a_end)
    else:
        mask = (angle_eff >= a_start) | (angle_eff < a_end)

    mask[shape_pa0 // 2, shape_pa1 // 2] = True
    return mask


def build_missing_wedge_mask(
    vol_shape: Tuple[int, int, int],
    angular_range_deg: float,
    start_angle_deg: float = 0.0,
    tilt_axis: Axis = 0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return a boolean Fourier keep-mask for a 3D missing-wedge geometry."""
    if tilt_axis not in (0, 1, 2):
        raise ValueError("tilt_axis must be 0, 1, or 2")
    if device is None:
        device = torch.device("cpu")

    plane_axes = [axis for axis in range(3) if axis != tilt_axis]
    pa0, pa1 = plane_axes
    mask_2d = _build_mask_2d_torch(
        vol_shape[pa0], vol_shape[pa1], angular_range_deg, start_angle_deg, device
    )
    return mask_2d.unsqueeze(tilt_axis).expand(vol_shape).contiguous()


def apply_missing_wedge(
    volume: torch.Tensor,
    angular_range_deg: float,
    start_angle_deg: float = 0.0,
    tilt_axis: Axis = 0,
) -> torch.Tensor:
    """FFT a volume, zero the missing wedge, and return the real inverse FFT."""
    if not isinstance(volume, torch.Tensor):
        raise TypeError(f"volume must be a torch.Tensor, got {type(volume)}")

    vol = volume.float()
    mask = build_missing_wedge_mask(
        tuple(vol.shape), angular_range_deg, start_angle_deg, tilt_axis, device=vol.device
    )
    fft_shifted = torch.fft.fftshift(torch.fft.fftn(vol))
    fft_shifted = fft_shifted * mask
    return torch.fft.ifftn(torch.fft.ifftshift(fft_shifted)).real.float()


def inpaint_fourier_wedge(
    target_volume: torch.Tensor,
    source_volume: torch.Tensor,
    angular_range_deg: float,
    start_angle_deg: float = 0.0,
    tilt_axis: Axis = 0,
) -> torch.Tensor:
    """Fill the missing wedge of ``target_volume`` with Fourier data from ``source_volume``."""
    if target_volume.shape != source_volume.shape:
        raise ValueError(
            f"target shape {tuple(target_volume.shape)} != source shape {tuple(source_volume.shape)}"
        )

    tgt = target_volume.float()
    src = source_volume.float().to(tgt.device)
    keep_mask = build_missing_wedge_mask(
        tuple(tgt.shape), angular_range_deg, start_angle_deg, tilt_axis, device=tgt.device
    )

    fft_tgt = torch.fft.fftshift(torch.fft.fftn(tgt))
    fft_src = torch.fft.fftshift(torch.fft.fftn(src))
    fft_out = torch.where(keep_mask, fft_tgt, fft_src)
    return torch.fft.ifftn(torch.fft.ifftshift(fft_out)).real.float()


def enforce_known_fourier(
    estimate_volume: torch.Tensor,
    measured_volume: torch.Tensor,
    angular_range_deg: float,
    start_angle_deg: float = 0.0,
    tilt_axis: Axis = 0,
) -> torch.Tensor:
    """Preserve known Fourier coefficients from ``measured_volume`` in an estimate."""
    return inpaint_fourier_wedge(
        target_volume=measured_volume,
        source_volume=estimate_volume,
        angular_range_deg=angular_range_deg,
        start_angle_deg=start_angle_deg,
        tilt_axis=tilt_axis,
    )


def make_norm_fns(norm_min: float, norm_max: float) -> Tuple[Callable, Callable]:
    """Return linear normalize/denormalize callables for tensors."""
    denom = float(norm_max) - float(norm_min)
    if denom <= 0:
        raise ValueError("norm_max must be greater than norm_min")

    def normalize_fn(x: torch.Tensor) -> torch.Tensor:
        return (x - norm_min) / denom

    def denormalize_fn(x: torch.Tensor) -> torch.Tensor:
        return x * denom + norm_min

    return normalize_fn, denormalize_fn


def load_norm_json(path: Path) -> Tuple[float, float]:
    with open(path) as f:
        cfg = json.load(f)
    return float(cfg["norm_min"]), float(cfg["norm_max"])
