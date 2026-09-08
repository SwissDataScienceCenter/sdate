"""Thin per-window FBP reconstruction, built directly on the project's
existing ASTRA wrapper (``sdate.tr_diffusion.reconstruct``) -- no new
reconstruction code, just streaming-shaped calls into it.
"""
from __future__ import annotations

import time
from typing import NamedTuple, Optional, Tuple

import numpy as np
import torch

from .sources import WindowSource


class ReconResult(NamedTuple):
    volume: torch.Tensor
    read_seconds: float   # source.read_window -- decode/IO cost, source-dependent
    recon_seconds: float  # attenuation + FBP -- GPU cost


def reconstruct_window(source: WindowSource, idx: np.ndarray, deg_per_frame: float, I0: float,
                       det_bin: int, device: torch.device,
                       vol_shape: Optional[Tuple[int, int, int]] = None) -> ReconResult:
    """Counts window -> attenuation -> FBP volume, clamped >= 0. ``(D, H, W)`` on ``device``.

    Timed in two pieces (``read_seconds``/``recon_seconds``) so a caller can
    tell whether a slow window is decode/IO-bound (the source) or GPU-bound
    (the reconstruction) -- the whole point when comparing a memmap source
    against a direct-``.mov`` ffmpeg source (see ``FfmpegWindowSource``).
    """
    from sdate.tr_diffusion.reconstruct import counts_to_attenuation, projection_angles, reconstruct

    t0 = time.time()
    counts = source.read_window(idx, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.time()

    angles = projection_angles(idx, deg_per_frame=deg_per_frame)
    atten = counts_to_attenuation(counts, I0)
    vol = reconstruct(atten, angles, det_bin=det_bin, device=device, vol_shape=vol_shape)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t2 = time.time()

    return ReconResult(vol, t1 - t0, t2 - t1)
