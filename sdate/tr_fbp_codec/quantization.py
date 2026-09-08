"""Int16 (or float count) -> 12-bit (4096-level) quantization + loss check.

Implements the "quick check on how much we lose" the design session asked
for: truncate/rescale to 12 bits, measure PSNR/SSIM against the original on
a sample of real projections, plus basic clipping/histogram statistics.

This does NOT decide the mapping mode for you -- run
:func:`evaluate_quantization_loss` on whatever dataset you point this
package at before assuming ``mode="truncate"`` is safe (it is dataset/
detector-dependent; see README.md and
``context/compression_data_recon/DATA_REPORT.md`` for a real counter-
example where it isn't).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .config import QuantConfig


def quantize_to_12bit(x: np.ndarray, cfg: QuantConfig) -> np.ndarray:
    """Map raw counts to integer class indices in [0, cfg.n_levels)."""
    n_max = cfg.n_levels - 1
    if cfg.mode == "truncate":
        q = np.clip(np.round(x), 0, n_max)
    elif cfg.mode == "rescale":
        if cfg.data_min is None or cfg.data_max is None:
            raise ValueError("QuantConfig.data_min/data_max required for mode='rescale'")
        span = cfg.data_max - cfg.data_min
        if span <= 0:
            raise ValueError(f"degenerate range [{cfg.data_min}, {cfg.data_max}]")
        q = np.clip(np.round((x - cfg.data_min) / span * n_max), 0, n_max)
    else:
        raise ValueError(f"unknown QuantConfig.mode={cfg.mode!r}")
    return q.astype(np.int64)


def dequantize_from_12bit(q: np.ndarray, cfg: QuantConfig) -> np.ndarray:
    """Inverse of :func:`quantize_to_12bit` (identity for ``mode='truncate'``)."""
    n_max = cfg.n_levels - 1
    if cfg.mode == "truncate":
        return q.astype(np.float64)
    elif cfg.mode == "rescale":
        span = cfg.data_max - cfg.data_min
        return q.astype(np.float64) / n_max * span + cfg.data_min
    raise ValueError(f"unknown QuantConfig.mode={cfg.mode!r}")


@dataclass
class QuantizationLossReport:
    n_frames: int
    global_min: float
    global_max: float
    bits_used: float  # log2(global_max + 1)
    clip_fraction: float  # fraction of pixels that hit the [0, n_max] clip bound
    psnr_mean: float
    psnr_std: float
    psnr_min: float
    ssim_mean: float
    ssim_std: float
    ssim_min: float

    def verdict(self, psnr_floor: float = 45.0, ssim_floor: float = 0.995) -> str:
        if self.clip_fraction > 0:
            return (
                f"LOSSY: {self.clip_fraction:.4%} of pixels clip under this mapping "
                f"(global_max={self.global_max}, bits_used={self.bits_used:.2f})."
            )
        if self.psnr_min < psnr_floor or self.ssim_min < ssim_floor:
            return (
                f"MARGINAL: worst-case frame PSNR={self.psnr_min:.1f}dB / "
                f"SSIM={self.ssim_min:.4f} -- inspect outlier frames before trusting the mean."
            )
        return (
            f"SAFE: effectively lossless (mean PSNR={self.psnr_mean:.1f}dB, "
            f"worst-case PSNR={self.psnr_min:.1f}dB, bits_used={self.bits_used:.2f}/12)."
        )


def evaluate_quantization_loss(
    frames: Sequence[np.ndarray], cfg: QuantConfig
) -> QuantizationLossReport:
    """Quantize each frame in ``frames`` to 12 bits and measure the damage.

    ``frames`` should be real, per-frame raw-count arrays (float or int),
    NOT normalized to [0, 1] -- normalization would hide genuine clipping.
    """
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    stacked = np.stack([np.asarray(f, dtype=np.float64) for f in frames])
    global_min, global_max = float(stacked.min()), float(stacked.max())
    bits_used = float(np.log2(global_max + 1)) if global_max > 0 else 0.0

    q = quantize_to_12bit(stacked, cfg)
    recon = dequantize_from_12bit(q, cfg)

    # Only the upper bound is genuine information loss (a value that gets
    # pushed down to n_max). Touching 0 is a normal CT dark-count floor, not
    # clipping -- do not flag it (an earlier version of this check did, and
    # produced false "LOSSY" verdicts on real data that simply has dark
    # pixels at 0).
    n_max = cfg.n_levels - 1
    clip_fraction = float(np.mean(stacked > n_max)) if cfg.mode == "truncate" else 0.0

    data_range = max(global_max - global_min, 1e-6)
    # Cap PSNR at a sentinel: a bit-exact round-trip (expected for
    # mode="truncate" on already-integer data under 4096) gives literal
    # inf, which would poison mean/std with nan once mixed with any
    # finite-PSNR frame. 100 dB reads as "effectively exact" without that.
    psnr_cap = 100.0
    psnrs, ssims = [], []
    for orig, rec in zip(stacked, recon):
        psnrs.append(min(peak_signal_noise_ratio(orig, rec, data_range=data_range), psnr_cap))
        ssims.append(structural_similarity(orig, rec, data_range=data_range))
    psnrs, ssims = np.array(psnrs), np.array(ssims)

    return QuantizationLossReport(
        n_frames=len(frames),
        global_min=global_min,
        global_max=global_max,
        bits_used=bits_used,
        clip_fraction=clip_fraction,
        psnr_mean=float(psnrs.mean()),
        psnr_std=float(psnrs.std()),
        psnr_min=float(psnrs.min()),
        ssim_mean=float(ssims.mean()),
        ssim_std=float(ssims.std()),
        ssim_min=float(ssims.min()),
    )
