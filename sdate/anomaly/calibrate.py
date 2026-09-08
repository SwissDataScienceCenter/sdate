"""Estimate acquisition geometry from a warm-up slice of the stream.

Used *before* detection when calibration values are unknown.  Ports the three
independent methods validated in ``notebooks/wunderkerze_rotation_calibration.ipynb``:

* **period / deg-per-frame** — identity autocorrelation of the row-summed
  horizontal profiles: frame ``k`` vs ``k+P`` best-matches at the 360° period,
  refined to sub-frame precision by parabolic interpolation of the peak.
* **rotation axis** — column maximising the 180°-mirror overlap (NCC of
  ``proj(θ)`` vs ``fliplr_c(proj(θ+180°))``).

Feed a stack of ``N`` *consecutive* frames (a few turns is plenty).  Returns a
partially-populated :class:`~sdate.anomaly.config.CalibrationConfig`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import CalibrationConfig


def _profiles(frames: np.ndarray) -> np.ndarray:
    """Row-summed horizontal profile per frame: ``(N, W)``."""
    return frames.reshape(frames.shape[0], frames.shape[1], frames.shape[2]).sum(axis=1)


def estimate_period(frames: np.ndarray, lag_lo: int = 150, lag_hi: int = 260,
                    stride: int = 3) -> float:
    """Frames per 360° turn, via identity-autocorrelation of the profiles."""
    prof = _profiles(np.asarray(frames, dtype=np.float64))
    pm = prof - prof.mean(1, keepdims=True)
    nrm = np.sqrt((pm * pm).sum(1)) + 1e-12
    lags = np.arange(lag_lo, lag_hi)
    sc = np.array([
        ((pm[:-L] * pm[L:]).sum(1) / (nrm[:-L] * nrm[L:]))[::stride].mean()
        for L in lags
    ])
    j = int(sc.argmax())
    if 0 < j < len(sc) - 1:                       # parabolic sub-frame refine
        y0, y1, y2 = sc[j - 1], sc[j], sc[j + 1]
        denom = (y0 - 2 * y1 + y2)
        frac = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    else:
        frac = 0.0
    return float(lags[j] + frac)


def estimate_axis(frames: np.ndarray, period: float,
                  lo_frac: float = 0.45, hi_frac: float = 0.55,
                  step: float = 0.25, n_pairs: int = 20) -> float:
    """Rotation-axis detector column, via 180°-mirror NCC maximisation."""
    F = np.asarray(frames, dtype=np.float64)
    N, H, W = F.shape
    half = int(round(period / 2.0))
    cols = np.arange(W)

    def flip(img, c):
        src = 2 * c - cols
        return np.stack([np.interp(src, cols, img[r]) for r in range(H)])

    def ncc(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-12))

    idx = np.arange(0, N - half, max(1, (N - half) // n_pairs))
    pairs = [(F[i], F[i + half]) for i in idx]
    cs = np.arange(lo_frac * W, hi_frac * W, step)
    sc = np.array([np.mean([ncc(a, flip(b, c)) for a, b in pairs]) for c in cs])
    j = int(sc.argmax())
    if 0 < j < len(sc) - 1:
        y0, y1, y2 = sc[j - 1], sc[j], sc[j + 1]
        denom = (y0 - 2 * y1 + y2)
        refine = 0.5 * (y0 - y2) / denom * step if abs(denom) > 1e-12 else 0.0
    else:
        refine = 0.0
    return float(cs[j] + refine)


def calibrate_from_frames(frames: np.ndarray, ref_frame: int = 0,
                          with_axis: bool = True,
                          flat: Optional[np.ndarray] = None,
                          dark: Optional[np.ndarray] = None) -> CalibrationConfig:
    """Build a :class:`CalibrationConfig` from a warm-up stack of consecutive frames."""
    frames = np.asarray(frames, dtype=np.float64)
    period = estimate_period(frames)
    axis = estimate_axis(frames, period) if with_axis else None
    return CalibrationConfig(
        frame_shape=(frames.shape[1], frames.shape[2]),
        deg_per_frame=360.0 / period,
        period=period,
        ref_frame=ref_frame,
        axis=axis,
        flat=flat,
        dark=dark,
    )
