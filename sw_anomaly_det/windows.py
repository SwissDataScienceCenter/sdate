"""Centered revolution windows for the streaming anomaly detector.

Every window is a *contiguous* block of frames spanning some number of full
revolutions, centered on a timestep ``t`` -- unlike
``sdate.tr_diffusion.reconstruct.sliding_windows`` (which anchors windows at
their start), the detector always needs several different-length windows
that share the same center frame, so they're directly comparable.
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np


def revolution_frames(n_revolutions: float, deg_per_frame: float) -> int:
    """Number of frames spanning ``n_revolutions`` full turns."""
    return int(round(n_revolutions * 360.0 / deg_per_frame))


def centered_window(t: int, n_frames: int) -> np.ndarray:
    """Contiguous frame indices of length ``n_frames`` centered on ``t``.

    For even ``n_frames`` the extra frame goes on the trailing side (``t``
    sits just left of center) -- an arbitrary but fixed convention, and the
    same one every window (any ``T``) uses, so they stay aligned to a common
    center.
    """
    lo = int(t) - n_frames // 2
    return np.arange(lo, lo + n_frames)


def window_centers(start: int, end: int, stride: int, window_frame_counts: Sequence[int]) -> List[int]:
    """Timesteps ``t`` in ``[start, end)`` (stride ``stride``) whose LARGEST
    window (over ``window_frame_counts``) stays fully inside ``[start, end)``.

    Callers pass every arm's frame count (all ``T`` revolutions' windows plus
    the single-revolution reference) so the margin covers every arm at once --
    downstream code never has to re-check per arm.
    """
    margin = max(int(np.ceil(n / 2.0)) for n in window_frame_counts)
    lo, hi = start + margin, end - margin
    return list(range(lo, hi, int(stride))) if hi > lo else []
