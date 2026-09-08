"""Route a raw acquisition stream into per-angle fixed-view sub-streams.

The sample rotates continuously and the period is *non-integer*
(≈199.844 frames/360°), so a tracked angle ``α`` is essentially never hit by an
integer acquired frame.  For each ``α`` we schedule the continuous crossing
positions ``x_m(α) = ref + α/deg_per_frame + m·period`` and, as consecutive
frames stream in, emit the **linear interpolation** of the two frames bracketing
the next crossing — exactly the sub-frame angle compensation used to build the
fixed-angle movies (see ``notebooks/wunderkerze_rotation_cache/make_movies.py``).
Each tracked angle therefore yields one aligned frame per turn.

Interpolation happens on *raw* frames (a linear op that commutes with flat/dark
correction); per-frame flux normalisation is applied downstream, after the
fixed-angle frame is synthesised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .config import CalibrationConfig, DetectorConfig


@dataclass
class AngleEmission:
    angle_index: int
    angle: float
    turn: int
    x_pos: float          # continuous source-frame position sampled
    vector: np.ndarray    # interpolated raw frame, flattened (D,)


class _AngleStream:
    """Tracks the next crossing position for one fixed angle."""

    def __init__(self, angle_index: int, angle: float, next_x: float, period: float):
        self.angle_index = angle_index
        self.angle = angle
        self.next_x = next_x
        self.period = period
        self.turn = 0


class AngleMultiplexer:
    """Consumes ``(idx, frame)`` pairs, emits interpolated fixed-angle frames."""

    def __init__(self, calib: CalibrationConfig, cfg: DetectorConfig):
        self.calib = calib
        self.cfg = cfg
        self._streams: List[_AngleStream] = []
        self._prev_idx: Optional[int] = None
        self._prev_vec: Optional[np.ndarray] = None
        self.skipped_gaps = 0

    def _init_streams(self, start_idx: int) -> None:
        c = self.calib
        period = c.period
        for ai, angle in enumerate(self.cfg.target_angles):
            first = c.ref_frame + angle / c.deg_per_frame
            # smallest crossing position >= start_idx
            m0 = int(np.ceil((start_idx - first) / period))
            next_x = first + m0 * period
            self._streams.append(_AngleStream(ai, angle, next_x, period))

    def push(self, idx: int, frame: np.ndarray) -> List[AngleEmission]:
        """Feed one raw acquired frame; return any fixed-angle frames it completes."""
        vec = np.asarray(frame, dtype=np.float64).reshape(-1)
        emissions: List[AngleEmission] = []

        if not self._streams:
            self._init_streams(idx)

        if self._prev_idx is not None:
            if idx == self._prev_idx + 1:
                for s in self._streams:
                    # a crossing bracketed by [prev, cur] : prev <= x < cur
                    if self._prev_idx <= s.next_x < idx:
                        frac = s.next_x - self._prev_idx
                        interp = (1.0 - frac) * self._prev_vec + frac * vec
                        emissions.append(AngleEmission(
                            s.angle_index, s.angle, s.turn, s.next_x, interp))
                        s.turn += 1
                        s.next_x += s.period
            else:
                # non-consecutive frame: cannot bracket crossings inside the gap
                self.skipped_gaps += 1
                for s in self._streams:
                    while s.next_x < idx:
                        s.next_x += s.period
                        s.turn += 1  # count the missed turn so timelines stay aligned

        self._prev_idx = idx
        self._prev_vec = vec
        return emissions
