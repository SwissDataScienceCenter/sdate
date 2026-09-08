"""Configuration objects for streaming PCA anomaly detection.

The library is deliberately *source-agnostic*: it never reads a ``.mov`` / HDF5 /
detector socket itself.  An external orchestrator decodes frames and pushes them
through :meth:`AnomalyDetector.push`.  Everything the detector needs to know in
advance about the acquisition geometry lives in :class:`CalibrationConfig`; the
detector's own knobs live in :class:`DetectorConfig`.

If a calibration value is unknown (typically ``deg_per_frame`` and the rotation
``axis``, occasionally ``flat`` / ``dark``) it can be estimated from a warm-up
slice of the stream with :mod:`sdate.anomaly.calibrate` *before* detection starts.
During detection the calibration is assumed fixed and known.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np


@dataclass
class CalibrationConfig:
    """Everything about the acquisition that the detector treats as *given*.

    Parameters
    ----------
    frame_shape : (H, W) of a single projection frame.
    deg_per_frame : angular increment per acquired frame (constant-speed
        continuous rotation).  For ``212_Wunderkerze2`` this is ``1.801402``.
    period : frames per full 360° turn.  Derived from ``deg_per_frame`` if not
        given (``360 / deg_per_frame``).
    ref_frame : the frame index whose (relative) rotation angle is defined as 0°.
        Angles are relative unless the caller ties one frame to a known
        orientation.
    axis : detector column of the rotation axis (center of rotation).  Only
        needed for 180°-partner mirror alignment; not required for plain
        per-angle 360° streams.
    flat, dark : optional flat-field / dark-field frames (H, W) for
        transmission normalisation.  If absent, preprocessing falls back to
        per-frame flux normalisation + the per-angle PCA mean, which absorbs a
        static background.
    """

    frame_shape: Tuple[int, int]
    deg_per_frame: float
    ref_frame: int = 0
    period: Optional[float] = None
    axis: Optional[float] = None
    flat: Optional[np.ndarray] = None
    dark: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.period is None:
            self.period = 360.0 / float(self.deg_per_frame)

    @property
    def n_pixels(self) -> int:
        h, w = self.frame_shape
        return int(h) * int(w)

    def angle_of(self, frame_idx) -> np.ndarray:
        """Relative rotation angle (deg, wrapped to ``[0, 360)``) of a frame."""
        return (np.asarray(frame_idx, dtype=np.float64) - self.ref_frame) * self.deg_per_frame % 360.0

    def target_positions(self, angle_deg: float, x_min: float, x_max: float) -> np.ndarray:
        """Continuous frame positions ``x_m`` at which ``angle_deg`` recurs.

        ``x_m = ref + angle/deg_per_frame + m * period`` for integer ``m`` such
        that ``x_min <= x_m <= x_max``.  These land *between* integer frames, so
        the multiplexer interpolates the two bracketing acquired frames.
        """
        first = self.ref_frame + (angle_deg / self.deg_per_frame)
        # smallest m with x_m >= x_min
        m0 = int(np.ceil((x_min - first) / self.period))
        xs = []
        m = m0
        while True:
            x = first + m * self.period
            if x > x_max:
                break
            if x >= x_min:
                xs.append(x)
            m += 1
        return np.asarray(xs, dtype=np.float64)


@dataclass
class DetectorConfig:
    """Knobs for the per-angle streaming PCA models and the detector wiring."""

    # --- angles to track (a few for v1; scale to a uniform subset / all later) ---
    target_angles: Tuple[float, ...] = (0.0, 90.0, 180.0)

    # --- streaming PCA ---
    n_modes: int = 8               # retained subspace rank R
    window: int = 40               # sliding window length (turns) kept per angle
    min_turns: int = 12            # turns to observe before scoring begins
    refit_every: int = 1           # refit the SVD every k accepted turns

    # --- robust thresholding / gating (control-chart style, per angle) ---
    gate: bool = True              # withhold flagged frames from the reference
    gate_k: float = 6.0            # threshold = median + gate_k * (1.4826*MAD)
    thr_history: int = 200         # rolling window of accepted scores for stats
    thr_min_history: int = 12      # scores needed before a finite threshold exists
    max_consecutive_reject: int = 8  # force re-baseline after a run of rejects (regime change)
    ema: float = 0.3               # EWMA smoothing applied to the score curves (0 = off)

    # --- preprocessing ---
    flux_normalize: bool = True    # divide each frame by its own mean flux
    use_log: bool = False          # -log after flat/dark (absorption domain)

    # --- recording (for the viewers) ---
    record_frames: bool = True     # keep preprocessed frames + masks for scrubber/mp4
    record_dtype: str = "float16"  # storage dtype for recorded masks/frames

    def __post_init__(self) -> None:
        self.target_angles = tuple(float(a) % 360.0 for a in self.target_angles)
