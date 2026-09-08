"""Frame preprocessing prior to PCA.

Turns a raw projection frame ``(H, W)`` into the flat, mean-normalised vector
``(D,)`` that the per-angle PCA model consumes.  Kept intentionally light: the
per-angle PCA mean already absorbs any *static* background, so the only jobs
here are (1) transmission normalisation when flats/darks exist and (2) killing
*global* per-frame brightness drift (beam-flux fluctuation) which would
otherwise dominate the Q statistic.
"""

from __future__ import annotations

import numpy as np

from .config import CalibrationConfig, DetectorConfig


class FramePreprocessor:
    """Vectorise + normalise frames according to the calibration/detector config."""

    def __init__(self, calib: CalibrationConfig, cfg: DetectorConfig):
        self.calib = calib
        self.cfg = cfg
        self._flat = None
        self._dark = None
        if calib.dark is not None:
            self._dark = np.asarray(calib.dark, dtype=np.float64).reshape(-1)
        if calib.flat is not None:
            flat = np.asarray(calib.flat, dtype=np.float64).reshape(-1)
            if self._dark is not None:
                flat = flat - self._dark
            # guard against divide-by-zero in flat-field
            self._flat = np.where(np.abs(flat) < 1e-8, 1e-8, flat)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Return a 1-D ``float64`` feature vector for one ``(H, W)`` frame."""
        x = np.asarray(frame, dtype=np.float64).reshape(-1)

        if self._dark is not None:
            x = x - self._dark
        if self._flat is not None:
            x = x / self._flat                     # transmission
            if self.cfg.use_log:
                x = -np.log(np.clip(x, 1e-6, None))  # absorption / line integral

        if self.cfg.flux_normalize:
            m = x.mean()
            if abs(m) > 1e-12:
                x = x / m

        return x
