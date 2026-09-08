"""Top-level streaming anomaly detector.

Source-agnostic: an external orchestrator decodes frames (from ``.mov``, HDF5, a
detector socket, …) and calls :meth:`AnomalyDetector.push` once per acquired
frame.  Internally the detector

1. routes the raw frame through the :class:`AngleMultiplexer` into per-angle
   fixed-view sub-streams (sub-frame interpolation),
2. preprocesses each synthesised fixed-angle frame,
3. scores it with that angle's :class:`AnglePCAModel` (T² + Q + masks),
4. records the results for the viewers / cross-angle voting.

Calibration (``deg_per_frame``, axis, flats/darks) is assumed known here; use
:mod:`sdate.anomaly.calibrate` beforehand if any of it is missing.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from .config import CalibrationConfig, DetectorConfig
from .multiplex import AngleMultiplexer
from .pca import AnglePCAModel, Score
from .preprocess import FramePreprocessor


class AngleResults:
    """Accumulated per-angle timeline of scores (and optionally frames/masks)."""

    def __init__(self, angle: float, frame_shape, record: bool, dtype: str):
        self.angle = angle
        self.frame_shape = frame_shape
        self._record = record
        self._dtype = dtype
        self.scores: List[Score] = []
        self.frames: List[np.ndarray] = []     # preprocessed fixed-angle frames (H,W)
        self.masks_t2: List[np.ndarray] = []
        self.masks_q: List[np.ndarray] = []

    def append(self, score: Score, frame_vec: Optional[np.ndarray]) -> None:
        if self._record and frame_vec is not None:
            self.frames.append(frame_vec.reshape(self.frame_shape).astype(self._dtype))
            if score.mask_t2 is not None:
                self.masks_t2.append(score.mask_t2.astype(self._dtype))
                self.masks_q.append(score.mask_q.astype(self._dtype))
            else:
                z = np.zeros(self.frame_shape, dtype=self._dtype)
                self.masks_t2.append(z)
                self.masks_q.append(z)
        # drop the (large) mask arrays from the stored Score to save memory
        score.mask_t2 = None
        score.mask_q = None
        self.scores.append(score)

    # --------------------------------------------------------------- accessors
    def _col(self, attr) -> np.ndarray:
        return np.array([getattr(s, attr) for s in self.scores], dtype=np.float64)

    @property
    def turns(self) -> np.ndarray:      return self._col("turn")
    @property
    def x_pos(self) -> np.ndarray:      return self._col("x_pos")
    @property
    def t2(self) -> np.ndarray:         return self._col("t2")
    @property
    def q(self) -> np.ndarray:          return self._col("q")
    @property
    def t2_ema(self) -> np.ndarray:     return self._col("t2_ema")
    @property
    def q_ema(self) -> np.ndarray:      return self._col("q_ema")
    @property
    def t2_thr(self) -> np.ndarray:     return self._col("t2_thr")
    @property
    def q_thr(self) -> np.ndarray:      return self._col("q_thr")
    @property
    def t2_flags(self) -> np.ndarray:   return np.array([s.t2_flag for s in self.scores], dtype=bool)
    @property
    def q_flags(self) -> np.ndarray:    return np.array([s.q_flag for s in self.scores], dtype=bool)

    def frame_stack(self) -> np.ndarray:   return np.stack(self.frames) if self.frames else np.empty((0,))
    def mask_t2_stack(self) -> np.ndarray: return np.stack(self.masks_t2) if self.masks_t2 else np.empty((0,))
    def mask_q_stack(self) -> np.ndarray:  return np.stack(self.masks_q) if self.masks_q else np.empty((0,))


class AnomalyDetector:
    """Streaming PCA anomaly detector over several fixed projection angles."""

    def __init__(self, calib: CalibrationConfig, cfg: Optional[DetectorConfig] = None):
        self.calib = calib
        self.cfg = cfg or DetectorConfig()
        self.pre = FramePreprocessor(calib, self.cfg)
        self.mux = AngleMultiplexer(calib, self.cfg)
        self.models: Dict[int, AnglePCAModel] = {
            ai: AnglePCAModel(self.cfg, calib.frame_shape)
            for ai in range(len(self.cfg.target_angles))
        }
        self.results: Dict[int, AngleResults] = {
            ai: AngleResults(a, calib.frame_shape, self.cfg.record_frames, self.cfg.record_dtype)
            for ai, a in enumerate(self.cfg.target_angles)
        }
        self._auto_idx: Optional[int] = None
        self.n_pushed = 0

    def push(self, frame: np.ndarray, idx: Optional[int] = None) -> List[Score]:
        """Feed one raw acquired frame. Returns the scores emitted this step
        (one per tracked angle whose crossing was completed by this frame)."""
        if idx is None:
            self._auto_idx = self.calib.ref_frame if self._auto_idx is None else self._auto_idx + 1
            idx = self._auto_idx
        self.n_pushed += 1

        out: List[Score] = []
        for em in self.mux.push(idx, frame):
            x = self.pre(em.vector)                       # preprocess synthesised frame
            model = self.models[em.angle_index]
            score = model.push(x, em.turn, em.x_pos, want_masks=self.cfg.record_frames)
            self.results[em.angle_index].append(score, x if self.cfg.record_frames else None)
            out.append(score)
        return out

    def run(self, frame_iter, indices=None, progress: int = 0) -> "AnomalyDetector":
        """Convenience: drive the detector from an iterable of frames.

        ``frame_iter`` yields ``(idx, frame)`` pairs, or bare frames if
        ``indices`` is given / auto-indexing is desired.
        """
        for i, item in enumerate(frame_iter):
            if isinstance(item, tuple):
                idx, frame = item
            elif indices is not None:
                idx, frame = indices[i], item
            else:
                idx, frame = None, item
            self.push(frame, idx)
            if progress and (i + 1) % progress == 0:
                print(f"  pushed {i + 1} frames", flush=True)
        return self

    @property
    def angles(self):
        return self.cfg.target_angles
