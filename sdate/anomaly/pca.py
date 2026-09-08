"""Per-angle streaming PCA anomaly model.

Maintains a sliding window of recent *fixed-angle* frames (one per turn) as a
reference subspace and scores each new frame with the two classical
multivariate-SPC statistics:

* **Hotelling's T²** — ``Σ tᵢ²/λᵢ`` — how extreme the frame is *along directions
  the model already knows about* (an unusually large but expected-type
  fluctuation).
* **Q / SPE** — ``‖e‖² = ‖x−μ‖² − ‖t‖²`` — how much of the frame lies *outside*
  the modelled subspace (genuinely novel structure).

Both are cheap: one projection ``t = Pᵀ(x−μ)`` per frame.  Each also yields a
per-pixel **contribution map** that localises *where* on the detector the
anomaly sits (the two masks the deliverable asks for):

* Q-mask   ``c_Q[j]  = e[j]²``                         (Σ = Q)
* T²-mask  ``c_T²[j] = (x−μ)[j] · (P Λ⁻¹ Pᵀ (x−μ))[j]``  (Σ = T²)

The reference is maintained streaming-style: a ring buffer of the last ``W``
accepted turns, re-SVD'd on update.  Flagged frames are withheld from the
reference (gating) so anomalies do not poison the baseline, with a
force-rebaseline fallback after a long run of rejections (regime change).
Thresholds are robust and adaptive: ``median + k·(1.4826·MAD)`` over a rolling
window of accepted scores (a streaming control chart).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config import DetectorConfig

_EPS = 1e-12


@dataclass
class Score:
    """Result of scoring one fixed-angle frame."""

    turn: int                      # 0-based movie-frame (turn) index for this angle
    x_pos: float                   # continuous source-frame position it was sampled at
    t2: float                      # Hotelling's T²
    q: float                       # Q / SPE
    t2_ema: float                  # EWMA-smoothed T²
    q_ema: float                   # EWMA-smoothed Q
    t2_flag: bool                  # exceeded the T² control limit
    q_flag: bool                   # exceeded the Q control limit
    t2_thr: float                  # current T² limit
    q_thr: float                   # current Q limit
    accepted: bool                 # folded into the reference (not gated out)
    mask_t2: Optional[np.ndarray] = None  # (H, W) T² contribution map
    mask_q: Optional[np.ndarray] = None   # (H, W) Q contribution map
    scored: bool = True            # False during warm-up (no model yet)


class AnglePCAModel:
    """Streaming PCA reference + scorer for a single fixed projection angle."""

    def __init__(self, cfg: DetectorConfig, frame_shape: Tuple[int, int]):
        self.cfg = cfg
        self.frame_shape = frame_shape

        self._buf: deque = deque(maxlen=cfg.window)  # raw feature vectors (D,)
        self.mu: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None          # (D, R) orthonormal columns
        self.lam: Optional[np.ndarray] = None        # (R,) eigenvalues (variances)

        self._since_fit = 0
        self._reject_run = 0
        self.n_seen = 0

        self._t2_hist: deque = deque(maxlen=cfg.thr_history)
        self._q_hist: deque = deque(maxlen=cfg.thr_history)
        self.t2_thr = np.inf
        self.q_thr = np.inf
        self._t2_ema = np.nan
        self._q_ema = np.nan

    # ------------------------------------------------------------------ fitting
    @property
    def ready(self) -> bool:
        """True once a subspace has been fitted and enough turns observed."""
        return self.P is not None and self.n_seen >= self.cfg.min_turns

    def _refit(self) -> None:
        if len(self._buf) < 2:
            return
        X = np.stack(self._buf)                  # (n, D)
        self.mu = X.mean(axis=0)
        Xc = X - self.mu
        # economy SVD; rows = turns, so right singular vectors are pixel-space modes
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        r = min(self.cfg.n_modes, Vt.shape[0])
        self.P = Vt[:r].T                        # (D, r)
        self.lam = (S[:r] ** 2) / max(len(self._buf) - 1, 1)
        self.lam = np.maximum(self.lam, _EPS)
        self._since_fit = 0

    # ------------------------------------------------------------- thresholding
    def _update_thresholds(self) -> None:
        c = self.cfg
        if len(self._t2_hist) >= c.thr_min_history:
            self.t2_thr = _robust_limit(self._t2_hist, c.gate_k)
            self.q_thr = _robust_limit(self._q_hist, c.gate_k)

    # -------------------------------------------------------------------- score
    def _score_vector(self, x: np.ndarray):
        xc = x - self.mu
        t = self.P.T @ xc                        # (R,)
        proj_energy = float(t @ t)
        total_energy = float(xc @ xc)
        q = max(total_energy - proj_energy, 0.0)
        t2 = float(np.sum(t * t / self.lam))
        return xc, t, t2, q

    def _masks(self, xc: np.ndarray, t: np.ndarray):
        # Q contribution: residual energy per pixel
        resid = xc - self.P @ t
        mask_q = (resid * resid).reshape(self.frame_shape)
        # T² contribution: xc_j * (P Λ⁻¹ Pᵀ xc)_j   (diagonal/complete decomposition; Σ = T²)
        u = t / self.lam
        m_xc = self.P @ u
        mask_t2 = (xc * m_xc).reshape(self.frame_shape)
        return mask_t2, mask_q

    def push(self, x: np.ndarray, turn: int, x_pos: float, want_masks: bool = True) -> Score:
        """Score one fixed-angle frame ``x`` (already preprocessed, shape ``(D,)``)."""
        c = self.cfg
        scored = self.ready

        if scored:
            xc, t, t2, q = self._score_vector(x)
            # EWMA smoothing of the score curves
            self._t2_ema = t2 if np.isnan(self._t2_ema) else (1 - c.ema) * self._t2_ema + c.ema * t2
            self._q_ema = q if np.isnan(self._q_ema) else (1 - c.ema) * self._q_ema + c.ema * q
            t2_flag = t2 > self.t2_thr
            q_flag = q > self.q_thr
            mask_t2, mask_q = self._masks(xc, t) if want_masks else (None, None)
        else:
            t2 = q = np.nan
            t2_flag = q_flag = False
            mask_t2 = mask_q = None

        # ---- gating: decide whether this frame joins the reference ----
        flagged = t2_flag or q_flag
        if c.gate and scored and flagged and self._reject_run < c.max_consecutive_reject:
            accepted = False
            self._reject_run += 1
        else:
            accepted = True
            self._reject_run = 0

        if accepted:
            self._buf.append(np.asarray(x, dtype=np.float64))
            self.n_seen += 1
            self._since_fit += 1
            if len(self._buf) >= 2 and (self.P is None or self._since_fit >= c.refit_every):
                self._refit()
            if scored:
                self._t2_hist.append(t2)
                self._q_hist.append(q)
                self._update_thresholds()
            elif self.P is not None and self.n_seen >= c.min_turns:
                # transitioning into "ready": seed thresholds on next scored frames
                pass

        return Score(
            turn=turn, x_pos=float(x_pos),
            t2=float(t2), q=float(q),
            t2_ema=float(self._t2_ema), q_ema=float(self._q_ema),
            t2_flag=bool(t2_flag), q_flag=bool(q_flag),
            t2_thr=float(self.t2_thr), q_thr=float(self.q_thr),
            accepted=bool(accepted), mask_t2=mask_t2, mask_q=mask_q, scored=scored,
        )


def _robust_limit(hist, k: float) -> float:
    a = np.fromiter(hist, dtype=np.float64)
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    return float(med + k * 1.4826 * mad)
