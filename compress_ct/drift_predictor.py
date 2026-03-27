"""
Drift-based predictor for CT projection compression.

Instead of a neural network, this predictor estimates the inter-frame
translation (drift) from two consecutive frames and extrapolates it
to predict the next frame.

Three estimation modes are provided:

- ``phase_correlation`` – sub-pixel precision via the Fourier shift theorem.
  Fast, robust, and recommended as the default.
- ``optical_flow`` – dense Farnebäck optical flow averaged over the frame.
  Captures more complex motion but is slower.
- ``cross_correlation`` – spatial cross-correlation in a central window.
  Simple and fast but limited to integer-pixel accuracy.

All modes produce a single (dy, dx) drift vector that is applied to the
last input frame with ``scipy.ndimage.shift`` (cubic interpolation) to
generate the prediction.

The public API (``predict_frame``) matches ``BlockPredictor`` so that
``CTCompressor`` and ``CTDecompressor`` can use either predictor
interchangeably.
"""

import numpy as np
from enum import Enum
from typing import List, Optional, Tuple

from scipy.ndimage import shift as ndimage_shift

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class DriftMode(str, Enum):
    """Supported drift-estimation algorithms."""
    PHASE_CORRELATION = "phase_correlation"
    OPTICAL_FLOW = "optical_flow"
    CROSS_CORRELATION = "cross_correlation"


class DriftPredictor:
    """
    Predict the next projection frame by estimating and extrapolating
    the inter-frame drift (translation).

    Parameters
    ----------
    mode : str or DriftMode
        Drift estimation algorithm.  One of ``"phase_correlation"``,
        ``"optical_flow"``, or ``"cross_correlation"``.
    num_input_projections : int
        Number of previous frames the compressor will feed.  Must be
        ≥ 2 (drift needs at least two frames).  The predictor uses
        only the **last two** frames regardless of this value.
    interpolation_order : int
        Spline interpolation order for ``scipy.ndimage.shift``.
        3 = cubic (default), 1 = linear (faster), 0 = nearest.
    search_range : int
        Only used by ``cross_correlation`` mode — half-width of the
        search window in pixels.  Default 64.
    flow_winsize : int
        Only used by ``optical_flow`` mode — Farnebäck window size.
        Default 15.
    upsample_factor : int
        Only used by ``phase_correlation`` — upsampling factor for
        sub-pixel refinement.  Default 10 (≈ 0.1-pixel accuracy).
    verbose : bool
        Print the estimated drift for each call.  Default False.
    """

    # Attributes expected by CTCompressor / CTDecompressor
    use_conditioning: bool = False

    def __init__(
        self,
        mode: str = "phase_correlation",
        num_input_projections: int = 3,
        interpolation_order: int = 3,
        search_range: int = 64,
        flow_winsize: int = 15,
        upsample_factor: int = 10,
        verbose: bool = False,
    ):
        self.mode = DriftMode(mode)
        if num_input_projections < 2:
            raise ValueError("num_input_projections must be >= 2 for drift prediction")
        self.num_input_projections = num_input_projections
        self.interpolation_order = interpolation_order
        self.search_range = search_range
        self.flow_winsize = flow_winsize
        self.upsample_factor = upsample_factor
        self.verbose = verbose

        if self.mode == DriftMode.OPTICAL_FLOW and not _HAS_CV2:
            raise ImportError("cv2 (opencv-python) is required for optical_flow mode")

    # ------------------------------------------------------------------ #
    #  Public API  (same signature as BlockPredictor.predict_frame)
    # ------------------------------------------------------------------ #

    def predict_frame(
        self,
        previous_frames: List[np.ndarray],
        batch_size: int = 16,           # ignored – kept for API compat
        center_x: Optional[float] = None,  # ignored – kept for API compat
    ) -> np.ndarray:
        """
        Predict the next frame from *k* previous frames.

        Only the last two frames (``previous_frames[-2]`` and
        ``previous_frames[-1]``) are used to estimate the drift.

        Parameters
        ----------
        previous_frames : list of np.ndarray
            List of k 2-D arrays (H, W), each in [0, 1].

        Returns
        -------
        predicted : np.ndarray
            2-D array of shape (H, W) with predicted pixel values
            (clipped to [0, 1]).
        """
        k = self.num_input_projections
        if len(previous_frames) < 2:
            raise ValueError(
                f"Need at least 2 previous frames for drift estimation, "
                f"got {len(previous_frames)}"
            )

        x_prev = previous_frames[-2]  # x_{-2}
        x_last = previous_frames[-1]  # x_{-1}

        # Estimate drift from x_{-2} → x_{-1}
        dy, dx = self._estimate_drift(x_prev, x_last)

        if self.verbose:
            print(f"    Drift: dy={dy:+.3f}  dx={dx:+.3f} px  (mode={self.mode.value})")

        # Extrapolate: apply the same drift to x_{-1}
        predicted = ndimage_shift(
            x_last,
            shift=(dy, dx),
            order=self.interpolation_order,
            mode="nearest",
        )

        return np.clip(predicted, 0.0, 1.0)

    # ------------------------------------------------------------------ #
    #  Drift estimation back-ends
    # ------------------------------------------------------------------ #

    def _estimate_drift(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> Tuple[float, float]:
        """Dispatch to the selected estimation algorithm."""
        if self.mode == DriftMode.PHASE_CORRELATION:
            return self._drift_phase_correlation(img1, img2)
        elif self.mode == DriftMode.OPTICAL_FLOW:
            return self._drift_optical_flow(img1, img2)
        elif self.mode == DriftMode.CROSS_CORRELATION:
            return self._drift_cross_correlation(img1, img2)
        else:
            raise ValueError(f"Unknown drift mode: {self.mode}")

    # -- Phase correlation ------------------------------------------------

    def _drift_phase_correlation(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> Tuple[float, float]:
        """
        Sub-pixel drift via Fourier phase correlation.

        Uses ``skimage.registration.phase_cross_correlation`` when
        available; otherwise falls back to a pure-NumPy implementation.
        """
        try:
            from skimage.registration import phase_cross_correlation
            shift_vec, _error, _phasediff = phase_cross_correlation(
                img1, img2,
                upsample_factor=self.upsample_factor,
            )
            # shift_vec is (dy, dx) — the shift to map img1 onto img2
            return float(shift_vec[0]), float(shift_vec[1])
        except ImportError:
            return self._phase_correlation_numpy(img1, img2)

    @staticmethod
    def _phase_correlation_numpy(
        img1: np.ndarray, img2: np.ndarray
    ) -> Tuple[float, float]:
        """Fallback phase-correlation without scikit-image."""
        F1 = np.fft.fft2(img1)
        F2 = np.fft.fft2(img2)

        cross_power = (F1 * np.conj(F2)) / (np.abs(F1 * np.conj(F2)) + 1e-10)
        correlation = np.real(np.fft.ifft2(cross_power))

        peak = np.unravel_index(np.argmax(correlation), correlation.shape)
        H, W = img1.shape
        dy = float(peak[0]) if peak[0] <= H // 2 else float(peak[0]) - H
        dx = float(peak[1]) if peak[1] <= W // 2 else float(peak[1]) - W
        return dy, dx

    # -- Optical flow (Farnebäck) -----------------------------------------

    def _drift_optical_flow(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> Tuple[float, float]:
        """Average dense Farnebäck optical flow."""
        def _to_u8(img: np.ndarray) -> np.ndarray:
            return np.clip(img * 255, 0, 255).astype(np.uint8)

        flow = cv2.calcOpticalFlowFarneback(
            _to_u8(img1), _to_u8(img2),
            flow=None,
            pyr_scale=0.5,
            levels=3,
            winsize=self.flow_winsize,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        # flow[..., 0] = dx, flow[..., 1] = dy  (OpenCV convention)
        dx = float(np.median(flow[..., 0]))
        dy = float(np.median(flow[..., 1]))
        return dy, dx

    # -- Spatial cross-correlation ----------------------------------------

    def _drift_cross_correlation(
        self, img1: np.ndarray, img2: np.ndarray
    ) -> Tuple[float, float]:
        """
        Integer-pixel drift via normalized cross-correlation on a
        central sub-region.
        """
        H, W = img1.shape
        sr = self.search_range

        # Use a central region for speed
        region_size = min(512, H, W)
        cy, cx = H // 2, W // 2
        y1, y2 = cy - region_size // 2, cy + region_size // 2
        x1, x2 = cx - region_size // 2, cx + region_size // 2

        template = img1[y1:y2, x1:x2]

        # Wider search area in img2
        sy1 = max(0, y1 - sr)
        sy2 = min(H, y2 + sr)
        sx1 = max(0, x1 - sr)
        sx2 = min(W, x2 + sr)
        search_area = img2[sy1:sy2, sx1:sx2]

        # Normalized cross-correlation via OpenCV if available
        if _HAS_CV2:
            result = cv2.matchTemplate(
                (search_area * 255).astype(np.float32),
                (template * 255).astype(np.float32),
                cv2.TM_CCOEFF_NORMED,
            )
            _, _, _, max_loc = cv2.minMaxLoc(result)
            # max_loc is (x, y) in search_area coordinates
            match_x = sx1 + max_loc[0]
            match_y = sy1 + max_loc[1]
            dx = float(match_x - x1)
            dy = float(match_y - y1)
        else:
            # Pure NumPy fallback using scipy
            from scipy.signal import fftconvolve

            template_norm = template - template.mean()
            search_norm = search_area - search_area.mean()

            corr = fftconvolve(search_norm, template_norm[::-1, ::-1], mode="valid")
            peak = np.unravel_index(np.argmax(corr), corr.shape)
            dy = float(sy1 + peak[0] - y1)
            dx = float(sx1 + peak[1] - x1)

        return dy, dx

    # ------------------------------------------------------------------ #
    #  Utility
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return (
            f"DriftPredictor(mode={self.mode.value!r}, "
            f"k={self.num_input_projections}, "
            f"interp_order={self.interpolation_order})"
        )
