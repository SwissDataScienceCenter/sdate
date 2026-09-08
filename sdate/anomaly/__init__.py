"""Streaming PCA anomaly detection for time-resolved projection acquisitions.

Builds a per-angle PCA reference from a sliding window of fixed-view frames and
scores every new frame on the fly with two complementary multivariate-SPC
statistics — **Hotelling's T²** (extreme along known modes) and **Q / SPE**
(novel structure outside the modelled subspace) — each with a per-pixel
**contribution mask** that localises *where* on the detector the anomaly sits.

The detector is *source-agnostic*: an external orchestrator decodes frames (from
``.mov`` / HDF5 / a live detector) and calls :meth:`AnomalyDetector.push` once
per acquired frame.  Acquisition geometry (``deg_per_frame``, rotation axis,
flats/darks) is supplied via :class:`CalibrationConfig`; :mod:`.calibrate`
estimates whatever is missing from a warm-up slice of the stream.

See the ``project-wunderkerze2-rotation`` memory / calibration notebook for the
rotation geometry this builds on.
"""

from .config import CalibrationConfig, DetectorConfig
from .preprocess import FramePreprocessor
from .pca import AnglePCAModel, Score
from .multiplex import AngleMultiplexer, AngleEmission
from .detector import AnomalyDetector, AngleResults
from .calibrate import calibrate_from_frames, estimate_period, estimate_axis
from .vote import aggregate_votes, VoteResult, mask_centroid

__all__ = [
    "CalibrationConfig", "DetectorConfig",
    "FramePreprocessor",
    "AnglePCAModel", "Score",
    "AngleMultiplexer", "AngleEmission",
    "AnomalyDetector", "AngleResults",
    "calibrate_from_frames", "estimate_period", "estimate_axis",
    "aggregate_votes", "VoteResult", "mask_centroid",
    # viewers imported lazily (matplotlib / ipywidgets / ffmpeg) via
    #   from sdate.anomaly.viz import scrubber, plot_all_angles
    #   from sdate.anomaly.movies import export_overlay_movie
]
