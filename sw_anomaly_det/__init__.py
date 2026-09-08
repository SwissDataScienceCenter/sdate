"""sw_anomaly_det -- streaming multi-scale FBP-consistency anomaly detection.

For a sliding window centered at frame ``t``, reconstruct a single-revolution
FBP and one or more T-revolution "joint" FBPs (``T`` in revolutions, e.g.
21/11/5), and score the error between them: low error = static scene, high
error = the scene changed within that window. Repeating at several ``T``
gives a rough time-sensitivity readout. A trivial fixed-reference baseline
(``t`` vs the FIRST evaluated window, rather than a moving T-joint window) is
scored alongside for comparison, on the expectation that both should catch
real anomalies but the T-joint approach should be better calibrated to
genuine scene dynamism.

v1 runs on native (un-denoised) projections, offline over a cached ``.mov``
(``sw_anomaly_det.sources.MovWindowSource``). Running this same pipeline on
noisy/dose-thinned projections -- to see whether anomalies remain detectable
amid more noise -- is an explicit v2, not implemented here.

Independent from ``sdate.anomaly`` (the existing per-angle streaming-PCA
anomaly detector): different technique, decoupled for now; the two signals
could be fused later once both are proven out.

See ``sw_anomaly_det.detector.stream_anomaly_scores`` for the main API, and
``scripts/sw_anomaly_det_run.py`` for a CLI driver.
"""
from .detector import DetectorConfig, fieldnames_for, stream_anomaly_scores  # noqa: F401
from .sources import FfmpegWindowSource, MovWindowSource, SequentialFfmpegWindowSource, WindowSource  # noqa: F401
