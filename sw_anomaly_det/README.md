# sw_anomaly_det

Streaming, multi-scale FBP-consistency anomaly detection for time-resolved CT
projection streams.

## Idea

For a sliding window centered at frame `t`:

1. Reconstruct a **single-revolution FBP** (~1 turn) centered on `t` — the
   shared reference.
2. Reconstruct one or more **T-revolution joint FBPs** centered on the same
   `t`, for `T` in a configurable list (default `21, 11, 5` revolutions).
3. Score masked MSE/MAE between the single-revolution reference and each
   T-joint reconstruction.

If the scene is static across the window, the single-turn and T-joint
reconstructions agree; if it moved, they disagree, and the disagreement
scales with how much motion happened. Using a **fixed single-revolution
reference** (e.g. just the first or the most recent frame) is not well
calibrated, because the object's own state can be genuinely different at
different points in the sequence — a T-joint reference adapts to "what the
scene currently looks like on average" instead. Running a few `T` values
side by side gives a rough time-sensitivity readout: small `T` reacts to
short-timescale change, large `T` is smoother/more robust to single-turn
noise but blurs together change that happens within its own window.

A trivial **fixed-reference baseline** is scored alongside for comparison:
the same single-revolution FBP at `t`, but diffed against one single-revolution
FBP anchored at the very first evaluated timestep (not a moving reference).
Both are expected to catch real anomalies; the T-joint approach is expected
to be better calibrated (less confounded by drift-from-datum).

Everything runs on GPU on top of the project's existing ASTRA-based FBP
wrapper (`sdate.tr_diffusion.reconstruct`) — no new reconstruction code, just
a streaming-shaped API around it. Only masked MSE/MAE are computed (SSIM is
too expensive for streaming rates on GPU); both mask to the central sample
disk (`sdate.tr_naf.metrics.make_circular_mask`), matching the project's
existing pseudo-GT masking convention.

## v1 scope

- Native (un-denoised) projections only. Running the same pipeline on
  noisy/dose-thinned projections — to see whether anomalies are still
  detectable amid more noise — is an explicit **v2**, not implemented here.
- Offline replay over a cached `.mov` (`sw_anomaly_det.sources.MovWindowSource`,
  wrapping the project's `MemmapFrameSource`), not live acquisition. The
  `WindowSource` interface is deliberately minimal so an h5-backed or
  live-streaming source can be added later without changing the detector.
- Strictly sequential: one window's reconstruction/scoring on the GPU at a
  time, nothing buffered across windows except scalar results and one small
  cached baseline-reference volume. No reconstructed volumes are stored to
  disk. Revisit only if a real run shows this isn't fast enough.
- Independent from `sdate.anomaly` (the existing per-angle streaming-PCA
  detector) — different technique, decoupled for now.

## Usage

```
python scripts/sw_anomaly_det_run.py --profile wunderkerze2 \
    --frame_start 412000 --frame_end 468000 --T 21 11 5 \
    --out_dir /myhome/data/sdate/shared/time_resolved/sw_anomaly_det
```

Writes an append-only, crash-safe CSV (`sw_anomaly_<tag>.csv`, one row per
window, `flush()`+`fsync()`'d after every write) and a periodically-refreshed
anomaly-vs-time plot (`sw_anomaly_<tag>.png`). Safe to kill and resubmit
(e.g. RunAI preemption) — on restart it validates/drops a possibly-torn
trailing row and resumes right after the last genuinely complete window, so
no window is silently skipped or wastefully recomputed. See
`sw_anomaly_det/logio.py` for the resume mechanics.

## API

```python
from sw_anomaly_det.sources import MovWindowSource
from sw_anomaly_det.detector import DetectorConfig, stream_anomaly_scores

source = MovWindowSource(memmap_path, mov_path, crop, axis_col)
cfg = DetectorConfig(T_list=(21, 11, 5))
for row in stream_anomaly_scores(source, deg_per_frame, frame_start, frame_end, cfg):
    ...  # row: {"t", "mse_T21", "mae_T21", ..., "mse_baseline", "mae_baseline", "window_seconds"}
```
