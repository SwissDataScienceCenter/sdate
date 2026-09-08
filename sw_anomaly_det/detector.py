"""Streaming multi-T sliding-window anomaly detector.

At each timestep ``t`` (stride ``stride``), reconstruct:
  * a single-revolution FBP centered on ``t`` (the shared reference), and
  * a T-revolution "joint" FBP centered on ``t``, for each ``T`` in ``T_list``,
and score masked MSE/MAE between the two -- if the scene is static within the
window, the joint (averaged-over-more-revolutions) reconstruction agrees with
the single-turn one; if it's moving, they disagree, and the disagreement
grows with how much motion happened. Repeating at a few ``T``s gives a rough
time-sensitivity readout: small ``T`` reacts to short-timescale change, large
``T`` is smoother and more robust to single-turn noise but blurs together
change that happens within its own (longer) window.

A trivial baseline is scored alongside: the SAME single-revolution FBP at
``t``, but compared against one FIXED single-revolution FBP anchored at the
very first evaluated timestep (not a moving reference). This is expected to
also catch anomalies, but be less calibrated to genuine scene dynamism --
drift from a datum rather than from "now" -- giving a reference point for
whether the multi-T approach is actually better calibrated, not just
different.

Strictly sequential (v1): exactly one window's reconstruction/scoring lives
on the GPU at a time; nothing carries over between steps except the one
cached baseline-reference volume (small: one det_bin-ed single-revolution
volume) and the running scalar results. No ``torch.cuda.empty_cache()`` is
called per window -- freed tensors return to PyTorch's caching allocator for
immediate reuse by the next window (faster than forcing driver-level
reallocation every step); the ASTRA wrapper itself already releases its own
working memory at the end of each reconstruction call. Revisit only if a
real run shows this isn't fast enough (v1 is deliberately not pipelined).

v1 runs on native (un-denoised) projections. Running this same pipeline on
noisy/dose-thinned projections, to see whether anomalies are still
detectable amid more noise, is an explicit v2 -- not implemented here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch

from .metrics import masked_mae, masked_mse
from .recon import reconstruct_window
from .sources import WindowSource
from .windows import centered_window, revolution_frames, window_centers


@dataclass
class DetectorConfig:
    T_list: Sequence[float] = field(default_factory=lambda: (21, 11, 5))
    stride: Optional[int] = None          # frames; default = 1 revolution
    det_bin: int = 2
    I0_pct: float = 99.5
    mask_radius_frac: float = 0.95
    device: Optional[torch.device] = None


def fieldnames_for(T_list: Sequence[float]) -> List[str]:
    """CSV column order for a given ``T_list`` -- shared by the log writer/reader and the plot."""
    names = ["t", "baseline_ref_t", "mse_baseline", "mae_baseline"]
    for T in T_list:
        key = f"T{T:g}"
        names += [f"mse_{key}", f"mae_{key}"]
    names += ["window_seconds", "read_seconds", "recon_seconds"]
    return names


def stream_anomaly_scores(source: WindowSource, deg_per_frame: float,
                          frame_start: int, frame_end: int,
                          cfg: DetectorConfig = DetectorConfig(),
                          resume_after_t: Optional[int] = None,
                          log_every: int = 20) -> Iterator[Dict]:
    """Yield one dict of scalar scores per evaluated timestep ``t`` (see module docstring).

    ``resume_after_t``: if given, skip straight to the first valid ``t``
    strictly greater than this (the caller's log-resume point) -- windows at
    or before it are never reconstructed, so resuming a killed run costs
    nothing beyond the skip itself.
    """
    device = cfg.device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    period_360 = 360.0 / deg_per_frame
    stride = int(cfg.stride or round(period_360))

    rev_frames = {T: revolution_frames(T, deg_per_frame) for T in cfg.T_list}
    one_rev = revolution_frames(1.0, deg_per_frame)
    all_frame_counts = [one_rev] + list(rev_frames.values())
    max_half_span = max(int(np.ceil(n / 2.0)) for n in all_frame_counts)

    lo = max(int(frame_start), source.first_index)
    hi = min(int(frame_end), source.last_index)
    centers = window_centers(lo, hi, stride, all_frame_counts)
    if resume_after_t is not None:
        centers = [t for t in centers if t > resume_after_t]
    if not centers:
        return

    # I0 estimated once from a sparse sample of the whole eval range and reused
    # for every window's counts->attenuation conversion (matches the existing
    # run_windows() convention in sdate.tr_diffusion.reconstruct).
    I0 = source.estimate_I0(np.arange(lo, hi, max(1, (hi - lo) // 64)), cfg.I0_pct)

    hplane = source.crop[1] // cfg.det_bin
    from sdate.tr_naf.metrics import make_circular_mask
    mask = make_circular_mask(hplane, hplane, radius=cfg.mask_radius_frac * hplane / 2.0, device=device)

    baseline_ref = None
    baseline_ref_t = None
    n = len(centers)
    for i, t in enumerate(centers):
        tw0 = time.time()
        idx1 = centered_window(t, one_rev)
        vol1, read_s, recon_s = reconstruct_window(source, idx1, deg_per_frame, I0, cfg.det_bin, device)

        if baseline_ref is None:
            baseline_ref = vol1.clone()
            baseline_ref_t = t

        row = {
            "t": t, "baseline_ref_t": baseline_ref_t,
            "mse_baseline": masked_mse(vol1, baseline_ref, mask),
            "mae_baseline": masked_mae(vol1, baseline_ref, mask),
        }

        for T in cfg.T_list:
            idxT = centered_window(t, rev_frames[T])
            volT, read_sT, recon_sT = reconstruct_window(source, idxT, deg_per_frame, I0, cfg.det_bin, device)
            read_s += read_sT
            recon_s += recon_sT
            key = f"T{T:g}"
            row[f"mse_{key}"] = masked_mse(vol1, volT, mask)
            row[f"mae_{key}"] = masked_mae(vol1, volT, mask)
            del volT

        del vol1
        # Safe once every arm for this `t` has been read: no future (larger) `t`
        # will ever need a frame before its own `t' - max_half_span`, and
        # `t' >= t` for every remaining center -- see SequentialFfmpegWindowSource.
        source.trim_before(t - max_half_span)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        row["window_seconds"] = time.time() - tw0
        row["read_seconds"] = read_s
        row["recon_seconds"] = recon_s
        if log_every and i % log_every == 0:
            print(f"  [sw_anomaly_det] {i + 1}/{n}  t={t}  {row['window_seconds']:.2f}s/window "
                 f"(read={read_s:.2f}s recon={recon_s:.2f}s)", flush=True)
        yield row
