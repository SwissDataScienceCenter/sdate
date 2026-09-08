#!/usr/bin/env python
"""Driver: stream the sw_anomaly_det detector over a frame range, writing a
crash-safe incremental CSV log + a periodically-refreshed anomaly-vs-time
plot. Safe to kill and resubmit (e.g. RunAI preemption) -- it resumes right
after the last logged window.

Example (matches the T5-native/T11 eval range already analyzed elsewhere):
    python scripts/sw_anomaly_det_run.py --profile wunderkerze2 \\
        --frame_start 412000 --frame_end 468000 --T 21 11 5 \\
        --out_dir /myhome/data/sdate/shared/time_resolved/sw_anomaly_det
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

# `python scripts/sw_anomaly_det_run.py` sets sys.path[0] to scripts/, not the
# repo root -- unlike `python -m sdate.foo` (which resolves via cwd), a plain
# script invocation needs the repo root added explicitly for `sdate`/
# `sw_anomaly_det` to import regardless of whether the editable pip install
# actually took effect in this container.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, "/myhome/astra-torch")  # astra_torch is not pip-installed, see reconstruct.py

from sdate.tr_diffusion.profiles import DatasetProfile
from sw_anomaly_det.detector import DetectorConfig, fieldnames_for, stream_anomaly_scores
from sw_anomaly_det.logio import AnomalyLog
from sw_anomaly_det.plotting import plot_anomaly_vs_time
from sw_anomaly_det.sources import FfmpegWindowSource, MovWindowSource, SequentialFfmpegWindowSource
from sw_anomaly_det.windows import revolution_frames


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="wunderkerze2")
    ap.add_argument("--source", choices=["memmap", "mov", "mov_sequential"], default="memmap",
                    help="'memmap': profile's pre-extracted memmap (fast, needs disk). "
                         "'mov': read straight from the raw .mov, one independent ffmpeg decode per "
                         "window (CPU-decode bound, and redundant across overlapping windows). "
                         "'mov_sequential': also reads straight from the raw .mov, no extraction/copy, "
                         "but via one persistent ffmpeg process + rolling buffer -- each frame is "
                         "decoded once for the whole run instead of once per overlapping window "
                         "(recommended over 'mov' for any real sliding-window range).")
    ap.add_argument("--memmap_path", default=None, help="override the profile's memmap (e.g. a chunk extraction)")
    ap.add_argument("--frame_start", type=int, default=None)
    ap.add_argument("--frame_end", type=int, default=None)
    ap.add_argument("--T", type=float, nargs="+", default=[21, 11, 5])
    ap.add_argument("--stride", type=int, default=None, help="frames; default = 1 revolution")
    ap.add_argument("--det_bin", type=int, default=2)
    ap.add_argument("--out_dir", default="/myhome/data/sdate/shared/time_resolved/sw_anomaly_det")
    ap.add_argument("--tag", default=None, help="output basename; default derived from range+T")
    ap.add_argument("--plot_every", type=int, default=200, help="refresh the plot every N new windows (0=only at end)")
    ap.add_argument("--log_every", type=int, default=20, help="stdout progress print interval")
    ap.add_argument("--max_windows", type=int, default=None, help="stop after N new windows (for quick perf smoke tests)")
    args = ap.parse_args()

    prof = DatasetProfile.load(args.profile)
    frame_start = args.frame_start if args.frame_start is not None else prof.frame_start
    frame_end = args.frame_end if args.frame_end is not None else prof.frame_end
    T_list = tuple(args.T)
    tag = args.tag or f"{prof.name}_{frame_start}_{frame_end}_T{'-'.join(str(int(t)) for t in T_list)}"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"sw_anomaly_{tag}.csv"
    plot_path = out_dir / f"sw_anomaly_{tag}.png"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    print(f"[sw_anomaly_det] profile={args.profile} source={args.source} range=[{frame_start},{frame_end}) "
         f"T={T_list} det_bin={args.det_bin} device={device}", flush=True)

    cfg = DetectorConfig(T_list=T_list, stride=args.stride, det_bin=args.det_bin, device=device)

    # Resolved before constructing the source: a `mov_sequential` source needs
    # to know the resume point to seek to (skip straight past already-logged
    # windows instead of wastefully decoding through them again).
    fields = fieldnames_for(T_list)
    log = AnomalyLog(log_path, fields)
    resume_after = log.last_t()
    if resume_after is not None:
        print(f"[sw_anomaly_det] resuming after t={resume_after} (log: {log_path})", flush=True)

    if args.source == "mov_sequential":
        rev_counts = [revolution_frames(1.0, prof.deg_per_frame)] + \
            [revolution_frames(T, prof.deg_per_frame) for T in T_list]
        max_half_span = max(-(-n // 2) for n in rev_counts)  # ceil division
        seq_start = frame_start if resume_after is None else max(frame_start, resume_after - max_half_span + 1)
        source = SequentialFfmpegWindowSource(prof.mov_path, prof.crop, prof.rot_axis_col,
                                              height=prof.height, width=prof.width,
                                              frame_start=seq_start, fps=prof.fps)
    elif args.source == "mov":
        source = FfmpegWindowSource(prof.mov_path, prof.crop, prof.rot_axis_col,
                                    height=prof.height, width=prof.width, fps=prof.fps)
    else:
        memmap_path = args.memmap_path or prof.memmap_path
        source = MovWindowSource(memmap_path, prof.mov_path, prof.crop, prof.rot_axis_col)

    t0 = time.time()
    n_done = 0
    read_sum = recon_sum = 0.0
    for row in stream_anomaly_scores(source, prof.deg_per_frame, frame_start, frame_end, cfg,
                                     resume_after_t=resume_after, log_every=args.log_every):
        log.append(row)
        n_done += 1
        read_sum += row["read_seconds"]
        recon_sum += row["recon_seconds"]
        if args.plot_every and n_done % args.plot_every == 0:
            plot_anomaly_vs_time(log_path, plot_path, T_list=T_list)
        if args.max_windows and n_done >= args.max_windows:
            print(f"[sw_anomaly_det] stopping after --max_windows={args.max_windows}", flush=True)
            break
    log.close()
    if hasattr(source, "close"):
        source.close()
    plot_anomaly_vs_time(log_path, plot_path, T_list=T_list)
    elapsed = time.time() - t0
    per_window = elapsed / n_done if n_done else float("nan")
    print(f"[sw_anomaly_det] done: {n_done} new windows in {elapsed:.1f}s "
         f"({per_window:.2f}s/window; avg read={read_sum / n_done if n_done else float('nan'):.2f}s "
         f"recon={recon_sum / n_done if n_done else float('nan'):.2f}s) -> {log_path}", flush=True)
    if device.type == "cuda":
        print(f"[sw_anomaly_det] peak GPU memory: "
             f"{torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB allocated, "
             f"{torch.cuda.max_memory_reserved(device) / 2**30:.2f} GiB reserved", flush=True)


if __name__ == "__main__":
    main()
