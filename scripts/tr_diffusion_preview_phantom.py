#!/usr/bin/env python3
"""Quick look at a phantom.default_scene design: GT|noisy recon movies at a few
z-slices, over a sliding-window time series. No training, no dataset write --
renders projections in memory straight from the analytic phantom and FBP-
reconstructs each window, so a scene design can be reviewed before spending
time generating the full dataset / retraining.

  python scripts/tr_diffusion_preview_phantom.py --tag mydesign --frame_end 24000
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate"); sys.path.insert(0, "/myhome/astra-torch")
from sdate.tr_diffusion import reconstruct as R
from sdate.tr_diffusion.noise import add_poisson_noise
from sdate.tr_diffusion.phantom import attenuation_to_counts, default_scene, render_projections


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--height", type=int, default=128)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--deg_per_frame", type=float, default=2.0)
    p.add_argument("--frame_end", type=int, default=24000, help="span of the phantom to preview")
    p.add_argument("--stride", type=int, default=1600, help="frames between window starts")
    p.add_argument("--window_deg", type=float, default=180.0)
    p.add_argument("--dose", type=float, default=0.025)
    p.add_argument("--I0", type=float, default=700.0)
    p.add_argument("--rows", type=float, nargs="+", default=[0.25, 0.5, 0.75],
                   help="z-slice rows to preview, as fractions of height")
    p.add_argument("--det_bin", type=int, default=1)
    p.add_argument("--out_dir", default="/myhome/data/sdate/shared/time_resolved/phantom_preview")
    p.add_argument("--tag", default="preview")
    p.add_argument("--vmax_pctile", type=float, default=99.0)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    a = parse_args()
    dev = torch.device(a.device)
    OUT = Path(a.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    caps = default_scene(frame_start=0, frame_end=a.frame_end, height=a.height, width=a.width, seed=a.seed)
    print(f"{len(caps)} capsules, preview span [0,{a.frame_end})", flush=True)

    windows = R.sliding_windows(0, a.frame_end, a.stride, window_deg=a.window_deg, deg_per_frame=a.deg_per_frame)
    print(f"{len(windows)} windows", flush=True)

    row_idx = [int(round(f * a.height)) for f in a.rows]
    movies = {r: [] for r in row_idx}
    starts = []
    for wi, (s, idx) in enumerate(windows):
        p_clean = render_projections(caps, idx.astype(np.float64), a.height, a.width,
                                     a.deg_per_frame, device=dev)
        counts = attenuation_to_counts(p_clean, a.I0)
        g = torch.Generator(device=dev).manual_seed(12345 + int(s))
        counts_noisy = add_poisson_noise(counts, a.dose, generator=g)
        p_noisy = R.counts_to_attenuation(counts_noisy, a.I0)

        ang = R.projection_angles(idx, deg_per_frame=a.deg_per_frame)
        vol_clean = R.reconstruct(p_clean, ang, det_bin=a.det_bin, method="fbp", device=dev)
        vol_noisy = R.reconstruct(p_noisy, ang, det_bin=a.det_bin, method="fbp", device=dev)
        for r in row_idx:
            row = min(r // a.det_bin, vol_clean.shape[0] - 1)
            combo = torch.cat([vol_clean[row], vol_noisy[row]], dim=-1)
            movies[r].append(combo.detach().cpu())
        starts.append(s)
        print(f"  window {wi + 1}/{len(windows)} start {s}", flush=True)

    for r in row_idx:
        stack = np.stack([f.numpy() for f in movies[r]])
        vmin, vmax = np.percentile(stack, [100 - a.vmax_pctile, a.vmax_pctile])
        out_path = OUT / f"phantom_{a.tag}_row{r}.mov"
        R.write_slice_movie(movies[r], out_path, float(vmin), float(vmax))
        print(f"wrote {out_path}  (clean | noisy, {len(movies[r])} frames)", flush=True)

    print(f"minutes {round((time.time() - t0) / 60, 1)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
