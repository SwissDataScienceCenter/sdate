#!/usr/bin/env python3
"""Cache T same-angle joint-FBP reprojection context taps for a new
joint-FBP-conditioned baseline denoiser.

The joint-FBP baseline (`tr_diffusion_jointfbp_k_sweep.py`) reconstructs a
volume from T consecutive revolutions of projections, kept at their own
exact wrapped angle (no averaging) -- very sharp wherever the sample is
static across the window. That reconstruction lives in VOLUME space, but the
denoiser's context channels live in PROJECTION space, so this script forward-
projects ("reprojects") each such volume back out through ASTRA.

Design (confirmed with the user): for a given real frame `i` at its own exact
angle theta_i, build T DIFFERENT nearby T-revolution-wide joint-FBP volumes
(windows sliding by exactly 1 revolution each step) and reproject EVERY one
of them at theta_i -- i.e. `i` gets T context values, each from a different
surrounding-data window, but ALL evaluated at `i`'s own exact angle (no
angular mismatch at all between context and target, unlike the raw rotation/
temporal taps the current baseline uses -- the point of this whole change is
that removing that mismatch is what should let sharpness survive).

Since consecutive window starts are exactly 1 revolution (`stride`) apart and
each window spans `T` revolutions (`win_T = T*stride` frames -- an EXACT
multiple of `stride`, not `round(T*period_360)`, else the slot-scatter below
periodically drops to T-1 filled slots instead of T; confirmed via an offline
coverage-counting check), a single sliding sweep (stride=1 revolution)
already gives every interior frame exactly T covering windows for free -- no
need to run T separate sweeps.
Windows are reconstructed ONCE and their ENTIRE angle set is reprojected in
one batched forward-projection call (matches "project it down into all the
projections used to reconstruct it"); each frame's contribution from a given
window is scattered into the channel ("tap") slot corresponding to that
window's position relative to the frame (slot 0 = the LATEST-starting window
covering this frame, i.e. the window this frame is nearest the START of;
slot T-1 = the EARLIEST-starting one, i.e. nearest this frame's END).

Output: T separate float16 `(n_out, H, W)` memmaps (`{tag}_tap{c}.f16`, one
per slot) + a shared `.meta.npz` (`first_index`, `num_frames`, `crop`) -- the
exact multi-memmap convention `TimeResolvedFrameDataset`'s generalised
`aux_channel_memmap` (a list of paths) expects, so these plug in directly:

  python scripts/tr_diffusion_jointfbp_context_cache.py --T 11 \
      --frame_start 410000 --frame_end 430000
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")
os.environ["PATH"] = f"/myhome/bin:{os.environ.get('PATH', '')}"

from sdate.tr_diffusion import reconstruct as R
from sdate.tr_diffusion.profiles import DatasetProfile
from sdate.tr_diffusion.frames import MemmapFrameSource


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="wunderkerze2")
    p.add_argument("--T", type=int, default=11,
                   help="number of same-angle reprojection context taps = number of revolutions "
                        "spanned by each underlying joint-FBP reconstruction")
    p.add_argument("--det_bin", type=int, default=1,
                   help="1 = full native detector resolution -- required for pixel-exact alignment "
                        "with the raw frame stream used as context elsewhere in this project")
    p.add_argument("--dose", type=float, default=0.05)
    p.add_argument("--native_noise", action="store_true",
                   help="skip synthetic dose-thinning entirely and reconstruct from the raw measured "
                        "counts as-is -- for the native-noise regime, NOT the same as --dose 1.0 "
                        "(which still applies an extra Poisson resample on top of the real counts)")
    p.add_argument("--noise_seed", type=int, default=12345)
    p.add_argument("--stride", type=int, default=None,
                   help="frames between consecutive window starts = chunk width; default = 1 "
                        "revolution (round(period_360)). Set smaller (e.g. 20) for finer time "
                        "granularity -- win_T = T*stride shrinks correspondingly.")
    p.add_argument("--frame_start", type=int, default=410_000)
    p.add_argument("--frame_end", type=int, default=430_000)
    p.add_argument("--out_dir", default="/myhome/data/sdate/shared/time_resolved/jointfbp_context")
    p.add_argument("--tag", default=None)
    p.add_argument("--log_every", type=int, default=10)
    return p.parse_args()


def main():
    a = parse_args()
    prof = DatasetProfile.load(a.profile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{prof.name}_jointfbpctx_T{a.T}"
    t0 = time.time()

    from astra_torch.lamino import build_lamino_projector

    dark_mov = Path(prof.mov_path).with_name(f"{prof.name}_darks.mov")
    flat_mov = Path(prof.mov_path).with_name(f"{prof.name}_flats.mov")
    dark_native = torch.from_numpy(R.load_calibration_average(
        str(dark_mov), prof.crop, prof.rot_axis_col, height=prof.height, width=prof.width,
    )).to(device=device, dtype=torch.float32)
    flat_native = torch.from_numpy(R.load_calibration_average(
        str(flat_mov), prof.crop, prof.rot_axis_col, height=prof.height, width=prof.width,
    )).to(device=device, dtype=torch.float32)
    dark_b = R.bin_detector(dark_native.unsqueeze(0), a.det_bin)
    flat_b = R.bin_detector(flat_native.unsqueeze(0), a.det_bin)
    print(f"flat/dark loaded: dark mean={dark_native.mean():.4g} flat mean={flat_native.mean():.4g}", flush=True)

    src = MemmapFrameSource(prof.memmap_path, prof.mov_path)
    stride = a.stride if a.stride is not None else int(round(prof.period_360))
    # win_T MUST be an exact multiple of stride -- otherwise the slot-scatter below
    # (slot = j // stride) periodically drops to T-1 covering windows instead of T at
    # positions where win_T/stride's fractional remainder crosses a slot boundary
    # (confirmed via an offline coverage-counting check: round(T*period_360) alone
    # left ~1% of frames with only T-1 filled slots). The tiny resulting mismatch vs
    # T exact revolutions (a few frames, well under 1 revolution) is harmless.
    win_T = a.T * stride
    det_shape = (prof.crop[0] // a.det_bin, prof.crop[1] // a.det_bin)
    vol_shape = (det_shape[0], det_shape[1], det_shape[1])

    window_starts = np.arange(a.frame_start, a.frame_end - win_T + 1, stride)
    assert len(window_starts) >= a.T, (
        f"range [{a.frame_start},{a.frame_end}) too small for T={a.T} -- need at least T={a.T} "
        f"overlapping windows, got {len(window_starts)}"
    )

    # --- coverage pass: which output frames get contributions from all T windows ---
    lo_c, hi_c = int(window_starts[0]), int(window_starts[-1] + win_T)
    coverage = np.zeros(hi_c - lo_c, dtype=np.int32)
    for s in window_starts:
        coverage[int(s) - lo_c: int(s) - lo_c + win_T] += 1
    full = np.flatnonzero(coverage >= a.T)
    out_lo, out_hi = lo_c + int(full[0]), lo_c + int(full[-1]) + 1
    n_out = out_hi - out_lo
    print(f"T={a.T}  stride={stride}f  win_T={win_T}f  windows={len(window_starts)}  "
          f"full-coverage output range [{out_lo},{out_hi})  ({n_out} frames)", flush=True)

    paths = [out_dir / f"{tag}_tap{c}.f16" for c in range(a.T)]
    mms = [np.memmap(p, dtype=np.float16, mode="w+", shape=(n_out, *prof.crop)) for p in paths]

    for wi, s in enumerate(window_starts):
        s = int(s)
        idx = np.arange(s, s + win_T)
        angles = R.projection_angles(idx, deg_per_frame=prof.deg_per_frame)

        native = R.native_window_gpu(src, idx, prof.crop, prof.rot_axis_col, device)
        if a.native_noise:
            noisy_counts = native
        else:
            gen = torch.Generator(device=device).manual_seed(a.noise_seed + s)
            noisy_counts = R.noisy_window_gpu(native, a.dose, generator=gen)
        noisy_atten = R.counts_to_attenuation_flatdark(noisy_counts, dark_native, flat_native)
        joint_vol = R.reconstruct(noisy_atten, angles, det_bin=a.det_bin, method="fbp", device=device)

        proj_layer = build_lamino_projector(
            vol_shape=vol_shape, det_shape=det_shape, angles_deg=angles,
            lamino_angle_deg=0.0, voxel_size_mm=1.0, det_spacing_mm=1.0, device=device,
        )
        with torch.no_grad():
            reproj_atten = proj_layer(joint_vol.unsqueeze(0).unsqueeze(0))[0]  # (win_T, r, c)
            reproj_counts = R.attenuation_to_counts_flatdark(reproj_atten, dark_b, flat_b)
            if a.det_bin != 1:
                reproj_counts = F.interpolate(reproj_counts.unsqueeze(1), size=prof.crop,
                                              mode="bilinear", align_corners=False).squeeze(1)

        lo_j = max(0, out_lo - s)
        hi_j = min(win_T, out_hi - s)
        if lo_j < hi_j:
            block = reproj_counts[lo_j:hi_j].cpu().numpy().astype(np.float16)
            frame_idx = s + np.arange(lo_j, hi_j)
            slots = np.minimum(np.arange(lo_j, hi_j) // stride, a.T - 1)
            for slot in np.unique(slots):
                sel = slots == slot
                mms[slot][frame_idx[sel] - out_lo] = block[sel]

        if a.log_every and wi % a.log_every == 0:
            print(f"  window {wi + 1}/{len(window_starts)}  start {s}", flush=True)

    for mm in mms:
        mm.flush()
    meta = dict(first_index=out_lo, num_frames=n_out, crop=list(prof.crop),
               dose=(1.0 if a.native_noise else a.dose), native_noise=a.native_noise,
               noise_seed=a.noise_seed, T=a.T, stride=stride, win_T=win_T, mode="jointfbp_context")
    for p in paths:
        np.savez(str(p) + ".meta.npz", **meta)
    print(f"wrote {a.T} tap memmaps to {out_dir} ({tag}_tap0..{a.T - 1}.f16), "
          f"{n_out} frames [{out_lo},{out_hi})", flush=True)
    print(f"minutes={round((time.time() - t0) / 60, 1)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
