#!/usr/bin/env python3
"""Leak-free version of `tr_diffusion_jointfbp_context_cache.py`: for each
target CHUNK (a stride=20 frame block on the aligned grid), every one of its
T context volumes now EXCLUDES that chunk's own real projections from its
FBP input (a genuine limited-angle reconstruction, ~180/200 views), instead
of the original design where all T windows fully contained the target chunk.

Why this exists (see conversation, 2026-08-29): the original design built
each of the T windows as a plain contiguous win_T-frame FBP. Since windows
are staggered by exactly `stride` and span `T*stride` frames, EVERY one of
the T windows for a target chunk C fully contains C's own real 20 frames --
not just some of them. Because FBP is approximately data-consistent at its
own input angles, reprojecting those volumes (even at OTHER real frames'
angles within the same window) leaks the target's own real measurements back
out through the context, undermining both the context_only training signal
and, more importantly, the super-time-resolution inference goal (most of the
"180 missing" angles are themselves real frames covered by several of the T
windows as real input data).

Design change: for target chunk C, each of the T windows s=C-k*stride
(k=0..T-1) is reconstructed from its normal 200-frame real span MINUS
chunk C's own [C, C+stride) frames (removed, not zeroed -- a genuine
angular gap, not just fewer counts at those angles). This CANNOT be shared
across neighbouring target chunks the way the original cache was (a given
window's volume is exclusion-specific to whichever chunk it's serving), so
this iterates per TARGET CHUNK (on the aligned stride=20 grid) and redoes
all T reconstructions each time -- ~10x the per-output-frame cost of the
original script, but still just a per-chunk cost comparable to one iteration
of tr_diffusion_super_tr_movie.py's inference loop.

Output: same convention as the original script -- T float16 (n_out,H,W)
memmaps (`{tag}_tap{c}.f16`) + shared `.meta.npz` (first_index, num_frames,
crop), one tap value per REAL frame within each target chunk (the training
targets), reprojected at that real frame's own angle.

  python scripts/tr_diffusion_jointfbp_context_cache_excl.py --T 10 --stride 20 \\
      --frame_start 415180 --frame_end 445820 --tag 212_Wunderkerze2_jointfbpctx_T10_str20_native_excl
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")
os.environ["PATH"] = f"/myhome/bin:{os.environ.get('PATH', '')}"

from sdate.tr_diffusion import reconstruct as R
from sdate.tr_diffusion.profiles import DatasetProfile
from sdate.tr_diffusion.frames import MemmapFrameSource
from astra_torch.lamino import build_lamino_projector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="wunderkerze2")
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--det_bin", type=int, default=1,
                   help="1 = full native detector resolution -- required for pixel-exact alignment "
                        "with the raw frame stream used as context elsewhere in this project")
    p.add_argument("--stride", type=int, default=20,
                   help="chunk width = frames between consecutive target-chunk starts")
    p.add_argument("--frame_start", type=int, default=415_180)
    p.add_argument("--frame_end", type=int, default=445_820)
    p.add_argument("--out_dir", default="/myhome/data/sdate/shared/time_resolved/jointfbp_context")
    p.add_argument("--tag", default=None)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--checkpoint_every", type=int, default=200)
    return p.parse_args()


def main():
    a = parse_args()
    assert a.det_bin == 1, "det_bin must be 1 to match the model's training distribution"
    prof = DatasetProfile.load(a.profile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{prof.name}_jointfbpctx_T{a.T}_str{a.stride}_native_excl"
    t0 = time.time()

    dark_mov = Path(prof.mov_path).with_name(f"{prof.name}_darks.mov")
    flat_mov = Path(prof.mov_path).with_name(f"{prof.name}_flats.mov")
    dark_native = torch.from_numpy(R.load_calibration_average(
        str(dark_mov), prof.crop, prof.rot_axis_col, height=prof.height, width=prof.width,
    )).to(device=device, dtype=torch.float32)
    flat_native = torch.from_numpy(R.load_calibration_average(
        str(flat_mov), prof.crop, prof.rot_axis_col, height=prof.height, width=prof.width,
    )).to(device=device, dtype=torch.float32)
    print(f"flat/dark loaded: dark mean={dark_native.mean():.4g} flat mean={flat_native.mean():.4g}", flush=True)

    src = MemmapFrameSource(prof.memmap_path, prof.mov_path)
    win_T = a.T * a.stride

    chunk_starts = np.arange(a.frame_start, a.frame_end, a.stride)
    lo_needed = int(chunk_starts.min()) - (a.T - 1) * a.stride
    hi_needed = int(chunk_starts.max()) + a.stride
    assert lo_needed >= prof.frame_start and hi_needed <= prof.frame_end, (
        f"chunk range needs real data [{lo_needed},{hi_needed}) outside usable "
        f"profile range [{prof.frame_start},{prof.frame_end})")
    n_out = a.frame_end - a.frame_start
    print(f"T={a.T}  stride={a.stride}f  win_T={win_T}f  chunks={len(chunk_starts)}  "
          f"output range [{a.frame_start},{a.frame_end})  ({n_out} frames)  "
          f"needs real data [{lo_needed},{hi_needed})", flush=True)

    paths = [out_dir / f"{tag}_tap{c}.f16" for c in range(a.T)]
    meta_path = Path(str(paths[0]) + ".meta.npz")
    mms = [np.memmap(p, dtype=np.float16, mode="r+" if p.exists() else "w+",
                     shape=(n_out, *prof.crop)) for p in paths]

    done_path = out_dir / f"{tag}_done_chunks.npy"
    done_mask = np.zeros(len(chunk_starts), dtype=bool)
    if done_path.exists():
        prev = np.load(done_path)
        if prev.shape == done_mask.shape:
            done_mask = prev
            print(f"resuming: {done_mask.sum()}/{len(chunk_starts)} chunks already done", flush=True)
        else:
            print(f"existing done-mask shape mismatch -- starting fresh", flush=True)

    for wi, c in enumerate(chunk_starts):
        if done_mask[wi]:
            continue
        c = int(c)
        real_idx = np.arange(c, c + a.stride)
        real_angles = R.projection_angles(real_idx, deg_per_frame=prof.deg_per_frame)
        det_shape = (prof.crop[0] // a.det_bin, prof.crop[1] // a.det_bin)
        vol_shape = (det_shape[0], det_shape[1], det_shape[1])
        # reprojection target (chunk C's own 20 real angles) is the same for every one
        # of the T volumes -- build the projector once per chunk, not once per (chunk,k).
        proj_layer = build_lamino_projector(
            vol_shape=vol_shape, det_shape=det_shape, angles_deg=real_angles,
            lamino_angle_deg=0.0, voxel_size_mm=1.0, det_spacing_mm=1.0, device=device,
        )

        for k in range(a.T):
            s = c - k * a.stride
            # idx_keep = [s,c) + [c+stride,s+win_T) -- NON-contiguous (a genuine angular gap
            # where chunk c's own frames would be). native_window_gpu only does ONE contiguous
            # read (_mm[idx[0]:idx[-1]+1]), so the two pieces must be fetched separately --
            # passing a non-contiguous idx straight through would silently read back the
            # excluded frames too (see feedback_native_window_gpu_contiguous_only in project memory).
            pieces = []
            if c > s:
                pieces.append(np.arange(s, c))
            if s + win_T > c + a.stride:
                pieces.append(np.arange(c + a.stride, s + win_T))
            idx_keep = np.concatenate(pieces)
            angles_keep = R.projection_angles(idx_keep, deg_per_frame=prof.deg_per_frame)
            native = torch.cat([
                R.native_window_gpu(src, piece, prof.crop, prof.rot_axis_col, device) for piece in pieces
            ], dim=0)
            atten = R.counts_to_attenuation_flatdark(native, dark_native, flat_native)
            vol = R.reconstruct(atten, angles_keep, det_bin=a.det_bin, method="fbp", device=device)

            with torch.no_grad():
                reproj_atten = proj_layer(vol.unsqueeze(0).unsqueeze(0))[0]
                reproj_counts = R.attenuation_to_counts_flatdark(reproj_atten, dark_native, flat_native)

            out_idx = real_idx - a.frame_start
            mms[k][out_idx] = reproj_counts.cpu().numpy().astype(np.float16)

        done_mask[wi] = True
        if a.log_every and wi % a.log_every == 0:
            print(f"  chunk {wi + 1}/{len(chunk_starts)}  c={c}  "
                  f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
        if a.checkpoint_every and (wi + 1) % a.checkpoint_every == 0:
            for mm in mms:
                mm.flush()
            np.save(done_path, done_mask)
            print(f"  checkpoint saved at {wi + 1}/{len(chunk_starts)}", flush=True)

    for mm in mms:
        mm.flush()
    np.save(done_path, done_mask)
    meta = dict(first_index=a.frame_start, num_frames=n_out, crop=list(prof.crop),
               dose=1.0, native_noise=True, T=a.T, stride=a.stride, win_T=win_T,
               mode="jointfbp_context_excl")
    for p in paths:
        np.savez(str(p) + ".meta.npz", **meta)
    print(f"wrote {a.T} tap memmaps to {out_dir} ({tag}_tap0..{a.T - 1}.f16), "
          f"{n_out} frames [{a.frame_start},{a.frame_end})", flush=True)
    print(f"minutes={round((time.time() - t0) / 60, 1)}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
