#!/usr/bin/env python3
"""Super-time-resolution inference for the `context_only` T=10/stride=20 model.

For a sliding "central chunk" of 20 real consecutive frames (~36 deg of native
coverage), synthesize the 180 projections needed to complete a full win_T=200
(360 deg) sinogram AT THAT CHUNK'S TIME INSTANT, then FBP it -- a genuinely
limited-angle reconstruction problem: only 20/200 projections are real, the
other 180 have no ground truth (see project discussion, 2026-08-28).

Mechanics (corrected 2026-09-04, see project memory -- a first version of this
script got this wrong: it built ONE set of T volumes excluding only chunk C
and reused them, reprojected, for all 180 missing frames -- but frames
[C+20, C+199) belong to 9 OTHER chunks, none of which were ever excluded, so
those context volumes structurally contained each missing frame's own real
data. That's not just a leak, it's out-of-distribution input the model was
never trained to handle -- training NEVER shows the model a context channel
that structurally contains its own prediction target):

* The chunk's own 20 REAL measured frames [C, C+stride) are used directly,
  unchanged -- never run through the model.
* The 180 "missing" frames [C+stride, C+win_T) are split into 9 sub-chunks
  of `stride` frames each: C+20, C+40, ..., C+180. EACH sub-chunk is treated
  exactly like a normal training/eval target: its own T=10 context volumes
  (windows s = sub_chunk - k*stride for k=0..9) each EXCLUDE that
  sub-chunk's own [sub_chunk, sub_chunk+stride) real frames -- identical
  construction to `tr_diffusion_jointfbp_context_cache_excl.py`, just called
  fresh per sub-chunk instead of once for C. Reproject each sub-chunk's own
  T volumes at ITS OWN real angles (not C's), feed through the model
  (central zeroed, class_label=0) exactly like context_only training -- 100%
  context, no measured-projection pass-through (native noise is low enough
  this isn't needed, see project memory 2026-09-03 correction).
* This means the 180 predicted frames are genuine leak-free contextual
  estimates of each sub-chunk's OWN real value (using nearby real revolutions,
  matching the model's training distribution exactly) -- combined with C's
  20 real measurements into one 200-view sinogram, this is the same
  static-object-across-one-revolution assumption an ordinary single-
  revolution FBP already makes, just with 9/10 of the revolution now backed
  by leak-free model estimates instead of raw (noisier, and here simply
  unavailable/future-relative-to-C) single-shot measurements.
* Cost: ~9x the limited-angle FBP reconstructions per output chunk (90
  instead of 10), since each of the 9 sub-chunks needs its own full T=10
  exclusion set -- no compute can be shared between them or across slide
  positions (every quantity depends on exact frame boundaries that shift by
  1 frame per slide step).
* Reference for the movie = slot 4's own real single-revolution FBP volume
  (`s=C-80`, plain contiguous FBP, no exclusion -- it's a display reference,
  not model input), recomputed fresh at each slide position C (NOT a static
  reference -- both panels evolve per chunk).

Output: one float16 stack of (n_c, 2, D, H) central-slice pairs (predicted |
reference) as a comparison .mov, plus a JSON summary.

  python scripts/tr_diffusion_super_tr_movie.py --c_start 442210 --n_chunks 100
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")
os.environ["PATH"] = f"/myhome/bin:{os.environ.get('PATH', '')}"

from sdate.tr_diffusion import reconstruct as R  # noqa: E402
from sdate.tr_diffusion.frames import MemmapFrameSource  # noqa: E402
from sdate.tr_diffusion.load import load_denoiser, make_norm_fns  # noqa: E402
from sdate.tr_diffusion.pipeline import denoise_frames_baseline  # noqa: E402
from sdate.tr_diffusion.profiles import DatasetProfile  # noqa: E402

CKDIR = "/mydata/sdate/shared/checkpoints"
TAG = "context_only_T10_str20_native_excl"
OUT_DIR = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="wunderkerze2")
    p.add_argument("--c_start", type=int, default=442210,
                   help="first central-chunk start frame (the roughest 100-frame patch "
                        "found by scanning the dynamic-region fit-quality curve)")
    p.add_argument("--n_chunks", type=int, default=100,
                   help="number of central-chunk slide positions (stride=1 frame each)")
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--ref_slot", type=int, default=4)
    p.add_argument("--det_bin", type=int, default=1,
                   help="MUST stay 1 -- the model was trained on det_bin=1 context taps; "
                        "changing this would put the aux channels out-of-distribution")
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--tag", default=None)
    p.add_argument("--ckpt_tag", default=TAG,
                   help="checkpoint tag -- selects /mydata/.../tr_denoise_{ckpt_tag}.pt "
                        "(default: the leak-free context_only checkpoint)")
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--checkpoint_every", type=int, default=100,
                   help="save accumulated slices to disk every N chunks, and resume from it on "
                        "restart -- this loop takes long enough at n_chunks=1000 to risk a RunAI "
                        "preemption mid-run (see project memory on preemption cadence)")
    return p.parse_args()


def leak_free_volumes(target_chunk, T, stride, win_T, deg_per_frame, src, crop, axis_col,
                       dark_native, flat_native, det_bin, device):
    """T=10 context volumes for `target_chunk`: window k is s=target_chunk-k*stride,
    EXCLUDING target_chunk's own [target_chunk, target_chunk+stride) real frames --
    identical construction to tr_diffusion_jointfbp_context_cache_excl.py, callable
    for any target chunk (training's own target C, or one of the 9 missing
    sub-chunks at inference)."""
    volumes = []
    for k in range(T):
        s = target_chunk - k * stride
        pieces = []
        if target_chunk > s:
            pieces.append(np.arange(s, target_chunk))
        if s + win_T > target_chunk + stride:
            pieces.append(np.arange(target_chunk + stride, s + win_T))
        idx_keep = np.concatenate(pieces)
        angles_keep = R.projection_angles(idx_keep, deg_per_frame=deg_per_frame)
        native = torch.cat([
            R.native_window_gpu(src, piece, crop, axis_col, device) for piece in pieces
        ], dim=0)
        atten = R.counts_to_attenuation_flatdark(native, dark_native, flat_native)
        vol = R.reconstruct(atten, angles_keep, det_bin=det_bin, method="fbp", device=device)
        volumes.append(vol)
    return volumes


def main():
    a = parse_args()
    assert a.det_bin == 1, "det_bin must be 1 to match the model's training distribution"
    prof = DatasetProfile.load(a.profile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{prof.name}_super_tr_c{a.c_start}_n{a.n_chunks}"
    t0 = time.time()

    from astra_torch.lamino import build_lamino_projector

    ckpt = f"{CKDIR}/tr_denoise_{a.ckpt_tag}.pt"
    model, cfg = load_denoiser(ckpt, device=device)
    assert cfg["mode"] == "context_only" and cfg["conditioning_probability"] == 0.0
    normalize, denormalize = make_norm_fns(cfg)
    lo_n, hi_n = float(cfg["norm_min"]), float(cfg["norm_max"])
    win_T = a.T * a.stride
    print(f"loaded {ckpt}  T={a.T} stride={a.stride} win_T={win_T} ref_slot={a.ref_slot}", flush=True)

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
    det_shape = prof.crop
    vol_shape = (prof.crop[0], prof.crop[1], prof.crop[1])

    c_positions = a.c_start + np.arange(a.n_chunks)
    # Farthest sub-chunk is c_max + (T-1)*stride (j=T-1); ITS OWN k=0 window
    # (s=target_chunk) reaches target_chunk + win_T -- i.e. c_max + (T-1)*stride + win_T,
    # well past the old (pre-fix) c_max + win_T bound.
    lo_needed = int(c_positions.min()) - (a.T - 1) * a.stride
    hi_needed = int(c_positions.max()) + (a.T - 1) * a.stride + win_T
    assert lo_needed >= prof.frame_start and hi_needed <= prof.frame_end, (
        f"chunk sweep [{lo_needed},{hi_needed}) exceeds usable profile range "
        f"[{prof.frame_start},{prof.frame_end})")
    print(f"chunk sweep: c in [{c_positions[0]},{c_positions[-1]}]  "
          f"needs real data [{lo_needed},{hi_needed})", flush=True)

    raw_path = out_dir / f"{tag}_slices.pt"
    pred_slices, ref_slices = [], []
    start_wi = 0
    if raw_path.exists():
        ckpt_data = torch.load(raw_path, map_location="cpu", weights_only=False)
        prev_c = np.asarray(ckpt_data["c_positions"])
        if len(prev_c) <= len(c_positions) and np.array_equal(prev_c, c_positions[:len(prev_c)]):
            pred_slices = list(ckpt_data["pred"].unbind(0))
            ref_slices = list(ckpt_data["ref"].unbind(0))
            start_wi = len(pred_slices)
            print(f"resuming from checkpoint: {start_wi}/{a.n_chunks} chunks already done", flush=True)
        else:
            print(f"existing checkpoint at {raw_path} doesn't match this sweep's c_positions "
                  f"-- ignoring, starting fresh", flush=True)

    for wi, c in enumerate(c_positions):
        if wi < start_wi:
            continue
        c = int(c)

        # --- reference volume: slot 4's own PLAIN contiguous single-revolution
        # FBP (display-only ground truth, not fed to the model, so no leak
        # concern here) ---
        s_ref = c - a.ref_slot * a.stride
        idx_ref = np.arange(s_ref, s_ref + win_T)
        angles_ref = R.projection_angles(idx_ref, deg_per_frame=prof.deg_per_frame)
        native_ref = R.native_window_gpu(src, idx_ref, prof.crop, prof.rot_axis_col, device)
        atten_ref = R.counts_to_attenuation_flatdark(native_ref, dark_native, flat_native)
        ref_vol = R.reconstruct(atten_ref, angles_ref, det_bin=a.det_bin, method="fbp", device=device)

        # --- 180 missing frames = 9 sub-chunks (c+20, c+40, ..., c+180), each
        # predicted like a normal training/eval target: its OWN T=10
        # exclusion-aware context volumes, reprojected at ITS OWN real angles
        # (not c's) -- see corrected mechanics in the module docstring. ---
        pred_pieces = []
        for j in range(1, a.T):
            target_chunk = c + j * a.stride
            volumes_j = leak_free_volumes(
                target_chunk, a.T, a.stride, win_T, prof.deg_per_frame,
                src, prof.crop, prof.rot_axis_col, dark_native, flat_native, a.det_bin, device,
            )
            real_idx_j = np.arange(target_chunk, target_chunk + a.stride)
            real_angles_j = R.projection_angles(real_idx_j, deg_per_frame=prof.deg_per_frame)
            proj_layer_j = build_lamino_projector(
                vol_shape=vol_shape, det_shape=det_shape, angles_deg=real_angles_j,
                lamino_angle_deg=0.0, voxel_size_mm=1.0, det_spacing_mm=1.0, device=device,
            )
            vol_batch_j = torch.stack(volumes_j, dim=0).unsqueeze(1)  # (T,1,D,H,W)
            with torch.no_grad():
                reproj_atten_j = proj_layer_j(vol_batch_j)  # (T, stride, R, C)
                reproj_counts_j = R.attenuation_to_counts_flatdark(reproj_atten_j, dark_native, flat_native)

            aux_j = reproj_counts_j.permute(1, 0, 2, 3)  # (stride, T, H, W)
            aux_norm_j = normalize(aux_j)
            central_j = torch.zeros(a.stride, 1, *prof.crop, device=device)
            context_j = torch.zeros(a.stride, 0, *prof.crop, device=device)
            out_j = denoise_frames_baseline(model, central_j, context_j, present=False,
                                            aux_channels=aux_norm_j, poisson_head=False,
                                            norm_min=lo_n, norm_max=hi_n)
            pred_pieces.append(denormalize(out_j.clamp(-1, 1))[:, 0])  # (stride, H, W)
        pred_missing_counts = torch.cat(pred_pieces, dim=0)  # (180, H, W) counts, ordered c+20..c+199

        # --- real 20 central-chunk frames (measured, no model needed) ---
        real_idx = np.arange(c, c + a.stride)
        real_counts = R.native_window_gpu(src, real_idx, prof.crop, prof.rot_axis_col, device)

        # --- combine into a full win_T-frame synthetic sinogram and FBP it ---
        full_idx = np.arange(c, c + win_T)
        full_angles = R.projection_angles(full_idx, deg_per_frame=prof.deg_per_frame)
        full_counts = torch.cat([real_counts, pred_missing_counts], dim=0)
        full_atten = R.counts_to_attenuation_flatdark(full_counts, dark_native, flat_native)
        pred_vol = R.reconstruct(full_atten, full_angles, det_bin=a.det_bin, method="fbp", device=device)

        mid = pred_vol.shape[0] // 2
        pred_slices.append(pred_vol[mid].cpu())
        ref_slices.append(ref_vol[mid].cpu())

        if a.log_every and wi % a.log_every == 0:
            print(f"  chunk {wi + 1}/{a.n_chunks}  c={c}  "
                  f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)

        if a.checkpoint_every and (wi + 1) % a.checkpoint_every == 0:
            torch.save({"pred": torch.stack(pred_slices), "ref": torch.stack(ref_slices),
                       "c_positions": c_positions[:wi + 1]}, raw_path)
            print(f"  checkpoint saved at {wi + 1}/{a.n_chunks}", flush=True)

    stacked_pred = torch.stack(pred_slices)
    stacked_ref = torch.stack(ref_slices)
    # checkpoint the raw slices BEFORE the movie-writing step, so a crash there
    # (e.g. torch.quantile's 2**24-element limit, hit once already) doesn't lose
    # the whole GPU sweep -- ~100 FBP reconstructions each are not cheap to redo.
    torch.save({"pred": stacked_pred, "ref": stacked_ref, "c_positions": c_positions}, raw_path)
    print(f"raw slices checkpointed -> {raw_path}", flush=True)
    # torch.quantile errors out above 2**24 elements; use numpy instead.
    vmin = float(np.percentile(stacked_ref.numpy(), 1))
    vmax = float(np.percentile(stacked_ref.numpy(), 99))
    combined = [torch.cat([p, r], dim=1) for p, r in zip(pred_slices, ref_slices)]
    movie_path = out_dir / f"{tag}.mov"
    R.write_slice_movie(combined, movie_path, vmin, vmax)
    print(f"movie written -> {movie_path}  panels: predicted(full-synth) | reference(slot{a.ref_slot})", flush=True)

    summary = {
        "tag": tag, "c_start": int(c_positions[0]), "c_end": int(c_positions[-1]),
        "n_chunks": a.n_chunks, "T": a.T, "stride": a.stride, "ref_slot": a.ref_slot,
        "movie": str(movie_path), "minutes": round((time.time() - t0) / 60, 1),
        "note": "predicted volume uses 20 real + 180 SYNTHESIZED (no-GT) projections per chunk; "
                "reference volume is slot %d's own real single-revolution FBP, recomputed per chunk" % a.ref_slot,
    }
    summary_path = out_dir / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("SUMMARY " + json.dumps(summary, indent=2), flush=True)
    print(f"TOTAL runtime: {(time.time() - t0) / 60:.1f} min", flush=True)
    print("PIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
