#!/usr/bin/env python3
"""3-chunk-pooled super-time-resolution reconstruction: for central chunk C,
instead of FBP-ing only C's own 200-projection synthesized sinogram (20 real +
180 synthesized), pool the FULL 200-projection sinograms of C-1, C, and C+1
(600 projections total) into ONE FBP.

Motivation (from conversation, 2026-08-29): each chunk's own 20 REAL
projections always sit at the SAME chunk-relative angular position (the first
stride=20 of its 200), so as chunk C slides forward the absolute angle of
that real-data arc sweeps around the volume too -- a plausible source of a
"rotating" reconstruction artifact tied to chunk position. Pooling 3
neighbouring chunks' full sinograms spreads the real-data influence across a
wider arc and averages 3 independently-anchored model predictions at the
(mostly, not exactly) shared angle set, which should suppress a per-chunk
anchor-specific artifact even though the 3 sinograms' angles overlap heavily
(neighbouring chunks here are 1 native frame apart, i.e. shifted by exactly
one angular step).

Each of the 3 pooled chunks needs its own full sinogram (T=10 real volumes +
missing-angle reprojection + model inference -- the same per-chunk cost as
tr_diffusion_super_tr_movie.py), computed via a rolling 3-chunk buffer so
each UNIQUE chunk's sinogram is generated exactly once, not 3x.

The other two comparison panels (single-chunk predicted, and the slot-N real
single-revolution reference) are NOT recomputed -- they're sliced straight out
of the existing full-sweep checkpoint (`_slices.pt`) from
tr_diffusion_super_tr_movie.py, which must already exist for the requested
--c_start / range.

  python scripts/tr_diffusion_super_tr_pool3_movie.py \\
      --base_tag 212_Wunderkerze2_super_tr_c435400_n3000 --lo_rel 750 --hi_rel 1250
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
TAG = "context_only_T10_str20_native"
OUT_DIR = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="wunderkerze2")
    p.add_argument("--base_tag", required=True,
                   help="tag of the existing tr_diffusion_super_tr_movie.py sweep to pull "
                        "single-chunk predicted + reference slices from (its _slices.pt)")
    p.add_argument("--lo_rel", type=int, required=True,
                   help="first output position, as an index into the base sweep's c_positions")
    p.add_argument("--hi_rel", type=int, required=True,
                   help="last output position (inclusive), same indexing")
    p.add_argument("--T", type=int, default=10)
    p.add_argument("--stride", type=int, default=20)
    p.add_argument("--det_bin", type=int, default=1)
    p.add_argument("--model_batch", type=int, default=64)
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--tag", default=None)
    p.add_argument("--log_every", type=int, default=25)
    p.add_argument("--checkpoint_every", type=int, default=100)
    return p.parse_args()


def compute_chunk_sinogram(c, T, stride, prof, src, dark_native, flat_native, model, normalize,
                           denormalize, lo_n, hi_n, det_bin, model_batch, device, build_lamino_projector):
    """Full win_T=T*stride-frame synthesized sinogram (angles, counts) for central chunk c."""
    win_T = T * stride
    volumes = []
    for k in range(T):
        s = c - k * stride
        idx_w = np.arange(s, s + win_T)
        angles_w = R.projection_angles(idx_w, deg_per_frame=prof.deg_per_frame)
        native = R.native_window_gpu(src, idx_w, prof.crop, prof.rot_axis_col, device)
        atten = R.counts_to_attenuation_flatdark(native, dark_native, flat_native)
        vol = R.reconstruct(atten, angles_w, det_bin=det_bin, method="fbp", device=device)
        volumes.append(vol)

    det_shape = prof.crop
    vol_shape = (prof.crop[0], prof.crop[1], prof.crop[1])
    missing_idx = np.arange(c + stride, c + win_T)
    missing_angles = R.projection_angles(missing_idx, deg_per_frame=prof.deg_per_frame)
    proj_layer = build_lamino_projector(
        vol_shape=vol_shape, det_shape=det_shape, angles_deg=missing_angles,
        lamino_angle_deg=0.0, voxel_size_mm=1.0, det_spacing_mm=1.0, device=device,
    )
    vol_batch = torch.stack(volumes, dim=0).unsqueeze(1)
    with torch.no_grad():
        reproj_atten = proj_layer(vol_batch)
        reproj_counts = R.attenuation_to_counts_flatdark(reproj_atten, dark_native, flat_native)

    aux = reproj_counts.permute(1, 0, 2, 3)
    aux_norm = normalize(aux)
    n_missing = aux_norm.shape[0]
    preds = []
    for b0 in range(0, n_missing, model_batch):
        aux_b = aux_norm[b0:b0 + model_batch]
        bsz = aux_b.shape[0]
        central_b = torch.zeros(bsz, 1, *prof.crop, device=device)
        context_b = torch.zeros(bsz, 0, *prof.crop, device=device)
        out = denoise_frames_baseline(model, central_b, context_b, present=False,
                                      aux_channels=aux_b, poisson_head=False,
                                      norm_min=lo_n, norm_max=hi_n)
        preds.append(denormalize(out.clamp(-1, 1))[:, 0])
    pred_missing_counts = torch.cat(preds, dim=0)

    real_idx = np.arange(c, c + stride)
    real_counts = R.native_window_gpu(src, real_idx, prof.crop, prof.rot_axis_col, device)

    full_idx = np.arange(c, c + win_T)
    full_angles = R.projection_angles(full_idx, deg_per_frame=prof.deg_per_frame)
    full_counts = torch.cat([real_counts, pred_missing_counts], dim=0)
    return full_angles, full_counts


def main():
    a = parse_args()
    assert a.det_bin == 1, "det_bin must be 1 to match the model's training distribution"
    prof = DatasetProfile.load(a.profile)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = a.tag or f"{a.base_tag}_pool3_{a.lo_rel}_{a.hi_rel}"
    t0 = time.time()

    from astra_torch.lamino import build_lamino_projector

    base_slices_path = out_dir / f"{a.base_tag}_slices.pt"
    assert base_slices_path.exists(), f"base sweep slices not found: {base_slices_path}"
    base = torch.load(base_slices_path, map_location="cpu", weights_only=False)
    base_c = np.asarray(base["c_positions"])
    assert a.lo_rel >= 0 and a.hi_rel < len(base_c), \
        f"[lo_rel,hi_rel]=[{a.lo_rel},{a.hi_rel}] out of range for base sweep of {len(base_c)} chunks"
    c_start_abs = int(base_c[a.lo_rel])
    c_end_abs = int(base_c[a.hi_rel])
    single_pred = base["pred"][a.lo_rel:a.hi_rel + 1]
    ref = base["ref"][a.lo_rel:a.hi_rel + 1]
    print(f"base sweep {a.base_tag}: pulling single-chunk predicted + reference for "
          f"c in [{c_start_abs},{c_end_abs}]  ({single_pred.shape[0]} positions)", flush=True)

    ckpt = f"{CKDIR}/tr_denoise_{TAG}.pt"
    model, cfg = load_denoiser(ckpt, device=device)
    assert cfg["mode"] == "context_only" and cfg["conditioning_probability"] == 0.0
    normalize, denormalize = make_norm_fns(cfg)
    lo_n, hi_n = float(cfg["norm_min"]), float(cfg["norm_max"])
    win_T = a.T * a.stride
    print(f"loaded {ckpt}  T={a.T} stride={a.stride} win_T={win_T}", flush=True)

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

    out_c_positions = np.arange(c_start_abs, c_end_abs + 1)
    lo_needed = int(out_c_positions.min()) - 1 - (a.T - 1) * a.stride
    hi_needed = int(out_c_positions.max()) + 1 + win_T
    assert lo_needed >= prof.frame_start and hi_needed <= prof.frame_end, (
        f"pooled sweep needs real data [{lo_needed},{hi_needed}) outside usable "
        f"profile range [{prof.frame_start},{prof.frame_end})")
    print(f"pooled sweep: c in [{out_c_positions[0]},{out_c_positions[-1]}]  "
          f"needs real data [{lo_needed},{hi_needed})", flush=True)

    raw_path = out_dir / f"{tag}_slices.pt"
    pooled_slices = []
    start_wi = 0
    if raw_path.exists():
        ckpt_data = torch.load(raw_path, map_location="cpu", weights_only=False)
        prev_c = np.asarray(ckpt_data["c_positions"])
        if len(prev_c) <= len(out_c_positions) and np.array_equal(prev_c, out_c_positions[:len(prev_c)]):
            pooled_slices = list(ckpt_data["pooled"].unbind(0))
            start_wi = len(pooled_slices)
            print(f"resuming from checkpoint: {start_wi}/{len(out_c_positions)} chunks already done", flush=True)
        else:
            print(f"existing checkpoint at {raw_path} doesn't match this sweep -- starting fresh", flush=True)

    # rolling buffer: sino_cache[c] = (angles, counts) for chunk c
    sino_cache = {}

    def get_sino(c):
        if c not in sino_cache:
            sino_cache[c] = compute_chunk_sinogram(
                c, a.T, a.stride, prof, src, dark_native, flat_native, model, normalize,
                denormalize, lo_n, hi_n, a.det_bin, a.model_batch, device, build_lamino_projector)
        return sino_cache[c]

    for wi, c in enumerate(out_c_positions):
        if wi < start_wi:
            continue
        c = int(c)
        angles_list, counts_list = [], []
        for cc in (c - 1, c, c + 1):
            ang, cnt = get_sino(cc)
            angles_list.append(ang)
            counts_list.append(cnt)
        # evict anything more than 1 away from the current centre -- only c-1,c,c+1 are
        # ever needed again (c+1 becomes next iteration's c, i.e. its c-1).
        for cached_c in list(sino_cache.keys()):
            if cached_c < c:
                del sino_cache[cached_c]

        pool_angles = np.concatenate(angles_list)
        pool_counts = torch.cat(counts_list, dim=0)
        pool_atten = R.counts_to_attenuation_flatdark(pool_counts, dark_native, flat_native)
        pool_vol = R.reconstruct(pool_atten, pool_angles, det_bin=a.det_bin, method="fbp", device=device)
        mid = pool_vol.shape[0] // 2
        pooled_slices.append(pool_vol[mid].cpu())

        if a.log_every and wi % a.log_every == 0:
            print(f"  chunk {wi + 1}/{len(out_c_positions)}  c={c}  "
                  f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)

        if a.checkpoint_every and (wi + 1) % a.checkpoint_every == 0:
            torch.save({"pooled": torch.stack(pooled_slices), "c_positions": out_c_positions[:wi + 1]}, raw_path)
            print(f"  checkpoint saved at {wi + 1}/{len(out_c_positions)}", flush=True)

    stacked_pooled = torch.stack(pooled_slices)
    torch.save({"pooled": stacked_pooled, "c_positions": out_c_positions}, raw_path)
    print(f"raw pooled slices checkpointed -> {raw_path}", flush=True)

    vmin = float(np.percentile(ref.numpy(), 1))
    vmax = float(np.percentile(ref.numpy(), 99))
    combined = [torch.cat([sp, pl, rf], dim=1) for sp, pl, rf in zip(single_pred, pooled_slices, ref)]
    movie_path = out_dir / f"{tag}.mov"
    R.write_slice_movie(combined, movie_path, vmin, vmax)
    print(f"movie written -> {movie_path}  panels: single-chunk-predicted | 3-chunk-pooled | reference(slot4)",
          flush=True)

    summary = {
        "tag": tag, "base_tag": a.base_tag, "c_start": int(out_c_positions[0]), "c_end": int(out_c_positions[-1]),
        "n_chunks": len(out_c_positions), "T": a.T, "stride": a.stride,
        "movie": str(movie_path), "minutes": round((time.time() - t0) / 60, 1),
        "note": "panels: single-chunk predicted (20 real+180 synth) | 3-chunk pooled (600 total "
                "projections from chunks C-1,C,C+1) | reference (slot4 real single-revolution FBP)",
    }
    summary_path = out_dir / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print("SUMMARY " + json.dumps(summary, indent=2), flush=True)
    print(f"TOTAL runtime: {(time.time() - t0) / 60:.1f} min", flush=True)
    print("PIPELINE COMPLETE", flush=True)


if __name__ == "__main__":
    main()
