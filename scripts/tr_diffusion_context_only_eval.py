#!/usr/bin/env python3
"""In-sample fit-quality check for a `--mode context_only` checkpoint: how well
does it recover the REAL measured central projection frame from the T=10
joint-FBP context taps ALONE (central always zeroed at input, native noise
throughout)? This is NOT a held-out generalization test -- no eval split was
reserved (see project discussion, 2026-08-27) -- it's a sanity check that
training actually converged to something useful before moving on to the real
target: synthesising NEVER-measured projection angles.

Reuses `reconstruct.denoise_sequence` as-is: the checkpoint's own saved
`conditioning_probability=0.0` makes it auto-detect the context-only
(`present=False`) inference regime, and `poisson_head=False` means the raw
model output IS the prediction (no posterior combination needed). Passing
`dose=None` keeps the dataset in the native (no synthetic noise) regime,
matching training exactly.

    python scripts/tr_diffusion_context_only_eval.py
"""
from __future__ import annotations

import os

os.environ["PATH"] = "/myhome/bin:" + os.environ.get("PATH", "")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

from sdate.tr_diffusion import reconstruct as R  # noqa: E402
from sdate.tr_diffusion.frames import MemmapFrameSource  # noqa: E402
from sdate.tr_naf.metrics import masked_psnr, masked_ssim  # noqa: E402

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"
CACHE = "/myhome/data/sdate/shared/time_resolved/tr_recon_cache"
CKDIR = "/mydata/sdate/shared/checkpoints"
TAG = "context_only_T10_str20_native"

MOV = f"{DATA}/212_Wunderkerze2.mov"
MEMMAP = f"{DATA}/frames_400k_500k.u16"
CKPT = f"{CKDIR}/tr_denoise_{TAG}.pt"

# PRED_MM/SUMMARY/MOVIE are built in main() once --out_suffix is known

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frame_start", type=int, default=None,
                   help="override eval range start (default: the checkpoint's own full training range)")
    p.add_argument("--frame_end", type=int, default=None,
                   help="override eval range end (default: the checkpoint's own full training range)")
    p.add_argument("--out_suffix", default="",
                   help="append to output filenames -- use for a smoke-test run so it doesn't clobber "
                        "the full-range eval's outputs")
    p.add_argument("--tag", default=TAG,
                   help="checkpoint tag -- selects /mydata/.../tr_denoise_{tag}.pt and namespaces "
                        "all output filenames (default: the original leaky context_only checkpoint)")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    t0 = time.time()
    tag = a.tag
    ckpt = f"{CKDIR}/tr_denoise_{tag}.pt"
    suffix = f"_{a.out_suffix}" if a.out_suffix else ""
    pred_mm_path = f"{DATA}/denoised_212_Wunderkerze2_{tag}{suffix}.f16"
    summary_path = Path(CACHE) / f"eval_summary_{tag}{suffix}.json"
    movie_path = Path(CACHE) / f"eval_movie_{tag}{suffix}.mov"
    assert Path(ckpt).exists(), f"checkpoint not found: {ckpt}"
    cfg = json.loads(Path(str(ckpt).replace(".pt", "_config.json")).read_text())
    log(f"checkpoint config: mode={cfg.get('mode')} k={cfg.get('k')} "
        f"conditioning_probability={cfg.get('conditioning_probability')} "
        f"aux_channel_memmap={len(cfg.get('aux_channel_memmap') or [])} taps "
        f"frame_range=[{cfg.get('frame_start')},{cfg.get('frame_end')})")
    assert cfg.get("conditioning_probability") == 0.0, \
        "checkpoint was not trained/saved with conditioning_probability=0.0 -- inference would be wrong"

    frame_start = a.frame_start if a.frame_start is not None else int(cfg["frame_start"])
    frame_end = a.frame_end if a.frame_end is not None else int(cfg["frame_end"])

    if not Path(str(pred_mm_path) + ".meta.npz").exists():
        log(f"=== inferring context_only predictions over [{frame_start},{frame_end}) ===")
        R.denoise_sequence(
            ckpt, MOV, MEMMAP, pred_mm_path,
            frame_start=frame_start, frame_end=frame_end,
            dose=None, batch=96, num_workers=8, device=device, log_every=200,
        )
        log(f"[infer] done in {(time.time() - t0) / 60:.1f} min -> {pred_mm_path}")
    else:
        log(f"[skip] predictions already exist at {pred_mm_path}")

    meta = np.load(str(pred_mm_path) + ".meta.npz")
    first, n = int(meta["first_index"]), int(meta["num_frames"])
    crop = tuple(meta["crop"])
    pred_mm = np.memmap(pred_mm_path, dtype=np.float16, mode="r", shape=(n, crop[0], crop[1]))

    log(f"=== scoring {n} predicted frames [{first},{first + n}) vs native GT ===")
    src = MemmapFrameSource(MEMMAP, MOV)
    axis_col = float(cfg.get("axis_col", 269.85))
    chunk = 500
    all_psnr, all_ssim = [], []
    movie_rows = []
    n_movie_samples = 6
    sample_idx = set(np.linspace(0, n - 1, n_movie_samples, dtype=int))
    for c0 in range(0, n, chunk):
        idx = np.arange(first + c0, first + min(c0 + chunk, n))
        gt_t = R.native_window_gpu(src, idx, crop, axis_col, device).cpu()
        pred_t = torch.from_numpy(np.asarray(pred_mm[c0:c0 + len(idx)]).astype(np.float32))
        mask = torch.ones(crop, dtype=torch.bool)
        for i in range(len(idx)):
            dr = float(gt_t[i].max() - gt_t[i].min())
            all_psnr.append(masked_psnr(gt_t[i:i + 1], pred_t[i:i + 1], mask, dr))
            all_ssim.append(masked_ssim(gt_t[i:i + 1], pred_t[i:i + 1], mask, dr))
            local_i = c0 + i
            if local_i in sample_idx:
                movie_rows.append((int(idx[i]), gt_t[i].clone(), pred_t[i].clone()))
        if c0 % (chunk * 10) == 0:
            log(f"  scored {c0 + len(idx)}/{n}  running PSNR {np.mean(all_psnr):.2f}dB")

    all_psnr = np.array(all_psnr)
    all_ssim = np.array(all_ssim)
    summary = {
        "tag": tag, "n_frames": n, "frame_range": [first, first + n],
        "minutes": round((time.time() - t0) / 60, 1),
        "psnr_mean": float(all_psnr.mean()), "psnr_std": float(all_psnr.std()),
        "psnr_min": float(all_psnr.min()), "psnr_max": float(all_psnr.max()),
        "ssim_mean": float(all_ssim.mean()), "ssim_std": float(all_ssim.std()),
        "note": "IN-SAMPLE fit-quality check, no held-out split -- sanity check that "
                "training converged, NOT the real generalisation eval",
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    log("SUMMARY " + json.dumps(summary, indent=2))

    if movie_rows:
        movie_rows.sort(key=lambda r: r[0])
        gts = np.stack([r[1].numpy() for r in movie_rows])
        vmin, vmax = np.percentile(gts, [1, 99])
        combined = [torch.cat([r[1], r[2]], dim=1) for r in movie_rows]
        R.write_slice_movie(combined, movie_path, float(vmin), float(vmax))
        log(f"movie written -> {movie_path}  panels: GT | predicted  frames: {[r[0] for r in movie_rows]}")

    log(f"TOTAL runtime: {(time.time() - t0) / 60:.1f} min")
    log(f"summary: {summary_path}")
    log("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
