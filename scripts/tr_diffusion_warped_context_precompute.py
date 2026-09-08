#!/usr/bin/env python3
"""Precompute motion-compensated ("warped") temporal context for the
angular-resolution-gap experiment (see the "Motion-compensated warped
temporal context" plan / README section).

Two stages:

1. Extend the existing production checkpoint's own denoised-output cache
   (posterior-mean, dose=0.05) to the FULL training range -- needed purely as
   a registration aid for flow estimation, never fed to any model as an
   actual input, so no target-leak concern applies to it.
2. For each of the 4 "temporal" context taps (k=1, temporal_raw_pairs=True),
   compute dense optical flow (validated interactively this session:
   Farneback on the phase-1 proxies, NOT raw noisy frames -- flow on raw data
   is 2x larger in magnitude and incoherent) between each central frame's
   phase-1 proxy and its bracket frame's phase-1 proxy, then warp the RAW
   dose=0.05 bracket value (read exactly as that central frame's own context
   tap -- see data.py's per-ci generator seeding) with that flow. Writes one
   float16 memmap per tap (``warped_ctx_{tap.name}.f16``), same
   first_index/num_frames/crop convention as every other cache in this
   project (see data.py's aux_channel_memmap/clean_target_memmap).

    python scripts/tr_diffusion_warped_context_precompute.py
"""
from __future__ import annotations

import os

os.environ["PATH"] = "/myhome/bin:" + os.environ.get("PATH", "")

import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

from sdate.tr_diffusion import reconstruct as R  # noqa: E402
from sdate.tr_diffusion.data import TimeResolvedFrameDataset  # noqa: E402
from sdate.tr_diffusion.geometry import usable_frame_range  # noqa: E402
from sdate.tr_diffusion.load import load_config  # noqa: E402
from sdate.tr_diffusion.profiles import REGISTRY  # noqa: E402

REAL = REGISTRY["wunderkerze2"]
CKPT = "/myhome/data/sdate/shared/checkpoints/tr_denoise_baseline_k1_dose005_poissonhead.pt"
TR_DIR = Path(REAL.memmap_path).parent
PHASE1_CACHE = str(TR_DIR / "phase1_posteriormean_fullrange_dose05.f16")
WARPED_DIR = TR_DIR / "warped_context_dose05"
DOSE, NOISE_SEED = 0.05, 12345
FB_KW = dict(pyr_scale=0.5, levels=4, winsize=21, iterations=5, poly_n=7, poly_sigma=1.5, flags=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def stage_extend_phase1_cache() -> None:
    if Path(PHASE1_CACHE + ".meta.npz").exists():
        log(f"[skip] phase-1 cache already exists at {PHASE1_CACHE}")
        return
    log("=== stage 1/2: extend phase-1 (posterior-mean) cache to the full training range ===")
    t0 = time.time()
    R.denoise_sequence(
        CKPT, REAL.mov_path, REAL.memmap_path, PHASE1_CACHE,
        frame_start=REAL.frame_start, frame_end=REAL.frame_end,
        dose=DOSE, noise_seed=NOISE_SEED, axis_col=REAL.rot_axis_col, deg_per_frame=REAL.deg_per_frame,
        poisson_posterior=True, var_out_path=None,
        batch=64, num_workers=8, device=device, log_every=500,
    )
    log(f"[extend] done in {(time.time() - t0) / 60:.1f} min -> {PHASE1_CACHE}")


# --- multiprocessing worker state (set once per process via _init_worker) --- #
_W = {}


def _init_worker(cfg: dict) -> None:
    cv2.setNumThreads(1)  # avoid oversubscription: the Pool itself provides the parallelism
    torch.set_num_threads(1)  # same -- torch's own OMP pool inside ds[ci] (poisson/normalize)
                              # otherwise oversubscribes 16x, measured ~22x slowdown per item
    _W["phase1_mm"] = np.memmap(PHASE1_CACHE, dtype=np.float16, mode="r",
                                shape=(cfg["phase1_n"], *cfg["crop"]))
    _W["phase1_first"] = cfg["phase1_first"]
    ds = TimeResolvedFrameDataset(
        REAL.mov_path, memmap_path=REAL.memmap_path, k=cfg["k"], include_mirror=cfg["include_mirror"],
        frame_start=REAL.frame_start, frame_end=REAL.frame_end, crop=cfg["crop"],
        neighborhoods=cfg["neighborhoods"], norm_range=(cfg["norm_min"], cfg["norm_max"]),
        extra_noise_dose=DOSE, noise_seed=NOISE_SEED, temporal_raw_pairs=cfg["temporal_raw_pairs"],
        axis_col=REAL.rot_axis_col, deg_per_frame=REAL.deg_per_frame,
    )
    _W["ds"] = ds
    _W["ds_first"] = int(ds.indices.min())
    _W["tap_idx"] = cfg["tap_idx"]
    _W["tap_offset"] = cfg["tap_offset"]


def _process_one(ci: int):
    phase1_mm, phase1_first = _W["phase1_mm"], _W["phase1_first"]
    ds, ds_first, tap_idx, tap_offset = _W["ds"], _W["ds_first"], _W["tap_idx"], _W["tap_offset"]
    bracket_ci = ci + tap_offset

    central_p1 = np.asarray(phase1_mm[ci - phase1_first]).astype(np.float32)
    bracket_p1 = np.asarray(phase1_mm[bracket_ci - phase1_first]).astype(np.float32)
    lo, hi = np.percentile(central_p1, [1, 99])
    if hi - lo < 1e-6:
        hi = lo + 1.0
    to_u8 = lambda x: np.clip((x - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)  # noqa: E731
    flow = cv2.calcOpticalFlowFarneback(to_u8(central_p1), to_u8(bracket_p1), None, **FB_KW)

    item = ds[ci - ds_first]
    bracket_raw = ds.denormalize(item["context"][tap_idx]).numpy().astype(np.float32)

    h, w = bracket_raw.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x, map_y = gx + flow[..., 0], gy + flow[..., 1]
    warped = cv2.remap(bracket_raw, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return ci, warped.astype(np.float16)


def stage_warp_context(num_workers: int = 16) -> None:
    WARPED_DIR.mkdir(parents=True, exist_ok=True)
    ck_cfg = load_config(CKPT)
    crop = tuple(ck_cfg["crop"])
    include_mirror = bool(ck_cfg.get("include_mirror", False))
    ds_probe = TimeResolvedFrameDataset(
        REAL.mov_path, memmap_path=REAL.memmap_path, k=int(ck_cfg["k"]), include_mirror=include_mirror,
        frame_start=REAL.frame_start, frame_end=REAL.frame_end, crop=crop,
        neighborhoods=ck_cfg.get("neighborhoods", "both"), norm_range=(ck_cfg["norm_min"], ck_cfg["norm_max"]),
        extra_noise_dose=DOSE, noise_seed=NOISE_SEED, temporal_raw_pairs=bool(ck_cfg.get("temporal_raw_pairs", False)),
        axis_col=REAL.rot_axis_col, deg_per_frame=REAL.deg_per_frame,
    )
    temporal_taps = [t for t in ds_probe.layout if t.kind == "temporal"]
    log(f"temporal taps to warp: {[t.name for t in temporal_taps]}")

    p1meta = np.load(PHASE1_CACHE + ".meta.npz")
    phase1_first, phase1_n = int(p1meta["first_index"]), int(p1meta["num_frames"])

    # Final trainable ci range: every ci here must have BOTH itself and every tap's
    # bracket frame inside the phase-1 cache's own valid range -- reuse
    # usable_frame_range (same margin logic the dataset itself already relies on)
    # rather than hand-rolling the arithmetic.
    lo, hi = usable_frame_range(
        phase1_first, phase1_first + phase1_n, int(ck_cfg["k"]), include_mirror=include_mirror,
        neighborhoods=ck_cfg.get("neighborhoods", "both"), period_360=360.0 / REAL.deg_per_frame,
        temporal_raw_pairs=bool(ck_cfg.get("temporal_raw_pairs", False)),
    )
    ci_list = list(range(lo, hi))
    n = len(ci_list)
    log(f"trainable ci range: [{lo}, {hi}) -- {n} frames (phase-1 cache covers "
        f"[{phase1_first}, {phase1_first + phase1_n}))")

    for tap_idx, tap in enumerate(ds_probe.layout):
        if tap.kind != "temporal":
            continue
        out_path = WARPED_DIR / f"warped_ctx_{tap.name}.f16"
        if Path(str(out_path) + ".meta.npz").exists():
            log(f"[skip] {out_path} already exists")
            continue
        log(f"=== warping tap {tap.name} (frame_offset={tap.frame_offset}) -- {n} frames, "
            f"{num_workers} workers ===")
        t0 = time.time()
        out_mm = np.memmap(str(out_path), dtype=np.float16, mode="w+", shape=(n, *crop))
        cfg = dict(phase1_n=phase1_n, phase1_first=phase1_first, crop=crop,
                  k=int(ck_cfg["k"]), include_mirror=include_mirror,
                  neighborhoods=ck_cfg.get("neighborhoods", "both"),
                  norm_min=ck_cfg["norm_min"], norm_max=ck_cfg["norm_max"],
                  temporal_raw_pairs=bool(ck_cfg.get("temporal_raw_pairs", False)),
                  tap_idx=tap_idx, tap_offset=int(tap.frame_offset))
        done = 0
        with mp.get_context("spawn").Pool(num_workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for ci, warped in pool.imap_unordered(_process_one, ci_list, chunksize=64):
                out_mm[ci - lo] = warped
                done += 1
                if done % 20000 == 0:
                    log(f"  {tap.name}: {done}/{n}")
        out_mm.flush()
        np.savez(str(out_path) + ".meta.npz", first_index=lo, num_frames=n, crop=np.array(crop))
        log(f"[warp {tap.name}] done in {(time.time() - t0) / 60:.1f} min -> {out_path}")


def main() -> None:
    log(f"device={device}")
    stage_extend_phase1_cache()
    stage_warp_context()
    log("PIPELINE COMPLETE (warped context precompute)")


if __name__ == "__main__":
    main()
