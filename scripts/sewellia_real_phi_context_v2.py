#!/usr/bin/env python3
"""Build the T=5 phi-gated joint-FBP context cache for the REAL full Sewellia
lineolata acquisition, v2: FULL native resolution (det_bin=1, 580x576) using
DISJOINT-GROUP reconstruction to work around the hard per-reconstruction
GPU-memory ceiling (empirically confirmed: 900 views safe, 1000 OOMs on an
80GB A100 -- see sdate/tr_diffusion/phi_context.py's
reconstruct_phi_gated_groups docstring and project memory
project-sewellia-phi-context.md).

Supersedes the v1 script (sewellia_real_phi_context.py, T=11 at det_bin=4,
145x144) -- abandoned as insufficient resolution/detail after comparing
against the PeriodRecon paper's figures (arXiv:2506.03792).

Design (confirmed with the user):
- T=5 levels, target real-view counts (400, 900, 1800, 2700, 3600), each
  reconstructed as up to (1,1,2,3,4) DISJOINT 900-view groups.
- Groups within a level are AVERAGED (clamp=False per group, single clamp_min(0)
  on the average -- see reconstruct.py's reconstruct() docstring on why
  clamping-then-averaging biases the result) into ONE stored channel per
  level -- NOT stored separately. This was a deliberate storage-budget
  tradeoff: storing all 11 disjoint groups separately at full resolution
  would need ~147GB (more than the ~117GB free on the shared NFS quota this
  data lives on), while averaging keeps it at 5 channels (~67GB) and still
  gets the full SNR benefit of using all the available data at each level.
- Real per-pixel dark/flat calibration (not the trivial dark=0,flat=1 used
  for the small precorrected preview file).

Per-bin compute cost measured in smoke testing: ~2.1s per 900-view
full-resolution reconstruction -> ~22s/bin (1+1+2+3+4=11 reconstructions) ->
~132min for all 360 bins, plus the one-time ~5.5min raw-data load.

    python scripts/sewellia_real_phi_context_v2.py --max_bins 3   # smoke test
    python scripts/sewellia_real_phi_context_v2.py                # full 360-bin build
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cupy as cp
import h5py
import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")
os.environ["PATH"] = f"/myhome/bin:{os.environ.get('PATH', '')}"

from sdate.tr_diffusion import phi_context as PC

FULL_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01"
DATA_PATH = f"{FULL_DIR}/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5"
PHASE_TXT = f"{FULL_DIR}/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01_sound_phase.txt"
OUT_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/phi_context"
CALIB_PATH = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/sewellia_real_calibration.npz"

TARGET_VIEWS = [400, 900, 1800, 2700, 3600]
GROUP_SIZE = 900


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=DATA_PATH)
    p.add_argument("--phase_txt", default=PHASE_TXT)
    p.add_argument("--calib_path", default=CALIB_PATH)
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--tag", default="sewellia_v2_phictx")
    p.add_argument("--n_bins", type=int, default=360)
    p.add_argument("--group_size", type=int, default=GROUP_SIZE)
    p.add_argument("--target_views", default=",".join(str(x) for x in TARGET_VIEWS),
                   help="comma-separated target real-view count per T-level -- T = len(...)")
    p.add_argument("--max_bins", type=int, default=None,
                   help="cap the number of distinct phi-bins processed, for a quick smoke test")
    p.add_argument("--bins_per_invocation", type=int, default=None,
                   help="process at most this many NEW (not-already-done) bins, then exit cleanly -- "
                        "works around a persistent low-level GPU memory leak (confirmed NOT fixed by "
                        "clearing torch/cupy pools; manifests as a raw ASTRA CUDA allocation failure "
                        "after ~30-35 full-resolution reconstructions in one process) by forcing a "
                        "fresh CUDA context via process restart, using the existing checkpoint/resume "
                        "mechanism to continue where it left off. Pair with a shell loop that re-invokes "
                        "this script until done_bins covers all needed bins.")
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--slice_targets", default="4000,10000,16000")
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_or_compute_calibration(a, device):
    if Path(a.calib_path).exists():
        log(f"loading cached calibration -> {a.calib_path}")
        z = np.load(a.calib_path)
        return (torch.from_numpy(z["dark_mean"]).to(device=device, dtype=torch.float32),
                torch.from_numpy(z["white_mean"]).to(device=device, dtype=torch.float32))
    log("computing dark/flat calibration from exchange/data_dark, exchange/data_white ...")
    with h5py.File(a.data_path, "r") as f:
        dark_stack = f["exchange/data_dark"][:].astype(np.float32)
        white_stack = f["exchange/data_white"][:].astype(np.float32)
    dark_mean = dark_stack.mean(axis=0)
    white_mean = white_stack.mean(axis=0)
    Path(a.calib_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.calib_path, dark_mean=dark_mean, white_mean=white_mean, dark_var=dark_stack.var(axis=0))
    return (torch.from_numpy(dark_mean).to(device=device, dtype=torch.float32),
            torch.from_numpy(white_mean).to(device=device, dtype=torch.float32))


def open_tap_memmap(path: Path, shape) -> np.memmap:
    expected_bytes = int(np.prod(shape)) * np.dtype(np.float16).itemsize
    if path.exists() and path.stat().st_size == expected_bytes:
        log(f"  resuming existing tap cache -> {path.name} ({path.stat().st_size/1e9:.2f}GB)")
        return np.memmap(path, dtype=np.float16, mode="r+", shape=shape)
    return np.memmap(path, dtype=np.float16, mode="w+", shape=shape)


def load_progress(done_bins_path: Path):
    if done_bins_path.exists():
        with open(done_bins_path) as f:
            return set(json.load(f))
    return set()


def save_progress(done_bins_path: Path, done_bins: set):
    tmp = done_bins_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(sorted(done_bins), f)
    tmp.replace(done_bins_path)


def load_partial_slices(partial_path: Path):
    if not partial_path.exists():
        return {}, {}
    z = np.load(partial_path, allow_pickle=True)
    target_idx = [int(x) for x in z["target_idx"]]
    slices = {ti: z[f"target_{ti}"] for ti in target_idx}
    proj_samples = {}
    for ti in target_idx:
        if f"proj_{ti}_raw" in z:
            proj_samples[ti] = dict(raw=z[f"proj_{ti}_raw"], transmission=z[f"proj_{ti}_transmission"],
                                    theta=float(z[f"proj_{ti}_theta"]), phase=float(z[f"proj_{ti}_phase"]))
    return slices, proj_samples


def save_partial_slices(partial_path: Path, slices: dict, proj_samples: dict):
    if not slices:
        return
    save_kwargs = {f"target_{ti}": arr for ti, arr in slices.items()}
    for ti, d in proj_samples.items():
        save_kwargs[f"proj_{ti}_raw"] = d["raw"]
        save_kwargs[f"proj_{ti}_transmission"] = d["transmission"]
        save_kwargs[f"proj_{ti}_theta"] = d["theta"]
        save_kwargs[f"proj_{ti}_phase"] = d["phase"]
    tmp = partial_path.with_name(partial_path.stem + ".tmp.npz")  # np.savez appends .npz -- must already end in it
    np.savez(tmp, target_idx=np.array(list(slices.keys())), **save_kwargs)
    tmp.replace(partial_path)


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        log(f"GPU: {props.name}, total_memory={props.total_memory/1e9:.1f}GB")

    target_views = [int(x) for x in a.target_views.split(",")]
    max_groups_per_level = [max(1, math.ceil(t / a.group_size)) for t in target_views]
    T = len(target_views)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"T={T} target_views={target_views} max_groups_per_level={max_groups_per_level} "
        f"group_size={a.group_size}")

    log("loading raw counts FULLY into RAM (uint16, ~12.4GB) ...")
    t_load = time.time()
    with h5py.File(a.data_path, "r") as f:
        sinogram = f["exchange/data"][:]
        theta = f["exchange/theta"][:].astype(np.float64)
    phase = np.loadtxt(a.phase_txt).astype(np.float64)
    log(f"loaded sinogram {sinogram.shape} in {time.time()-t_load:.1f}s")
    n_proj, n_rows, n_pix = sinogram.shape
    vol_shape = (n_rows, n_pix, n_pix)  # det_bin=1 -- native full resolution
    det_shape = (n_rows, n_pix)

    dark_full, flat_full = load_or_compute_calibration(a, device)

    bin_centers = PC.phi_bin_centers(a.n_bins)
    bin_idx = PC.snap_to_bins(phase, a.n_bins)
    needed_bins = np.unique(bin_idx)
    log(f"n_bins={a.n_bins}  {len(needed_bins)}/{a.n_bins} bins have >=1 real projection snapped to them")

    if a.max_bins is not None:
        needed_bins = needed_bins[: a.max_bins]
        log(f"--max_bins={a.max_bins}: smoke-test mode, only processing {len(needed_bins)} bins")

    paths = [out_dir / f"{a.tag}_tap{c}.f16" for c in range(T)]
    mms = [open_tap_memmap(p, (n_proj, n_rows, n_pix)) for p in paths]

    done_bins_path = out_dir / f"{a.tag}_done_bins.json"
    diag_jsonl_path = out_dir / f"{a.tag}_diag.jsonl"
    partial_slices_path = out_dir / f"{a.tag}_slices_partial.npz"
    done_bins = load_progress(done_bins_path)
    filled = np.isin(bin_idx, np.array(sorted(done_bins), dtype=np.int64)) if done_bins else np.zeros(n_proj, dtype=bool)
    if done_bins:
        log(f"RESUMING: {len(done_bins)} bins already completed ({filled.sum()}/{n_proj} projections filled)")

    slice_targets = [int(x) for x in a.slice_targets.split(",") if x.strip() != ""]
    slice_row = n_rows // 2
    bin_to_target = {int(bin_idx[i]): i for i in slice_targets}
    slices, proj_samples = load_partial_slices(partial_slices_path)

    diag = []
    if diag_jsonl_path.exists():
        with open(diag_jsonl_path) as f:
            diag = [json.loads(line) for line in f if line.strip()]

    t0 = time.time()
    crash_at = None
    new_bins_done = 0
    for bi, b in enumerate(needed_bins):
        b = int(b)
        if b in done_bins:
            continue
        if a.bins_per_invocation is not None and new_bins_done >= a.bins_per_invocation:
            log(f"reached --bins_per_invocation={a.bins_per_invocation}, exiting cleanly for a fresh restart")
            break
        target_idx = np.flatnonzero(bin_idx == b)
        target_theta = theta[target_idx]
        phi_c = bin_centers[b]
        bin_diag_records = []

        if b in bin_to_target:
            ti = bin_to_target[b]
            raw_frame = sinogram[ti].astype(np.float32)
            span = np.clip((flat_full - dark_full).cpu().numpy(), 1e-3, None)
            proj_samples[ti] = dict(raw=raw_frame, transmission=(raw_frame - dark_full.cpu().numpy()) / span,
                                    theta=float(theta[ti]), phase=float(phase[ti]))

        for c, (target_h, max_groups) in enumerate(zip(target_views, max_groups_per_level)):
            t1 = time.time()
            try:
                radius = target_h * np.pi / n_proj  # uniform-density formula (confirmed accurate to <1% empirically)
                volumes, n_sel = PC.reconstruct_phi_gated_groups(
                    sinogram, theta, phase, phi_c, radius, dark_full, flat_full, device,
                    det_bin=1, vol_shape=vol_shape, group_size=a.group_size, max_groups=max_groups, clamp=False,
                )
                if not volumes:
                    log(f"  bin {b} tap{c} (target_h={target_h}): NO projections in gate -- skipping")
                    continue
                vol = torch.stack(volumes).mean(dim=0).clamp_min(0.0)
                n_groups_used = len(volumes)
                del volumes
                torch.cuda.empty_cache()

                if b in bin_to_target:
                    ti = bin_to_target[b]
                    slices.setdefault(ti, np.zeros((T, n_pix, n_pix), dtype=np.float32))
                    slices[ti][c] = vol[slice_row].cpu().numpy()
                recon_s = time.time() - t1

                t2 = time.time()
                reproj = PC.reproject_to_counts(vol, target_theta, det_shape, device, dark_full, flat_full)
                reproj_s = time.time() - t2

                block = reproj.cpu().numpy().astype(np.float16)
                mms[c][target_idx] = block
                filled[target_idx] = True

                rec = dict(bin=b, tap=c, target_h=target_h, n_sel=n_sel, n_groups_used=n_groups_used,
                          n_targets=len(target_idx), recon_s=round(recon_s, 3), reproj_s=round(reproj_s, 3),
                          vol_min=float(vol.min()), vol_max=float(vol.max()), vol_mean=float(vol.mean()))
                diag.append(rec)
                bin_diag_records.append(rec)
                if bi % a.log_every == 0 or c == T - 1:
                    log(f"  bin {b} tap{c} (target_h={target_h}, n_sel={n_sel}, groups={n_groups_used}): "
                        f"recon={recon_s:.2f}s reproj={reproj_s:.2f}s "
                        f"vol[min={rec['vol_min']:.4g} max={rec['vol_max']:.4g} mean={rec['vol_mean']:.4g}]")

                del vol, reproj, block
                torch.cuda.empty_cache()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception as e:
                crash_at = dict(bin=b, tap=c, target_h=target_h, error=repr(e))
                log(f"  CRASH at bin={b} tap={c} target_h={target_h}: {e!r}")
                raise

        for mm in mms:
            mm.flush()
        done_bins.add(b)
        new_bins_done += 1
        save_progress(done_bins_path, done_bins)
        with open(diag_jsonl_path, "a") as f:
            for rec in bin_diag_records:
                f.write(json.dumps(rec) + "\n")
        if b in bin_to_target:
            save_partial_slices(partial_slices_path, slices, proj_samples)

        if bi % a.log_every == 0:
            elapsed = time.time() - t0
            log(f"progress {bi+1}/{len(needed_bins)} bins, elapsed={elapsed/60:.1f}min, "
                f"est. total for all {len(needed_bins)} bins = {elapsed/(bi+1)*len(needed_bins)/60:.1f}min")

    for mm in mms:
        mm.flush()
    n_unfilled = int((~filled).sum())
    log(f"DONE. {n_proj - n_unfilled}/{n_proj} projections filled ({n_unfilled} unfilled)")

    meta = dict(
        data_path=a.data_path, phase_txt=a.phase_txt, n_proj=n_proj, det_shape=list(det_shape),
        vol_shape=list(vol_shape), n_bins=a.n_bins, T=T, target_views=target_views,
        max_groups_per_level=max_groups_per_level, group_size=a.group_size,
        needed_bins=[int(x) for x in needed_bins], num_frames=n_proj, mode="phi_context_v2_fullres_groups",
    )
    for p in paths:
        np.savez(str(p) + ".meta.npz", **meta)
    with open(out_dir / f"{a.tag}_diag.json", "w") as fh:
        json.dump(dict(meta=meta, crash_at=crash_at, records=diag), fh, indent=2)

    if slices:
        slice_path = out_dir / f"{a.tag}_slices.npz"
        save_kwargs = {f"target_{ti}": arr for ti, arr in slices.items()}
        for ti, d in proj_samples.items():
            save_kwargs[f"proj_{ti}_raw"] = d["raw"]
            save_kwargs[f"proj_{ti}_transmission"] = d["transmission"]
        np.savez_compressed(
            slice_path, target_views=np.array(target_views), row=slice_row,
            target_idx=np.array(list(slices.keys())),
            target_theta=theta[np.array(list(slices.keys()))],
            target_phase=phase[np.array(list(slices.keys()))],
            **save_kwargs,
        )
        log(f"wrote slices -> {slice_path}")

    log(f"total minutes={(time.time()-t0)/60:.1f}")
    log("SUCCESS" if crash_at is None else "FAILED")


if __name__ == "__main__":
    main()
