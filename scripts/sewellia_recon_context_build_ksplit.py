#!/usr/bin/env python3
"""v3: rebuild the reconstruction-domain anchor as a K-WAY Noise2Inverse-style
split instead of the v2 fixed Noise2Noise PAIR (a1, a2) -- see
project-sewellia-recon-space-n2v.md for the full v2->v3 rationale.

Diagnosis (user, visual inspection of the v2 FINAL checkpoint): even though
v2's a1/a2 pair gave genuinely independent noise, the pairing was FIXED once
per bin and reused identically across all 30 epochs. With only 180 unique
bins and context that near-uniquely identifies which bin a sample comes
from, the network had room to partially memorize a bin-specific correction
that reproduces a1/a2's own shared artifact fingerprint rather than learning
a general denoising operator -- consistent with Noise2Inverse's own stated
motivation (Hendriksen et al. 2020, https://arxiv.org/abs/2001.11801): a
FIXED single input/target combination lets a denoiser learn to reproduce
sample-specific reconstruction artifacts, whereas training across MANY
different combinations of which subset plays input vs. target breaks that.

Fix: split the SAME anchor pool (same phi center, same radius, same
ANCHOR_PAIR_VIEWS=440 target -- i.e. the exact same `exclude_idx` as v2, so
the existing ctx0/ctx1/ctx2 caches remain valid and do NOT need to be
rebuilt) into K=4 disjoint sub-reconstructions instead of 2. The dataset
(SewelliaReconContextKSplitDataset) then randomly re-draws, on EVERY
`__getitem__` call, which of the K subsets is this step's regression target
and averages the other K-1 as the (lower-noise) input -- Noise2Inverse's
"X:1" strategy. Since real projection data can't be redrawn, this is the
correct analogue of the paper's "train across all combinations" idea: we
can't resample new NOISE, but we can resample which precomputed split plays
which role, every training step, so no single (input, target) pairing is
ever memorized.

This script ONLY writes the K sub-anchor arrays (`{tag}_asub{k}.f16`,
k=0..K-1) -- it reuses the EXACT SAME per-bin anchor gate/radius as v2's
build (`sewellia_recon_context_build.py`), so `anchor_exclude_idx` (and
therefore the already-built ctx0/ctx1/ctx2 caches) is byte-for-byte
unchanged. Old `{tag}_a1.f16`/`{tag}_a2.f16` are left in place (not read,
not overwritten) until the new K-split cache is validated -- delete them by
hand afterward to reclaim ~2x17.3GB.

    python scripts/sewellia_recon_context_build_ksplit.py --max_bins 5   # smoke test
    python scripts/sewellia_recon_context_build_ksplit.py                # full 180-bin build
"""
from __future__ import annotations

import argparse
import json
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
OUT_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/recon_context"
CALIB_PATH = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/sewellia_real_calibration.npz"

ANCHOR_PAIR_VIEWS = 440  # MUST match v2's build exactly -- same radius -> same exclude_idx -> ctx caches stay valid
K_SPLITS = 4
SPLIT_GROUP_SIZE = 100  # K*100=400 <= observed n_sel floor of ~438 with comfortable margin (see v2 smoke test)
CROP_ROWS = 145


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=DATA_PATH)
    p.add_argument("--phase_txt", default=PHASE_TXT)
    p.add_argument("--calib_path", default=CALIB_PATH)
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--tag", default="sewellia_reconctx")
    p.add_argument("--n_bins", type=int, default=180)
    p.add_argument("--anchor_pair_views", type=int, default=ANCHOR_PAIR_VIEWS,
                   help="MUST match the v2 build's value -- determines the gate radius, and therefore "
                        "anchor_exclude_idx, which must stay identical to what ctx0/ctx1/ctx2 were built with")
    p.add_argument("--k_splits", type=int, default=K_SPLITS)
    p.add_argument("--split_group_size", type=int, default=SPLIT_GROUP_SIZE)
    p.add_argument("--crop_rows", type=int, default=CROP_ROWS)
    p.add_argument("--max_bins", type=int, default=None)
    p.add_argument("--bins_per_invocation", type=int, default=None)
    p.add_argument("--log_every", type=int, default=5)
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_or_compute_calibration(a, device):
    if Path(a.calib_path).exists():
        log(f"loading cached calibration -> {a.calib_path}")
        z = np.load(a.calib_path)
        return (torch.from_numpy(z["dark_mean"]).to(device=device, dtype=torch.float32),
                torch.from_numpy(z["white_mean"]).to(device=device, dtype=torch.float32))
    raise FileNotFoundError(f"expected cached calibration at {a.calib_path} (built by the v2 script)")


def open_bin_memmap(path: Path, shape) -> np.memmap:
    expected_bytes = int(np.prod(shape)) * np.dtype(np.float16).itemsize
    if path.exists() and path.stat().st_size == expected_bytes:
        log(f"  resuming existing cache -> {path.name} ({path.stat().st_size/1e9:.2f}GB)")
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


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} K_splits={a.k_splits} split_group_size={a.split_group_size} "
        f"(total {a.k_splits * a.split_group_size} of ~{a.anchor_pair_views} anchor pool views used)")

    log("loading raw counts FULLY into RAM (uint16, ~12.4GB) ...")
    t_load = time.time()
    with h5py.File(a.data_path, "r") as f:
        sinogram_full = f["exchange/data"][:]
        theta = f["exchange/theta"][:].astype(np.float64)
    phase = np.loadtxt(a.phase_txt).astype(np.float64)
    log(f"loaded sinogram {sinogram_full.shape} in {time.time()-t_load:.1f}s")
    n_proj, n_rows_full, n_pix = sinogram_full.shape

    row_center = n_rows_full // 2
    row_lo = row_center - a.crop_rows // 2
    row_hi = row_lo + a.crop_rows
    sinogram = np.ascontiguousarray(sinogram_full[:, row_lo:row_hi, :])
    del sinogram_full
    vol_shape = (a.crop_rows, n_pix, n_pix)

    dark_full, flat_full = load_or_compute_calibration(a, device)
    dark = dark_full[row_lo:row_hi].contiguous()
    flat = flat_full[row_lo:row_hi].contiguous()

    bin_centers = PC.phi_bin_centers(a.n_bins)
    bin_idx = PC.snap_to_bins(phase, a.n_bins)
    needed_bins = np.unique(bin_idx)
    log(f"n_bins={a.n_bins}  {len(needed_bins)}/{a.n_bins} bins have >=1 real projection snapped to them")

    if a.max_bins is not None:
        needed_bins = needed_bins[: a.max_bins]
        log(f"--max_bins={a.max_bins}: smoke-test mode, only processing {len(needed_bins)} bins")

    asub_paths = [Path(a.out_dir) / f"{a.tag}_asub{k}.f16" for k in range(a.k_splits)]
    bin_shape = (a.n_bins, a.crop_rows, n_pix, n_pix)
    asub_mms = [open_bin_memmap(p, bin_shape) for p in asub_paths]

    done_bins_path = Path(a.out_dir) / f"{a.tag}_ksplit_done_bins.json"
    diag_jsonl_path = Path(a.out_dir) / f"{a.tag}_ksplit_diag.jsonl"
    done_bins = load_progress(done_bins_path)
    if done_bins:
        log(f"RESUMING: {len(done_bins)}/{len(needed_bins)} bins already completed")

    t0 = time.time()
    new_bins_done = 0
    for bi, b in enumerate(needed_bins):
        b = int(b)
        if b in done_bins:
            continue
        if a.bins_per_invocation is not None and new_bins_done >= a.bins_per_invocation:
            log(f"reached --bins_per_invocation={a.bins_per_invocation}, exiting cleanly for a fresh restart")
            break
        phi_c = bin_centers[b]

        radius_anchor = a.anchor_pair_views * np.pi / n_proj  # MUST match v2 exactly
        t1 = time.time()
        volumes, n_sel = PC.reconstruct_phi_gated_groups(
            sinogram, theta, phase, phi_c, radius_anchor, dark, flat, device,
            det_bin=1, vol_shape=vol_shape, group_size=a.split_group_size, max_groups=a.k_splits, clamp=True,
        )
        if len(volumes) < a.k_splits:
            log(f"  bin {b}: only {len(volumes)}/{a.k_splits} disjoint splits available "
                f"(n_sel={n_sel}, need >= {a.k_splits*a.split_group_size}) -- skipping bin")
            continue
        for k in range(a.k_splits):
            asub_mms[k][b] = volumes[k].cpu().numpy().astype(np.float16)
        recon_s = time.time() - t1
        rec = dict(bin=b, n_sel=n_sel, k_splits=a.k_splits, split_group_size=a.split_group_size,
                  recon_s=round(recon_s, 3))
        del volumes
        torch.cuda.empty_cache()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()

        for mm in asub_mms:
            mm.flush()
        done_bins.add(b)
        new_bins_done += 1
        save_progress(done_bins_path, done_bins)
        with open(diag_jsonl_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

        if bi % a.log_every == 0:
            elapsed = time.time() - t0
            log(f"progress {bi+1}/{len(needed_bins)} bins (n_sel={n_sel}), elapsed={elapsed/60:.1f}min, "
                f"est. total for all {len(needed_bins)} bins = {elapsed/(bi+1)*len(needed_bins)/60:.1f}min")

    for mm in asub_mms:
        mm.flush()
    log(f"DONE. {len(done_bins)}/{len(needed_bins)} bins completed this run")

    meta = dict(
        data_path=a.data_path, phase_txt=a.phase_txt, n_proj=n_proj, n_rows_full=n_rows_full, n_pix=n_pix,
        row_lo=row_lo, row_hi=row_hi, crop_rows=a.crop_rows, vol_shape=list(vol_shape), n_bins=a.n_bins,
        k_splits=a.k_splits, split_group_size=a.split_group_size, anchor_pair_views=a.anchor_pair_views,
        needed_bins=[int(x) for x in needed_bins],
        mode="recon_context_v3_noise2inverse_ksplit",
    )
    for p in asub_paths:
        np.savez(str(p) + ".meta.npz", **meta)

    log(f"total minutes={(time.time()-t0)/60:.1f}")
    log("SUCCESS")


if __name__ == "__main__":
    main()
