#!/usr/bin/env python3
"""Build the T-tap phi-gated joint-FBP context cache for the REAL full Sewellia
lineolata acquisition (20000 x 580 x 576 raw counts + real darks/flats),
superseding the small precorrected-preview prototype (scripts/sewellia_phi_context_prototype.py).

Two differences from the preview prototype that matter enough to need a
separate script rather than just new CLI args:

1. REAL per-pixel dark/flat calibration (not the trivial dark=0,flat=1 used
   for the precorrected preview) -- computed once from exchange/data_dark
   (50 frames) and exchange/data_white (100 frames), applied at full
   resolution (raw counts -> attenuation) BEFORE any detector binning, exactly
   mirroring the wunderkerze2 flat/dark convention.

2. Storage AND GPU memory: at FULL resolution (580x576, float16, T=11,
   n_proj=20000) this cache would be ~147GB -- more than the ENTIRE available
   quota on the shared NFS mount this data lives on (/mydata/sdate/shared,
   aliased at /myhome/data/..., 83GB free as of this run). Worse, the
   ramp-filter FFT inside FBP needs GPU memory scaling with n_sel (view
   count) x det_shape -- confirmed empirically to OOM at det_bin=2 (290x288)
   with just n_sel=3200 views, nowhere near the widest pct=50% gate's
   max_views=12000. --det_bin 4 reconstructs AND reprojects at QUARTER
   resolution (145x144) -- the cached tap size drops ~16x (to ~9GB total,
   comfortably inside the shared quota) and the ramp-filter buffer ~4x /
   backprojection volume buffer ~64x, which empirically fits max_views=12000.
   The dark/flat maps are average-pooled by the same factor for the
   (already-approximate, auxiliary-conditioning) counts round-trip on the way
   back out of reconstruct_phi_gated/reproject_to_counts. Training-time code
   must upsample these low-res taps back to the central frame's native
   580x576 before concatenation (not done here -- this script only builds
   the cache).

The raw 13GB exchange/data array is loaded FULLY into RAM once (as uint16,
~12.4GB) rather than repeatedly fancy-indexed off NFS -- avoids re-reading
gigabytes over the network for every one of the ~360*T gated reconstructions.

    python scripts/sewellia_real_phi_context.py --max_bins 3   # smoke test
    python scripts/sewellia_real_phi_context.py                # full 360-bin build
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
import torch.nn.functional as F

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")
os.environ["PATH"] = f"/myhome/bin:{os.environ.get('PATH', '')}"

from sdate.tr_diffusion import phi_context as PC

FULL_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01"
DATA_PATH = f"{FULL_DIR}/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5"
PHASE_TXT = f"{FULL_DIR}/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01_sound_phase.txt"
OUT_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/phi_context"
CALIB_PATH = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/sewellia_real_calibration.npz"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=DATA_PATH)
    p.add_argument("--phase_txt", default=PHASE_TXT)
    p.add_argument("--calib_path", default=CALIB_PATH)
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--tag", default="sewellia_real_phictx")
    p.add_argument("--n_bins", type=int, default=360)
    p.add_argument("--pcts", default=",".join(str(x) for x in PC.DEFAULT_PCTS),
                   help="comma-separated fractions (NOT percent) -- T = len(pcts)")
    p.add_argument("--det_bin", type=int, default=4,
                   help="detector/volume binning for BOTH reconstruction and the cached tap "
                        "resolution (580x576 -> 145x144 at the default 4). Originally tried "
                        "det_bin=2 (290x288) for disk-quota reasons, but the ramp-filter FFT's "
                        "GPU memory scales with n_sel (view count) x det_shape -- confirmed "
                        "empirically to OOM (CuPy alloc of 7.6GB failed, 36.9GB already "
                        "allocated) at just n_sel=3200 (pct=8% gate) with det_bin=2, well before "
                        "the widest pct=50% gate (n_sel up to max_views=12000). det_bin=4 cuts "
                        "the ramp-filter buffer ~4x and the 3D backprojection volume buffer ~64x "
                        "(voxel count scales as det_bin^-3), comfortably fitting max_views=12000 "
                        "at the widest gate.")
    p.add_argument("--max_views", type=int, default=12000,
                   help="ASTRA CUDA backend view-count ceiling safety net, see phi_context.py")
    p.add_argument("--max_bins", type=int, default=None,
                   help="cap the number of distinct phi-bins processed, for a quick smoke test")
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
    dark_var = dark_stack.var(axis=0)
    Path(a.calib_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.calib_path, dark_mean=dark_mean, white_mean=white_mean, dark_var=dark_var)
    log(f"saved calibration -> {a.calib_path}")
    return (torch.from_numpy(dark_mean).to(device=device, dtype=torch.float32),
            torch.from_numpy(white_mean).to(device=device, dtype=torch.float32))


def bin_map(x: torch.Tensor, det_bin: int) -> torch.Tensor:
    """Average-pool a (H, W) calibration map by det_bin, matching bin_detector's
    behaviour on the projection data (same avg-pool convention, see reconstruct.py)."""
    if det_bin <= 1:
        return x
    return F.avg_pool2d(x.view(1, 1, *x.shape), kernel_size=det_bin, stride=det_bin).view(
        x.shape[0] // det_bin, x.shape[1] // det_bin)


def open_tap_memmap(path: Path, shape) -> np.memmap:
    """Resume an existing tap cache file in-place (mode='r+') if it already has the
    right size (a previous pod incarnation got partway through before being killed --
    see main()'s bin-level checkpointing), otherwise create fresh (mode='w+').

    This job runs on a cluster where the pod has been observed to be killed and
    auto-restarted by an external sidecar (an S3-mount container, "s3-fs-0") on a
    roughly fixed ~6.5min cadence UNRELATED to this script's own correctness --
    confirmed by seeing it happen even after fixing the real bugs (GPU OOM, then an
    UnboundLocalError) that caused the FIRST few crashes. Since reloading the raw
    13GB sinogram alone costs ~5.5min, resuming (not restarting from bin 0) is the
    difference between this job ever finishing and looping forever on the first few
    bins.
    """
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
    tmp.replace(done_bins_path)  # atomic on the same filesystem -- never a half-written file


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
    # np.savez silently APPENDS ".npz" to any filename that doesn't already end in
    # ".npz" -- naming the tmp file "*.npz.tmp" (ending in ".tmp") meant the file
    # numpy actually wrote was "*.npz.tmp.npz", so the rename below always failed
    # to find its source and this function silently lost every target's slice data
    # (confirmed: the full 360-bin run finished with zero recovered slices because
    # of exactly this). Name the tmp file so it ALREADY ends in ".npz".
    tmp = partial_path.with_name(partial_path.stem + ".tmp.npz")
    np.savez(tmp, target_idx=np.array(list(slices.keys())), **save_kwargs)
    tmp.replace(partial_path)


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        log(f"GPU: {props.name}, total_memory={props.total_memory/1e9:.1f}GB")
    pcts = [float(x) for x in a.pcts.split(",")]
    T = len(pcts)
    radii = PC.radii_from_pcts(pcts)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("loading raw counts FULLY into RAM (uint16, ~12.4GB) ...")
    t_load = time.time()
    with h5py.File(a.data_path, "r") as f:
        sinogram = f["exchange/data"][:]  # (20000, 580, 576) uint16
        theta = f["exchange/theta"][:].astype(np.float64)
    phase = np.loadtxt(a.phase_txt).astype(np.float64)
    log(f"loaded sinogram {sinogram.shape} dtype={sinogram.dtype} in {time.time()-t_load:.1f}s, "
        f"theta range [{theta.min():.2f},{theta.max():.2f}]deg, "
        f"phase range [{phase.min():.3f},{phase.max():.3f}]rad")
    n_proj, n_rows, n_pix = sinogram.shape
    assert phase.shape[0] == n_proj, f"phase length {phase.shape[0]} != n_proj {n_proj}"

    dark_full, flat_full = load_or_compute_calibration(a, device)
    dark_bin = bin_map(dark_full, a.det_bin)
    flat_bin = bin_map(flat_full, a.det_bin)
    rows_bin, pix_bin = n_rows // a.det_bin, n_pix // a.det_bin
    log(f"calibration: dark_full mean={dark_full.mean():.2f} flat_full mean={flat_full.mean():.2f} "
        f"-> binned to ({rows_bin},{pix_bin}) dark_bin mean={dark_bin.mean():.2f} flat_bin mean={flat_bin.mean():.2f}")

    bin_centers = PC.phi_bin_centers(a.n_bins)
    bin_idx = PC.snap_to_bins(phase, a.n_bins)
    needed_bins = np.unique(bin_idx)
    log(f"n_bins={a.n_bins}  T={T}  pcts={pcts}  det_bin={a.det_bin} -> tap resolution ({rows_bin},{pix_bin})")
    log(f"{len(needed_bins)}/{a.n_bins} bins have >=1 real projection snapped to them "
        f"(mean {n_proj/len(needed_bins):.1f} proj/bin)")

    if a.max_bins is not None:
        needed_bins = needed_bins[: a.max_bins]
        log(f"--max_bins={a.max_bins}: smoke-test mode, only processing {len(needed_bins)} bins")

    vol_shape = (rows_bin, pix_bin, pix_bin)
    det_shape_bin = (rows_bin, pix_bin)

    paths = [out_dir / f"{a.tag}_tap{c}.f16" for c in range(T)]
    mms = [open_tap_memmap(p, (n_proj, rows_bin, pix_bin)) for p in paths]

    done_bins_path = out_dir / f"{a.tag}_done_bins.json"
    diag_jsonl_path = out_dir / f"{a.tag}_diag.jsonl"
    partial_slices_path = out_dir / f"{a.tag}_slices_partial.npz"
    done_bins = load_progress(done_bins_path)
    filled = np.isin(bin_idx, np.array(sorted(done_bins), dtype=np.int64)) if done_bins else np.zeros(n_proj, dtype=bool)
    if done_bins:
        log(f"RESUMING: {len(done_bins)} bins already completed in a previous incarnation of this job "
            f"({filled.sum()}/{n_proj} projections already filled) -- skipping them.")

    slice_targets = [int(x) for x in a.slice_targets.split(",") if x.strip() != ""]
    slice_row = rows_bin // 2
    bin_to_target = {int(bin_idx[i]): i for i in slice_targets}
    slices, proj_samples = load_partial_slices(partial_slices_path)

    diag = []
    if diag_jsonl_path.exists():
        with open(diag_jsonl_path) as f:
            diag = [json.loads(line) for line in f if line.strip()]
    t0 = time.time()
    crash_at = None
    for bi, b in enumerate(needed_bins):
        b = int(b)
        if b in done_bins:
            continue
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

        for c, (pct, r) in enumerate(zip(pcts, radii)):
            t1 = time.time()
            try:
                max_views = a.max_views if a.max_views and a.max_views > 0 else None
                vol, n_sel = PC.reconstruct_phi_gated(
                    sinogram, theta, phase, phi_c, r, dark_full, flat_full, device,
                    det_bin=a.det_bin, vol_shape=vol_shape, max_views=max_views,
                )
                if vol is None:
                    log(f"  bin {b} tap{c} (pct={pct:.0%}): NO projections in gate -- skipping")
                    continue
                if b in bin_to_target:
                    ti = bin_to_target[b]
                    slices.setdefault(ti, np.zeros((T, pix_bin, pix_bin), dtype=np.float32))
                    slices[ti][c] = vol[slice_row].cpu().numpy()
                n_used = min(n_sel, max_views) if max_views else n_sel
                subsampled = n_used < n_sel
                n_nan = int(torch.isnan(vol).sum().item())
                n_inf = int(torch.isinf(vol).sum().item())
                recon_s = time.time() - t1

                t2 = time.time()
                reproj = PC.reproject_to_counts(vol, target_theta, det_shape_bin, device, dark_bin, flat_bin)
                reproj_s = time.time() - t2

                block = reproj.cpu().numpy().astype(np.float16)
                mms[c][target_idx] = block
                filled[target_idx] = True

                rec = dict(bin=b, tap=c, pct=pct, n_sel=n_sel, n_used=n_used, subsampled=subsampled,
                          n_targets=len(target_idx),
                          recon_s=round(recon_s, 3), reproj_s=round(reproj_s, 3),
                          vol_min=float(vol.min()), vol_max=float(vol.max()), vol_mean=float(vol.mean()),
                          n_nan=n_nan, n_inf=n_inf)
                diag.append(rec)
                bin_diag_records.append(rec)
                tag_sub = f" [subsampled {n_sel}->{n_used}]" if subsampled else ""
                if n_nan or n_inf:
                    log(f"  !! bin {b} tap{c} (pct={pct:.0%}, n_sel={n_sel}){tag_sub}: NaN/Inf! {rec}")
                elif bi % a.log_every == 0 or pct == max(pcts):
                    log(f"  bin {b} tap{c} (pct={pct:.0%}, n_sel={n_sel}){tag_sub}: "
                        f"recon={recon_s:.2f}s reproj={reproj_s:.2f}s "
                        f"vol[min={rec['vol_min']:.4g} max={rec['vol_max']:.4g} mean={rec['vol_mean']:.4g}]")

                # PyTorch and CuPy each keep their OWN caching allocator pool on the same
                # physical GPU; lamino.py's fbp_reconstruction_masked frees CuPy's pool at the
                # end of a successful call, but the reprojection path (reproject_at_angles /
                # build_lamino_projector, used every tap here) does not -- confirmed
                # empirically: without explicitly freeing BOTH pools here, allocation climbs
                # tap over tap within a bin (32GB already resident by tap7/pct=23% before that
                # tap's own ~5.5GB ramp-filter buffer even asks for space) and OOMs well under
                # the GPU's real 80GB capacity.
                del vol, reproj, block
                torch.cuda.empty_cache()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception as e:
                crash_at = dict(bin=b, tap=c, pct=pct, error=repr(e))
                log(f"  CRASH at bin={b} tap={c} pct={pct:.0%}: {e!r}")
                raise

        # Checkpoint after every fully-completed bin (all T taps written) -- the pod on
        # this cluster gets killed and auto-restarted by an external sidecar on a ~6.5min
        # cadence unrelated to this script (see open_tap_memmap's docstring), so progress
        # must survive a restart at bin granularity, not just at the very end of the run.
        for mm in mms:
            mm.flush()
        done_bins.add(b)
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
        data_path=a.data_path, phase_txt=a.phase_txt, n_proj=n_proj,
        det_shape_full=[n_rows, n_pix], det_shape=list(det_shape_bin), vol_shape=list(vol_shape),
        n_bins=a.n_bins, T=T, pcts=pcts, det_bin=a.det_bin,
        dark_bin_mean=float(dark_bin.mean()), flat_bin_mean=float(flat_bin.mean()),
        max_views=a.max_views, max_bins=a.max_bins, needed_bins=[int(x) for x in needed_bins], first_index=0,
        num_frames=n_proj, mode="phi_context_real",
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
            slice_path, pcts=np.array(pcts), row=slice_row,
            target_idx=np.array(list(slices.keys())),
            target_theta=theta[np.array(list(slices.keys()))],
            target_phase=phase[np.array(list(slices.keys()))],
            **save_kwargs,
        )
        log(f"wrote {len(slices)} target(s) x {T} reconstructed slice(s) + raw/transmission samples -> {slice_path}")

    total_min = (time.time() - t0) / 60
    log(f"wrote {T} tap memmaps ({rows_bin}x{pix_bin}) to {out_dir} ({a.tag}_tap0..{T-1}.f16), "
        f"diag -> {a.tag}_diag.json")
    log(f"total minutes={total_min:.1f}")
    log("SUCCESS" if crash_at is None else "FAILED")


if __name__ == "__main__":
    main()
