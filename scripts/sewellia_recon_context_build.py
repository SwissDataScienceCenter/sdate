#!/usr/bin/env python3
"""Build the reconstruction-domain N2V+context cache for the Sewellia
lineolata real full acquisition -- a PARALLEL, INDEPENDENT track alongside
the projection-domain N2V+context model (owned by another agent this
session -- this script does not read or write anything under
`phi_context/`; see project memory project-sewellia-recon-space-n2v.md).

Unlike the projection-domain cache (which reprojects each context level back
to real target ANGLES and stores PROJECTIONS), this one stores the
reconstructed VOLUMES themselves (cropped in the row/z axis), because the
whole point of this track is to denoise reconstructed SLICES, not
projections. No `reproject_to_counts` step anywhere in this script.

Design (confirmed with the user):
- T=3 context levels (not 5): target real-view counts (900, 1800, 3600),
  each up to (1, 2, 4) disjoint 900-view groups, averaged (clamp=False per
  group, single clamp_min(0) on the average).
- A 180-bin phi grid (not 360).
- Detector rows cropped to a 145-row window CENTERED on the full detector's
  own center (row ~290 of 580) before reconstruction -- not a top/bottom
  half. Because the crop is symmetric about the true center, the cropped
  volume's own local center coincides with the true z=290 (the row the
  earlier zscan confirmed matches the PeriodRecon paper's anatomy) with no
  extra geometry-offset handling needed. In-plane (axial) resolution stays
  native 576x576 -- only the z-extent shrinks 4x (580 -> 145).
- ALSO caches an independent NOISE2NOISE ANCHOR PAIR (a1, a2): gate to
  ~400 views, split into 2 disjoint ~200-view halves (reusing the exact
  same disjoint-groups mechanism as the context levels), reconstruct each
  independently. a1 is the model's input, a2 is its training target --
  genuinely independent noise realizations of the same underlying structure,
  no blind-spot masking needed downstream (see v1 postmortem below).
- Context levels EXCLUDE the anchor pair's ~400 views (`exclude_idx` in
  `phi_context.reconstruct_phi_gated_groups`) -- anchor and context gates
  are nested disks around the same phi center, so without this exclusion
  the context would silently contain the anchor's own photon draws,
  breaking the independence the whole design depends on.

**v1 postmortem (superseded by this v2 design)**: the original version
cached a SINGLE ~200-view anchor and used blind-spot N2V masking
(BaselineN2VLoss, window=5) instead of a genuine independent pair. Two bugs
found after training showed near-zero denoising (0.98 input/output
correlation at step 19k): (1) the 5x5 swap window drew replacement pixels
from within the ~2px noise-correlation length measured by
scripts/sewellia_recon_noise_autocorr_check.py, leaking the true value back
in; (2) the context gates being nested supersets of the anchor's gate meant
ctx0 in particular contained 100% of the anchor's own ~200 views. Both are
fixed by the noise2noise-anchor-pair + exclude_idx design above -- see
project-sewellia-recon-space-n2v.md.

Storage: 3 context arrays + 2 anchor arrays (a1, a2), each
(n_bins, 145, 576, 576) float16 ~= 17.3GB -> ~86.6GB total. See
project-sewellia-recon-space-n2v.md for the full shared-mount storage ledger.

Per-bin compute is cheap (rows are 4x smaller than the projection-domain
build) -- still uses --bins_per_invocation (same self-restarting-process
workaround as sewellia_real_phi_context_v2.py) since the ASTRA leak is about
repeated allocations, not size.

    python scripts/sewellia_recon_context_build.py --max_bins 3       # smoke test
    python scripts/sewellia_recon_context_build.py                    # full 180-bin build
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
OUT_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/recon_context"
CALIB_PATH = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/sewellia_real_calibration.npz"

TARGET_VIEWS = [900, 1800, 3600]
GROUP_SIZE = 900
ANCHOR_PAIR_VIEWS = 440  # target with ~10% margin -- gate right at 400 (2x200) undershot to n_sel=399
# on 3/5 smoke-test bins (the uniform-density radius formula is only approximate), silently
# skipping the whole bin (context included). 440 gives comfortable headroom above the 400 floor.
ANCHOR_GROUP_SIZE = 200
CROP_ROWS = 145  # centered on the full detector's own center (~row 290 of 580)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=DATA_PATH)
    p.add_argument("--phase_txt", default=PHASE_TXT)
    p.add_argument("--calib_path", default=CALIB_PATH)
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--tag", default="sewellia_reconctx")
    p.add_argument("--n_bins", type=int, default=180)
    p.add_argument("--group_size", type=int, default=GROUP_SIZE)
    p.add_argument("--target_views", default=",".join(str(x) for x in TARGET_VIEWS),
                   help="comma-separated target real-view count per context level -- T = len(...)")
    p.add_argument("--anchor_pair_views", type=int, default=ANCHOR_PAIR_VIEWS,
                   help="total views gathered for the anchor pair, split into two disjoint halves (a1, a2)")
    p.add_argument("--anchor_group_size", type=int, default=ANCHOR_GROUP_SIZE,
                   help="views per anchor half (a1/a2) -- anchor_pair_views should be 2x this")
    p.add_argument("--crop_rows", type=int, default=CROP_ROWS,
                   help="number of detector rows to keep, centered on the full detector's own center")
    p.add_argument("--max_bins", type=int, default=None,
                   help="cap the number of distinct phi-bins processed, for a quick smoke test")
    p.add_argument("--bins_per_invocation", type=int, default=None,
                   help="process at most this many NEW bins then exit cleanly -- see module docstring")
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--slice_target_bins", default=None,
                   help="comma-separated bin indices to snapshot (local center slice, all levels+anchor) "
                        "for visual inspection; default: 8 bins evenly spaced across n_bins")
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
        f"group_size={a.group_size} anchor_pair_views={a.anchor_pair_views} "
        f"anchor_group_size={a.anchor_group_size} crop_rows={a.crop_rows}")

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
    assert 0 <= row_lo and row_hi <= n_rows_full, f"crop [{row_lo}:{row_hi}) out of [0,{n_rows_full})"
    log(f"row crop [{row_lo}:{row_hi}) of [0,{n_rows_full}) -- centered on row {row_center} "
        f"(local center row {a.crop_rows//2} <-> global row {row_lo + a.crop_rows//2})")

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

    ctx_paths = [out_dir / f"{a.tag}_ctx{c}.f16" for c in range(T)]
    a1_path = out_dir / f"{a.tag}_a1.f16"
    a2_path = out_dir / f"{a.tag}_a2.f16"
    bin_shape = (a.n_bins, a.crop_rows, n_pix, n_pix)
    ctx_mms = [open_bin_memmap(p, bin_shape) for p in ctx_paths]
    a1_mm = open_bin_memmap(a1_path, bin_shape)
    a2_mm = open_bin_memmap(a2_path, bin_shape)

    done_bins_path = out_dir / f"{a.tag}_done_bins.json"
    diag_jsonl_path = out_dir / f"{a.tag}_diag.jsonl"
    done_bins = load_progress(done_bins_path)
    if done_bins:
        log(f"RESUMING: {len(done_bins)}/{len(needed_bins)} bins already completed")

    if a.slice_target_bins:
        slice_target_bins = [int(x) for x in a.slice_target_bins.split(",")]
    else:
        n_snap = min(8, len(needed_bins))
        slice_target_bins = [int(x) for x in needed_bins[np.linspace(0, len(needed_bins) - 1, n_snap, dtype=int)]]
    local_center = a.crop_rows // 2
    slice_snapshots = {}

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
        phi_c = bin_centers[b]
        bin_diag_records = []
        snap_this_bin = b in slice_target_bins

        try:
            # -- anchor pair: gate to ~anchor_pair_views, split into 2 disjoint
            #    ~anchor_group_size halves (a1, a2) -- independent noise, same
            #    underlying structure (Noise2Noise). Reuses the exact same
            #    disjoint-groups mechanism as the context levels below.
            t1 = time.time()
            radius_anchor = a.anchor_pair_views * np.pi / n_proj
            anchor_mask = PC.phi_gate_mask(phase, phi_c, radius_anchor)
            anchor_exclude_idx = np.flatnonzero(anchor_mask)  # excluded from ALL context gates below
            volumes_a, n_sel_a = PC.reconstruct_phi_gated_groups(
                sinogram, theta, phase, phi_c, radius_anchor, dark, flat, device,
                det_bin=1, vol_shape=vol_shape, group_size=a.anchor_group_size, max_groups=2, clamp=True,
            )
            if len(volumes_a) < 2:
                log(f"  bin {b} anchor: only {len(volumes_a)} disjoint half(s) available "
                    f"(n_sel={n_sel_a}, need >= {2*a.anchor_group_size}) -- skipping bin")
                continue
            vol_a1, vol_a2 = volumes_a[0], volumes_a[1]
            a1_mm[b] = vol_a1.cpu().numpy().astype(np.float16)
            a2_mm[b] = vol_a2.cpu().numpy().astype(np.float16)
            recon_s = time.time() - t1
            rec = dict(bin=b, tap="anchor_pair", target_h=a.anchor_pair_views, n_sel=n_sel_a,
                      recon_s=round(recon_s, 3), vol_min=float(min(vol_a1.min(), vol_a2.min())),
                      vol_max=float(max(vol_a1.max(), vol_a2.max())))
            if snap_this_bin:
                slice_snapshots.setdefault(b, {})["a1"] = vol_a1[local_center].cpu().numpy()
                slice_snapshots.setdefault(b, {})["a2"] = vol_a2[local_center].cpu().numpy()
            del volumes_a, vol_a1, vol_a2
            torch.cuda.empty_cache()
            bin_diag_records.append(rec)

            # -- T context levels: disjoint-groups, averaged, EXCLUDING the anchor pair's views --
            for c, (target_h, max_groups) in enumerate(zip(target_views, max_groups_per_level)):
                t1 = time.time()
                radius = target_h * np.pi / n_proj
                volumes, n_sel = PC.reconstruct_phi_gated_groups(
                    sinogram, theta, phase, phi_c, radius, dark, flat, device,
                    det_bin=1, vol_shape=vol_shape, group_size=a.group_size, max_groups=max_groups, clamp=False,
                    exclude_idx=anchor_exclude_idx,
                )
                if not volumes:
                    log(f"  bin {b} ctx{c} (target_h={target_h}): NO projections in gate -- skipping")
                    continue
                vol = torch.stack(volumes).mean(dim=0).clamp_min(0.0)
                n_groups_used = len(volumes)
                del volumes
                torch.cuda.empty_cache()

                ctx_mms[c][b] = vol.cpu().numpy().astype(np.float16)
                if snap_this_bin:
                    slice_snapshots.setdefault(b, {})[f"ctx{c}"] = vol[local_center].cpu().numpy()
                recon_s = time.time() - t1
                rec = dict(bin=b, tap=f"ctx{c}", target_h=target_h, n_sel=n_sel, n_groups_used=n_groups_used,
                          recon_s=round(recon_s, 3), vol_min=float(vol.min()), vol_max=float(vol.max()),
                          vol_mean=float(vol.mean()))
                bin_diag_records.append(rec)

                del vol
                torch.cuda.empty_cache()
                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

            if bi % a.log_every == 0:
                log(f"  bin {b}: anchor_pair n_sel={n_sel_a} (excluded {anchor_exclude_idx.size} from context) " +
                    " ".join(f"ctx{c}[n_sel={r['n_sel']},groups={r.get('n_groups_used','-')}]"
                            for c, r in enumerate(bin_diag_records[1:])))
        except Exception as e:
            crash_at = dict(bin=b, error=repr(e))
            log(f"  CRASH at bin={b}: {e!r}")
            raise

        for mm in ctx_mms:
            mm.flush()
        a1_mm.flush()
        a2_mm.flush()
        done_bins.add(b)
        new_bins_done += 1
        save_progress(done_bins_path, done_bins)
        with open(diag_jsonl_path, "a") as f:
            for rec in bin_diag_records:
                f.write(json.dumps(rec) + "\n")

        if bi % a.log_every == 0:
            elapsed = time.time() - t0
            log(f"progress {bi+1}/{len(needed_bins)} bins, elapsed={elapsed/60:.1f}min, "
                f"est. total for all {len(needed_bins)} bins = {elapsed/(bi+1)*len(needed_bins)/60:.1f}min")

    for mm in ctx_mms:
        mm.flush()
    a1_mm.flush()
    a2_mm.flush()
    log(f"DONE. {len(done_bins)}/{len(needed_bins)} bins completed this run")

    meta = dict(
        data_path=a.data_path, phase_txt=a.phase_txt, n_proj=n_proj, n_rows_full=n_rows_full, n_pix=n_pix,
        row_lo=row_lo, row_hi=row_hi, crop_rows=a.crop_rows, vol_shape=list(vol_shape), n_bins=a.n_bins, T=T,
        target_views=target_views, max_groups_per_level=max_groups_per_level, group_size=a.group_size,
        anchor_pair_views=a.anchor_pair_views, anchor_group_size=a.anchor_group_size,
        needed_bins=[int(x) for x in needed_bins],
        mode="recon_context_v2_noise2noise_anchor",
    )
    for p in ctx_paths + [a1_path, a2_path]:
        np.savez(str(p) + ".meta.npz", **meta)

    if slice_snapshots:
        slice_path = out_dir / f"{a.tag}_slices.npz"
        save_kwargs = {}
        for b, d in slice_snapshots.items():
            for k, arr in d.items():
                save_kwargs[f"bin{b}_{k}"] = arr
        np.savez_compressed(slice_path, bins=np.array(list(slice_snapshots.keys())),
                            local_center=local_center, **save_kwargs)
        log(f"wrote slice snapshots -> {slice_path}")

    log(f"total minutes={(time.time()-t0)/60:.1f}")
    log("SUCCESS" if crash_at is None else "FAILED")


if __name__ == "__main__":
    main()
