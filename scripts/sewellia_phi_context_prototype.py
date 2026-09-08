#!/usr/bin/env python3
"""Prototype the phi-domain joint-FBP context-tap mechanism on the Sewellia
lineolata (theta, phi) preview file.

This is a mechanism smoke test / first build, NOT a final training-ready
cache: it runs against the small h5 preview (20000 projections, 6 detector
rows, already flat/dark-corrected, single ~180deg theta half-turn) to
validate the phi-gating + growing-radius FBP + exact-angle reprojection
pipeline (see sdate/tr_diffusion/phi_context.py) end-to-end, including
whether the r=100% (whole-dataset) reconstruction runs without crashing --
an open question flagged before building this.

Output mirrors the existing time-domain tap-cache convention (one float16
memmap per tap channel + a meta.npz) so it plugs into the same
aux_channel_memmap/context+N2V training code once this becomes an actual
training run against real full-projection data.

    python scripts/sewellia_phi_context_prototype.py --max_bins 5   # smoke test
    python scripts/sewellia_phi_context_prototype.py                # full 360-bin build
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")
os.environ["PATH"] = f"/myhome/bin:{os.environ.get('PATH', '')}"

from sdate.tr_diffusion import phi_context as PC

DATA_PATH = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5"
OUT_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/phi_context"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=DATA_PATH)
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--tag", default="sewellia_phictx_proto")
    p.add_argument("--n_bins", type=int, default=360)
    p.add_argument("--pcts", default=",".join(str(x) for x in PC.DEFAULT_PCTS),
                   help="comma-separated fractions (NOT percent, e.g. 0.01 = 1%%) -- T = len(pcts)")
    p.add_argument("--det_bin", type=int, default=1)
    p.add_argument("--max_views", type=int, default=12000,
                   help="ASTRA CUDA backend view-count ceiling safety net (confirmed empirically "
                        "to crash between 15999 and 20000 views on this detector/volume shape) -- "
                        "deterministically stride-subsample down to this many views when a gate "
                        "selects more. Set <=0 to disable (will crash on any gate >~16k views).")
    p.add_argument("--max_bins", type=int, default=None,
                   help="cap the number of distinct phi-bins processed, for a quick smoke test "
                        "(default: process every bin that has >=1 real projection snapped to it)")
    p.add_argument("--log_every", type=int, default=5)
    p.add_argument("--slice_targets", default="4000,10000,16000",
                   help="comma-separated real projection indices: also save the reconstructed "
                        "volume's middle-row 2D slice (all T taps) for each one's bin, purely "
                        "for visualization -- the main tap cache only stores reprojected "
                        "PROJECTIONS, not the intermediate 3D volumes")
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")
    pcts = [float(x) for x in a.pcts.split(",")]
    T = len(pcts)
    radii = PC.radii_from_pcts(pcts)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(a.data_path, "r") as f:
        sinogram = f["sinogram"][:]
        theta = f["theta"][:].astype(np.float64)
        phase = f["phase"][:].astype(np.float64)
    n_proj, n_rows, n_pix = sinogram.shape
    log(f"loaded sinogram {sinogram.shape}, theta range [{theta.min():.2f},{theta.max():.2f}]deg, "
        f"phase range [{phase.min():.3f},{phase.max():.3f}]rad")

    bin_centers = PC.phi_bin_centers(a.n_bins)
    bin_idx = PC.snap_to_bins(phase, a.n_bins)
    needed_bins = np.unique(bin_idx)
    log(f"n_bins={a.n_bins}  T={T}  pcts={pcts}")
    log(f"{len(needed_bins)}/{a.n_bins} bins have >=1 real projection snapped to them "
        f"(mean {n_proj/len(needed_bins):.1f} proj/bin)")

    if a.max_bins is not None:
        needed_bins = needed_bins[: a.max_bins]
        log(f"--max_bins={a.max_bins}: smoke-test mode, only processing {len(needed_bins)} bins")

    det_shape = (n_rows, n_pix)
    vol_shape = (n_rows, n_pix, n_pix)
    # This h5's sinogram is already flat/dark-corrected (ratio done), but NOT yet
    # -log'd into attenuation. Trivial identity calibration (dark=0, flat=1) makes
    # counts_to_attenuation_flatdark/attenuation_to_counts_flatdark reduce to plain
    # -log(x)/exp(-x) -- exactly the missing round trip that keeps context taps in
    # the same (transmission) domain as the raw target. See phi_context.py docstring.
    dark = torch.zeros((), device=device, dtype=torch.float32)
    flat = torch.ones((), device=device, dtype=torch.float32)

    paths = [out_dir / f"{a.tag}_tap{c}.f16" for c in range(T)]
    mms = [np.memmap(p, dtype=np.float16, mode="w+", shape=(n_proj, n_rows, n_pix)) for p in paths]
    filled = np.zeros(n_proj, dtype=bool)

    slice_targets = [int(x) for x in a.slice_targets.split(",") if x.strip() != ""]
    slice_row = n_rows // 2
    bin_to_target = {int(bin_idx[i]): i for i in slice_targets}
    slices = {}  # target_idx -> (T, n_pix, n_pix) float32

    diag = []  # per (bin, tap) timing/sanity records
    t0 = time.time()
    crash_at = None
    for bi, b in enumerate(needed_bins):
        b = int(b)
        target_idx = np.flatnonzero(bin_idx == b)  # real projections that will use this bin's taps
        target_theta = theta[target_idx]
        phi_c = bin_centers[b]

        for c, (pct, r) in enumerate(zip(pcts, radii)):
            t1 = time.time()
            try:
                max_views = a.max_views if a.max_views and a.max_views > 0 else None
                vol, n_sel = PC.reconstruct_phi_gated(
                    sinogram, theta, phase, phi_c, r, dark, flat, device,
                    det_bin=a.det_bin, vol_shape=vol_shape, max_views=max_views,
                )
                if vol is None:
                    log(f"  bin {b} tap{c} (pct={pct:.0%}): NO projections in gate -- skipping")
                    continue
                if b in bin_to_target:
                    ti = bin_to_target[b]
                    slices.setdefault(ti, np.zeros((T, n_pix, n_pix), dtype=np.float32))
                    slices[ti][c] = vol[slice_row].cpu().numpy()
                n_used = min(n_sel, max_views) if max_views else n_sel
                subsampled = n_used < n_sel
                n_nan = int(torch.isnan(vol).sum().item())
                n_inf = int(torch.isinf(vol).sum().item())
                recon_s = time.time() - t1

                t2 = time.time()
                reproj = PC.reproject_to_counts(vol, target_theta, det_shape, device, dark, flat)
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
                tag_sub = f" [subsampled {n_sel}->{n_used}]" if subsampled else ""
                if n_nan or n_inf:
                    log(f"  !! bin {b} tap{c} (pct={pct:.0%}, n_sel={n_sel}){tag_sub}: NaN/Inf in reconstruction! {rec}")
                elif bi % a.log_every == 0 or pct == max(pcts):
                    log(f"  bin {b} tap{c} (pct={pct:.0%}, n_sel={n_sel}){tag_sub}: "
                        f"recon={recon_s:.2f}s reproj={reproj_s:.2f}s "
                        f"vol[min={rec['vol_min']:.4g} max={rec['vol_max']:.4g} mean={rec['vol_mean']:.4g}]")
            except Exception as e:
                crash_at = dict(bin=b, tap=c, pct=pct, error=repr(e))
                log(f"  CRASH at bin={b} tap={c} pct={pct:.0%}: {e!r}")
                raise

        if bi % a.log_every == 0:
            elapsed = time.time() - t0
            log(f"progress {bi+1}/{len(needed_bins)} bins, elapsed={elapsed/60:.1f}min, "
                f"est. total for all {len(needed_bins)} bins = {elapsed/(bi+1)*len(needed_bins)/60:.1f}min")

    for mm in mms:
        mm.flush()
    n_unfilled = int((~filled).sum())
    log(f"DONE. {n_proj - n_unfilled}/{n_proj} projections filled ({n_unfilled} unfilled -- "
        f"should be 0 if every real phi value snapped to a processed bin)")

    meta = dict(
        data_path=a.data_path, n_proj=n_proj, det_shape=list(det_shape), vol_shape=list(vol_shape),
        n_bins=a.n_bins, T=T, pcts=pcts, det_bin=a.det_bin, dark=0.0, flat=1.0,
        max_views=a.max_views, max_bins=a.max_bins, needed_bins=[int(x) for x in needed_bins], first_index=0,
        num_frames=n_proj, mode="phi_context_prototype",
    )
    for p in paths:
        np.savez(str(p) + ".meta.npz", **meta)
    with open(out_dir / f"{a.tag}_diag.json", "w") as fh:
        json.dump(dict(meta=meta, crash_at=crash_at, records=diag), fh, indent=2)

    if slices:
        slice_path = out_dir / f"{a.tag}_slices.npz"
        save_kwargs = {f"target_{ti}": arr for ti, arr in slices.items()}
        np.savez_compressed(
            slice_path, pcts=np.array(pcts), row=slice_row,
            target_idx=np.array(list(slices.keys())),
            target_theta=theta[np.array(list(slices.keys()))],
            target_phase=phase[np.array(list(slices.keys()))],
            **save_kwargs,
        )
        log(f"wrote {len(slices)} target(s) x {T} reconstructed slice(s) -> {slice_path}")

    total_min = (time.time() - t0) / 60
    log(f"wrote {T} tap memmaps to {out_dir} ({a.tag}_tap0..{T-1}.f16), diag -> {a.tag}_diag.json")
    log(f"total minutes={total_min:.1f}")
    log("SUCCESS" if crash_at is None else "FAILED")


if __name__ == "__main__":
    main()
