#!/usr/bin/env python3
"""Cheap diagnostic for the reconstruction-domain N2V idea (see
project-sewellia-recon-space-n2v.md, open risk section) -- BEFORE committing
to a full training run, check whether post-FBP noise is still local enough
for blind-spot masking to force genuine denoising, or whether backprojection
correlates neighboring pixels enough that a network could "cheat" by reading
the masked pixel's value off its neighbors.

Method: reconstruct TWO INDEPENDENT disjoint-view-subset FBPs of the exact
same phi (reusing reconstruct_phi_gated_groups, which already returns
multiple disjoint groups from one gate) -- the underlying clean structure is
identical between them (same phi, same object), so their difference cancels
the signal to first order and isolates ~2x the independent per-group noise.
Then:
  1. 2D spatial autocorrelation of that difference image (normalized,
     radially averaged) -- how fast does it decay away from lag 0?
  2. A concrete "cheating shortcut" number: residual variance of (pixel -
     mean of its 4 immediate neighbors) as a fraction of total variance --
     if neighbors barely predict a pixel's noise, this ratio is close to the
     value expected from the autocorrelation, and blind-spot masking should
     still work; if neighbors predict a lot of it, the ratio is small and we
     should be suspicious of N2V's validity here.
  3. For comparison/context, the SAME diagnostic run on the raw PROJECTION
     domain (two disjoint real-projection subsets at nominally the same
     phi, no reconstruction) -- this is the domain N2V already demonstrably
     works in (the projection-domain model is training right now), so it's
     the reference point for "what a working case looks like", not
     something we expect to fail.

    python scripts/sewellia_recon_noise_autocorr_check.py --target_phi_bin 90
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

from sdate.tr_diffusion import phi_context as PC

FULL_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01"
DATA_PATH = f"{FULL_DIR}/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5"
PHASE_TXT = f"{FULL_DIR}/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01_sound_phase.txt"
CALIB_PATH = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/sewellia_real_calibration.npz"
OUT_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/recon_context"

CROP_ROWS = 145
GROUP_SIZE = 900
TARGET_VIEWS_FOR_PAIR = 1800  # -> 2 disjoint 900-view groups, the noise-pair source
MAX_LAG = 12


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target_phi_bin", type=int, default=90, help="bin out of --n_bins to test")
    p.add_argument("--n_bins", type=int, default=180)
    p.add_argument("--out_dir", default=OUT_DIR)
    return p.parse_args()


def radial_autocorr(img: np.ndarray, max_lag: int) -> np.ndarray:
    """Normalized 2D autocorrelation of ``img`` (zero-mean), radially averaged
    into 1D bins ``0..max_lag`` (pixels). ``img[H,W]`` should already have its
    mean removed. Uses FFT for the full autocorrelation, then averages by
    integer radius."""
    img = img - img.mean()
    H, W = img.shape
    F = np.fft.fft2(img, s=(2 * H, 2 * W))
    ac = np.fft.ifft2(F * np.conj(F)).real
    ac = np.fft.fftshift(ac)
    cy, cx = ac.shape[0] // 2, ac.shape[1] // 2
    ac = ac[cy - max_lag: cy + max_lag + 1, cx - max_lag: cx + max_lag + 1]
    ac = ac / ac[max_lag, max_lag]  # normalize by lag-0

    yy, xx = np.mgrid[-max_lag:max_lag + 1, -max_lag:max_lag + 1]
    r = np.round(np.sqrt(yy ** 2 + xx ** 2)).astype(int)
    profile = np.zeros(max_lag + 1)
    for lag in range(max_lag + 1):
        profile[lag] = ac[r == lag].mean()
    return profile


def neighbor_residual_ratio(img: np.ndarray) -> float:
    """Var(pixel - mean of its 4-neighbors) / Var(pixel), interior only."""
    c = img[1:-1, 1:-1]
    nbr_mean = (img[:-2, 1:-1] + img[2:, 1:-1] + img[1:-1, :-2] + img[1:-1, 2:]) / 4.0
    resid = c - nbr_mean
    return float(resid.var() / c.var())


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    with h5py.File(DATA_PATH, "r") as f:
        sinogram_full = f["exchange/data"][:]
        theta = f["exchange/theta"][:].astype(np.float64)
    phase = np.loadtxt(PHASE_TXT).astype(np.float64)
    n_proj, n_rows_full, n_pix = sinogram_full.shape
    row_center = n_rows_full // 2
    row_lo = row_center - CROP_ROWS // 2
    row_hi = row_lo + CROP_ROWS
    sinogram = np.ascontiguousarray(sinogram_full[:, row_lo:row_hi, :])
    del sinogram_full
    vol_shape = (CROP_ROWS, n_pix, n_pix)
    local_center = CROP_ROWS // 2

    z = np.load(CALIB_PATH)
    dark_full = torch.from_numpy(z["dark_mean"]).to(device=device, dtype=torch.float32)
    flat_full = torch.from_numpy(z["white_mean"]).to(device=device, dtype=torch.float32)
    dark = dark_full[row_lo:row_hi].contiguous()
    flat = flat_full[row_lo:row_hi].contiguous()

    bin_centers = PC.phi_bin_centers(a.n_bins)
    phi_c = bin_centers[a.target_phi_bin]
    log(f"target phi_bin={a.target_phi_bin} phi_c={phi_c:.4f}")

    # -- reconstruction-domain noise pair: 2 disjoint 900-view groups, same phi --
    radius = TARGET_VIEWS_FOR_PAIR * np.pi / n_proj
    volumes, n_sel = PC.reconstruct_phi_gated_groups(
        sinogram, theta, phase, phi_c, radius, dark, flat, device,
        det_bin=1, vol_shape=vol_shape, group_size=GROUP_SIZE, max_groups=2, clamp=False,
    )
    assert len(volumes) == 2, f"expected 2 disjoint groups, got {len(volumes)} (n_sel={n_sel})"
    slice0 = volumes[0][local_center].cpu().numpy()
    slice1 = volumes[1][local_center].cpu().numpy()
    log(f"recon-domain pair: n_sel={n_sel}, slice0 mean={slice0.mean():.4g}, slice1 mean={slice1.mean():.4g}")

    recon_diff = (slice0 - slice1) / np.sqrt(2)  # normalize so var(diff) ~= var(single-group noise)
    recon_profile = radial_autocorr(recon_diff, MAX_LAG)
    recon_neighbor_ratio = neighbor_residual_ratio(recon_diff)

    # -- projection-domain noise pair, for reference: 2 disjoint raw-projection subsets at the same phi --
    mask = PC.phi_gate_mask(phase, phi_c, radius)
    idx = np.flatnonzero(mask)
    m = min(len(idx) // 2, 900)
    idx0, idx1 = idx[:m], idx[m:2 * m]
    # use the raw central detector row as a representative 1D-ish "image": stack a modest
    # number of adjacent-in-selection projections' central row into a 2D image for the same
    # autocorrelation machinery (rows=viewIndex, cols=detector column) -- a reasonable proxy for
    # "an image with genuinely independent per-pixel Poisson noise", not meant to be anatomically
    # meaningful, just a noise-character reference.
    n_img_rows = min(m, 145)
    proj0 = sinogram[idx0[:n_img_rows], local_center, :].astype(np.float32)
    proj1 = sinogram[idx1[:n_img_rows], local_center, :].astype(np.float32)
    proj_diff = (proj0 - proj1) / np.sqrt(2)
    proj_profile = radial_autocorr(proj_diff, MAX_LAG)
    proj_neighbor_ratio = neighbor_residual_ratio(proj_diff)

    log("=== radial autocorrelation (normalized, lag 0..%d px) ===" % MAX_LAG)
    log(f"  recon-domain: {np.round(recon_profile, 3).tolist()}")
    log(f"  proj-domain : {np.round(proj_profile, 3).tolist()}")
    log("=== neighbor-residual-ratio (1.0 = neighbors predict nothing, 0.0 = neighbors predict everything) ===")
    log(f"  recon-domain: {recon_neighbor_ratio:.4f}")
    log(f"  proj-domain : {proj_neighbor_ratio:.4f}")

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "recon_noise_autocorr_check.npz"
    np.savez_compressed(out_path, target_phi_bin=a.target_phi_bin, n_bins=a.n_bins,
                        recon_diff=recon_diff, proj_diff=proj_diff,
                        recon_profile=recon_profile, proj_profile=proj_profile,
                        recon_neighbor_ratio=recon_neighbor_ratio, proj_neighbor_ratio=proj_neighbor_ratio)
    log(f"saved -> {out_path}")

    verdict = ("OK -- recon-domain neighbor-residual-ratio is close to (or above) the proj-domain reference; "
              "blind-spot N2V should still force real denoising"
              if recon_neighbor_ratio >= 0.8 * proj_neighbor_ratio else
              "CAUTION -- recon-domain noise is substantially more locally predictable than proj-domain; "
              "a plain N2V mask may partially collapse to a local-smoothing shortcut, consider a wider "
              "mask neighborhood or a structured-mask variant before trusting this approach")
    log(f"VERDICT: {verdict}")
    log("SUCCESS")


if __name__ == "__main__":
    main()
