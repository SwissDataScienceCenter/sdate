#!/usr/bin/env python3
"""Per-voxel truncated B-spline-in-time MLE fit for a window of the real
Wunderkerze2 dataset (dose=0.05), the time-domain analogue of the Sewellia
per-voxel Fourier-in-phi MLE fit (scripts/sewellia_fourier_mle_slice.py).

Model: every voxel is

    f(x,y,z, tau) = sum_{j=0}^{K-1} c_j(x,y,z) * B_j(tau)

where ``tau`` is the RAW acquisition frame index (continuous, NOT binned per
revolution) and ``{B_j}`` is a fixed clamped cubic B-spline basis with K
control points spanning the chosen window. Every single raw projection in
the window is forward-projected from the volume evaluated at its OWN exact
(theta, tau) -- so within-revolution motion is captured too, not just
revolution-to-revolution change (unlike the existing static-per-window
k-joint-FBP / jointfbpctx baselines, see project-jointfbp-k21-baseline.md /
project-tr-diffusion-jointfbpctx.md).

Splines, not Fourier: Wunderkerze's motion is a one-off transient, not
periodic, so a global truncated Fourier series would (a) impose a false
periodic-wraparound assumption across the window's ends and (b) need many
harmonics to represent a localized event without ringing. A clamped B-spline
basis has local support and no periodicity assumption, at the same
"project only K coefficients, reassemble cheaply" cost as Fourier: weights
are a fixed (K, N) matrix (a partition-of-unity: rows sum to 1 everywhere),
so evaluating a frame at any tau (including a finer output grid than the
input data) is one matmul against a fixed basis matrix.

Redundancy: a window of ``n_rev`` revolutions gives ``n_rev*period_360``
(~200/rev) raw projections, each at its own (theta, tau), all constraining
only K << n_rev*period_360 unknowns per voxel -- e.g. n_rev=100, K=6 is a
~3300x data-to-parameter ratio, comfortably over-determined (unlike
Sewellia's photon-starved regime, where the redundancy came from many
DIFFERENT revolutions sharing the same phi due to phase/rotation
non-integer-ratio aliasing -- Wunderkerze has no such aliasing to exploit;
the redundancy here comes purely from K being small, not from repeat
sampling of any one tau).

Noise model: proper Poisson transmission NLL
(sdate.tr_diffusion.mle_reconstruct.poisson_data_loss), matching the dose
convention every other Wunderkerze2 MLE/denoiser baseline in this project
uses (see mle_reconstruct.py's module docstring) -- NOT the Gaussian
approximation the Sewellia script uses (Sewellia has no raw counts in a
convenient per-row array + an explicit read-noise floor; Wunderkerze already
has an established Poisson+dose convention here).

Interior-tomography halo fix (--pad_factor) and stabilization
(--pos_weight/--precondition/--grad_clip): ported unchanged from
scripts/sewellia_fourier_mle_slice.py and the ct_padded_recon skill -- widen
ONLY the reconstructed volume's in-plane (rotation-plane) axes, never the
detector; never fabricate sinogram data. Zero-init is known to diverge on
this dataset via l_clip saturation (see mle_reconstruct.gd_mle_reconstruct's
docstring) -- every one of the K coefficient volumes is instead initialized
to the SAME physically-sane warm-start volume (a small joint-FBP
reconstruction centred in the fit window), so l_pred(tau) = warm_start
exactly at step 0 for every tau (the B-spline basis is a partition of unity,
sum_j B_j(tau) == 1 everywhere in-domain).

  python scripts/wunderkerze_spline_mle.py --n_rev 100 --n_coeffs 6 --steps 3000
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
import torch.nn.functional as F

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

os.environ["PATH"] = f"/myhome/bin:{os.environ.get('PATH', '')}"

# astra_torch.lamino is imported LAZILY (inside the functions that need it),
# never at module level: this image's sdate.tr_diffusion package pulls in
# diffusers/torch._dynamo through its __init__ (load.py -> model.py), and
# importing astra_torch.lamino BEFORE that finishes registers a custom
# autograd kernel for the "wait_tensor" op that collides with torch._dynamo's
# own registration of the same op ("already a kernel registered ... for
# Autograd dispatch key and _c10d_functional namespace"). Every other script
# in this project that uses both (tr_diffusion_mle_kjoint_grid.py,
# tr_diffusion_jointfbp_k_sweep.py, mle_reconstruct.py itself) avoids this by
# importing sdate.tr_diffusion first and only reaching astra_torch.lamino
# deep inside function bodies, well after tr_diffusion's own import is done --
# same convention followed here (see build_chunk/compute_sensitivity below).
# sewellia_fourier_mle_slice.py instead avoided the collision by never
# importing sdate.tr_diffusion at all (it inlines the 2 functions it needs).

from sdate.tr_diffusion import reconstruct as R
from sdate.tr_diffusion.mle_reconstruct import poisson_data_loss
from sdate.tr_diffusion.map_reconstruct import _tv_loss
from sdate.tr_diffusion.profiles import DatasetProfile
from sdate.tr_diffusion.frames import MemmapFrameSource

OUT_DIR = "/myhome/data/sdate/shared/time_resolved/wunderkerze_spline_mle"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="wunderkerze2")
    p.add_argument("--frame_start", type=int, default=410_000,
                   help="first frame of the fit window (default matches "
                        "tr_diffusion_mle_kjoint_grid.py's own default: ~1/10 into the "
                        "profile's usable [400000,500000) range, inside the 'still dynamic' "
                        "region identified in project-tr-diffusion-jointfbpctx.md).")
    p.add_argument("--n_rev", type=float, default=100.0,
                   help="fit-window length in revolutions; win_frames = round(n_rev * period_360).")
    p.add_argument("--n_coeffs", type=int, default=6, help="number of B-spline control points (K).")
    p.add_argument("--spline_degree", type=int, default=3)
    p.add_argument("--det_bin", type=int, default=2)
    p.add_argument("--dose", type=float, default=0.05)
    p.add_argument("--noise_seed", type=int, default=12345)
    p.add_argument("--pad_factor", type=float, default=1.2,
                   help="widen ONLY the reconstructed volume's in-plane (H,W) axes; the "
                        "detector/slice (D) axis is never touched. See ct_padded_recon skill.")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.001,
                   help="kept conservative on purpose -- see sewellia_fourier_mle_slice.py's "
                        "--lr help for the exact failure mode (Adam's first step from an "
                        "unstable init overshoots l_clip and permanently zeros the gradient). "
                        "Warm-start init here should already avoid the worst of this, but the "
                        "lr itself is left at the same known-safe value.")
    p.add_argument("--batch_size", type=int, default=2000,
                   help="fresh random subset of the full held-in view pool drawn every step "
                        "(true mini-batch SGD, not a fixed shard -- see project memory "
                        "feedback on this exact bug in the Sewellia script).")
    p.add_argument("--max_views_per_call", type=int, default=4000,
                   help="chunk cap for the ASTRA projector call -- kept lower than Sewellia's "
                        "8000 default since these are full 3D (multi-slice) volumes, not single "
                        "rows, so each view costs more memory per call.")
    p.add_argument("--holdout_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tv_weight", type=float, default=0.0,
                   help="0 by default -- prior TV sweeps on this project's other MLE work found "
                        "TV=0 optimal by held-out NLL at every weight tried; enable explicitly "
                        "to sweep here too.")
    p.add_argument("--grad_clip", type=float, default=1e4)
    p.add_argument("--l_clip", type=float, default=20.0,
                   help="clamp on the linear predictor before exp(), matches "
                        "mle_reconstruct.poisson_data_loss's own default.")
    p.add_argument("--pos_weight", type=float, default=1e4,
                   help="soft positivity penalty weight, see sewellia_fourier_mle_slice.py's "
                        "--pos_weight help for the exact rationale (nothing else keeps "
                        "sum_j c_j(x,y,z)*B_j(tau) non-negative for every tau).")
    p.add_argument("--n_tau_probe", type=int, default=16)
    p.add_argument("--precondition", action="store_true", default=True)
    p.add_argument("--no_precondition", dest="precondition", action="store_false")
    p.add_argument("--sens_floor", type=float, default=0.05)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--n_out_frames", type=int, default=100,
                   help="number of evenly-spaced tau points the fitted basis is evaluated at "
                        "for the output movie/frames -- purely a display/comparison grid, "
                        "independent of --n_coeffs.")
    p.add_argument("--warm_start_rev", type=float, default=5.0,
                   help="revolutions in the small joint-FBP window used to build the warm-start "
                        "volume every coefficient is initialized to (centred in the fit window).")
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--tag", default=None)
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build_spline_knots(n_coeffs: int, degree: int = 3) -> np.ndarray:
    n_interior = n_coeffs - degree - 1
    if n_interior < 0:
        raise ValueError(f"n_coeffs={n_coeffs} must be >= degree+1={degree + 1}")
    interior = np.linspace(0.0, 1.0, n_interior + 2)[1:-1]
    return np.concatenate([np.zeros(degree + 1), interior, np.ones(degree + 1)])


def build_spline_weights(tau_norm: np.ndarray, n_coeffs: int, degree: int = 3) -> np.ndarray:
    """(n_coeffs, N) basis matrix: row j = B_j(tau_norm), clamped cubic B-spline.

    Clamped knots pin the curve to interpolate near the first/last control
    point exactly, avoiding a Fourier basis's periodic-wraparound assumption.
    The basis is a partition of unity (columns sum to 1) -- used at init time
    to guarantee every coefficient volume starting at the same warm-start
    volume reproduces exactly that volume at every tau, step 0.
    """
    from scipy.interpolate import BSpline
    t = build_spline_knots(n_coeffs, degree)
    tau_c = np.clip(tau_norm, 1e-9, 1.0 - 1e-9)
    weights = np.zeros((n_coeffs, len(tau_c)), dtype=np.float64)
    for j in range(n_coeffs):
        c = np.zeros(n_coeffs)
        c[j] = 1.0
        spl = BSpline(t, c, degree, extrapolate=False)
        weights[j] = np.nan_to_num(spl(tau_c), nan=0.0)
    return weights


def positivity_penalty(coeffs: torch.Tensor, weights_probe: torch.Tensor) -> torch.Tensor:
    """Soft penalty keeping f(x,y,z,tau) >= 0 at a fixed set of probe taus.
    weights_probe: (n_coeffs, P)."""
    frame_probe = torch.einsum("bp,bdhw->pdhw", weights_probe, coeffs)
    return torch.relu(-frame_probe).pow(2).mean()


def build_chunk(local_idx, theta_all, y_all, device, vol_shape, det_shape):
    """One chunk: fresh ASTRA projector geometry for exactly these views (cheap
    -- geometry construction is pure NumPy, the CUDA work happens inside the
    projector's forward call regardless of whether the geometry is fresh),
    plus their real dose-noised counts. local_idx indexes into theta_all/y_all
    (0-based within the fit window, NOT global frame numbers)."""
    from astra_torch.lamino import build_lamino_projector
    projector = build_lamino_projector(vol_shape=vol_shape, det_shape=det_shape,
                                       angles_deg=theta_all[local_idx], lamino_angle_deg=0.0,
                                       device=device)
    y_chunk = y_all[local_idx].to(device=device, dtype=torch.float32)
    return dict(projector=projector, y=y_chunk, local_idx=local_idx, n=len(local_idx))


def make_chunks(local_idx, theta_all, y_all, max_views, device, vol_shape, det_shape):
    return [build_chunk(local_idx[start:start + max_views], theta_all, y_all, device, vol_shape, det_shape)
            for start in range(0, len(local_idx), max_views)]


def compute_sensitivity(theta_all, local_idx, vol_shape, det_shape, device, max_views):
    """Sensitivity image A^T.1 over held-in views, chunked. See
    sewellia_fourier_mle_slice.py's compute_sensitivity -- identical logic,
    generalized to a multi-slice (D>1) vol_shape/det_shape."""
    from astra_torch.lamino import _create_lamino_geometry, _AstraLaminoOp
    det_rows, det_cols = det_shape
    sens = torch.zeros(vol_shape, device=device, dtype=torch.float32)
    for start in range(0, len(local_idx), max_views):
        sub = local_idx[start:start + max_views]
        vol_geom, proj_geom = _create_lamino_geometry(
            vol_shape=vol_shape, det_shape=det_shape,
            angles_deg=theta_all[sub], lamino_angle_deg=0.0)
        op = _AstraLaminoOp(vol_geom, proj_geom, vol_shape, det_rows, det_cols, len(sub))
        ones_sino = torch.ones((det_rows, len(sub), det_cols), device=device, dtype=torch.float32)
        sens += op.adjoint(ones_sino)
    return sens


def chunk_poisson_nll(chunk, coeffs, weights_all, I0, dark, dose, l_clip):
    """weights_all: (n_coeffs, N_total) precomputed once for the whole window;
    sliced here by the chunk's local_idx. Returns the RAW SUM Poisson NLL
    (matches mle_reconstruct.poisson_data_loss's own convention -- the data
    term is a sum over every ray, see that module's docstring on why TV/rho
    scaling has to be auto-derived rather than hand-picked)."""
    n_basis = coeffs.shape[0]
    weights = weights_all[:, chunk["local_idx"]]  # (n_basis, n)
    vol = coeffs.view(n_basis, 1, *coeffs.shape[1:])
    basis_proj = chunk["projector"](vol)
    # basis_proj: (n_basis, n, D, C) -- one forward-projection call per basis
    # coefficient volume, THEN combine per-view via the known spline weights
    # (projection is linear in the volume -- same trick as the Fourier script).
    l_pred = torch.einsum("bn,bndc->ndc", weights, basis_proj)
    return poisson_data_loss(l_pred, chunk["y"], I0, dark, dose, l_clip=l_clip)


def fit(coeffs, held_in_idx, rng, theta_all, y_all, weights_all, device, vol_shape, det_shape,
        batch_size, chunks_out, I0, dark, dose, steps, lr, tv_weight, eval_every, grad_clip,
        l_clip, pos_weight=0.0, weights_probe=None, sensitivity=None, log_prefix=""):
    n_basis = coeffs.shape[0]
    optimizer = torch.optim.Adam([coeffs], lr=lr)
    history = []
    best_holdout = float("inf")
    best_coeffs = coeffs.detach().clone()
    best_step = 0
    for step in range(steps):
        sub = rng.choice(held_in_idx, size=min(batch_size, len(held_in_idx)), replace=False)
        batch = build_chunk(sub, theta_all, y_all, device, vol_shape, det_shape)

        optimizer.zero_grad(set_to_none=True)
        loss = chunk_poisson_nll(batch, coeffs, weights_all, I0, dark, dose, l_clip)
        n_elem = batch["y"].numel()
        if tv_weight > 0:
            loss = loss + tv_weight * sum(_tv_loss(coeffs[i]) for i in range(n_basis))
        if pos_weight > 0 and weights_probe is not None:
            loss = loss + pos_weight * positivity_penalty(coeffs, weights_probe)
        loss.backward()
        if sensitivity is not None:
            coeffs.grad.mul_(sensitivity)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([coeffs], max_norm=grad_clip)
        optimizer.step()
        train_nll_per_ray = float(loss.item()) / max(n_elem, 1)

        if step % eval_every == 0 or step == steps - 1:
            with torch.no_grad():
                denom_out = sum(c["y"].numel() for c in chunks_out)
                holdout_nll = sum(
                    chunk_poisson_nll(c, coeffs, weights_all, I0, dark, dose, l_clip).item()
                    for c in chunks_out
                ) / max(denom_out, 1)
            rec = dict(step=step, train_nll_per_ray=train_nll_per_ray, holdout_nll_per_ray=holdout_nll)
            history.append(rec)
            if holdout_nll < best_holdout:
                best_holdout = holdout_nll
                best_step = step
                best_coeffs = coeffs.detach().clone()
            log(f"{log_prefix} step={step} train_nll_per_ray={train_nll_per_ray:.5f} "
                f"holdout_nll_per_ray={holdout_nll:.5f}{'  *best*' if step == best_step else ''}")
    return history, best_coeffs, best_step, best_holdout


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    prof = DatasetProfile.load(a.profile)
    tag = a.tag or f"{prof.name}_f{a.frame_start}_nrev{a.n_rev:g}_k{a.n_coeffs}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(a.out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"output -> {out_dir}")

    win_frames = int(round(a.n_rev * prof.period_360))
    frame_lo = a.frame_start
    frame_hi = frame_lo + win_frames
    assert frame_lo >= prof.frame_start and frame_hi <= prof.frame_end, (
        f"fit window [{frame_lo},{frame_hi}) must lie within the profile's usable range "
        f"[{prof.frame_start},{prof.frame_end})")
    idx_global = np.arange(frame_lo, frame_hi)
    log(f"fit window: frames [{frame_lo},{frame_hi}) = {win_frames} frames "
        f"({a.n_rev:g} revolutions @ period_360={prof.period_360:.3f})")

    # --- real per-pixel flat/dark calibration ---
    dark_mov = Path(prof.mov_path).with_name(f"{prof.name}_darks.mov")
    flat_mov = Path(prof.mov_path).with_name(f"{prof.name}_flats.mov")
    dark_native = torch.from_numpy(R.load_calibration_average(
        str(dark_mov), prof.crop, prof.rot_axis_col, height=prof.height, width=prof.width,
    )).to(device=device, dtype=torch.float32)
    flat_native = torch.from_numpy(R.load_calibration_average(
        str(flat_mov), prof.crop, prof.rot_axis_col, height=prof.height, width=prof.width,
    )).to(device=device, dtype=torch.float32)
    log(f"flat/dark loaded: dark mean={dark_native.mean():.4g} flat mean={flat_native.mean():.4g}")

    src = MemmapFrameSource(prof.memmap_path, prof.mov_path)

    theta_all = R.projection_angles(idx_global, deg_per_frame=prof.deg_per_frame)
    tau_norm = (idx_global - frame_lo).astype(np.float64) / max(win_frames - 1, 1)

    # --- dose-noised counts for the WHOLE window, built in chunks to avoid a
    # large transient native-resolution buffer, one continuous noise stream ---
    log("loading + noising the full fit window...")
    t0 = time.time()
    gen = torch.Generator(device=device).manual_seed(a.noise_seed + frame_lo)
    y_chunks = []
    load_chunk = 4000
    for start in range(0, win_frames, load_chunk):
        sub_idx = idx_global[start:start + load_chunk]
        native = R.native_window_gpu(src, sub_idx, prof.crop, prof.rot_axis_col, device)
        noisy = R.noisy_window_gpu(native, a.dose, generator=gen)
        y_chunks.append(R.bin_detector(noisy, a.det_bin))
    y_all = torch.cat(y_chunks, dim=0)
    del y_chunks
    log(f"loaded {y_all.shape} in {time.time()-t0:.1f}s")

    I0 = R.bin_detector(flat_native.unsqueeze(0), a.det_bin)
    dark = R.bin_detector(dark_native.unsqueeze(0), a.det_bin)

    n_rows, n_cols = y_all.shape[1], y_all.shape[2]  # (D, C) after det_bin
    n_cols_recon = int(round(a.pad_factor * n_cols))
    pad = (n_cols_recon - n_cols) // 2
    n_cols_recon = n_cols + 2 * pad
    fov_lo, fov_hi = pad, pad + n_cols
    vol_shape = (n_rows, n_cols_recon, n_cols_recon)
    det_shape = (n_rows, n_cols)  # real detector width ALWAYS -- never widened/padded
    log(f"pad_factor={a.pad_factor} -> volume widened to {n_rows}x{n_cols_recon}x{n_cols_recon}, "
        f"detector stays at real {n_rows}x{n_cols}; true FOV crop [{fov_lo}:{fov_hi},{fov_lo}:{fov_hi}]")

    n_holdout = int(round(a.holdout_frac * win_frames))
    perm = rng.permutation(win_frames)
    holdout_idx = np.sort(perm[:n_holdout])
    held_in_idx = np.sort(perm[n_holdout:])
    log(f"held_in={len(held_in_idx)} holdout={len(holdout_idx)} (seed={a.seed})")

    weights_all = torch.from_numpy(
        build_spline_weights(tau_norm, a.n_coeffs, a.spline_degree)
    ).to(device=device, dtype=torch.float32)
    log(f"spline basis: n_coeffs={a.n_coeffs} degree={a.spline_degree} "
        f"(partition-of-unity check: max|sum-1|={float((weights_all.sum(0)-1).abs().max()):.2e})")

    assert a.batch_size <= a.max_views_per_call, "--batch_size must not exceed --max_views_per_call"
    chunks_out = make_chunks(holdout_idx, theta_all, y_all, a.max_views_per_call, device, vol_shape, det_shape)
    log(f"holdout into {len(chunks_out)} eval chunk(s) ({[c['n'] for c in chunks_out]})")

    weights_probe = None
    if a.pos_weight > 0:
        tau_probe = np.linspace(0.0, 1.0, a.n_tau_probe)
        weights_probe = torch.from_numpy(
            build_spline_weights(tau_probe, a.n_coeffs, a.spline_degree)
        ).to(device=device, dtype=torch.float32)
        log(f"positivity penalty enabled: pos_weight={a.pos_weight}, {a.n_tau_probe} probe taus")

    sensitivity = None
    if a.precondition:
        log("computing sensitivity image (A^T . 1) over held-in views...")
        t0 = time.time()
        sens_raw = compute_sensitivity(theta_all, held_in_idx, vol_shape, det_shape, device, a.max_views_per_call)
        sensitivity = (sens_raw / sens_raw.mean().clamp_min(1e-8)).clamp(min=a.sens_floor, max=1.0)
        log(f"sensitivity image computed in {time.time()-t0:.1f}s "
            f"(raw mean={sens_raw.mean().item():.3f}, floor={a.sens_floor})")

    # --- warm start: small joint-FBP volume centred in the fit window, embedded
    # into the padded canvas (periphery stays zero -- no fabricated data there,
    # same convention as --pad_factor's periphery). Every coefficient starts
    # identical to this volume, so l_pred(tau) == warm_start for every tau at
    # step 0 (the spline basis is a partition of unity). ---
    log(f"building warm-start volume ({a.warm_start_rev:g}-revolution joint-FBP, window centre)...")
    warm_win = int(round(a.warm_start_rev * prof.period_360))
    warm_center = frame_lo + win_frames // 2
    warm_lo = warm_center - warm_win // 2
    warm_idx = np.arange(warm_lo, warm_lo + warm_win)
    warm_angles = R.projection_angles(warm_idx, deg_per_frame=prof.deg_per_frame)
    warm_native = R.native_window_gpu(src, warm_idx, prof.crop, prof.rot_axis_col, device)
    warm_gen = torch.Generator(device=device).manual_seed(a.noise_seed + warm_lo)
    warm_noisy = R.noisy_window_gpu(warm_native, a.dose, generator=warm_gen)
    warm_atten = R.counts_to_attenuation_flatdark(warm_noisy, dark_native, flat_native)
    warm_vol_native = R.reconstruct(warm_atten, warm_angles, det_bin=a.det_bin, method="fbp", device=device,
                                     vol_shape=(n_rows, n_cols, n_cols))
    warm_vol = torch.zeros(vol_shape, device=device, dtype=torch.float32)
    warm_vol[:, fov_lo:fov_hi, fov_lo:fov_hi] = warm_vol_native
    log(f"warm start built from frames [{warm_idx[0]},{warm_idx[-1]}], "
        f"native range [{warm_vol_native.min():.4g},{warm_vol_native.max():.4g}]")

    coeffs = warm_vol.unsqueeze(0).repeat(a.n_coeffs, 1, 1, 1).clone().requires_grad_(True)

    log(f"=== fitting n_coeffs={a.n_coeffs} over {win_frames} views "
        f"({a.n_rev:g} revolutions), vol_shape={vol_shape} ===")
    t0 = time.time()
    hist, best_coeffs, best_step, best_holdout = fit(
        coeffs, held_in_idx, rng, theta_all, y_all, weights_all, device, vol_shape, det_shape,
        a.batch_size, chunks_out, I0, dark, a.dose, a.steps, a.lr, a.tv_weight, a.eval_every,
        a.grad_clip if a.grad_clip > 0 else None, a.l_clip, pos_weight=a.pos_weight,
        weights_probe=weights_probe, sensitivity=sensitivity, log_prefix=f"[K={a.n_coeffs}]")
    log(f"fit done in {time.time()-t0:.1f}s -- best holdout_nll_per_ray={best_holdout:.5f} "
        f"at step={best_step} (final={hist[-1]['holdout_nll_per_ray']:.5f})")

    np.save(out_dir / "coeffs_best.npy", best_coeffs.cpu().numpy())
    np.save(out_dir / "coeffs_best_fov.npy", best_coeffs.cpu().numpy()[:, :, fov_lo:fov_hi, fov_lo:fov_hi])

    results = dict(
        profile=prof.name, frame_start=frame_lo, frame_end=frame_hi, n_rev=a.n_rev,
        win_frames=win_frames, n_coeffs=a.n_coeffs, spline_degree=a.spline_degree,
        det_bin=a.det_bin, dose=a.dose, pad_factor=a.pad_factor, n_rows=n_rows, n_cols=n_cols,
        n_cols_recon=n_cols_recon, fov_crop=[fov_lo, fov_hi], pos_weight=a.pos_weight,
        precondition=a.precondition, tv_weight=a.tv_weight, lr=a.lr, batch_size=a.batch_size,
        steps=a.steps, best_step=best_step, best_holdout_nll_per_ray=best_holdout,
        warm_start_rev=a.warm_start_rev, history=hist,
    )
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # --- reassemble: evaluate the fitted basis at n_out_frames evenly-spaced
    # tau across the window (or any finer/coarser grid -- one matmul, cheap) ---
    log(f"=== reassembling {a.n_out_frames} output frames from the fitted coefficients ===")
    tau_out = np.linspace(0.0, 1.0, a.n_out_frames)
    weights_out = build_spline_weights(tau_out, a.n_coeffs, a.spline_degree)  # (K, n_out_frames)
    coeffs_np = best_coeffs.cpu().numpy()  # (K, D, Hp, Wp)
    mid_slice = n_rows // 2
    frames_mid = np.einsum("kt,kwh->twh", weights_out,
                           coeffs_np[:, mid_slice, fov_lo:fov_hi, fov_lo:fov_hi])
    np.savez(out_dir / "tau_sweep_mid_fov.npz", frames=frames_mid, tau_out=tau_out,
             frame_indices=frame_lo + tau_out * (win_frames - 1))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        vmin, vmax = np.percentile(frames_mid, [1, 99])
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(frames_mid[0], cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"mid-slice, K={a.n_coeffs}, frame={frame_lo:.0f}")
        ax.axis("off")

        def update(i):
            im.set_data(frames_mid[i])
            ax.set_title(f"mid-slice, K={a.n_coeffs}, frame={frame_lo + tau_out[i]*(win_frames-1):.0f}")
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=a.n_out_frames, interval=80)
        ani.save(out_dir / "tau_sweep_mid.gif", writer="pillow")
        plt.close(fig)
        log(f"saved tau-sweep GIF -> {out_dir / 'tau_sweep_mid.gif'}")
    except Exception as e:
        log(f"GIF generation skipped ({type(e).__name__}: {e})")

    log(f"DONE. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
