#!/usr/bin/env python3
"""Non-deep-learning baseline: per-voxel truncated Fourier-in-phi MLE fit for
a single 2D slice of the real Sewellia lineolata dataset (arXiv:2506.03792,
PeriodRecon, single-value-of-k debug test -- see project memory
project-sewellia-phi-context.md and the design grilled with the user before
this script existed).

Model: every pixel (x, y) of slice z_slice is a periodic function of phase
phi,

    f(x, y, phi) = a0(x,y) + sum_{j=1..k} a_j(x,y) cos(j*phi) + b_j(x,y) sin(j*phi)

The (2k+1) coefficient maps {a0, a_1..a_k, b_1..b_k} are the only free
parameters. They are fit by gradient descent through the project's existing
differentiable ASTRA parallel-beam projector (astra_torch.lamino), using a
Poisson-shot-noise + Gaussian-read-noise-floor NLL against the REAL raw
detector counts at that row (no synthetic noise, no log-transform of the
data -- the noise model is applied directly in count space).

Efficiency: projection is linear in the volume, so instead of evaluating a
fresh image per measurement and projecting it, we forward-project the (2k+1)
BASIS coefficient maps ONCE per step (through every angle in the current
batch in a single batched ASTRA call) and combine per-measurement via the
known cos(j*phi_n)/sin(j*phi_n) weights -- O(2k+1) projector calls per step
regardless of how many thousand measurements are in the batch, not O(N).

Held-out projections (fixed split, reused for every future k/TV comparison)
give the only overfitting signal available here since Sewellia has no ground
truth: if held-out NLL for k>0 is not better than a k=0 (phi-independent)
baseline fit on the same split, the harmonics are just fitting noise.

Interior-tomography halo fix (--pad_factor): the reconstructed volume is
widened beyond the real detector width so real, physical rays that graze the
object's periphery (which extends past the small synchrotron FOV) have
somewhere to deposit that density instead of biasing the interior. The
detector itself is NEVER widened/padded with fabricated data -- an earlier
version of this script extrapolated the raw sinogram outward and fed that
synthetic data through the same Poisson NLL as if it were real measured
counts, which is wrong (it asserts a specific, false shot-noise variance
around invented values) and was a likely contributor to the instabilities
below. Peripheral voxels now get zero fabricated supervision -- only
whatever real, sparse, oblique rays happen to graze them -- stabilized
instead by: a soft positivity penalty (see --pos_weight), a sensitivity-image
(A^T.1) gradient preconditioner (see --precondition), and a tighter,
physically-motivated clamp on the linear predictor (see --l_clamp). See the
ct_padded_recon skill for the full writeup (/root/.claude/skills/ct_padded_recon).

    python scripts/sewellia_fourier_mle_slice.py --k 8 --steps 3000
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

from astra_torch.lamino import build_lamino_projector, _create_lamino_geometry, _AstraLaminoOp

# Inlined rather than imported from sdate.tr_diffusion: that package's __init__
# unconditionally pulls in the diffusers/UNet stack, which collides with
# torch._dynamo/torch.distributed when imported alongside astra_torch in this
# image ("kernel already registered ... wait_tensor"). Not worth touching the
# currently-running production training job's import path to chase that down --
# these two functions are a few lines each and stable (see nb_head.py,
# map_reconstruct.py).


def nb_nll_gaussian(y, mu, var, sigma_read2, dose: float = 1.0, eps: float = 1e-6):
    total_var = (mu * float(dose) + var + sigma_read2).clamp_min(eps)
    return 0.5 * torch.log(total_var) + (y - mu).pow(2) / (2.0 * total_var)


def _tv_loss(mu: torch.Tensor) -> torch.Tensor:
    dz = (mu[1:, :, :] - mu[:-1, :, :]).abs().sum()
    dy = (mu[:, 1:, :] - mu[:, :-1, :]).abs().sum()
    dx = (mu[:, :, 1:] - mu[:, :, :-1]).abs().sum()
    return dz + dy + dx

FULL_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01"
DATA_PATH = f"{FULL_DIR}/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5"
PHASE_TXT = f"{FULL_DIR}/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01_sound_phase.txt"
CALIB_PATH = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/sewellia_real_calibration.npz"
OUT_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/fourier_mle"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=DATA_PATH)
    p.add_argument("--phase_txt", default=PHASE_TXT)
    p.add_argument("--calib_path", default=CALIB_PATH)
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--z_slice", type=int, default=29)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--fit_k0_baseline", action="store_true", default=True)
    p.add_argument("--no_k0_baseline", dest="fit_k0_baseline", action="store_false")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--holdout_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tv_weight", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=2000,
                   help="Mini-batch SGD: every step draws a FRESH random size-batch_size subset "
                        "from the full held-in pool (rng.choice, without replacement within the "
                        "batch; see fit()) -- a real mini-batch, not a fixed shard. An earlier "
                        "version pre-split held_in_idx into fixed shards of this size ONCE and only "
                        "shuffled the ORDER they were visited in every epoch, so the same exact set "
                        "of views recurred identically every time a given shard came up -- any "
                        "weakness specific to one such fixed shard's geometry would never be "
                        "averaged out the way genuine resampling does (flagged by the user "
                        "2026-09-06; fixed same day). Also keeps every ASTRA call safely under the "
                        "~16384-view CUDA 3D-array cap -- see --max_views_per_call. Checked "
                        "empirically (2026-09-05, under the old fixed-shard scheme, but the "
                        "conclusion -- phi is well-covered by any ~2000-view window regardless of "
                        "how it's drawn -- still holds): even a contiguous ~batch_size-frame window "
                        "(a narrow slice of theta) has close to uniform phi coverage here, since the "
                        "550Hz phase oscillation completes many cycles within any such window, at "
                        "the start/middle/end of the scan alike. Re-check on any future dataset with "
                        "a slower stimulus/rotation ratio.")
    p.add_argument("--max_views_per_call", type=int, default=8000,
                   help="Hard safety cap on views passed to a single ASTRA projector call. "
                        "Empirically, ASTRA's BP3D_CUDA adjoint fails with 'CUDA error 1: invalid "
                        "argument' allocating a 3D array once the view count exceeds ~16384 (a CUDA "
                        "3D-array dimension cap unrelated to GPU memory). --batch_size should "
                        "already be well under this; it's also used to chunk the held-out "
                        "evaluation set and the sensitivity-image computation, neither of which is "
                        "otherwise batched.")
    p.add_argument("--grad_clip", type=float, default=1e4,
                   help="torch.nn.utils.clip_grad_norm_ max_norm on the coeffs gradient before "
                        "each Adam step (applied AFTER sensitivity preconditioning, see "
                        "--precondition), or 0/negative to disable. An independent safety net, not "
                        "a fix for the underlying bias-toward-runaway pressure.")
    p.add_argument("--l_clamp", type=float, default=5.0,
                   help="Clamp the linear predictor l_pred to [-l_clamp, l_clamp] before exp() in "
                        "mu = I0*exp(-l_pred) + dark. Tightened from an earlier +-20 (which allows "
                        "exp(20) ~ 5e8x the open-beam level -- never physically real for this sample) "
                        "to a still-generous but physically-motivated bound. This does not fix any "
                        "underlying bias toward extreme l values, it only stops catastrophic overflow "
                        "-- turns 'diverges to NaN' into 'saturates, keeps training'. mu is also "
                        "hard-clamped to [1e-6, 1e6] as a second, blunt float32-overflow safety net.")
    p.add_argument("--pos_weight", type=float, default=1e4,
                   help="Weight on a soft positivity penalty: f(x,y,phi) = a0 + sum_j(aj*cos(j*phi) "
                        "+ bj*sin(j*phi)) must stay non-negative for every phi the data actually "
                        "cover, but nothing otherwise constrains aj/bj individually -- the harmonic "
                        "sum can dip negative at some phase even with a perfectly reasonable-looking "
                        "a0. Evaluated as mean(relu(-f)^2) at --n_phi_probe fixed phase points, added "
                        "to the training loss every step. 0 disables. This is a soft penalty, not a "
                        "hard reparametrization (e.g. log-domain a0 + a harmonic-amplitude cap "
                        "sqrt(aj^2+bj^2) <= a0) -- the reparametrization would be the more rigorous "
                        "fix but requires restructuring how coeffs is defined and consumed "
                        "everywhere; not yet implemented.")
    p.add_argument("--n_phi_probe", type=int, default=16,
                   help="Number of fixed, evenly-spaced phase points used to evaluate --pos_weight's "
                        "positivity penalty every step.")
    p.add_argument("--precondition", action="store_true", default=True,
                   help="Precondition the coeffs gradient by multiplying (elementwise, broadcast "
                        "over all n_basis channels) by a normalized, [sens_floor, 1.0]-clamped "
                        "sensitivity image A^T.1 -- the backprojection of an all-ones sinogram over "
                        "the held-in views, i.e. a per-voxel map of how much real ray-weight "
                        "actually constrains that voxel. Computed once (not per-step) before "
                        "training starts. Voxels at the edge of the FOV or under-sampled angularly "
                        "are the ones most likely to be first to wander into a bad region during "
                        "optimization, since a large raw step there isn't damped by data "
                        "consistency elsewhere the way a well-sampled voxel's is; multiplying by "
                        "this (rather than dividing, which would AMPLIFY exactly those voxels' "
                        "steps -- tried and confirmed empirically worse, see the ct_padded_recon "
                        "skill's 'Known status') damps their updates instead, reducing the chance "
                        "of any single voxel taking the big first step that starts a runaway. See "
                        "--sens_floor for the minimum damping factor.")
    p.add_argument("--no_precondition", dest="precondition", action="store_false")
    p.add_argument("--sens_floor", type=float, default=0.05,
                   help="Floor applied to the normalized sensitivity image (sensitivity / its own "
                        "mean, capped at 1.0) before it's used as a gradient-damping multiplier, so "
                        "voxels with ~zero real ray coverage (e.g. far corners of a widened "
                        "--pad_factor volume, outside the swept circle entirely) still get a small "
                        "but nonzero update rather than being frozen at exactly 0 forever.")
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--phi_sweep_frames", type=int, default=60)
    p.add_argument("--pad_factor", type=float, default=1.0,
                   help="Fix for the local/interior-tomography halo: the synchrotron FOV is "
                        "smaller than the fish cross-section, so real material outside the "
                        "n_cols x n_cols square still attenuates the rays we DO measure (the "
                        "object rotates real density in and out of the small reconstructed square "
                        "at different angles). If the model can't place that density anywhere, it "
                        "biases the voxels it does have -- a halo/cupping that grows toward the "
                        "edges. pad_factor widens ONLY the reconstructed volume (vol_shape); the "
                        "detector geometry (det_shape) always stays at the real n_cols -- no "
                        "fabricated/extrapolated sinogram data is ever created or fed through the "
                        "NLL. An earlier version of this script padded the raw sinogram with a "
                        "cosine-tapered extrapolation and trained against it through the same "
                        "Poisson NLL as if it were real measured counts -- that's wrong (it asserts "
                        "a specific, false shot-noise variance around invented values) and is a "
                        "likely contributor to the instabilities that version showed. Peripheral "
                        "voxels now receive zero fabricated supervision -- only whatever genuine, "
                        "sparse, oblique real rays happen to graze them, the same rays that were "
                        "already part of the measured data before padding -- stabilized instead by "
                        "--pos_weight, --precondition, and --l_clamp. 1.0 = old behavior (no extra "
                        "FOV, halo present). Only the central n_cols x n_cols crop (saved as "
                        "*_fov.npy / phi_sweep_fov.npz) is the physically-supported region -- the "
                        "periphery is a stabilization aid, not a trustworthy reconstruction in its "
                        "own right.")
    p.add_argument("--tag", default=None)
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_row_data(a):
    with h5py.File(a.data_path, "r") as f:
        counts_row = f["exchange/data"][:, a.z_slice, :].astype(np.float32)
        theta_deg = f["exchange/theta"][:].astype(np.float64)
    phi = np.loadtxt(a.phase_txt).astype(np.float64)
    calib = np.load(a.calib_path)
    dark_row = calib["dark_mean"][a.z_slice].astype(np.float32)
    white_row = calib["white_mean"][a.z_slice].astype(np.float32)
    dark_var_row = calib["dark_var"][a.z_slice].astype(np.float32)
    return counts_row, theta_deg, phi, dark_row, white_row, dark_var_row


def build_weights(phi: torch.Tensor, k: int) -> torch.Tensor:
    """(n_basis, N) rows: [1, cos(1*phi)..cos(k*phi), sin(1*phi)..sin(k*phi)]."""
    rows = [torch.ones_like(phi)]
    for j in range(1, k + 1):
        rows.append(torch.cos(j * phi))
    for j in range(1, k + 1):
        rows.append(torch.sin(j * phi))
    return torch.stack(rows, dim=0)


def positivity_penalty(coeffs: torch.Tensor, k: int, phi_probe: torch.Tensor) -> torch.Tensor:
    """Soft penalty keeping f(x,y,phi) = a0 + sum_j aj*cos(j*phi) + bj*sin(j*phi)
    non-negative at a fixed set of probe phases, evaluated over every voxel at
    once. Nothing else constrains a_j/b_j individually -- the harmonic sum can
    dip negative at some phase even with a perfectly reasonable-looking a0
    (see --pos_weight help)."""
    weights = build_weights(phi_probe, k)          # (n_basis, P)
    frame_probe = torch.einsum("bp,bhw->phw", weights, coeffs)  # (P, H, W)
    return torch.relu(-frame_probe).pow(2).mean()


def project_basis(projector, coeffs: torch.Tensor, n_cols: int) -> torch.Tensor:
    """coeffs: (n_basis, H, W) -> (n_basis, V, W) line integrals, one row per view."""
    n_basis = coeffs.shape[0]
    vol = coeffs.view(n_basis, 1, 1, coeffs.shape[1], coeffs.shape[2])
    proj = projector(vol)  # (n_basis, V, 1, n_cols)
    return proj[:, :, 0, :]


def nll_from_l(l_pred, counts, I0, dark, sigma_read2, l_clamp=5.0):
    lc = l_pred.clamp(-l_clamp, l_clamp)
    mu = I0.unsqueeze(0) * torch.exp(-lc) + dark.unsqueeze(0)
    mu = mu.clamp(min=1e-6, max=1e6)  # blunt float32-overflow safety net, independent of l_clamp
    zero_var = torch.zeros_like(mu)
    return nb_nll_gaussian(counts, mu, zero_var, sigma_read2.unsqueeze(0), dose=1.0).mean()


def build_chunk(idx, theta_deg, phi, counts_row, device, vol_shape, det_shape):
    """One chunk: a fresh ASTRA projector geometry for exactly these views, plus
    their phi values and real measured counts. Cheap to call every step --
    _create_lamino_geometry (inside build_lamino_projector) is pure NumPy, the
    actual CUDA work (FP3D_CUDA/BP3D_CUDA) only happens inside the projector's
    forward/adjoint calls, which happen every step regardless of whether the
    geometry itself is fresh or reused."""
    projector = build_lamino_projector(vol_shape=vol_shape, det_shape=det_shape,
                                       angles_deg=theta_deg[idx], lamino_angle_deg=0.0,
                                       device=device)
    phi_t = torch.from_numpy(phi[idx]).to(device=device, dtype=torch.float32)
    counts_t = torch.from_numpy(np.ascontiguousarray(counts_row[idx])).to(device=device, dtype=torch.float32)
    return dict(projector=projector, phi=phi_t, counts=counts_t, n=len(idx))


def make_chunks(idx, theta_deg, phi, counts_row, max_views, device, vol_shape, det_shape):
    """Split idx into <= max_views pieces, each with its own fixed-geometry projector
    (see --max_views_per_call for why: ASTRA's adjoint hits a hard CUDA 3D-array
    dimension cap well before GPU memory would). Used for the (fixed, not
    resampled) held-out evaluation set -- see --batch_size help for why the
    TRAINING side no longer uses this."""
    return [build_chunk(idx[start:start + max_views], theta_deg, phi, counts_row, device, vol_shape, det_shape)
            for start in range(0, len(idx), max_views)]


def compute_sensitivity(theta_deg, idx, vol_shape, det_shape, device, max_views):
    """Sensitivity image A^T.1: backproject an all-ones sinogram over every
    held-in view, chunked the same way as training (see --max_views_per_call).
    A per-voxel map of how much real ray-weight actually constrains that
    voxel -- used to precondition the gradient (see --precondition help). No
    fabricated data anywhere here: this is purely a function of the real
    acquisition geometry (angles, detector width), not of any measured or
    invented counts."""
    det_rows, det_cols = det_shape
    sens = torch.zeros(vol_shape, device=device, dtype=torch.float32)
    for start in range(0, len(idx), max_views):
        sub = idx[start:start + max_views]
        vol_geom, proj_geom = _create_lamino_geometry(
            vol_shape=vol_shape, det_shape=det_shape,
            angles_deg=theta_deg[sub], lamino_angle_deg=0.0)
        op = _AstraLaminoOp(vol_geom, proj_geom, vol_shape, det_rows, det_cols, len(sub))
        ones_sino = torch.ones((det_rows, len(sub), det_cols), device=device, dtype=torch.float32)
        sens += op.adjoint(ones_sino)
    return sens


def chunk_nll(chunk, coeffs, k, n_cols, I0, dark, sigma_read2, l_clamp=5.0):
    weights = build_weights(chunk["phi"], k)
    basis_proj = project_basis(chunk["projector"], coeffs, n_cols)
    l_pred = torch.einsum("bn,bnc->nc", weights, basis_proj)
    return nll_from_l(l_pred, chunk["counts"], I0, dark, sigma_read2, l_clamp=l_clamp)


def fit(coeffs, held_in_idx, rng, theta_deg, phi, counts_row, device, vol_shape, det_shape,
        batch_size, chunks_out, I0, dark, sigma_read2, k, steps, lr, tv_weight,
        eval_every, grad_clip, log_prefix, pos_weight=0.0, phi_probe=None,
        sensitivity=None, l_clamp=5.0):
    n_basis = coeffs.shape[0]
    n_cols = coeffs.shape[2]

    optimizer = torch.optim.Adam([coeffs], lr=lr)
    history = []
    best_holdout = float("inf")
    best_coeffs = coeffs.detach().clone()
    best_step = 0
    for step in range(steps):
        # Fresh random subset of the FULL held-in pool every step -- true
        # mini-batch SGD, not a fixed partition cycled in shuffled order (an
        # earlier version pre-split held_in_idx into --batch_size shards ONCE
        # and only shuffled the ORDER they were visited in, so the same exact
        # set of views recurred identically every epoch; any weakness
        # specific to one such fixed shard's geometry would never be averaged
        # out the way genuine resampling does). Rebuilding the projector every
        # step is not the performance cost it looks like: geometry
        # construction is pure NumPy, the actual CUDA work happens in the
        # forward/adjoint calls either way -- see build_chunk.
        sub = rng.choice(held_in_idx, size=min(batch_size, len(held_in_idx)), replace=False)
        batch = build_chunk(sub, theta_deg, phi, counts_row, device, vol_shape, det_shape)

        optimizer.zero_grad(set_to_none=True)
        # Real measured columns only -- no fabricated/extrapolated data ever
        # enters this loss (see module docstring and --pad_factor help).
        loss = chunk_nll(batch, coeffs, k, n_cols, I0, dark, sigma_read2, l_clamp=l_clamp)
        if tv_weight > 0:
            loss = loss + tv_weight * sum(_tv_loss(coeffs[i:i + 1]) for i in range(n_basis))
        if pos_weight > 0 and phi_probe is not None:
            loss = loss + pos_weight * positivity_penalty(coeffs, k, phi_probe)
        loss.backward()
        if sensitivity is not None:
            coeffs.grad.mul_(sensitivity)
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_([coeffs], max_norm=grad_clip)
        optimizer.step()
        train_loss_val = float(loss.item())

        if step % eval_every == 0 or step == steps - 1:
            with torch.no_grad():
                denom_out = sum(c["n"] for c in chunks_out)
                holdout_nll = sum(
                    chunk_nll(c, coeffs, k, n_cols, I0, dark, sigma_read2, l_clamp=l_clamp).item()
                    * (c["n"] / denom_out)
                    for c in chunks_out
                )
            rec = dict(step=step, train_loss=float(train_loss_val), holdout_nll=holdout_nll)
            history.append(rec)
            if holdout_nll < best_holdout:
                best_holdout = holdout_nll
                best_step = step
                best_coeffs = coeffs.detach().clone()
            log(f"{log_prefix} step={step} train_loss={rec['train_loss']:.4f} holdout_nll={holdout_nll:.4f}"
                f"{'  *best*' if step == best_step else ''}")
    return history, best_coeffs, best_step, best_holdout


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    tag = a.tag or f"z{a.z_slice}_k{a.k}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(a.out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"output -> {out_dir}")

    counts_row, theta_deg, phi, dark_row, white_row, dark_var_row = load_row_data(a)
    n_proj, n_cols = counts_row.shape
    log(f"loaded slice z={a.z_slice}: counts {counts_row.shape}, theta {theta_deg.shape}, phi {phi.shape}")

    n_cols_recon = int(round(a.pad_factor * n_cols))
    pad = (n_cols_recon - n_cols) // 2
    n_cols_recon = n_cols + 2 * pad  # re-derive so it's always exactly n_cols + 2*pad
    fov_lo, fov_hi = pad, pad + n_cols
    log(f"pad_factor={a.pad_factor} -> volume widened to {n_cols_recon}x{n_cols_recon}, "
        f"detector stays at the real {n_cols} cols (no data padding); true FOV crop is "
        f"[{fov_lo}:{fov_hi}, {fov_lo}:{fov_hi}]")

    n_holdout = int(round(a.holdout_frac * n_proj))
    perm = rng.permutation(n_proj)
    holdout_idx = np.sort(perm[:n_holdout])
    held_in_idx = np.sort(perm[n_holdout:])
    log(f"held_in={len(held_in_idx)} holdout={len(holdout_idx)} (seed={a.seed})")

    to_t = lambda x: torch.from_numpy(np.ascontiguousarray(x)).to(device=device, dtype=torch.float32)
    I0 = to_t(white_row)
    dark = to_t(dark_row)
    sigma_read2 = to_t(dark_var_row)

    assert a.batch_size <= a.max_views_per_call, "--batch_size must not exceed --max_views_per_call"
    vol_shape = (1, n_cols_recon, n_cols_recon)
    det_shape = (1, n_cols)  # real detector width ALWAYS -- never widened/padded
    # Held-in views are NOT pre-split into fixed shards -- fit() draws a fresh
    # random --batch_size subset from held_in_idx every single step (true
    # mini-batch SGD; see fit()'s docstring comment for why a fixed-shards
    # version was wrong). Only the held-out evaluation set is fixed, since
    # "best held-out NLL" needs a consistent yardstick across steps.
    chunks_out = make_chunks(holdout_idx, theta_deg, phi, counts_row, a.max_views_per_call,
                             device, vol_shape, det_shape)
    log(f"held_in pool={len(held_in_idx)} views, resampled fresh (size {a.batch_size}) every step; "
        f"holdout into {len(chunks_out)} eval chunk(s) ({[c['n'] for c in chunks_out]})")

    phi_probe = None
    if a.pos_weight > 0:
        phi_probe = torch.linspace(0.0, 2.0 * np.pi, a.n_phi_probe + 1, device=device)[:-1]
        log(f"positivity penalty enabled: pos_weight={a.pos_weight}, {a.n_phi_probe} probe phases")

    sensitivity = None
    if a.precondition:
        log("computing sensitivity image (A^T . 1) over held-in views for gradient preconditioning...")
        t0 = time.time()
        sens_raw = compute_sensitivity(theta_deg, held_in_idx, vol_shape, det_shape, device,
                                        a.max_views_per_call)
        # Multiply (not divide): a voxel with LOW real ray coverage should have
        # its gradient DAMPED (smaller step), not amplified. Capped at 1.0 so
        # well-sampled voxels aren't boosted beyond their normal Adam-driven
        # update, only under-sampled ones are damped toward a smaller one.
        sensitivity = (sens_raw / sens_raw.mean().clamp_min(1e-8)).clamp(min=a.sens_floor, max=1.0)
        log(f"sensitivity image computed in {time.time()-t0:.1f}s "
            f"(raw mean={sens_raw.mean().item():.3f}, floor={a.sens_floor})")

    results = {}

    log(f"=== fitting k={a.k} ===")
    coeffs_k = torch.zeros(1 + 2 * a.k, n_cols_recon, n_cols_recon, device=device, requires_grad=True)
    t0 = time.time()
    hist_k, best_coeffs_k, best_step_k, best_holdout_k = fit(
        coeffs_k, held_in_idx, rng, theta_deg, phi, counts_row, device, vol_shape, det_shape,
        a.batch_size, chunks_out, I0, dark, sigma_read2, a.k, a.steps, a.lr,
        a.tv_weight, a.eval_every, a.grad_clip if a.grad_clip > 0 else None, log_prefix=f"[k={a.k}]",
        pos_weight=a.pos_weight, phi_probe=phi_probe, sensitivity=sensitivity, l_clamp=a.l_clamp)
    log(f"k={a.k} fit done in {time.time()-t0:.1f}s -- best holdout_nll={best_holdout_k:.4f} "
        f"at step={best_step_k} (final step holdout_nll={hist_k[-1]['holdout_nll']:.4f})")
    results["k"] = dict(k=a.k, history=hist_k, best_holdout_nll=best_holdout_k, best_step=best_step_k,
                        final_holdout_nll=hist_k[-1]["holdout_nll"])
    np.save(out_dir / f"coeffs_k{a.k}_best.npy", best_coeffs_k.cpu().numpy())
    np.save(out_dir / f"coeffs_k{a.k}_final.npy", coeffs_k.detach().cpu().numpy())
    np.save(out_dir / f"coeffs_k{a.k}_best_fov.npy",
            best_coeffs_k.cpu().numpy()[:, fov_lo:fov_hi, fov_lo:fov_hi])

    if a.fit_k0_baseline:
        log("=== fitting k=0 baseline ===")
        coeffs_0 = torch.zeros(1, n_cols_recon, n_cols_recon, device=device, requires_grad=True)
        t0 = time.time()
        hist_0, best_coeffs_0, best_step_0, best_holdout_0 = fit(
            coeffs_0, held_in_idx, rng, theta_deg, phi, counts_row, device, vol_shape, det_shape,
            a.batch_size, chunks_out, I0, dark, sigma_read2, 0, a.steps, a.lr,
            a.tv_weight, a.eval_every, a.grad_clip if a.grad_clip > 0 else None, log_prefix="[k=0]",
            pos_weight=a.pos_weight, phi_probe=phi_probe, sensitivity=sensitivity, l_clamp=a.l_clamp)
        log(f"k=0 fit done in {time.time()-t0:.1f}s -- best holdout_nll={best_holdout_0:.4f} "
            f"at step={best_step_0} (final step holdout_nll={hist_0[-1]['holdout_nll']:.4f})")
        results["k0"] = dict(k=0, history=hist_0, best_holdout_nll=best_holdout_0, best_step=best_step_0,
                             final_holdout_nll=hist_0[-1]["holdout_nll"])
        np.save(out_dir / "coeffs_k0_best.npy", best_coeffs_0.cpu().numpy())
        np.save(out_dir / "coeffs_k0_final.npy", coeffs_0.detach().cpu().numpy())
        np.save(out_dir / "coeffs_k0_best_fov.npy",
                best_coeffs_0.cpu().numpy()[:, fov_lo:fov_hi, fov_lo:fov_hi])

        delta = results["k0"]["best_holdout_nll"] - results["k"]["best_holdout_nll"]
        log(f"BEST held-out NLL: k=0 -> {results['k0']['best_holdout_nll']:.4f}, "
            f"k={a.k} -> {results['k']['best_holdout_nll']:.4f}  (improvement={delta:.4f}, "
            f"{'REAL SIGNAL' if delta > 0 else 'NO IMPROVEMENT -- likely overfitting to noise'})")

    results["pad_factor"] = a.pad_factor
    results["n_cols"] = n_cols
    results["n_cols_recon"] = n_cols_recon
    results["fov_crop"] = [fov_lo, fov_hi]
    results["pos_weight"] = a.pos_weight
    results["precondition"] = a.precondition
    results["l_clamp"] = a.l_clamp
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    log(f"=== phi sweep ({a.phi_sweep_frames} frames, using BEST checkpoint at step {best_step_k}) ===")
    coeffs_np = best_coeffs_k.cpu().numpy()
    phi_vals = np.linspace(0.0, 2.0 * np.pi, a.phi_sweep_frames, endpoint=False)
    frames = np.zeros((a.phi_sweep_frames, n_cols_recon, n_cols_recon), dtype=np.float32)
    for i, pv in enumerate(phi_vals):
        frame = coeffs_np[0].copy()
        for j in range(1, a.k + 1):
            frame += coeffs_np[j] * np.cos(j * pv) + coeffs_np[a.k + j] * np.sin(j * pv)
        frames[i] = frame
    np.savez(out_dir / "phi_sweep.npz", frames=frames, phi_vals=phi_vals)
    np.savez(out_dir / "phi_sweep_fov.npz",
             frames=frames[:, fov_lo:fov_hi, fov_lo:fov_hi], phi_vals=phi_vals)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        vmin, vmax = np.percentile(frames, [1, 99])
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(frames[0], cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"slice {a.z_slice}, k={a.k}, phi=0.00")
        ax.axis("off")

        def update(i):
            im.set_data(frames[i])
            ax.set_title(f"slice {a.z_slice}, k={a.k}, phi={phi_vals[i]:.2f}")
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=a.phi_sweep_frames, interval=80)
        ani.save(out_dir / "phi_sweep.gif", writer="pillow")
        plt.close(fig)
        log(f"saved phi-sweep GIF -> {out_dir / 'phi_sweep.gif'}")
    except Exception as e:
        log(f"GIF generation skipped ({type(e).__name__}: {e})")

    log(f"DONE. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
