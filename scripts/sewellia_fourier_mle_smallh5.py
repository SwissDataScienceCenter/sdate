#!/usr/bin/env python3
"""Per-voxel truncated Fourier-in-phi MLE fit, adapted for the Sewellia
lineolata SMALL PREVIEW h5 (arXiv:2506.03792, PeriodRecon) -- the original
6-detector-row test/comparison dataset used for the paper's own figures, as
opposed to the full 580-row acquisition sewellia_fourier_mle_slice.py fits.

This is NOT the same data regime as sewellia_fourier_mle_slice.py and can't
reuse its likelihood: the small h5's ``sinogram`` dataset is already
flat/dark-corrected TRANSMISSION (dark=0, flat=1 identity -- confirmed
against notebooks/Sewellia_TimeResolved_Preview.ipynb and
scripts/sewellia_phi_context_prototype.py, both of which establish this same
file has no raw counts, no per-pixel I0/dark/read-noise calibration
available). There is therefore no Poisson-shot-noise model to build a
counts-domain NLL from here. Instead:

  atten = -log(clip(sinogram, eps, None))    (transmission -> line integral)

and the per-voxel model f(x,y,phi) = a0 + sum_j(aj*cos(j*phi)+bj*sin(j*phi))
is fit by minimizing (unweighted) Gaussian NLL directly in attenuation space,
i.e. plain MSE between predicted and measured attenuation (sigma2=1, a
constant that doesn't affect the optimum or the k=0-vs-k=2 comparison, only
the absolute NLL scale -- NOT comparable in absolute terms to the
counts-domain held-out NLL numbers from the full-volume script).

Consequences of this domain switch for two features carried over from the
full-volume script:

- De-ringing (--destripe_kernel): the full-volume script folds the FBP-style
  ring-bias correction into I0 rather than editing the sinogram/attenuation
  the Poisson likelihood is scored against, because doing the latter there
  would assert false shot-noise confidence around edited values. That
  concern does not apply here -- there is no counts likelihood, no I0, and
  the loss is already an unweighted Gaussian/MSE on attenuation. Subtracting
  a deterministic per-column bias from a Gaussian-noise observation before
  an MSE fit is exactly the FBP-side recipe (see
  sdate.tr_diffusion.reconstruct.destripe_sinogram) and is applied directly
  to the attenuation sinogram here, once, before fitting.

- --l_clamp / exp() overflow guard: gone entirely -- there is no I0*exp(-l)
  count-rate model to overflow, l_pred is compared directly to measured
  attenuation.

Everything else -- --pad_factor (widen vol_shape only, never det_shape),
the positivity penalty, the sensitivity-image (A^T.1) gradient
preconditioner, and per-step mini-batch resampling from the FULL held-in
pool (never a fixed shard) -- carries over unchanged; see
scripts/sewellia_fourier_mle_slice.py and the ct_padded_recon skill for the
full reasoning.

    python scripts/sewellia_fourier_mle_smallh5.py --z_row 0 --k 2 --steps 1200 \
        --pad_factor 1.2 --destripe_kernel 31
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
import torch.nn.functional as F

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

from astra_torch.lamino import build_lamino_projector, _create_lamino_geometry, _AstraLaminoOp

DATA_PATH = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5"
OUT_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/fourier_mle_smallh5"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=DATA_PATH)
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--z_row", type=int, default=0, help="row index into the small h5's 6 detector rows (0-5).")
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--fit_k0_baseline", action="store_true", default=True)
    p.add_argument("--no_k0_baseline", dest="fit_k0_baseline", action="store_false")
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--holdout_frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tv_weight", type=float, default=0.0)
    p.add_argument("--batch_size", type=int, default=2000,
                   help="Fresh random resample from the full held-in pool every step -- see "
                        "scripts/sewellia_fourier_mle_slice.py's --batch_size help for why a "
                        "fixed-shard version was wrong (flagged by the user 2026-09-06).")
    p.add_argument("--max_views_per_call", type=int, default=8000)
    p.add_argument("--grad_clip", type=float, default=1e4)
    p.add_argument("--pos_weight", type=float, default=1e4,
                   help="Soft positivity penalty weight, same convention as the full-volume script.")
    p.add_argument("--n_phi_probe", type=int, default=16)
    p.add_argument("--precondition", action="store_true", default=True)
    p.add_argument("--no_precondition", dest="precondition", action="store_false")
    p.add_argument("--sens_floor", type=float, default=0.05)
    p.add_argument("--eval_every", type=int, default=50)
    p.add_argument("--phi_sweep_frames", type=int, default=60)
    p.add_argument("--pad_factor", type=float, default=1.0,
                   help="Widen the reconstructed volume only (never the detector) -- see "
                        "scripts/sewellia_fourier_mle_slice.py's --pad_factor help. 1.0 = old "
                        "behavior (no extra FOV, halo present).")
    p.add_argument("--atten_eps", type=float, default=1e-2,
                   help="Floor applied to the raw transmission ratio before -log(): this small "
                        "h5's sinogram is a noisy I/I0 ratio that occasionally dips at/below 0 "
                        "(detector noise, not physical), which -log() can't handle. 1e-2 treats "
                        "anything below 1%% transmission as fully opaque rather than letting a "
                        "handful of noise-driven negative samples blow up to +inf attenuation.")
    p.add_argument("--destripe_kernel", type=int, default=0,
                   help="If >0, subtract a per-detector-column ring-artifact bias directly from "
                        "the attenuation sinogram before fitting (estimated from held-in "
                        "view-averaged attenuation, same method as "
                        "sdate.tr_diffusion.reconstruct.destripe_sinogram). Unlike the full-volume "
                        "script's I0-correction workaround, editing the attenuation directly is "
                        "legitimate here -- see module docstring for why. 0 disables.")
    p.add_argument("--tag", default=None)
    return p.parse_args()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_row_data(a):
    with h5py.File(a.data_path, "r") as f:
        sinogram = f["sinogram"][:, a.z_row, :].astype(np.float32)  # (n_proj, n_pix), transmission I/I0
        theta_deg = f["theta"][:].astype(np.float64)
        phase = f["phase"][:].astype(np.float64)
    atten_row = -np.log(np.clip(sinogram, a.atten_eps, None)).astype(np.float32)
    return atten_row, theta_deg, phase


def build_weights(phi: torch.Tensor, k: int) -> torch.Tensor:
    rows = [torch.ones_like(phi)]
    for j in range(1, k + 1):
        rows.append(torch.cos(j * phi))
    for j in range(1, k + 1):
        rows.append(torch.sin(j * phi))
    return torch.stack(rows, dim=0)


def positivity_penalty(coeffs: torch.Tensor, k: int, phi_probe: torch.Tensor) -> torch.Tensor:
    weights = build_weights(phi_probe, k)
    frame_probe = torch.einsum("bp,bhw->phw", weights, coeffs)
    return torch.relu(-frame_probe).pow(2).mean()


def project_basis(projector, coeffs: torch.Tensor, n_cols: int) -> torch.Tensor:
    n_basis = coeffs.shape[0]
    vol = coeffs.view(n_basis, 1, 1, coeffs.shape[1], coeffs.shape[2])
    proj = projector(vol)
    return proj[:, :, 0, :]


def estimate_ring_bias(atten_row: np.ndarray, idx: np.ndarray, device, kernel: int = 31) -> torch.Tensor:
    """Per-detector-column ring bias directly in attenuation space -- see module
    docstring for why editing the attenuation is legitimate here (no counts
    likelihood to fabricate data under)."""
    sub = torch.from_numpy(np.ascontiguousarray(atten_row[idx])).to(device=device, dtype=torch.float32)
    col_mean = sub.mean(dim=0)  # (n_cols,)
    pad = kernel // 2
    padded = F.pad(col_mean.view(1, 1, -1), (pad, pad), mode="reflect")
    smoothed = F.avg_pool1d(padded, kernel, stride=1).view(-1)
    return col_mean - smoothed


def build_chunk(idx, theta_deg, phi, atten_row, device, vol_shape, det_shape):
    projector = build_lamino_projector(vol_shape=vol_shape, det_shape=det_shape,
                                       angles_deg=theta_deg[idx], lamino_angle_deg=0.0,
                                       device=device)
    phi_t = torch.from_numpy(phi[idx]).to(device=device, dtype=torch.float32)
    atten_t = torch.from_numpy(np.ascontiguousarray(atten_row[idx])).to(device=device, dtype=torch.float32)
    return dict(projector=projector, phi=phi_t, atten=atten_t, n=len(idx))


def make_chunks(idx, theta_deg, phi, atten_row, max_views, device, vol_shape, det_shape):
    return [build_chunk(idx[start:start + max_views], theta_deg, phi, atten_row, device, vol_shape, det_shape)
            for start in range(0, len(idx), max_views)]


def compute_sensitivity(theta_deg, idx, vol_shape, det_shape, device, max_views):
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


def chunk_mse(chunk, coeffs, k, n_cols):
    weights = build_weights(chunk["phi"], k)
    basis_proj = project_basis(chunk["projector"], coeffs, n_cols)
    l_pred = torch.einsum("bn,bnc->nc", weights, basis_proj)
    return 0.5 * (chunk["atten"] - l_pred).pow(2).mean()


def fit(coeffs, held_in_idx, rng, theta_deg, phi, atten_row, device, vol_shape, det_shape,
        batch_size, chunks_out, k, steps, lr, tv_weight, eval_every, grad_clip, log_prefix,
        pos_weight=0.0, phi_probe=None, sensitivity=None):
    n_basis = coeffs.shape[0]
    n_cols = coeffs.shape[2]

    optimizer = torch.optim.Adam([coeffs], lr=lr)
    history = []
    best_holdout = float("inf")
    best_coeffs = coeffs.detach().clone()
    best_step = 0
    for step in range(steps):
        sub = rng.choice(held_in_idx, size=min(batch_size, len(held_in_idx)), replace=False)
        batch = build_chunk(sub, theta_deg, phi, atten_row, device, vol_shape, det_shape)

        optimizer.zero_grad(set_to_none=True)
        loss = chunk_mse(batch, coeffs, k, n_cols)
        if tv_weight > 0:
            dz = (coeffs[1:, :, :] - coeffs[:-1, :, :]).abs().sum()
            dy = (coeffs[:, 1:, :] - coeffs[:, :-1, :]).abs().sum()
            dx = (coeffs[:, :, 1:] - coeffs[:, :, :-1]).abs().sum()
            loss = loss + tv_weight * (dz + dy + dx)
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
                    chunk_mse(c, coeffs, k, n_cols).item() * (c["n"] / denom_out)
                    for c in chunks_out
                )
            rec = dict(step=step, train_loss=float(train_loss_val), holdout_nll=holdout_nll)
            history.append(rec)
            if holdout_nll < best_holdout:
                best_holdout = holdout_nll
                best_step = step
                best_coeffs = coeffs.detach().clone()
                tag = " *best*"
            else:
                tag = ""
            log(f"{log_prefix} step={step} train_loss={train_loss_val:.4f} "
                f"holdout_nll={holdout_nll:.4f}{tag}")
    return history, best_coeffs, best_step, best_holdout


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")
    rng = np.random.default_rng(a.seed)

    tag = a.tag or f"row{a.z_row}_k{a.k}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(a.out_dir) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"output -> {out_dir}")

    atten_row, theta_deg, phi = load_row_data(a)
    n_proj, n_cols = atten_row.shape
    log(f"loaded row z_row={a.z_row}: atten {atten_row.shape}, theta {theta_deg.shape}, phi {phi.shape}")

    n_cols_recon = int(round(a.pad_factor * n_cols))
    pad = (n_cols_recon - n_cols) // 2
    n_cols_recon = n_cols + 2 * pad
    fov_lo, fov_hi = pad, pad + n_cols
    log(f"pad_factor={a.pad_factor} -> volume widened to {n_cols_recon}x{n_cols_recon}, "
        f"detector stays at the real {n_cols} cols; true FOV crop is [{fov_lo}:{fov_hi}, {fov_lo}:{fov_hi}]")

    n_holdout = int(round(a.holdout_frac * n_proj))
    perm = rng.permutation(n_proj)
    holdout_idx = np.sort(perm[:n_holdout])
    held_in_idx = np.sort(perm[n_holdout:])
    log(f"held_in={len(held_in_idx)} holdout={len(holdout_idx)} (seed={a.seed})")

    if a.destripe_kernel > 0:
        ring_bias = estimate_ring_bias(atten_row, held_in_idx, device, kernel=a.destripe_kernel)
        ring_bias_np = ring_bias.cpu().numpy()
        atten_row = atten_row - ring_bias_np[None, :]
        log(f"de-ringing: subtracted per-column bias from attenuation (kernel={a.destripe_kernel}), "
            f"bias range [{ring_bias.min().item():.4f}, {ring_bias.max().item():.4f}]")

    assert a.batch_size <= a.max_views_per_call, "--batch_size must not exceed --max_views_per_call"
    vol_shape = (1, n_cols_recon, n_cols_recon)
    det_shape = (1, n_cols)
    chunks_out = make_chunks(holdout_idx, theta_deg, phi, atten_row, a.max_views_per_call,
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
        sensitivity = (sens_raw / sens_raw.mean().clamp_min(1e-8)).clamp(min=a.sens_floor, max=1.0)
        log(f"sensitivity image computed in {time.time()-t0:.1f}s "
            f"(raw mean={sens_raw.mean().item():.3f}, floor={a.sens_floor})")

    results = {}

    log(f"=== fitting k={a.k} ===")
    coeffs_k = torch.zeros(1 + 2 * a.k, n_cols_recon, n_cols_recon, device=device, requires_grad=True)
    t0 = time.time()
    hist_k, best_coeffs_k, best_step_k, best_holdout_k = fit(
        coeffs_k, held_in_idx, rng, theta_deg, phi, atten_row, device, vol_shape, det_shape,
        a.batch_size, chunks_out, a.k, a.steps, a.lr,
        a.tv_weight, a.eval_every, a.grad_clip if a.grad_clip > 0 else None, log_prefix=f"[k={a.k}]",
        pos_weight=a.pos_weight, phi_probe=phi_probe, sensitivity=sensitivity)
    log(f"k={a.k} fit done in {time.time()-t0:.1f}s -- best holdout_nll={best_holdout_k:.4f} "
        f"at step={best_step_k} (final step holdout_nll={hist_k[-1]['holdout_nll']:.4f})")
    results["k"] = dict(k=a.k, history=hist_k, best_holdout_nll=best_holdout_k, best_step=best_step_k,
                        final_holdout_nll=hist_k[-1]["holdout_nll"])
    np.save(out_dir / f"coeffs_k{a.k}_best.npy", best_coeffs_k.cpu().numpy())
    np.save(out_dir / f"coeffs_k{a.k}_best_fov.npy",
            best_coeffs_k.cpu().numpy()[:, fov_lo:fov_hi, fov_lo:fov_hi])

    if a.fit_k0_baseline:
        log("=== fitting k=0 baseline ===")
        coeffs_0 = torch.zeros(1, n_cols_recon, n_cols_recon, device=device, requires_grad=True)
        t0 = time.time()
        hist_0, best_coeffs_0, best_step_0, best_holdout_0 = fit(
            coeffs_0, held_in_idx, rng, theta_deg, phi, atten_row, device, vol_shape, det_shape,
            a.batch_size, chunks_out, 0, a.steps, a.lr,
            a.tv_weight, a.eval_every, a.grad_clip if a.grad_clip > 0 else None, log_prefix="[k=0]",
            pos_weight=a.pos_weight, phi_probe=phi_probe, sensitivity=sensitivity)
        log(f"k=0 fit done in {time.time()-t0:.1f}s -- best holdout_nll={best_holdout_0:.4f} "
            f"at step={best_step_0} (final step holdout_nll={hist_0[-1]['holdout_nll']:.4f})")
        results["k0"] = dict(k=0, history=hist_0, best_holdout_nll=best_holdout_0, best_step=best_step_0,
                             final_holdout_nll=hist_0[-1]["holdout_nll"])
        np.save(out_dir / "coeffs_k0_best.npy", best_coeffs_0.cpu().numpy())
        np.save(out_dir / "coeffs_k0_best_fov.npy",
                best_coeffs_0.cpu().numpy()[:, fov_lo:fov_hi, fov_lo:fov_hi])

        delta = results["k0"]["best_holdout_nll"] - results["k"]["best_holdout_nll"]
        log(f"BEST held-out NLL: k=0 -> {results['k0']['best_holdout_nll']:.4f}, "
            f"k={a.k} -> {results['k']['best_holdout_nll']:.4f}  (improvement={delta:.4f}, "
            f"{'REAL SIGNAL' if delta > 0 else 'NO IMPROVEMENT -- likely overfitting to noise'})")

    results["z_row"] = a.z_row
    results["pad_factor"] = a.pad_factor
    results["n_cols"] = n_cols
    results["n_cols_recon"] = n_cols_recon
    results["fov_crop"] = [fov_lo, fov_hi]
    results["pos_weight"] = a.pos_weight
    results["precondition"] = a.precondition
    results["destripe_kernel"] = a.destripe_kernel
    results["atten_eps"] = a.atten_eps
    results["loss"] = "unweighted Gaussian NLL (sigma2=1) on attenuation -- NOT comparable in " \
                       "absolute scale to the full-volume script's counts-domain NLL"
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

    # Fixed data range over time: one vmin/vmax computed ONCE over the WHOLE
    # frame stack (all phi), reused for every frame -- never per-frame
    # autoscale (see the "Two Bugs, One Halo" dashboard for why that's wrong).
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.animation as animation

        fov_frames = frames[:, fov_lo:fov_hi, fov_lo:fov_hi]
        vmin, vmax = np.percentile(fov_frames, [1, 99])
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(fov_frames[0], cmap="gray", vmin=vmin, vmax=vmax)
        ax.set_title(f"row {a.z_row}, k={a.k}, phi=0.00")
        ax.axis("off")

        def update(i):
            im.set_data(fov_frames[i])
            ax.set_title(f"row {a.z_row}, k={a.k}, phi={phi_vals[i]:.2f}")
            return [im]

        ani = animation.FuncAnimation(fig, update, frames=a.phi_sweep_frames, interval=80)
        ani.save(out_dir / "phi_sweep_fov.gif", writer="pillow")
        plt.close(fig)
        log(f"saved phi-sweep GIF (fixed range [{vmin:.4f},{vmax:.4f}]) -> {out_dir / 'phi_sweep_fov.gif'}")

        # A few full-res stills at representative phi values, for visual
        # comparison against the paper's own figures.
        still_phis_deg = [0, 90, 180, 270]
        for pd in still_phis_deg:
            i = int(round(pd / 360.0 * a.phi_sweep_frames)) % a.phi_sweep_frames
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(fov_frames[i], cmap="gray", vmin=vmin, vmax=vmax)
            ax.axis("off")
            plt.tight_layout(pad=0)
            plt.savefig(out_dir / f"still_row{a.z_row}_phi{pd:03d}.png", dpi=150)
            plt.close(fig)
        log(f"saved {len(still_phis_deg)} full-res stills at phi={still_phis_deg}deg")
    except Exception as e:
        log(f"GIF/still generation skipped ({type(e).__name__}: {e})")

    log(f"DONE. All outputs in {out_dir}")


if __name__ == "__main__":
    main()
