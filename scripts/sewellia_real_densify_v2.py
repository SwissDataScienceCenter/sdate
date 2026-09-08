#!/usr/bin/env python3
"""Prototype: angular-densification reconstruction.

Idea (from the user): a narrow phi-gate (~200 real projections, minimal
motion blur) is too angularly sparse for a clean FBP -- it streaks. A wide
gate has enough views to avoid streaking but blurs across genuine phase
motion. This script tries to get BOTH: reconstruct the (wide, clean) T=5
context volumes at the target phi, then REPROJECT them to many more
synthetic angles (default 500) than the narrow window has real measurements,
denoise each synthetic angle's context-only projection with the trained
model (present=False -- exactly the pathway conditioning_probability=0.5
training exists for), and combine the synthetic + real projections into ONE
FBP. Hope: the combined view density suppresses sparse-view streaks while the
underlying context still reflects a near-single-phi structural estimate
(narrow-window sharpness), rather than a genuinely-averaged-over-motion wide
gate.

This is exploratory -- run it, look at the result, don't over-trust it
without comparing against the plain narrow-gate and wide-gate baselines
(both computed here too).

    python scripts/sewellia_real_densify_v2.py --target 10000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cupy as cp
import h5py
import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

from sdate.tr_diffusion import phi_context as PC
from sdate.tr_diffusion import reconstruct as R
from sdate.tr_diffusion.model import create_baseline_unet
from sdate.tr_diffusion.pipeline import denoise_frames_baseline
from sdate.tr_diffusion.sewellia_n2v import PaddedUNet

FULL_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01"
DATA_PATH = f"{FULL_DIR}/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5"
PHASE_TXT = f"{FULL_DIR}/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01_sound_phase.txt"
CTX_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/phi_context"
CTX_TAG = "sewellia_v2_phictx"
CALIB_PATH = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/sewellia_real_calibration.npz"
CKPT_DIR = "/mydata/sdate/shared/checkpoints"
OUT_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata"

TARGET_VIEWS_CONTEXT = [400, 900, 1800, 2700, 3600]  # must match training's v2 context levels
GROUP_SIZE = 900
NARROW_VIEWS = 200
N_SYNTHETIC = 500


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=10000)
    p.add_argument("--exp_name", default="sewellia_n2v_v2_fullres")
    p.add_argument("--ckpt_step", type=int, default=None)
    p.add_argument("--ckpt_dir", default=CKPT_DIR)
    p.add_argument("--n_synthetic", type=int, default=N_SYNTHETIC)
    p.add_argument("--narrow_views", type=int, default=NARROW_VIEWS)
    p.add_argument("--out_dir", default=OUT_DIR)
    return p.parse_args()


def chunked_denoise(model, central, empty_ctx, present, aux, norm_min, norm_max, chunk_size=16):
    """Full-res (608x576) UNet forward passes are large enough that a batch of
    a few hundred views OOMs an 80GB A100 outright -- run in chunks and concatenate."""
    outs = []
    n = central.shape[0]
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        with torch.no_grad():
            out = denoise_frames_baseline(model, central[start:end], empty_ctx[start:end], present=present,
                                          aux_channels=aux[start:end], poisson_head=True,
                                          norm_min=norm_min, norm_max=norm_max)
        outs.append(out)
    return torch.cat(outs, dim=0)


def find_checkpoint(ckpt_dir: Path, exp_name: str, step):
    if step is not None:
        return ckpt_dir / f"{exp_name}_step{step}.pt"
    cands = sorted(ckpt_dir.glob(f"{exp_name}_step*.pt"), key=lambda f: int(f.stem.split("step")[-1]))
    if not cands:
        raise FileNotFoundError(f"no checkpoints found for {exp_name} in {ckpt_dir}")
    return cands[-1]


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    ckpt_dir = Path(a.ckpt_dir)
    with open(ckpt_dir / f"{a.exp_name}_config.json") as f:
        cfg = json.load(f)
    ckpt_path = find_checkpoint(ckpt_dir, a.exp_name, a.ckpt_step)
    T = cfg["T"]
    norm_min, norm_max = cfg["norm_min"], cfg["norm_max"]

    def normalize(x):
        return 2.0 * (x - norm_min) / (norm_max - norm_min) - 1.0

    def denormalize(x):
        return (x + 1.0) * 0.5 * (norm_max - norm_min) + norm_min

    unet = create_baseline_unet(k=0, sample_size=(608, 576), extra_cond_channels=T, poisson_head=True)
    model = PaddedUNet(unet, orig_hw=(cfg["h"], cfg["w"])).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(sd["unet"])
    model.eval()
    log(f"loaded checkpoint {ckpt_path} (step={sd['step']})")

    with h5py.File(DATA_PATH, "r") as f:
        sinogram = f["exchange/data"][:]
        theta = f["exchange/theta"][:].astype(np.float64)
    phase = np.loadtxt(PHASE_TXT).astype(np.float64)
    n_proj, h, w = sinogram.shape
    z = np.load(CALIB_PATH)
    dark_full = torch.from_numpy(z["dark_mean"]).to(device=device, dtype=torch.float32)
    flat_full = torch.from_numpy(z["white_mean"]).to(device=device, dtype=torch.float32)
    det_shape = (h, w)
    vol_shape = (h, w, w)

    ti = a.target
    phi_c = phase[ti]
    log(f"target i={ti} theta={theta[ti]:.2f} phase={phi_c:.3f}")

    # -- reconstruct the T=5 context volumes at this phi (same recipe as the cache build) --
    context_vols = []
    for target_h in TARGET_VIEWS_CONTEXT:
        max_groups = max(1, -(-target_h // GROUP_SIZE))  # ceil
        radius = target_h * np.pi / n_proj
        volumes, n_sel = PC.reconstruct_phi_gated_groups(sinogram, theta, phase, phi_c, radius, dark_full, flat_full,
                                                          device, det_bin=1, vol_shape=vol_shape,
                                                          group_size=GROUP_SIZE, max_groups=max_groups, clamp=False)
        vol = torch.stack(volumes).mean(dim=0).clamp_min(0.0)
        context_vols.append(vol)
        log(f"  context level target_h={target_h}: n_sel={n_sel} groups={len(volumes)}")
        del volumes
        torch.cuda.empty_cache()

    # -- narrow real window (baseline #1: plain sparse FBP) --
    radius_narrow = a.narrow_views * np.pi / n_proj
    mask = PC.phi_gate_mask(phase, phi_c, radius_narrow)
    idx_real = np.flatnonzero(mask)
    if idx_real.size > a.narrow_views:
        stride = idx_real.size / a.narrow_views
        idx_real = idx_real[np.floor(np.arange(a.narrow_views) * stride).astype(np.int64)]
    theta_real = theta[idx_real]
    log(f"narrow real window: {idx_real.size} views")

    # -- wide gate baseline (#2: the widest context level's own real target angles, just for reference) --
    radius_wide = TARGET_VIEWS_CONTEXT[-1] * np.pi / n_proj
    mask_wide = PC.phi_gate_mask(phase, phi_c, radius_wide)
    idx_wide = np.flatnonzero(mask_wide)
    theta_wide = theta[idx_wide]

    # -- synthetic angles: fill the gaps the narrow window leaves, spanning the same theta range as
    #    the wide gate (so the densified reconstruction covers as much angular range as the wide one,
    #    but each individual projection is denoised from an aux built at the SAME narrow-window phi) --
    theta_synth = np.linspace(theta_wide.min(), theta_wide.max(), a.n_synthetic)

    def build_aux_for_angles(angles_deg):
        chans = []
        for vol in context_vols:
            reproj_counts = PC.reproject_to_counts(vol, angles_deg, det_shape, device, dark_full, flat_full)
            chans.append(reproj_counts)
        return torch.stack(chans, dim=1)  # (n_angles, T, H, W)

    log(f"building aux for {idx_real.size} real + {a.n_synthetic} synthetic angles ...")
    aux_real = build_aux_for_angles(theta_real)
    aux_synth = build_aux_for_angles(theta_synth)

    empty_ctx_real = torch.zeros((idx_real.size, 0, h, w), device=device)
    empty_ctx_synth = torch.zeros((a.n_synthetic, 0, h, w), device=device)

    central_real = normalize(torch.from_numpy(sinogram[idx_real][:, None].astype(np.float32)).to(device))
    aux_real_n = normalize(aux_real)
    aux_synth_n = normalize(aux_synth)

    den_real = chunked_denoise(model, central_real, empty_ctx_real, True, aux_real_n, norm_min, norm_max)
    zeros_central_synth = torch.zeros((a.n_synthetic, 1, h, w), device=device)
    den_synth = chunked_denoise(model, zeros_central_synth, empty_ctx_synth, False, aux_synth_n, norm_min, norm_max)
    den_real_np = denormalize(den_real).squeeze(1).cpu().numpy()
    den_synth_np = denormalize(den_synth).squeeze(1).cpu().numpy()
    log(f"den_real mean={den_real_np.mean():.2f}  den_synth mean={den_synth_np.mean():.2f}")

    # ASTRA's internal CUDA allocator doesn't fully release memory between
    # reconstructions within a process (see project memory
    # project-sewellia-phi-context); the context-building + reprojection
    # stages above are the biggest consumers, so drop everything no longer
    # needed and free both pools before the FBP-heavy section below.
    del context_vols, aux_real, aux_synth, aux_real_n, aux_synth_n
    del central_real, empty_ctx_real, empty_ctx_synth, zeros_central_synth, den_real, den_synth
    torch.cuda.empty_cache()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()

    def fbp_from(sino_np, angles_deg, det_bin=2):
        p = torch.from_numpy(sino_np).to(device=device, dtype=torch.float32)
        p_atten = R.counts_to_attenuation_flatdark(p, dark_full, flat_full)
        vs = (h // det_bin, w // det_bin, w // det_bin)
        vol = R.reconstruct(p_atten, angles_deg, det_bin=det_bin, method="fbp", vol_shape=vs, device=device)
        return vol[h // (2 * det_bin)].cpu().numpy()

    log("reconstructing: narrow-raw, wide-raw, densified (real-denoised + synthetic) ...")
    # det_bin=2 FBP has an empirically measured GPU memory ceiling around
    # n_sel~2000-3200 (see project memory project-sewellia-phi-context) --
    # idx_wide can have thousands of views, so subsample for this baseline
    # reconstruction only (theta_wide/theta_synth's range is unaffected).
    WIDE_RECON_VIEWS = 900
    idx_wide_recon = idx_wide
    if idx_wide_recon.size > WIDE_RECON_VIEWS:
        stride = idx_wide_recon.size / WIDE_RECON_VIEWS
        idx_wide_recon = idx_wide_recon[np.floor(np.arange(WIDE_RECON_VIEWS) * stride).astype(np.int64)]
    log(f"wide baseline: {idx_wide.size} views in gate, subsampled to {idx_wide_recon.size} for FBP")

    with h5py.File(DATA_PATH, "r") as f:
        raw_narrow = f["exchange/data"][idx_real].astype(np.float32)
        raw_wide = f["exchange/data"][idx_wide_recon].astype(np.float32)

    slice_narrow_raw = fbp_from(raw_narrow, theta_real)
    slice_wide_raw = fbp_from(raw_wide, theta[idx_wide_recon])
    slice_narrow_denoised = fbp_from(den_real_np, theta_real)

    combined_sino = np.concatenate([den_real_np, den_synth_np], axis=0)
    combined_theta = np.concatenate([theta_real, theta_synth], axis=0)
    slice_densified = fbp_from(combined_sino, combined_theta)

    out_path = Path(a.out_dir) / f"densify_v2_i{ti}.npz"
    np.savez_compressed(out_path, target=ti, theta=theta[ti], phase=phi_c,
                        n_real=idx_real.size, n_synthetic=a.n_synthetic,
                        slice_narrow_raw=slice_narrow_raw, slice_wide_raw=slice_wide_raw,
                        slice_narrow_denoised=slice_narrow_denoised, slice_densified=slice_densified)
    log(f"saved -> {out_path}")
    log("SUCCESS")


if __name__ == "__main__":
    main()
