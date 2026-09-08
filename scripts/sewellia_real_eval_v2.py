#!/usr/bin/env python3
"""Evaluate the trained N2V+v2-context denoiser in BOTH conditioning modes:

1. present=True: central projection passed (standard N2V denoising check).
2. present=False: central zeroed, denoised value comes ENTIRELY from the T=5
   context channels -- the "context-only" pathway the model was trained for
   50% of the time via conditioning_probability=0.5. This is the mode the
   angular-densification reconstruction (scripts/sewellia_real_densify_v2.py)
   depends on: hallucinating a plausible projection at an angle with NO real
   measurement, using only the phi-context.

For a few representative target frames: denoise in both modes, then
reconstruct narrow-phi-gate comparisons (raw vs present=True vs present=False)
so the effect of each mode is visible in the reconstruction domain, not just
projection-domain stats.

    python scripts/sewellia_real_eval_v2.py --ckpt_step <N>
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

TARGETS = [4000, 10000, 16000]
NARROW_VIEWS = 200  # "around 200 projections" -- the sparse narrow-window regime this whole exercise targets


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", default="sewellia_n2v_v2_fullres")
    p.add_argument("--ckpt_step", type=int, default=None, help="default: latest checkpoint found")
    p.add_argument("--ckpt_dir", default=CKPT_DIR)
    p.add_argument("--ctx_dir", default=CTX_DIR)
    p.add_argument("--ctx_tag", default=CTX_TAG)
    p.add_argument("--out_dir", default=OUT_DIR)
    return p.parse_args()


def chunked_denoise(model, central, empty_ctx, present, aux, norm_min, norm_max, chunk_size=16):
    """Full-res (608x576) UNet forward passes are large enough that a batch of
    ~200 views OOMs an 80GB A100 outright -- run in chunks and concatenate."""
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
    cands = sorted(ckpt_dir.glob(f"{exp_name}_step*.pt"),
                   key=lambda f: int(f.stem.split("step")[-1]))
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
    log(f"loading checkpoint -> {ckpt_path}")

    T = cfg["T"]
    unet = create_baseline_unet(k=0, sample_size=(608, 576), extra_cond_channels=T, poisson_head=True)
    model = PaddedUNet(unet, orig_hw=(cfg["h"], cfg["w"])).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(sd["unet"])
    model.eval()
    log(f"loaded checkpoint at step={sd['step']}")

    norm_min, norm_max = cfg["norm_min"], cfg["norm_max"]

    def normalize(x):
        return 2.0 * (x - norm_min) / (norm_max - norm_min) - 1.0

    def denormalize(x):
        return (x + 1.0) * 0.5 * (norm_max - norm_min) + norm_min

    with h5py.File(DATA_PATH, "r") as f:
        theta = f["exchange/theta"][:].astype(np.float64)
        raw_frames = {ti: f["exchange/data"][ti].astype(np.float32) for ti in TARGETS}
    phase = np.loadtxt(PHASE_TXT).astype(np.float64)
    z = np.load(CALIB_PATH)
    dark_full = torch.from_numpy(z["dark_mean"]).to(device=device, dtype=torch.float32)
    flat_full = torch.from_numpy(z["white_mean"]).to(device=device, dtype=torch.float32)

    tap_paths = [f"{a.ctx_dir}/{a.ctx_tag}_tap{c}.f16" for c in range(T)]
    n_proj, h, w = h5py.File(DATA_PATH, "r")["exchange/data"].shape
    aux_mms = [np.memmap(p, dtype=np.float16, mode="r", shape=(n_proj, h, w)) for p in tap_paths]

    vol_shape = (h, w, w)
    radius_narrow = NARROW_VIEWS * np.pi / n_proj

    from sdate.tr_diffusion import reconstruct as R

    results = {}
    for ti in TARGETS:
        log(f"target i={ti} theta={theta[ti]:.2f} phase={phase[ti]:.3f}")
        phi_c = phase[ti]
        mask = PC.phi_gate_mask(phase, phi_c, radius_narrow)
        idx = np.flatnonzero(mask)
        if idx.size > NARROW_VIEWS:
            stride = idx.size / NARROW_VIEWS
            idx = idx[np.floor(np.arange(NARROW_VIEWS) * stride).astype(np.int64)]
        log(f"  narrow window: {idx.size} real projections")

        with h5py.File(DATA_PATH, "r") as f:
            raw_window = f["exchange/data"][idx].astype(np.float32)  # (n_win, H, W)
        aux_window = np.stack([mm[idx] for mm in aux_mms], axis=1).astype(np.float32)  # (n_win, T, H, W)

        central = normalize(torch.from_numpy(raw_window[:, None]).to(device))
        aux = normalize(torch.from_numpy(aux_window).to(device))
        empty_ctx = torch.zeros((idx.size, 0, h, w), device=device)

        out_present = chunked_denoise(model, central, empty_ctx, True, aux, norm_min, norm_max)
        out_absent = chunked_denoise(model, central, empty_ctx, False, aux, norm_min, norm_max)
        den_present_window = denormalize(out_present).squeeze(1).cpu().numpy()
        den_absent_window = denormalize(out_absent).squeeze(1).cpu().numpy()
        log(f"  raw: mean={raw_window.mean():.2f} | present=True den: mean={den_present_window.mean():.2f} "
            f"| present=False den: mean={den_absent_window.mean():.2f}")

        recon = {}
        for name, sino_window in [("raw", raw_window), ("den_present", den_present_window),
                                  ("den_absent", den_absent_window)]:
            p = torch.from_numpy(sino_window).to(device=device, dtype=torch.float32)
            p_atten = R.counts_to_attenuation_flatdark(p, dark_full, flat_full)
            vol = R.reconstruct(p_atten, theta[idx], det_bin=2, method="fbp",
                               vol_shape=(h // 2, w // 2, w // 2), device=device)
            recon[name] = vol[h // 4].cpu().numpy()
            log(f"  {name}: n_views={idx.size} recon slice mean={recon[name].mean():.4g}")

        results[ti] = dict(recon_raw=recon["raw"], recon_den_present=recon["den_present"],
                           recon_den_absent=recon["den_absent"], theta=float(theta[ti]), phase=float(phase[ti]))

    out_path = Path(a.out_dir) / f"eval_v2_{ckpt_path.stem}.npz"
    save_kwargs = {}
    for ti, r in results.items():
        for k, v in r.items():
            save_kwargs[f"t{ti}_{k}"] = v
    np.savez_compressed(out_path, targets=np.array(TARGETS), **save_kwargs)
    log(f"saved -> {out_path}")
    log("SUCCESS")


if __name__ == "__main__":
    main()
