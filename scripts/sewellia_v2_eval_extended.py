#!/usr/bin/env python3
"""Extended eval: multiple z-slices per reconstruction (not just the single
zscan-confirmed row) AND the raw projection-domain comparison (denoising
effect BEFORE any reconstruction), for the same 3 target frames as
sewellia_real_eval_v2.py.

    python scripts/sewellia_v2_eval_extended.py --exp_name sewellia_n2v_v2_fullres
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
sys.path.insert(0, "/myhome/sdate/scripts")

from sdate.tr_diffusion import phi_context as PC
from sdate.tr_diffusion import reconstruct as R
from sdate.tr_diffusion.model import create_baseline_unet
from sdate.tr_diffusion.sewellia_n2v import PaddedUNet

from sewellia_real_eval_v2 import chunked_denoise, find_checkpoint  # reuse

FULL_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01"
DATA_PATH = f"{FULL_DIR}/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5"
PHASE_TXT = f"{FULL_DIR}/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01_sound_phase.txt"
CTX_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/phi_context"
CTX_TAG = "sewellia_v2_phictx"
CALIB_PATH = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/sewellia_real_calibration.npz"
CKPT_DIR = "/mydata/sdate/shared/checkpoints"
OUT_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata"

TARGETS = [4000, 10000, 16000]
NARROW_VIEWS = 200
SLICE_FRACTIONS = [0.25, 0.40, 0.50, 0.60, 0.75]  # of the det_bin=2 volume height


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", default="sewellia_n2v_v2_fullres")
    p.add_argument("--ckpt_step", type=int, default=None)
    p.add_argument("--ckpt_dir", default=CKPT_DIR)
    p.add_argument("--ctx_dir", default=CTX_DIR)
    p.add_argument("--ctx_tag", default=CTX_TAG)
    p.add_argument("--out_dir", default=OUT_DIR)
    return p.parse_args()


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
    phase = np.loadtxt(PHASE_TXT).astype(np.float64)
    z = np.load(CALIB_PATH)
    dark_full = torch.from_numpy(z["dark_mean"]).to(device=device, dtype=torch.float32)
    flat_full = torch.from_numpy(z["white_mean"]).to(device=device, dtype=torch.float32)

    tap_paths = [f"{a.ctx_dir}/{a.ctx_tag}_tap{c}.f16" for c in range(T)]
    n_proj, h, w = h5py.File(DATA_PATH, "r")["exchange/data"].shape
    aux_mms = [np.memmap(p, dtype=np.float16, mode="r", shape=(n_proj, h, w)) for p in tap_paths]

    radius_narrow = NARROW_VIEWS * np.pi / n_proj
    vol_h2 = h // 2
    slice_rows = [int(round(f * vol_h2)) for f in SLICE_FRACTIONS]
    log(f"saving slices at det_bin=2 rows: {slice_rows} (of {vol_h2})")

    save_kwargs = {}
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

        # -- projection-domain comparison: the window element closest to the target's own angle/phase --
        local = int(np.argmin(np.abs(idx - ti)))
        proj_raw = raw_window[local]
        proj_den_present = den_present_window[local]
        proj_den_absent = den_absent_window[local]
        log(f"  projection-domain @ local idx {local} (real i={idx[local]}): raw mean={proj_raw.mean():.2f} "
            f"den_present mean={proj_den_present.mean():.2f} den_absent mean={proj_den_absent.mean():.2f}")

        # -- reconstruct FULL volumes (det_bin=2) and keep multiple z-slices --
        recon_vols = {}
        for name, sino_window in [("raw", raw_window), ("den_present", den_present_window),
                                  ("den_absent", den_absent_window)]:
            p = torch.from_numpy(sino_window).to(device=device, dtype=torch.float32)
            p_atten = R.counts_to_attenuation_flatdark(p, dark_full, flat_full)
            vol = R.reconstruct(p_atten, theta[idx], det_bin=2, method="fbp",
                               vol_shape=(h // 2, w // 2, w // 2), device=device)
            recon_vols[name] = vol.cpu().numpy()
            del vol
            torch.cuda.empty_cache()

        for name, vol in recon_vols.items():
            for row in slice_rows:
                save_kwargs[f"t{ti}_recon_{name}_row{row}"] = vol[row]
        save_kwargs[f"t{ti}_proj_raw"] = proj_raw
        save_kwargs[f"t{ti}_proj_den_present"] = proj_den_present
        save_kwargs[f"t{ti}_proj_den_absent"] = proj_den_absent
        save_kwargs[f"t{ti}_theta"] = np.float64(theta[ti])
        save_kwargs[f"t{ti}_phase"] = np.float64(phase[ti])

    out_path = Path(a.out_dir) / f"eval_v2_extended_{ckpt_path.stem}.npz"
    np.savez_compressed(out_path, targets=np.array(TARGETS), slice_rows=np.array(slice_rows), **save_kwargs)
    log(f"saved -> {out_path}")
    log("SUCCESS")


if __name__ == "__main__":
    main()
