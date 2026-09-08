#!/usr/bin/env python3
"""Denoise ALL 20000 real projections (present=True: the real central
projection is passed, standard denoising mode -- NOT the context-only
present=False pathway used for synthetic/absent angles) using the trained
N2V+v2-context checkpoint, and write out a new HDF5 file that is a byte-for-
byte structural clone of the original DataExchange file (same groups,
datasets, dtypes, shapes, dark/white/theta arrays UNCHANGED) with only
`exchange/data` replaced by the denoised counts -- so any code that reads the
original file (DatasetProfile, phi_context, reconstruct.py, ...) can swap in
this file's directory instead with zero changes.

Output layout (mirrors the original FULL_DIR convention exactly, so the
phase-sidecar-derivation-by-basename pattern still works):

    <OUT_ROOT>/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01/
        SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5          <- clone, exchange/data denoised
        SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01_sound_phase.txt  <- copied unchanged (phase is a
                                                                      per-frame physical quantity,
                                                                      not affected by denoising)
        denoise_metadata.json                                  <- how this was generated

    python scripts/sewellia_v2_denoise_full_dataset.py --exp_name sewellia_n2v_v2_fullres
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")
sys.path.insert(0, "/myhome/sdate/scripts")

from sdate.tr_diffusion.model import create_baseline_unet
from sdate.tr_diffusion.pipeline import denoise_frames_baseline
from sdate.tr_diffusion.sewellia_n2v import PaddedUNet

FULL_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01"
DATA_PATH = f"{FULL_DIR}/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5"
PHASE_TXT = f"{FULL_DIR}/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01_sound_phase.txt"
CTX_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/phi_context"
CTX_TAG = "sewellia_v2_phictx"
CKPT_DIR = "/mydata/sdate/shared/checkpoints"
OUT_ROOT = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata"
BASENAME = "SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01"

CHUNK = 16  # matches the GPU-memory-safe chunk size established during eval


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--exp_name", default="sewellia_n2v_v2_fullres")
    p.add_argument("--ckpt_step", type=int, default=None)
    p.add_argument("--ckpt_dir", default=CKPT_DIR)
    p.add_argument("--ctx_dir", default=CTX_DIR)
    p.add_argument("--ctx_tag", default=CTX_TAG)
    p.add_argument("--out_root", default=OUT_ROOT)
    p.add_argument("--out_tag", default="denoised_v2_fullres_present_true")
    p.add_argument("--chunk", type=int, default=CHUNK)
    return p.parse_args()


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
    log(f"checkpoint={ckpt_path} T={T} norm_min={norm_min:.2f} norm_max={norm_max:.2f}")

    unet = create_baseline_unet(k=0, sample_size=(608, 576), extra_cond_channels=T, poisson_head=True)
    model = PaddedUNet(unet, orig_hw=(cfg["h"], cfg["w"])).to(device)
    sd = torch.load(ckpt_path, map_location=device)
    unet.load_state_dict(sd["unet"])
    model.eval()
    ckpt_step = sd["step"]
    log(f"loaded checkpoint at step={ckpt_step}")

    def normalize(x):
        return 2.0 * (x - norm_min) / (norm_max - norm_min) - 1.0

    def denormalize(x):
        return (x + 1.0) * 0.5 * (norm_max - norm_min) + norm_min

    n_proj, h, w = h5py.File(DATA_PATH, "r")["exchange/data"].shape
    tap_paths = [f"{a.ctx_dir}/{a.ctx_tag}_tap{c}.f16" for c in range(T)]
    for p in tap_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"context tap not found: {p}")
    aux_mms = [np.memmap(p, dtype=np.float16, mode="r", shape=(n_proj, h, w)) for p in tap_paths]

    out_dir = Path(a.out_root) / a.out_tag / BASENAME
    out_dir.mkdir(parents=True, exist_ok=True)
    out_h5_path = out_dir / f"{BASENAME}.h5"
    out_phase_path = out_dir / f"{BASENAME}_sound_phase.txt"

    assert out_h5_path.resolve() != Path(DATA_PATH).resolve(), (
        f"REFUSING to proceed: output path {out_h5_path} resolves to the same file as the "
        f"source {DATA_PATH} -- this would overwrite real measurement data in place.")

    if not out_h5_path.exists():
        log(f"cloning original file structure -> {out_h5_path} (this copies ~{Path(DATA_PATH).stat().st_size/1e9:.1f}GB) ...")
        t0 = time.time()
        shutil.copyfile(DATA_PATH, out_h5_path)
        log(f"clone done in {time.time()-t0:.1f}s")
    else:
        log(f"output file already exists, reusing -> {out_h5_path}")

    shutil.copyfile(PHASE_TXT, out_phase_path)
    log(f"copied phase sidecar -> {out_phase_path}")

    t0 = time.time()
    with h5py.File(DATA_PATH, "r") as fin, h5py.File(out_h5_path, "r+") as fout:
        ds_out = fout["exchange/data"]
        assert ds_out.shape == (n_proj, h, w) and ds_out.dtype == np.uint16
        for start in range(0, n_proj, a.chunk):
            end = min(start + a.chunk, n_proj)
            raw_chunk = fin["exchange/data"][start:end].astype(np.float32)  # (b,H,W)
            aux_chunk = np.stack([mm[start:end] for mm in aux_mms], axis=1).astype(np.float32)  # (b,T,H,W)

            central = normalize(torch.from_numpy(raw_chunk[:, None]).to(device))
            aux = normalize(torch.from_numpy(aux_chunk).to(device))
            empty_ctx = torch.zeros((end - start, 0, h, w), device=device)

            with torch.no_grad():
                out = denoise_frames_baseline(model, central, empty_ctx, present=True, aux_channels=aux,
                                              poisson_head=True, norm_min=norm_min, norm_max=norm_max)
            den = denormalize(out).squeeze(1).cpu().numpy()
            den_u16 = np.clip(np.round(den), 0, 65535).astype(np.uint16)
            ds_out[start:end] = den_u16

            if (start // a.chunk) % 50 == 0:
                elapsed = time.time() - t0
                rate = (start + (end - start)) / max(elapsed, 1e-6)
                eta = (n_proj - end) / max(rate, 1e-6)
                log(f"  {end}/{n_proj} done ({rate:.1f} frames/s, ETA {eta/60:.1f}min) "
                    f"raw_mean={raw_chunk.mean():.2f} den_mean={den.mean():.2f}")

    log(f"denoising complete in {(time.time()-t0)/60:.1f}min")

    meta = dict(
        generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        source_h5=DATA_PATH,
        source_phase_txt=PHASE_TXT,
        checkpoint_path=str(ckpt_path),
        checkpoint_step=int(ckpt_step),
        exp_name=a.exp_name,
        denoise_mode="present=True (real central projection passed; standard denoising, "
                     "NOT the context-only present=False pathway)",
        model_config=dict(T=T, poisson_head=True, gaussian_floor=cfg.get("gaussian_floor"),
                          conditioning_probability=cfg.get("conditioning_probability"),
                          norm_min=norm_min, norm_max=norm_max, k=0,
                          sample_size=[608, 576], native_hw=[cfg["h"], cfg["w"]]),
        context_taps=dict(ctx_dir=a.ctx_dir, ctx_tag=a.ctx_tag, T=T),
        output_encoding="uint16, same dtype/shape as exchange/data in the source file; denoised "
                        "posterior-mean counts rounded to nearest integer and clipped to [0, 65535]",
        unchanged_fields=["exchange/data_dark", "exchange/data_white", "exchange/theta",
                          "exchange/theta_dark", "exchange/theta_dark_raw", "exchange/theta_raw",
                          "exchange/theta_white", "exchange/theta_white_raw",
                          "measurement/instrument/acquisition/*", "phase sidecar .txt"],
        notes="This file is a structural clone of the original DataExchange HDF5 with ONLY "
              "exchange/data replaced. Point any existing DatasetProfile/loader at this "
              "directory instead of the original to use denoised projections transparently -- "
              "same file basename, same shape/dtype, same darks/flats/theta/phase.",
        script="scripts/sewellia_v2_denoise_full_dataset.py",
    )
    meta_path = out_dir / "denoise_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    log(f"wrote metadata -> {meta_path}")
    log(f"OUTPUT DIRECTORY: {out_dir}")
    log("SUCCESS")


if __name__ == "__main__":
    main()
