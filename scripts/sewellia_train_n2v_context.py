#!/usr/bin/env python3
"""Train an N2V + joint-FBP-context baseline denoiser on the Sewellia lineolata
(theta, phi) preview data, then run inference on every projection and
reconstruct a few 3D slice comparisons (raw-noisy vs. denoised, at a narrow
phi-gate, plus a wide-gate raw reference) to see the effect of denoising.

Differences from the rest of tr_diffusion's baseline N2V+context pipeline
(see sdate/tr_diffusion/sewellia_n2v.py for why): no `.mov`/rotation window
(k=0, context is empty -- ALL conditioning comes from the T=11 phi-gated
joint-FBP taps built by scripts/sewellia_phi_context_prototype.py, fed as
`aux_channel`), plain Gaussian/L2 loss (no raw counts/darks-flats available
for this h5 preview file, so no Poisson-based loss), and the model is wrapped
to handle the 6-pixel-tall detector (PaddedUNet, see sewellia_n2v.py).

Trained on the FULL dataset (no held-out split) -- self-supervised per-scene
denoiser, no train/test generalization goal (see project memory
feedback_no_generalization_expected.md).

    python scripts/sewellia_train_n2v_context.py --epochs 40
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
from torch.utils.data import DataLoader

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")
sys.path.insert(0, "/myhome/BaseTraining")

from diffusers.optimization import get_cosine_schedule_with_warmup

from sdate.tr_diffusion import phi_context as PC
from sdate.tr_diffusion.losses import BaselineN2VLoss
from sdate.tr_diffusion.model import create_baseline_unet
from sdate.tr_diffusion.pipeline import denoise_frames_baseline
from sdate.tr_diffusion.sewellia_n2v import PaddedUNet, SewelliaN2VDataset

DATA_PATH = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5"
CTX_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/phi_context"
CTX_TAG = "sewellia_phictx_proto"
CKPT_DIR = "/mydata/sdate/shared/checkpoints"
OUT_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=DATA_PATH)
    p.add_argument("--ctx_dir", default=CTX_DIR)
    p.add_argument("--ctx_tag", default=CTX_TAG)
    p.add_argument("--T", type=int, default=11)
    p.add_argument("--exp_name", default="sewellia_n2v_phictx")
    p.add_argument("--ckpt_dir", default=CKPT_DIR)
    p.add_argument("--load_checkpoint", default=None, help="resume from this .pt (same exp config)")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--n2v_ratio", type=float, default=0.02)
    p.add_argument("--n2v_window", type=int, default=5)
    p.add_argument("--conditioning_probability", type=float, default=0.5)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--out_dir", default=OUT_DIR)
    p.add_argument("--recon_targets", default="4000,10000,16000")
    p.add_argument("--recon_pct_narrow", type=float, default=0.01)
    p.add_argument("--recon_pct_wide", type=float, default=0.5)
    p.add_argument("--recon_max_views", type=int, default=12000)
    return p.parse_args()


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    tap_paths = [f"{a.ctx_dir}/{a.ctx_tag}_tap{c}.f16" for c in range(a.T)]
    log("loading dataset (sinogram + %d context taps)..." % a.T)
    ds = SewelliaN2VDataset(a.data_path, tap_paths)
    log(f"dataset ready: n={ds.n} h={ds.h} w={ds.w} T={ds.T} "
        f"norm_min={ds.norm_min:.4f} norm_max={ds.norm_max:.4f}")

    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
                        pin_memory=True, drop_last=True)

    unet = create_baseline_unet(k=0, sample_size=(32, 576), extra_cond_channels=ds.T, poisson_head=False)
    model = PaddedUNet(unet, orig_hw=(ds.h, ds.w)).to(device)

    ckpt_path = Path(a.ckpt_dir) / f"{a.exp_name}.pt"
    config_path = Path(a.ckpt_dir) / f"{a.exp_name}_config.json"
    start_epoch = 0
    if a.load_checkpoint:
        sd = torch.load(a.load_checkpoint, map_location=device)
        unet.load_state_dict(sd["unet"])
        start_epoch = sd.get("epoch", 0)
        log(f"resumed from {a.load_checkpoint} at epoch {start_epoch}")

    loss_fn = BaselineN2VLoss(device, ratio=a.n2v_ratio, window=a.n2v_window,
                              conditioning_probability=a.conditioning_probability,
                              loss_type="mse", poisson_head=False)

    opt = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    total_steps = a.epochs * len(loader)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=min(500, max(1, total_steps // 20)),
                                            num_training_steps=max(1, total_steps))

    log(f"training: {len(ds)} samples, batch={a.batch_size}, {len(loader)} steps/epoch, "
        f"{a.epochs} epochs ({total_steps} total steps)")
    model.train()
    for epoch in range(start_epoch, a.epochs):
        t0 = time.time()
        losses = []
        for batch in loader:
            opt.zero_grad(set_to_none=True)
            loss, stats = loss_fn.compute_loss(batch, model)
            loss.backward()
            opt.step()
            sched.step()
            losses.append(stats["loss"])
        log(f"epoch {epoch+1}/{a.epochs}: mean_loss={np.mean(losses):.6f} "
            f"(min={np.min(losses):.6f} max={np.max(losses):.6f}) {time.time()-t0:.1f}s")

    Path(a.ckpt_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"unet": unet.state_dict(), "epoch": a.epochs}, ckpt_path)
    config = dict(
        exp_name=a.exp_name, data_path=a.data_path, ctx_dir=a.ctx_dir, ctx_tag=a.ctx_tag, T=ds.T,
        h=ds.h, w=ds.w, norm_min=ds.norm_min, norm_max=ds.norm_max, k=0, poisson_head=False,
        loss_type="mse", n2v_ratio=a.n2v_ratio, n2v_window=a.n2v_window,
        conditioning_probability=a.conditioning_probability, epochs=a.epochs,
        batch_size=a.batch_size, learning_rate=a.learning_rate,
    )
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log(f"saved checkpoint -> {ckpt_path}, config -> {config_path}")

    # --------------------------------------------------------------------- #
    # Inference: denoise every projection.
    # --------------------------------------------------------------------- #
    log("running inference on all projections...")
    model.eval()
    denoised = np.empty((ds.n, ds.h, ds.w), dtype=np.float32)
    infer_bs = 256
    empty_context = torch.zeros((infer_bs, 0, ds.h, ds.w), device=device)
    with torch.no_grad():
        for s in range(0, ds.n, infer_bs):
            e = min(s + infer_bs, ds.n)
            bsz = e - s
            central = ds.normalize(torch.from_numpy(ds.sino[s:e][:, None]).float()).to(device)
            aux = ds.normalize(torch.from_numpy(ds.aux[s:e].astype(np.float32))).to(device)
            ctx = empty_context[:bsz]
            out = denoise_frames_baseline(model, central, ctx, present=True, aux_channels=aux,
                                          poisson_head=False)
            denoised[s:e] = ds.denormalize(out).squeeze(1).cpu().numpy()
            if s % (infer_bs * 20) == 0:
                log(f"  inferred {e}/{ds.n}")
    denoised_path = Path(a.out_dir) / f"denoised_{a.exp_name}.npy"
    np.save(denoised_path, denoised)
    log(f"denoised sinogram saved -> {denoised_path}")

    # --------------------------------------------------------------------- #
    # Reconstruction comparison: narrow-gate raw vs. narrow-gate denoised vs.
    # wide-gate raw (reference), for a few representative targets.
    # --------------------------------------------------------------------- #
    with h5py.File(a.data_path, "r") as f:
        theta = f["theta"][:].astype(np.float64)
        phase = f["phase"][:].astype(np.float64)
    dark = torch.zeros((), device=device, dtype=torch.float32)
    flat = torch.ones((), device=device, dtype=torch.float32)
    det_shape = (ds.h, ds.w)
    vol_shape = (ds.h, ds.w, ds.w)
    slice_row = ds.h // 2

    targets = [int(x) for x in a.recon_targets.split(",")]
    results = {}
    for ti in targets:
        phi_c = float(phase[ti])
        r_narrow = a.recon_pct_narrow * PC.TWO_PI
        r_wide = a.recon_pct_wide * PC.TWO_PI
        log(f"reconstructing target i={ti} theta={theta[ti]:.1f} phase={phase[ti]:.2f} ...")

        vol_raw_narrow, n1 = PC.reconstruct_phi_gated(ds.sino, theta, phase, phi_c, r_narrow, dark, flat,
                                                      device, vol_shape=vol_shape, max_views=a.recon_max_views)
        vol_den_narrow, n2 = PC.reconstruct_phi_gated(denoised, theta, phase, phi_c, r_narrow, dark, flat,
                                                      device, vol_shape=vol_shape, max_views=a.recon_max_views)
        vol_raw_wide, n3 = PC.reconstruct_phi_gated(ds.sino, theta, phase, phi_c, r_wide, dark, flat,
                                                    device, vol_shape=vol_shape, max_views=a.recon_max_views)
        log(f"  n_proj: narrow={n1} wide={n3}")
        results[ti] = dict(
            raw_narrow=vol_raw_narrow[slice_row].cpu().numpy(),
            den_narrow=vol_den_narrow[slice_row].cpu().numpy(),
            raw_wide=vol_raw_wide[slice_row].cpu().numpy(),
            theta=float(theta[ti]), phase=float(phase[ti]), n_narrow=n1, n_wide=n3,
        )

    recon_path = Path(a.out_dir) / f"recon_compare_{a.exp_name}.npz"
    save_kwargs = {}
    for ti, r in results.items():
        for k, v in r.items():
            save_kwargs[f"t{ti}_{k}"] = v
    np.savez_compressed(recon_path, targets=np.array(targets), **save_kwargs)
    log(f"reconstruction comparisons saved -> {recon_path}")
    log("SUCCESS")


if __name__ == "__main__":
    main()
