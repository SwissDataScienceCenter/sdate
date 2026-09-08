#!/usr/bin/env python3
"""Train an N2V + T=5 phi-context-groups baseline denoiser on the REAL full
Sewellia lineolata dataset (genuine raw counts + real darks/flats, full
native resolution). Supersedes scripts/sewellia_train_n2v_context.py (built
for the small precorrected preview file: plain MSE loss, det_bin=4 upsampled
context, k=0 PaddedUNet for a 6-pixel-tall detector).

conditioning_probability=0.5 (default) is the EXISTING N2V conditioning-dropout
mechanism (see sdate/tr_diffusion/losses.py's _N2VLossBase) -- already gives
the "50% central passed / 50% central zeroed, context-only" training regime
the user asked for. Nothing new needed there, just the right data + loss.

Checkpoints every --ckpt_every steps and auto-resumes from the latest one on
restart (this unattended run WILL be preempted on this shared cluster at some
point over many hours) -- see save_checkpoint/find_latest_checkpoint.

    python scripts/sewellia_real_train_v2.py --epochs 30
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")
sys.path.insert(0, "/myhome/BaseTraining")

from diffusers.optimization import get_cosine_schedule_with_warmup

from sdate.tr_diffusion.losses import BaselineN2VLoss
from sdate.tr_diffusion.model import create_baseline_unet
from sdate.tr_diffusion.sewellia_n2v import PaddedUNet
from sdate.tr_diffusion.sewellia_real_n2v import SewelliaRealN2VDataset

FULL_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01"
DATA_PATH = f"{FULL_DIR}/SL2022_3_SC_10kHz_550Hz_0deg_1p8Vpp_4D_01.h5"
CTX_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/phi_context"
CTX_TAG = "sewellia_v2_phictx"
CALIB_PATH = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/sewellia_real_calibration.npz"
CKPT_DIR = "/mydata/sdate/shared/checkpoints"
OUT_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", default=DATA_PATH)
    p.add_argument("--ctx_dir", default=CTX_DIR)
    p.add_argument("--ctx_tag", default=CTX_TAG)
    p.add_argument("--calib_path", default=CALIB_PATH)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--exp_name", default="sewellia_n2v_v2_fullres")
    p.add_argument("--ckpt_dir", default=CKPT_DIR)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--n2v_ratio", type=float, default=0.02)
    p.add_argument("--n2v_window", type=int, default=5)
    p.add_argument("--conditioning_probability", type=float, default=0.5)
    p.add_argument("--poisson_warmup_steps", type=int, default=2000)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--ckpt_every", type=int, default=500)
    p.add_argument("--max_steps", type=int, default=None, help="smoke-test: stop after this many steps total")
    p.add_argument("--out_dir", default=OUT_DIR)
    return p.parse_args()


def find_latest_checkpoint(ckpt_dir: Path, exp_name: str):
    pattern = re.compile(rf"^{re.escape(exp_name)}_step(\d+)\.pt$")
    best = None
    for f in ckpt_dir.glob(f"{exp_name}_step*.pt"):
        m = pattern.match(f.name)
        if m:
            step = int(m.group(1))
            if best is None or step > best[0]:
                best = (step, f)
    return best  # (step, path) or None


def save_checkpoint(ckpt_dir: Path, exp_name: str, step: int, epoch: int, unet, opt, sched):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{exp_name}_step{step}.pt"
    tmp = ckpt_dir / f"{exp_name}_step{step}.pt.tmp"
    torch.save({"unet": unet.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
               "step": step, "epoch": epoch}, tmp)
    tmp.replace(path)
    # keep only the 2 most recent checkpoints -- these are large (UNet + AdamW state)
    all_ckpts = sorted(ckpt_dir.glob(f"{exp_name}_step*.pt"),
                       key=lambda f: int(re.match(rf"^{re.escape(exp_name)}_step(\d+)\.pt$", f.name).group(1)))
    for old in all_ckpts[:-2]:
        old.unlink(missing_ok=True)
    return path


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    tap_paths = [f"{a.ctx_dir}/{a.ctx_tag}_tap{c}.f16" for c in range(a.T)]
    for p in tap_paths:
        if not Path(p).exists():
            raise FileNotFoundError(f"context tap not found: {p} -- has the v2 context build finished?")

    log("loading dataset (raw counts + %d full-res context taps)..." % a.T)
    t0 = time.time()
    ds = SewelliaRealN2VDataset(a.data_path, tap_paths, a.calib_path)
    log(f"dataset ready in {time.time()-t0:.1f}s: n={ds.n} h={ds.h} w={ds.w} T={ds.T} "
        f"norm_min={ds.norm_min:.2f} norm_max={ds.norm_max:.2f}")

    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
                        pin_memory=True, drop_last=True, persistent_workers=(a.num_workers > 0))

    unet = create_baseline_unet(k=0, sample_size=(608, 576), extra_cond_channels=ds.T, poisson_head=True,
                                poisson_init_mean=(ds.norm_min + ds.norm_max) / 2,
                                poisson_init_var=((ds.norm_max - ds.norm_min) / 4) ** 2)
    model = PaddedUNet(unet, orig_hw=(ds.h, ds.w)).to(device)

    sigma_read2 = torch.from_numpy(ds.dark_var).to(device=device, dtype=torch.float32)
    loss_fn = BaselineN2VLoss(device, ratio=a.n2v_ratio, window=a.n2v_window,
                              conditioning_probability=a.conditioning_probability,
                              poisson_head=True, norm_min=ds.norm_min, norm_max=ds.norm_max,
                              poisson_warmup_steps=a.poisson_warmup_steps,
                              gaussian_floor=True, sigma_read2=sigma_read2)

    opt = torch.optim.AdamW(model.parameters(), lr=a.learning_rate, weight_decay=a.weight_decay)
    total_steps = a.epochs * len(loader)
    sched = get_cosine_schedule_with_warmup(opt, num_warmup_steps=min(500, max(1, total_steps // 20)),
                                            num_training_steps=max(1, total_steps))

    ckpt_dir = Path(a.ckpt_dir)
    start_step, start_epoch = 0, 0
    latest = find_latest_checkpoint(ckpt_dir, a.exp_name)
    if latest is not None:
        step, path = latest
        log(f"resuming from checkpoint {path} (step {step})")
        sd = torch.load(path, map_location=device)
        unet.load_state_dict(sd["unet"])
        opt.load_state_dict(sd["opt"])
        sched.load_state_dict(sd["sched"])
        start_step = sd["step"]
        start_epoch = sd["epoch"]

    config_path = ckpt_dir / f"{a.exp_name}_config.json"
    config = dict(exp_name=a.exp_name, data_path=a.data_path, ctx_dir=a.ctx_dir, ctx_tag=a.ctx_tag, T=ds.T,
                 h=ds.h, w=ds.w, norm_min=ds.norm_min, norm_max=ds.norm_max, k=0, poisson_head=True,
                 gaussian_floor=True, conditioning_probability=a.conditioning_probability, epochs=a.epochs,
                 batch_size=a.batch_size, learning_rate=a.learning_rate)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    log(f"training: {len(ds)} samples, batch={a.batch_size}, {len(loader)} steps/epoch, "
        f"{a.epochs} epochs ({total_steps} total steps), starting at step {start_step}")
    model.train()
    step = start_step
    loss_log_path = Path(a.out_dir) / f"{a.exp_name}_loss.jsonl"
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
            step += 1
            if step % a.ckpt_every == 0:
                save_checkpoint(ckpt_dir, a.exp_name, step, epoch, unet, opt, sched)
                with open(loss_log_path, "a") as f:
                    f.write(json.dumps(dict(step=step, epoch=epoch, loss=float(np.mean(losses[-a.ckpt_every:])))) + "\n")
                log(f"  step {step}: mean_loss(last {min(a.ckpt_every,len(losses))})="
                    f"{np.mean(losses[-a.ckpt_every:]):.6f} checkpoint saved")
            if a.max_steps is not None and step >= a.max_steps:
                log(f"reached max_steps={a.max_steps}, stopping (smoke test)")
                save_checkpoint(ckpt_dir, a.exp_name, step, epoch, unet, opt, sched)
                log("SUCCESS")
                return
        log(f"epoch {epoch+1}/{a.epochs}: mean_loss={np.mean(losses):.6f} "
            f"(min={np.min(losses):.6f} max={np.max(losses):.6f}) {time.time()-t0:.1f}s")
        save_checkpoint(ckpt_dir, a.exp_name, step, epoch + 1, unet, opt, sched)

    log("training complete")
    log("SUCCESS")


if __name__ == "__main__":
    main()
