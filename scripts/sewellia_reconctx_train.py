#!/usr/bin/env python3
"""Train the reconstruction-domain Noise2Noise+context denoiser (parallel
track -- see project-sewellia-recon-space-n2v.md). v2 design: input = a1
(one independent ~200-view noisy reconstruction), target = a2 (a SECOND,
independent ~200-view reconstruction of the same phi bin -- disjoint views,
no shared photon draws), regressed with plain full-frame MSE. Context = T=3
phi-gated joint-FBP levels, built with the anchor pair's views excluded from
every context gate.

Supersedes the v1 blind-spot-N2V design (BaselineN2VLoss, single anchor):
that version showed near-zero denoising after training (0.98 input/output
correlation) traced to two leaks -- the swap-based masking drew replacement
pixels from within the measured ~2px noise-correlation length, AND the
context gates were nested supersets of the anchor's gate (ctx0 in particular
contained 100% of the anchor's own views). The Noise2Noise-pair design fixes
both at the data level: genuine independence, no masking needed at all. See
scripts/sewellia_recon_context_build.py's docstring for the full postmortem.

No PaddedUNet needed here (unlike the projection-domain tracks): slices are
576x576, already a multiple of the UNet's 32x downsampling factor.

Checkpoints every --ckpt_every steps, auto-resumes on restart.

    python scripts/sewellia_reconctx_train.py --epochs 30
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
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")
sys.path.insert(0, "/myhome/BaseTraining")

from diffusers.optimization import get_cosine_schedule_with_warmup

from sdate.tr_diffusion.model import create_baseline_unet
from sdate.tr_diffusion.sewellia_reconctx_n2v import SewelliaReconContextDataset

CTX_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/recon_context"
CTX_TAG = "sewellia_reconctx"
CKPT_DIR = "/mydata/sdate/shared/checkpoints"
OUT_DIR = "/myhome/data/sdate/shared/time_resolved/sewellia_lineolata/recon_context"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ctx_dir", default=CTX_DIR)
    p.add_argument("--ctx_tag", default=CTX_TAG)
    p.add_argument("--exp_name", default="sewellia_reconctx_n2v_v1")
    p.add_argument("--ckpt_dir", default=CKPT_DIR)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-6)
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
    return best


def save_checkpoint(ckpt_dir: Path, exp_name: str, step: int, epoch: int, unet, opt, sched):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{exp_name}_step{step}.pt"
    tmp = ckpt_dir / f"{exp_name}_step{step}.pt.tmp"
    torch.save({"unet": unet.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
               "step": step, "epoch": epoch}, tmp)
    tmp.replace(path)
    all_ckpts = sorted(ckpt_dir.glob(f"{exp_name}_step*.pt"),
                       key=lambda f: int(re.match(rf"^{re.escape(exp_name)}_step(\d+)\.pt$", f.name).group(1)))
    for old in all_ckpts[:-2]:
        old.unlink(missing_ok=True)
    return path


def main():
    a = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")

    log("loading dataset (a1/a2 noise2noise anchor pair + T=3 recon-domain context, memmapped)...")
    t0 = time.time()
    ds = SewelliaReconContextDataset(a.ctx_dir, tag=a.ctx_tag)
    log(f"dataset ready in {time.time()-t0:.1f}s: usable_bins={len(ds.usable_bins)} crop_rows={ds.crop_rows} "
        f"n_pix={ds.n_pix} T={ds.T} n_samples={len(ds)} norm_min={ds.norm_min:.4g} norm_max={ds.norm_max:.4g}")

    loader = DataLoader(ds, batch_size=a.batch_size, shuffle=True, num_workers=a.num_workers,
                        pin_memory=True, drop_last=True, persistent_workers=(a.num_workers > 0))

    unet = create_baseline_unet(k=0, sample_size=(ds.n_pix, ds.n_pix), extra_cond_channels=ds.T,
                                poisson_head=False)
    model = unet.to(device)

    def compute_loss(batch, model):
        """Plain full-frame Noise2Noise regression: predict a2 from [a1, context]."""
        a1 = batch["input"].to(device, non_blocking=True).float()
        a2 = batch["target"].to(device, non_blocking=True).float()
        aux = batch["aux_channel"].to(device, non_blocking=True).float()
        bsz = a1.shape[0]
        model_input = torch.cat([a1, aux], dim=1)
        timesteps = torch.zeros(bsz, device=device, dtype=torch.long)
        present = torch.ones(bsz, device=device, dtype=torch.long)  # vestigial class-embed input, always "present"
        raw = model(model_input, timestep=timesteps, class_labels=present, return_dict=False)[0]
        loss = F.mse_loss(raw, a2)
        return loss, {"loss": loss.item()}

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
    config = dict(exp_name=a.exp_name, ctx_dir=a.ctx_dir, ctx_tag=a.ctx_tag, T=ds.T, n_pix=ds.n_pix,
                 crop_rows=ds.crop_rows, norm_min=ds.norm_min, norm_max=ds.norm_max, k=0, poisson_head=False,
                 loss_type="noise2noise_mse", epochs=a.epochs, batch_size=a.batch_size,
                 learning_rate=a.learning_rate)
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
            loss, stats = compute_loss(batch, model)
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
