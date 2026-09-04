#!/usr/bin/env python3
"""Render a GT|baseline|pmB|noisy movie of raw 2D projection FRAMES over time,
for any dataset via --profile. Distinct from tr_diffusion_recon_pm.py's
reconstruction-SLICE movie: this shows the detector frames themselves, not
reconstructed volume slices. Reuses whatever denoised memmaps are already
cached by tr_diffusion_recon_pm.py (same cache-path convention); generates
them if missing.

  python scripts/tr_diffusion_projection_movie.py --profile ag10_c1mm --exp ag10_dose005 \
      --frame_start 150000 --frame_end 152000
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import torch

sys.path.insert(0, "/myhome/sdate"); sys.path.insert(0, "/myhome/astra-torch")
from sdate.tr_diffusion import reconstruct as R
from sdate.tr_diffusion.profiles import DatasetProfile

CK = "/myhome/sdate/checkpoints"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", required=True, help="DatasetProfile name or JSON path")
    p.add_argument("--exp", default="dose005", help="checkpoint exp_name suffix")
    p.add_argument("--diffusion_ckpt", default=None)
    p.add_argument("--baseline_ckpt", default=None)
    p.add_argument("--out_dir", default="/myhome/data/sdate/shared/time_resolved/tr_recon_cache")
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--ss_timestep", type=int, default=500)
    p.add_argument("--frame_start", type=int, required=True)
    p.add_argument("--frame_end", type=int, required=True)
    p.add_argument("--dose", type=float, default=0.05)
    p.add_argument("--tag", default=None)
    p.add_argument("--batch", type=int, default=96, help="baseline denoise batch; lower if GPU is shared with training")
    p.add_argument("--pm_batch", type=int, default=64, help="diffusion posterior-mean denoise batch")
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


def main():
    a = parse_args()
    prof = DatasetProfile.load(a.profile)
    dev = torch.device("cuda")
    OUT = Path(a.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    dcap = prof.mov_path.rsplit("/", 1)[0]
    diff_ck = a.diffusion_ckpt or f"{CK}/tr_denoise_diffusion_{a.exp}.pt"
    base_ck = a.baseline_ckpt or f"{CK}/tr_denoise_baseline_{a.exp}.pt"
    tag = a.tag or f"{prof.name}_proj_pm{a.num_samples}"
    kw = dict(deg_per_frame=prof.deg_per_frame, axis_col=prof.rot_axis_col)
    t0 = time.time()

    # cache paths match tr_diffusion_recon_pm.py's convention so both scripts
    # reuse the same denoised memmaps whenever frame ranges overlap.
    fs, fe = a.frame_start, a.frame_end
    base_mm = f"{dcap}/denoised_{prof.name}_baseline_dose{int(a.dose * 100):02d}_proj.f16"
    pm_mm = f"{dcap}/denoised_{prof.name}_pm{a.num_samples}_t{a.ss_timestep}_proj.f16"
    if not Path(base_mm + ".meta.npz").exists():
        print("=== baseline denoise sequence ===", flush=True)
        R.denoise_sequence(base_ck, prof.mov_path, prof.memmap_path, base_mm,
                           frame_start=fs, frame_end=fe, dose=a.dose,
                           batch=a.batch, num_workers=a.num_workers, device=dev, **kw)
    if not Path(pm_mm + ".meta.npz").exists():
        print(f"=== diffusion posterior-mean B={a.num_samples} denoise sequence ===", flush=True)
        R.denoise_sequence(diff_ck, prof.mov_path, prof.memmap_path, pm_mm,
                           frame_start=fs, frame_end=fe, dose=a.dose,
                           ss_timestep=a.ss_timestep, num_samples=a.num_samples,
                           batch=a.pm_batch, num_workers=a.num_workers, device=dev, **kw)

    variants = {"baseline": base_mm, f"pm{a.num_samples}": pm_mm}
    out_path = OUT / f"proj_{tag}.mov"
    print(f"=== projection movie frames [{fs},{fe}) -> {out_path} ===", flush=True)
    _, arms = R.write_projection_movie(prof.mov_path, prof.memmap_path, variants, out_path,
                                       frame_start=fs, frame_end=fe, dose=a.dose,
                                       crop=prof.crop, axis_col=prof.rot_axis_col)
    print(f"panels: {' | '.join(arms)}  ({(fe - fs)} frames, {round((time.time() - t0) / 60, 1)} min)")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
