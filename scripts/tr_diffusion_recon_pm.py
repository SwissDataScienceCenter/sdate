#!/usr/bin/env python3
"""Reconstruct with a posterior-mean diffusion denoiser (B samples) vs the baseline,
for any dataset via --profile. Generates the denoised memmaps it needs (baseline
single-pass + diffusion posterior-mean B) if not cached, then runs a full-res FBP
sliding-window reconstruction over odd windows (start from the 2nd; even windows
carry the mirror-half artifact) and renders a GT|baseline|pmB|noisy movie.

  # wunderkerze (defaults)                         # new dataset
  python scripts/tr_diffusion_recon_pm.py          python scripts/tr_diffusion_recon_pm.py \
      --profile wunderkerze2 ...                        --profile asc_thixo --num_samples 4 --det_bin 1
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate"); sys.path.insert(0, "/myhome/astra-torch")
from sdate.tr_diffusion import reconstruct as R
from sdate.tr_diffusion.geometry import usable_frame_range
from sdate.tr_diffusion.profiles import DatasetProfile

CK = "/myhome/sdate/checkpoints"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="asc_thixo", help="DatasetProfile name or JSON path")
    p.add_argument("--exp", default="asc_dose005", help="checkpoint exp_name suffix")
    p.add_argument("--diffusion_ckpt", default=None)
    p.add_argument("--baseline_ckpt", default=None)
    p.add_argument("--out_dir", default="/myhome/data/sdate/shared/time_resolved/tr_recon_cache")
    p.add_argument("--num_samples", type=int, default=4)
    p.add_argument("--ss_timestep", type=int, default=500)
    p.add_argument("--frame_start", type=int, default=None)   # default: first half of the profile range
    p.add_argument("--frame_end", type=int, default=None)
    p.add_argument("--det_bin", type=int, default=1)
    p.add_argument("--window_skip", type=int, default=2)
    p.add_argument("--start_window_offset", type=int, default=1)
    p.add_argument("--dose", type=float, default=0.05)
    p.add_argument("--tag", default=None)
    return p.parse_args()


def main():
    a = parse_args()
    prof = DatasetProfile.load(a.profile)
    dev = torch.device("cuda")
    OUT = Path(a.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    dcap = prof.mov_path.rsplit("/", 1)[0]  # dataset dir (keep memmaps beside the data)
    diff_ck = a.diffusion_ckpt or f"{CK}/tr_denoise_diffusion_{a.exp}.pt"
    base_ck = a.baseline_ckpt or f"{CK}/tr_denoise_baseline_{a.exp}.pt"
    fs = a.frame_start if a.frame_start is not None else prof.frame_start
    fe = a.frame_end if a.frame_end is not None else (prof.frame_start + prof.frame_end) // 2  # first half
    tag = a.tag or f"{prof.name}_pm{a.num_samples}_odd"
    kw = dict(deg_per_frame=prof.deg_per_frame, axis_col=prof.rot_axis_col)
    t0 = time.time()

    base_mm = f"{dcap}/denoised_{prof.name}_baseline_dose{int(a.dose*100):02d}.f16"
    pm_mm = f"{dcap}/denoised_{prof.name}_pm{a.num_samples}_t{a.ss_timestep}.f16"
    if not Path(base_mm + ".meta.npz").exists():
        print("=== baseline denoise sequence ===", flush=True)
        R.denoise_sequence(base_ck, prof.mov_path, prof.memmap_path, base_mm,
                           frame_start=fs, frame_end=fe, dose=a.dose,
                           batch=96, num_workers=8, device=dev, **kw)
    if not Path(pm_mm + ".meta.npz").exists():
        print(f"=== diffusion posterior-mean B={a.num_samples} denoise sequence ===", flush=True)
        R.denoise_sequence(diff_ck, prof.mov_path, prof.memmap_path, pm_mm,
                           frame_start=fs, frame_end=fe, dose=a.dose,
                           ss_timestep=a.ss_timestep, num_samples=a.num_samples,
                           batch=64, num_workers=8, device=dev, **kw)

    win = R.window_length_frames(180.0, prof.deg_per_frame)
    stride = a.window_skip * win
    ulo, _ = usable_frame_range(fs, fe, 1, period_360=prof.period_360)
    recon_start = ulo + a.start_window_offset * win
    variants = {"baseline": base_mm, f"pm{a.num_samples}": pm_mm}
    print(f"=== recon  det_bin={a.det_bin} stride={stride}f start={recon_start} ===", flush=True)
    res = R.run_windows(prof.mov_path, prof.memmap_path, variants, stride=stride, det_bin=a.det_bin,
                        method="fbp", frame_start=recon_start, frame_end=fe,
                        axis_col=prof.rot_axis_col, deg_per_frame=prof.deg_per_frame,
                        device=dev, log_every=25)
    nW = len(res["window_starts"])
    for arm in [f"pm{a.num_samples}", "baseline", "noisy"]:
        m = res["metrics"][arm]
        print(f"  {arm:12s} PSNR {m['psnr'].mean():6.2f}  SSIM {m['ssim'].mean():.3f}", flush=True)

    np.savez(OUT / f"recon_results_{tag}.npz", window_starts=np.array(res["window_starts"]),
             **{f"{arm}_psnr": res["metrics"][arm]["psnr"] for arm in res["metrics"]},
             **{f"{arm}_ssim": res["metrics"][arm]["ssim"] for arm in res["metrics"]})
    summary = {"tag": tag, "profile": prof.name, "n_windows": nW, "det_bin": a.det_bin,
               "num_samples": a.num_samples, "frame_range": [fs, fe], "minutes": round((time.time()-t0)/60, 1)}
    for arm in res["metrics"]:
        summary[f"{arm}_psnr"] = float(res["metrics"][arm]["psnr"].mean())
        summary[f"{arm}_ssim"] = float(res["metrics"][arm]["ssim"].mean())
    (OUT / f"recon_summary_{tag}.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)

    mid = len(res["movie_rows"]) // 2
    gt = np.stack([f[mid].numpy() for f in res["movie"]["GT"]])
    vmin, vmax = np.percentile(gt, [1, 99])
    combined = [torch.cat([res["movie"][arm][i][mid] for arm in res["arms"]], dim=1) for i in range(nW)]
    R.write_slice_movie(combined, OUT / f"recon_{tag}.mov", float(vmin), float(vmax))
    print("panels:", " | ".join(res["arms"]), "\nDONE", flush=True)


if __name__ == "__main__":
    main()
