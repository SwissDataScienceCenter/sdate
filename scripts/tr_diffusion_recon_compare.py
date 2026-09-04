#!/usr/bin/env python3
"""Compare denoiser variants via reconstruction: baseline vs single-shot diffusion
(t=500) vs diffusion->baseline cascade, plus the noisy floor, all vs GT.

Generates the diffusion-single-shot and cascade denoised sequences (cached), then
runs a multi-variant FBP sliding-window reconstruction and renders a side-by-side
movie (GT | baseline | diffusion_ss | cascade | noisy).

  python scripts/tr_diffusion_recon_compare.py --det_bin 2 --tag compare
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate"); sys.path.insert(0, "/myhome/astra-torch")
from sdate.tr_diffusion import reconstruct as R

DATA = "/myhome/data/sdate/shared/time_resolved/212_Wunderkerze2"
CK = "/myhome/sdate/checkpoints"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mov", default=f"{DATA}/212_Wunderkerze2.mov")
    p.add_argument("--memmap", default=f"{DATA}/frames_400k_500k.u16")
    p.add_argument("--baseline_ckpt", default=f"{CK}/tr_denoise_baseline_k1_dose005.pt")
    p.add_argument("--diffusion_ckpt", default=f"{CK}/tr_denoise_diffusion_k1_dose005.pt")
    p.add_argument("--baseline_mm", default=f"{DATA}/denoised_baseline_dose005.f16")
    p.add_argument("--diffusion_mm", default=f"{DATA}/denoised_diffusion_ss_t500_dose005.f16")
    p.add_argument("--cascade_mm", default=f"{DATA}/denoised_cascade_t500_dose005.f16")
    p.add_argument("--out_dir", default=f"/myhome/data/sdate/shared/time_resolved/tr_recon_cache")
    p.add_argument("--ss_timestep", type=int, default=500)
    p.add_argument("--det_bin", type=int, default=2)
    p.add_argument("--window_skip", type=int, default=1)
    p.add_argument("--frame_start", type=int, default=400_000)
    p.add_argument("--frame_end", type=int, default=500_000)
    p.add_argument("--tag", default="compare")
    return p.parse_args()


def main():
    a = parse_args()
    dev = torch.device("cuda")
    OUT = Path(a.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if not Path(a.diffusion_mm + ".meta.npz").exists():
        print("=== generating diffusion single-shot sequence ===", flush=True)
        R.denoise_sequence(a.diffusion_ckpt, a.mov, a.memmap, a.diffusion_mm,
                           ss_timestep=a.ss_timestep, batch=96, num_workers=8, device=dev)
    if not Path(a.cascade_mm + ".meta.npz").exists():
        print("=== generating cascade (baseline on diffusion) sequence ===", flush=True)
        R.cascade_sequence(a.baseline_ckpt, a.diffusion_mm, a.cascade_mm, batch=96, device=dev)

    variants = {"baseline": a.baseline_mm, "diffusion_ss": a.diffusion_mm, "cascade": a.cascade_mm}
    win = R.window_length_frames(180.0)
    stride = a.window_skip * win
    print(f"=== recon compare  det_bin={a.det_bin}  stride={stride}f ===", flush=True)
    res = R.run_windows(a.mov, a.memmap, variants, stride=stride, det_bin=a.det_bin, method="fbp",
                        frame_start=a.frame_start, frame_end=a.frame_end, device=dev, log_every=50)
    nW = len(res["window_starts"])
    print(f"windows {nW}", flush=True)
    for arm in ("baseline", "diffusion_ss", "cascade", "noisy"):
        m = res["metrics"][arm]
        print(f"  {arm:12s} PSNR {m['psnr'].mean():6.2f}  SSIM {m['ssim'].mean():.3f}", flush=True)

    np.savez(OUT / f"recon_results_{a.tag}.npz",
             window_starts=np.array(res["window_starts"]),
             **{f"{arm}_psnr": res["metrics"][arm]["psnr"] for arm in res["metrics"]},
             **{f"{arm}_ssim": res["metrics"][arm]["ssim"] for arm in res["metrics"]},
             det_bin=a.det_bin, ss_timestep=a.ss_timestep, stride=stride)
    summary = {"tag": a.tag, "n_windows": nW, "det_bin": a.det_bin, "ss_timestep": a.ss_timestep,
               "window_skip": a.window_skip, "minutes": round((time.time() - t0) / 60, 1)}
    for arm in res["metrics"]:
        summary[f"{arm}_psnr"] = float(res["metrics"][arm]["psnr"].mean())
        summary[f"{arm}_ssim"] = float(res["metrics"][arm]["ssim"].mean())
    (OUT / f"recon_summary_{a.tag}.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)

    # movie: GT | baseline | diffusion_ss | cascade | noisy  (middle slice)
    mid = len(res["movie_rows"]) // 2
    gt = np.stack([f[mid].numpy() for f in res["movie"]["GT"]])
    vmin, vmax = np.percentile(gt, [1, 99])
    combined = [torch.cat([res["movie"][arm][i][mid] for arm in res["arms"]], dim=1) for i in range(nW)]
    R.write_slice_movie(combined, OUT / f"recon_{a.tag}.mov", float(vmin), float(vmax))
    print("panels:", " | ".join(res["arms"]), flush=True)
    print("movie written", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
