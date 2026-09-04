#!/usr/bin/env python3
"""Classical 'sliding-window FBP' baseline vs the learned denoiser.

Averages (2k+1) INDEPENDENTLY-measured same-angle projections across k turns
before/after (no learning — pure temporal averaging), then FBP. Trades noise
reduction for blurred time resolution; k controls the tradeoff. Compared
against the learned baseline denoiser and the plain noisy floor, all vs GT.

  python scripts/tr_diffusion_recon_swfbp.py --profile wunderkerze2 --ks 1 2 4
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
    p.add_argument("--profile", default="wunderkerze2")
    p.add_argument("--baseline_ckpt", default=f"{CK}/tr_denoise_baseline_k1_dose005.pt")
    p.add_argument("--baseline_mm", default=None, help="cached learned-baseline memmap; generated if missing")
    p.add_argument("--out_dir", default="/myhome/data/sdate/shared/time_resolved/tr_recon_cache")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--frame_start", type=int, default=None)
    p.add_argument("--frame_end", type=int, default=None)   # default: first half of the profile range
    p.add_argument("--det_bin", type=int, default=1)        # full res, matches the pm4 run
    p.add_argument("--window_skip", type=int, default=2)    # every other window
    p.add_argument("--start_window_offset", type=int, default=1)  # start from the 2nd (odd)
    p.add_argument("--dose", type=float, default=0.05)
    p.add_argument("--tag", default=None)
    return p.parse_args()


def main():
    a = parse_args()
    prof = DatasetProfile.load(a.profile)
    dev = torch.device("cuda")
    OUT = Path(a.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    dcap = prof.mov_path.rsplit("/", 1)[0]
    fs = a.frame_start if a.frame_start is not None else prof.frame_start
    fe = a.frame_end if a.frame_end is not None else (prof.frame_start + prof.frame_end) // 2
    tag = a.tag or f"{prof.name}_swfbp"
    kw = dict(deg_per_frame=prof.deg_per_frame, axis_col=prof.rot_axis_col)
    t0 = time.time()

    base_mm = a.baseline_mm or f"{dcap}/denoised_{prof.name}_baseline_dose{int(a.dose*100):02d}.f16"
    if not Path(base_mm + ".meta.npz").exists():
        print("=== learned baseline denoise sequence ===", flush=True)
        R.denoise_sequence(a.baseline_ckpt, prof.mov_path, prof.memmap_path, base_mm,
                           frame_start=fs, frame_end=fe, dose=a.dose,
                           batch=96, num_workers=8, device=dev, **kw)

    variants = {"baseline": base_mm}
    for k in a.ks:
        mm_path = f"{dcap}/swfbp_{prof.name}_k{k}_dose{int(a.dose*100):02d}.f16"
        if not Path(mm_path + ".meta.npz").exists():
            print(f"=== swfbp k={k} sequence ===", flush=True)
            R.temporal_average_sequence(prof.mov_path, prof.memmap_path, mm_path, k=k,
                                        frame_start=fs, frame_end=fe, dose=a.dose,
                                        crop=prof.crop, device=dev, **kw)
        variants[f"swfbp_k{k}"] = mm_path

    win = R.window_length_frames(180.0, prof.deg_per_frame)
    stride = a.window_skip * win
    ulo, _ = usable_frame_range(fs, fe, 1, period_360=prof.period_360)
    recon_start = ulo + a.start_window_offset * win
    print(f"=== recon  det_bin={a.det_bin}  stride={stride}f  start={recon_start} ===", flush=True)
    res = R.run_windows(prof.mov_path, prof.memmap_path, variants, stride=stride, det_bin=a.det_bin,
                        method="fbp", frame_start=recon_start, frame_end=fe,
                        axis_col=prof.rot_axis_col, deg_per_frame=prof.deg_per_frame,
                        device=dev, log_every=25)
    nW = len(res["window_starts"])
    print(f"windows {nW}", flush=True)
    for arm in list(variants) + ["noisy"]:
        m = res["metrics"][arm]
        print(f"  {arm:12s} PSNR {m['psnr'].mean():6.2f}  SSIM {m['ssim'].mean():.3f}", flush=True)

    np.savez(OUT / f"recon_results_{tag}.npz", window_starts=np.array(res["window_starts"]),
             **{f"{arm}_psnr": res["metrics"][arm]["psnr"] for arm in res["metrics"]},
             **{f"{arm}_ssim": res["metrics"][arm]["ssim"] for arm in res["metrics"]})
    summary = {"tag": tag, "profile": prof.name, "n_windows": nW, "det_bin": a.det_bin,
               "ks": a.ks, "frame_range": [fs, fe], "minutes": round((time.time() - t0) / 60, 1)}
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
    print("panels:", " | ".join(res["arms"]), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
