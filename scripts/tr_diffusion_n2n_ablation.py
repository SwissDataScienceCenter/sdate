#!/usr/bin/env python3
"""N2N (binomial-split) vs N2V, in reconstruction space.

Exploratory comparison: the existing N2V baseline/diffusion (dose-0.05, blind-spot
masked loss) vs the new N2N baseline/diffusion (binomial-split of the SAME fixed
dose-0.05 measurement, full-frame loss, conditioned on the split fraction p).

For N2N models, inference sweeps the ASSUMED input fraction ``q`` (see
``pipeline.denoise_frames_n2n_baseline`` / ``pred_x0_n2n_ensemble``): ``q=1.0``
feeds the full measured frame (no further thinning, no information thrown away —
expected best); smaller ``q`` further binomial-thins the measurement to see
whether the "large fraction wins" hypothesis holds. Uses the SAME windows (frame
range / stride / start offset) as the other ablations for direct comparability.

``--include_diffusion`` adds the N2N diffusion model in TWO inference modes:
``single_shot`` (one forward pass / posterior mean at ``ss_timestep``, cheap,
swept across the full ``--qs``) and ``ancestral`` (full DDIM sampling from
``ss_timestep`` down to 0, ``ancestral_num_steps`` steps, initialised from the
noisy measurement — expensive, ~``ancestral_num_steps``x the cost per frame, so
by default only run at ``--ancestral_qs`` (default 1.0) rather than the full
sweep). N2V found ancestral sampling counterproductive (blind-spot conditioning
reconverges to the noisy input); N2N's conditioning mechanism differs (nothing is
deliberately hidden), so this re-tests whether that finding holds here.

  python scripts/tr_diffusion_n2n_ablation.py --profile wunderkerze2 \
      --n2n_baseline_ckpt checkpoints/tr_denoise_baseline_n2n_dose005.pt \
      --include_diffusion --qs 0.5,0.7,0.9,0.95,1.0
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
    p.add_argument("--n2v_baseline_ckpt", default=f"{CK}/tr_denoise_baseline_k1_dose005.pt")
    p.add_argument("--n2n_baseline_ckpt", default=f"{CK}/tr_denoise_baseline_n2n_dose005.pt")
    p.add_argument("--n2n_diffusion_ckpt", default=f"{CK}/tr_denoise_diffusion_n2n_dose005.pt")
    p.add_argument("--include_diffusion", action="store_true",
                   help="also denoise with the N2N diffusion model (single-shot AND ancestral)")
    p.add_argument("--ss_timestep", type=int, default=500)
    p.add_argument("--qs", default="0.5,0.7,0.9,0.95,1.0", help="comma-separated input fractions to ablate")
    p.add_argument("--ancestral_qs", default="1.0",
                   help="comma-separated input fractions for the (expensive) ancestral diffusion sampler")
    p.add_argument("--ancestral_num_steps", type=int, default=50,
                   help="DDIM steps from ss_timestep down to 0 for the ancestral sampler")
    p.add_argument("--skip_single_shot_diffusion", action="store_true",
                   help="skip the cheap single-shot diffusion sweep (only run ancestral)")
    p.add_argument("--n2v_mm", default=None, help="cached N2V baseline memmap (reused if present)")
    p.add_argument("--out_dir", default="/myhome/data/sdate/shared/time_resolved/tr_recon_cache")
    p.add_argument("--frame_start", type=int, default=None)
    p.add_argument("--frame_end", type=int, default=None)   # default: first half of the profile range
    p.add_argument("--det_bin", type=int, default=1)        # full res, matches the swfbp/pm4/ablation runs
    p.add_argument("--window_skip", type=int, default=2)    # every other window
    p.add_argument("--start_window_offset", type=int, default=1)  # start from the 2nd (odd)
    p.add_argument("--dose", type=float, default=0.05)
    p.add_argument("--tag", default=None)
    p.add_argument("--batch", type=int, default=96, help="inference batch size for baseline models")
    p.add_argument("--diff_batch", type=int, default=64, help="inference batch size for diffusion models")
    return p.parse_args()


def main():
    a = parse_args()
    prof = DatasetProfile.load(a.profile)
    dev = torch.device("cuda")
    OUT = Path(a.out_dir); OUT.mkdir(parents=True, exist_ok=True)
    dcap = prof.mov_path.rsplit("/", 1)[0]
    fs = a.frame_start if a.frame_start is not None else prof.frame_start
    fe = a.frame_end if a.frame_end is not None else (prof.frame_start + prof.frame_end) // 2
    tag = a.tag or f"{prof.name}_n2n_ablation"
    kw = dict(deg_per_frame=prof.deg_per_frame, axis_col=prof.rot_axis_col)
    qs = [float(x) for x in a.qs.split(",")]
    t0 = time.time()

    # N2V baseline reference (reuse the cached full-method memmap if it already exists).
    n2v_mm = a.n2v_mm or f"{dcap}/denoised_{prof.name}_baseline_dose{int(a.dose*100):02d}.f16"
    if not Path(n2v_mm + ".meta.npz").exists():
        print("=== N2V baseline sequence ===", flush=True)
        R.denoise_sequence(a.n2v_baseline_ckpt, prof.mov_path, prof.memmap_path, n2v_mm,
                           frame_start=fs, frame_end=fe, dose=a.dose,
                           batch=a.batch, num_workers=8, device=dev, **kw)
    variants = {"n2v_baseline": n2v_mm}

    for q in qs:
        qtag = f"n2n_base_q{int(round(q*100)):03d}"
        mm_path = f"{dcap}/denoised_{prof.name}_{qtag}_dose{int(a.dose*100):02d}.f16"
        if not Path(mm_path + ".meta.npz").exists():
            print(f"=== N2N baseline sequence, q={q} ===", flush=True)
            R.denoise_sequence(a.n2n_baseline_ckpt, prof.mov_path, prof.memmap_path, mm_path,
                               frame_start=fs, frame_end=fe, dose=a.dose, n2n_q=q,
                               batch=a.batch, num_workers=8, device=dev, **kw)
        variants[qtag] = mm_path

    if a.include_diffusion:
        if not a.skip_single_shot_diffusion:
            for q in qs:
                qtag = f"n2n_diff_ss_q{int(round(q*100)):03d}"
                mm_path = f"{dcap}/denoised_{prof.name}_{qtag}_t{a.ss_timestep}_dose{int(a.dose*100):02d}.f16"
                if not Path(mm_path + ".meta.npz").exists():
                    print(f"=== N2N diffusion single-shot sequence, q={q} ===", flush=True)
                    R.denoise_sequence(a.n2n_diffusion_ckpt, prof.mov_path, prof.memmap_path, mm_path,
                                       frame_start=fs, frame_end=fe, dose=a.dose, n2n_q=q,
                                       ss_timestep=a.ss_timestep, num_samples=1,
                                       diffusion_inference="single_shot",
                                       batch=a.diff_batch, num_workers=8, device=dev, **kw)
                variants[qtag] = mm_path

        ancestral_qs = [float(x) for x in a.ancestral_qs.split(",")] if a.ancestral_qs else []
        for q in ancestral_qs:
            qtag = f"n2n_diff_anc_q{int(round(q*100)):03d}"
            mm_path = (f"{dcap}/denoised_{prof.name}_{qtag}_t{a.ss_timestep}"
                       f"_s{a.ancestral_num_steps}_dose{int(a.dose*100):02d}.f16")
            if not Path(mm_path + ".meta.npz").exists():
                print(f"=== N2N diffusion ANCESTRAL sequence, q={q}, "
                      f"{a.ss_timestep}->0 in {a.ancestral_num_steps} steps "
                      f"(slow: ~{a.ancestral_num_steps}x a single-shot pass) ===", flush=True)
                R.denoise_sequence(a.n2n_diffusion_ckpt, prof.mov_path, prof.memmap_path, mm_path,
                                   frame_start=fs, frame_end=fe, dose=a.dose, n2n_q=q,
                                   ss_timestep=a.ss_timestep, num_samples=1,
                                   diffusion_inference="ancestral", ancestral_num_steps=a.ancestral_num_steps,
                                   batch=a.diff_batch, num_workers=8, device=dev, **kw)
            variants[qtag] = mm_path

    win = R.window_length_frames(180.0, prof.deg_per_frame)
    stride = a.window_skip * win
    ulo, _ = usable_frame_range(fs, fe, 1, period_360=prof.period_360)
    recon_start = ulo + a.start_window_offset * win
    print(f"=== recon  det_bin={a.det_bin}  stride={stride}f  start={recon_start}  variants={list(variants)} ===", flush=True)
    res = R.run_windows(prof.mov_path, prof.memmap_path, variants, stride=stride, det_bin=a.det_bin,
                        method="fbp", frame_start=recon_start, frame_end=fe,
                        axis_col=prof.rot_axis_col, deg_per_frame=prof.deg_per_frame,
                        device=dev, log_every=25)
    nW = len(res["window_starts"])
    print(f"windows {nW}", flush=True)
    for arm in list(variants) + ["noisy"]:
        m = res["metrics"][arm]
        print(f"  {arm:16s} PSNR {m['psnr'].mean():6.2f}  SSIM {m['ssim'].mean():.3f}", flush=True)

    np.savez(OUT / f"recon_results_{tag}.npz", window_starts=np.array(res["window_starts"]),
             **{f"{arm}_psnr": res["metrics"][arm]["psnr"] for arm in res["metrics"]},
             **{f"{arm}_ssim": res["metrics"][arm]["ssim"] for arm in res["metrics"]})
    summary = {"tag": tag, "profile": prof.name, "n_windows": nW, "det_bin": a.det_bin, "qs": qs,
               "frame_range": [fs, fe], "minutes": round((time.time() - t0) / 60, 1)}
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
