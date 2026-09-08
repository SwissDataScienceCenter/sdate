#!/usr/bin/env python3
"""Annealed-N2N: iterative renoising/annealing Noise2Noise denoiser (wunderkerze2, dose=0.05).

See sdate/tr_diffusion/CONTEXT.md (glossary) and docs/adr/0001, docs/adr/0002
for the design and the two deliberate deviations from the obvious path.

Each round: train the round denoiser D (plain MSE Noise2Noise, in Anscombe/
Gaussian z-space) on the current iterate x_hat_k, targeting y1; run D over the
full dataset (projection-domain PSNR/SSIM vs the native reference logged as
the trajectory diagnostic); renoise x_hat_{k+1} = alpha_k*y2 + (1-alpha_k)*
D(x_hat_k). y1/y2 are a fixed 50/50 binomial split of one Poisson draw at
combined dose=0.1 (ADR-0001), generated once and never resampled. Stops after
--rounds rounds; the final round's D(x_hat_k) (alpha ~= 0 by then) is the
pipeline's official output. Round 0 trains 5 epochs from scratch; every later
round warm-starts from the previous round's weights for 2 epochs (fixed
schedule, monitored via the PSNR/SSIM trajectory -- ADR-0002).

Resumable: after every round the model weights + a small state.json (last
completed round) are written to --run_dir, and the evolving x_hat iterate
lives in an on-disk memmap there too -- a restarted run picks up from the
last completed round rather than starting over.

At the end: writes the round-0 output (the internal MMSE-anchor baseline)
and the final round's output to raw-count memmaps in the same layout
sdate.tr_diffusion.reconstruct.denoise_sequence uses, then reconstructs both
via a standard single-180-window sliding FBP (reconstruct.run_windows) on the
--eval_frame_start/--eval_frame_end window -- by default 402149-497749, the
window used by the existing joint k=21 FBP baseline (27.10 dB / 0.715 SSIM,
see project memory project-jointfbp-k21-baseline). NOTE: that baseline number
comes from a genuinely different reconstruction method (joint-k=21 sliding
multi-revolution FBP), not this script's plain single-window FBP -- the same
window makes the two numbers a useful reference point, not a strictly
apples-to-apples comparison.

Needs a real GPU (UNet training/inference) + ASTRA (astra_torch, for the
final reconstruction eval): submit via RunAI, e.g.

  runai workspace submit sdate-anneal-n2n \\
    -i lfbarba/sdsc_image:1.0.1 -p sdate-luisb \\
    --gpu-request-type portion --gpu-portion-request 0.4 \\
    --node-type A100 --large-shm --cpu-core-request 4 --cpu-core-limit 10 \\
    --cpu-memory-limit 64G --preemptibility preemptible \\
    --command -- bash -c "cd /myhome/sdate && python -m pip install -e . -q && \\
      python scripts/tr_diffusion_anneal_n2n.py --run_dir /myhome/data/sdate/shared/time_resolved/tr_anneal_n2n/run1"

If preempted, resubmit the IDENTICAL command -- --run_dir's state.json makes
it resume from the last completed round automatically.
"""
from __future__ import annotations

import os

os.environ["PATH"] = "/myhome/bin:" + os.environ.get("PATH", "")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/myhome/sdate")
sys.path.insert(0, "/myhome/astra-torch")

from sdate.tr_diffusion.anneal import (
    AnnealDataset, alpha_schedule, create_counts_memmap, create_x_hat_memmap,
    fit_z_norm, infer_round, load_state, pick_trajectory_indices, save_state,
    train_round, write_counts_meta,
)
from sdate.tr_diffusion.data import TimeResolvedFrameDataset
from sdate.tr_diffusion.model import create_baseline_unet
from sdate.tr_diffusion.profiles import DatasetProfile

EVAL_DOSE = 0.05          # the standard per-projection dose used for the "noisy" floor arm
EVAL_NOISE_SEED = 12345   # matches reconstruct.denoise_sequence's default -- same "noisy" realization


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="wunderkerze2")
    p.add_argument("--run_dir", required=True, help="where checkpoints/x_hat memmap/trajectory/metrics live")
    p.add_argument("--k", type=int, default=3, help="context radius (see model.py; k=3 is this project's standard)")

    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--schedule", choices=["cosine", "linear"], default="cosine")
    p.add_argument("--combined_dose", type=float, default=0.1, help="total dose Poisson-thinned before the 50/50 split")
    p.add_argument("--split_p", type=float, default=0.5)
    p.add_argument("--round0_epochs", type=int, default=5)
    p.add_argument("--round_epochs", type=int, default=2, help="epochs for every round after round 0")
    p.add_argument("--epoch_overrides", default="", help="JSON dict {round_index: epochs} to bump specific rounds")

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--data_seed", type=int, default=0, help="fixes y1/y2/context for the whole run (ADR-0001)")

    p.add_argument("--traj_frames", type=int, default=6)
    p.add_argument("--traj_seed", type=int, default=0)

    p.add_argument("--eval_frame_start", type=int, default=402_149)
    p.add_argument("--eval_frame_end", type=int, default=497_749)
    p.add_argument("--det_bin", type=int, default=2)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    epoch_overrides = json.loads(args.epoch_overrides) if args.epoch_overrides else {}

    prof = DatasetProfile.load(args.profile)
    base_ds = TimeResolvedFrameDataset(
        mov_path=prof.mov_path, memmap_path=prof.memmap_path, k=args.k,
        frame_start=prof.frame_start, frame_end=prof.frame_end, crop=prof.crop,
        axis_col=prof.rot_axis_col, deg_per_frame=prof.deg_per_frame,
        norm_range=prof.norm_range,
        n2n=True, extra_noise_dose=args.combined_dose, p_range=(args.split_p, args.split_p),
        noise_seed=args.data_seed,
    )
    split_dose = args.combined_dose * args.split_p
    n = len(base_ds)
    crop = base_ds.crop
    print(f"dataset: {n} usable frames, crop={crop}, k={args.k}, "
          f"combined_dose={args.combined_dose}, split_dose={split_dose}", flush=True)

    state_path = run_dir / "model"
    x_hat_path = run_dir / "x_hat.f16"
    state = load_state(state_path)

    model = create_baseline_unet(k=args.k, sample_size=crop, poisson_head=False)
    assert model.config.in_channels == base_ds.in_channels_baseline
    model.to(device)

    if state is not None:
        start_round = int(state["last_completed_round"]) + 1
        z_min, z_max = float(state["z_min"]), float(state["z_max"])
        ckpt = torch.load(str(state_path) + ".pt", map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"resuming from round {start_round} (z_min={z_min:.4f}, z_max={z_max:.4f})", flush=True)
    else:
        start_round = 0
        z_min, z_max = fit_z_norm(base_ds, split_dose, seed=args.data_seed)
        print(f"fitted z-range from y1: z_min={z_min:.4f}, z_max={z_max:.4f}", flush=True)

    dataset = AnnealDataset(base_ds, x_hat_path, z_min, z_max, split_dose, args.combined_dose)

    if start_round == 0:
        fresh_mm = create_x_hat_memmap(x_hat_path, n, crop)
        fresh_mm.flush()
        del fresh_mm
        dataset.init_x_hat_to_y2(batch_size=args.eval_batch_size, num_workers=args.num_workers)
        print("initialised x_hat_0 = y2", flush=True)

    alphas = alpha_schedule(args.rounds, kind=args.schedule)
    print(f"alpha schedule ({args.schedule}): {np.round(alphas, 3).tolist()}", flush=True)

    traj_indices = pick_trajectory_indices(base_ds, args.eval_frame_start, args.eval_frame_end,
                                           n=args.traj_frames, seed=args.traj_seed)
    traj_frame_ids = base_ds.indices[traj_indices]
    print(f"trajectory frames: {traj_frame_ids.tolist()}", flush=True)

    first_index = int(base_ds.indices.min())
    data_range = base_ds.norm_max - base_ds.norm_min
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else []
    (run_dir / "trajectory").mkdir(exist_ok=True)

    mmse_anchor_path = run_dir / "mmse_anchor.f16"
    final_estimate_path = run_dir / "final_estimate.f16"

    for k in range(start_round, args.rounds):
        epochs = int(epoch_overrides.get(str(k), args.round0_epochs if k == 0 else args.round_epochs))
        is_last = (k == args.rounds - 1)
        print(f"\n=== round {k}/{args.rounds - 1}  epochs={epochs}  {'(final)' if is_last else ''} ===", flush=True)

        train_round(model, dataset, epochs, device, batch_size=args.batch_size,
                   lr=args.learning_rate, weight_decay=args.weight_decay, num_workers=args.num_workers)

        out_mm = None
        if k == 0:
            out_mm = create_counts_memmap(mmse_anchor_path, n, crop)
        elif is_last:
            out_mm = create_counts_memmap(final_estimate_path, n, crop)

        # alphas[k] is the FIRST-transition weight only in the sense that alphas[0]=1
        # describes x_hat_0=y2 itself, not a transition -- using alphas[k] here would
        # make the round-0 renoise step exactly alpha=1 (x_hat_1 = y2 = x_hat_0), a
        # total no-op that stalls the anneal for a full round. Shifting by one index
        # makes the LAST actual transition (round rounds-2 -> rounds-1) land on
        # alphas[rounds-1]=0 exactly, and the first transition already blend in D(x_hat_0).
        alpha_k = None if is_last else float(alphas[k + 1])
        result = infer_round(model, dataset, device, alpha_k, data_range,
                             batch_size=args.eval_batch_size, num_workers=args.num_workers,
                             trajectory_indices=traj_indices, out_counts_mm=out_mm)
        print(f"round {k}: PSNR {result['psnr']:.2f}  SSIM {result['ssim']:.3f}"
              + ("" if alpha_k is None else f"  (renoised with alpha={alpha_k:.3f})"), flush=True)

        metrics.append({"round": k, "epochs": epochs, "alpha": alpha_k,
                        "psnr": result["psnr"], "ssim": result["ssim"]})
        metrics_path.write_text(json.dumps(metrics, indent=2))

        np.savez(run_dir / "trajectory" / f"round_{k:02d}.npz",
                 frame_ids=traj_frame_ids,
                 x_hat=np.stack([result["trajectory"][int(i)]["x_hat"] for i in traj_indices]),
                 D=np.stack([result["trajectory"][int(i)]["D"] for i in traj_indices]))

        if k == 0:
            write_counts_meta(mmse_anchor_path, first_index, n, crop, EVAL_DOSE, EVAL_NOISE_SEED,
                              ckpt="round0_mmse_anchor")
        if is_last:
            write_counts_meta(final_estimate_path, first_index, n, crop, EVAL_DOSE, EVAL_NOISE_SEED,
                              ckpt=f"round{k}_final_estimate")

        save_state(state_path, model, k, {"z_min": z_min, "z_max": z_max, "rounds": args.rounds,
                                          "schedule": args.schedule, "k": args.k})

    print("\n=== round loop complete; reconstructing MMSE anchor + final estimate ===", flush=True)
    from sdate.tr_diffusion import reconstruct as R
    variants = {"annealed_n2n_final": str(final_estimate_path), "mmse_anchor": str(mmse_anchor_path)}
    res = R.run_windows(prof.mov_path, prof.memmap_path, variants, det_bin=args.det_bin, method="fbp",
                        frame_start=args.eval_frame_start, frame_end=args.eval_frame_end,
                        axis_col=prof.rot_axis_col, deg_per_frame=prof.deg_per_frame,
                        device=device, log_every=25)
    summary = {"n_windows": len(res["window_starts"]), "eval_frame_range": [args.eval_frame_start, args.eval_frame_end]}
    for arm in list(variants) + ["noisy"]:
        m = res["metrics"][arm]
        summary[f"{arm}_psnr"] = float(m["psnr"].mean())
        summary[f"{arm}_ssim"] = float(m["ssim"].mean())
        summary[f"{arm}_sharpness"] = float(m["sharpness"].mean())
        print(f"  {arm:20s} PSNR {m['psnr'].mean():6.2f}  SSIM {m['ssim'].mean():.3f}  "
              f"sharpness {m['sharpness'].mean():.3f}", flush=True)
    (run_dir / "recon_summary.json").write_text(json.dumps(summary, indent=2))
    print("SUMMARY", json.dumps(summary, indent=2), flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
