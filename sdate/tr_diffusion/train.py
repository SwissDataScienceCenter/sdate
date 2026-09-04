#!/usr/bin/env python3
"""Train the conditional time-resolved frame denoiser (diffusion or baseline).

Both models share the dataset, the N2V masking, and the conditioning-dropout
recipe; ``--mode`` selects which is trained:

* ``diffusion`` — ε-prediction conditional DDPM (in_channels = 2 + 4k).
* ``baseline``  — single-pass x_0 regressor (in_channels = 1 + 4k).

Example (native self-supervised, k=1, memmap-backed)::

    python -m sdate.tr_diffusion.train --mode diffusion \
        --mov /myhome/data/sdate/shared/time_resolved/212_Wunderkerze2/212_Wunderkerze2.mov \
        --memmap /myhome/data/sdate/shared/time_resolved/212_Wunderkerze2/frames_400k_600k.u16 \
        --k 1 --frame_start 400000 --frame_end 600000 \
        --batch_size 16 --epochs 100 --exp_name tr_diff_k1 --wandb

Reuses ``pytorch_base.PyTorchExperiment`` and the diffusers UNet2DModel, matching
``isodiffusion/train_conditional_2d.py``.
"""

from __future__ import annotations

import json
import os
import random
import sys
from argparse import ArgumentParser, BooleanOptionalAction
from pathlib import Path

import torch
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
for _c in (Path("/myhome/BaseTraining"), Path("/myhome/sdsc"), Path("/myhome/chip-project")):
    if _c.exists() and str(_c) not in sys.path:
        sys.path.insert(0, str(_c))

from pytorch_base.experiment import PyTorchExperiment  # noqa: E402

from sdate.tr_diffusion.data import TimeResolvedFrameDataset  # noqa: E402
from sdate.tr_diffusion.losses import (  # noqa: E402
    BaselineN2NLoss, BaselineN2VLoss, DiffusionN2NLoss, DiffusionN2VLoss,
)
from sdate.tr_diffusion.model import create_baseline_unet, create_diffusion_unet  # noqa: E402


def parse_args():
    p = ArgumentParser(description="Train time-resolved conditional frame denoiser.")
    p.add_argument("--mode", choices=["diffusion", "baseline"], default="diffusion")
    p.add_argument("--denoise_mode", choices=["n2v", "n2n"], default="n2v",
                   help="n2v = blind-spot masked loss (default); n2n = binomial-split "
                        "Noise2Noise (requires --extra_noise_dose; conditions on split fraction p).")
    p.add_argument("--p_min", type=float, default=0.1, help="n2n: min split fraction sampled per example.")
    p.add_argument("--p_max", type=float, default=0.9, help="n2n: max split fraction.")
    p.add_argument("--p_bins", type=int, default=100, help="n2n: discretisation of p for the class embedding.")
    p.add_argument("--n2n_prediction_type", choices=["epsilon", "sample"], default="epsilon",
                   help="n2n diffusion only: predict added noise (original) or the target-split x0 directly.")
    p.add_argument("--n2n_consistency_weight", type=float, default=0.0,
                   help="n2n diffusion only: weight on the swap-consistency term (0 = disabled, original behaviour). "
                        "Predicts BOTH directions (input->target, target->input) at independent timesteps and "
                        "penalises disagreement between the two x0 estimates.")
    p.add_argument("--profile", type=str, default=None,
                   help="DatasetProfile name (e.g. asc_thixo) or JSON path; supplies "
                        "mov/memmap/crop/frame-range/axis/deg_per_frame (explicit flags override).")
    p.add_argument("--mov", type=str, default=None, help="Path to the .mov (with .norm.npz sidecar).")
    p.add_argument("--memmap", type=str, default=None, help="Optional pre-extracted uint16 memmap for fast access.")
    p.add_argument("--k", type=int, default=1, help="Context radius: in_channels = (2 or 1) + 4k.")
    p.add_argument("--neighborhoods", choices=["both", "rotation", "temporal"], default="both",
                   help="Which context taps to keep: both (default), rotation-only (angular "
                        "neighbours, no same-angle-across-turns taps), or temporal-only "
                        "(same-angle-across-turns taps, no angular neighbours). Ablates the "
                        "two conditioning neighbourhoods independently.")
    p.add_argument("--frame_start", type=int, default=None)
    p.add_argument("--frame_end", type=int, default=None)
    p.add_argument("--crop_h", type=int, default=None)
    p.add_argument("--crop_w", type=int, default=None)
    p.add_argument("--deg_per_frame", type=float, default=None, help="Rotation rate (from profile if unset).")
    p.add_argument("--axis_col", type=float, default=None, help="Centre-of-rotation column (from profile if unset).")
    p.add_argument("--include_mirror", action="store_true", help="Add 180-degree half-turn mirror taps (+2 ch).")
    p.add_argument("--temporal_raw_pairs", action=BooleanOptionalAction, default=True,
                   help="Default ON: replace each interpolated same-angle temporal tap (PERIOD_360 is "
                        "never an integer, so a plain blend of the two bracketing frames at a fixed "
                        "ratio was the old behaviour -- confirmed to ghost at moving edges) with BOTH "
                        "bracketing frames as separate, un-blurred channels (temporal context doubles "
                        "2k -> 4k, total context 4k -> 6k; confirmed a small but consistent reconstruction "
                        "PSNR/SSIM improvement over the interpolated version). Pass --no-temporal_raw_pairs "
                        "to reproduce the old interpolated behaviour (e.g. to match an existing checkpoint's "
                        "recipe).")
    p.add_argument("--extra_noise_dose", type=float, default=None,
                   help="If set (0<dose<=1), train on extra-Poisson-noised frames; original kept as reference.")
    p.add_argument("--noise_seed", type=int, default=None,
                   help="Extra-noise RNG seed. Default None = fresh noise each epoch (train); "
                        "set an int for reproducible per-frame noise (eval).")
    p.add_argument("--max_samples", type=int, default=None, help="Subsample this many usable centres for the epoch.")
    p.add_argument("--norm_min", type=float, default=None)
    p.add_argument("--norm_max", type=float, default=None)

    p.add_argument("--n2v_ratio", type=float, default=0.02, help="Blind-spot pixel fraction.")
    p.add_argument("--n2v_window", type=int, default=5, help="Blind-spot neighbour window (odd).")
    p.add_argument("--conditioning_probability", type=float, default=0.5,
                   help="Probability of keeping the corrupted central frame (with/without-central).")
    p.add_argument("--cond_angle_time", action="store_true",
                   help="Baseline only: add 3 extra input channels -- sin(angle), cos(angle) of "
                        "the central frame's rotation angle, and its normalised position in "
                        "[frame_start, frame_end) -- so the model can learn angle/time-dependent "
                        "structure directly instead of only inferring it from the neighbour frames.")
    p.add_argument("--loss_type", choices=["mae", "mse", "huber"], default="huber")
    p.add_argument("--edge_weight", type=float, default=0.0,
                   help="Baseline (n2v) only: up-weight the masked pixel loss at high-gradient "
                        "(edge) GT pixels by 1+edge_weight*normalized_grad, to fight regression-to-"
                        "mean blur. 0 = disabled (original behaviour).")

    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--test_fraction", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--exp_name", type=str, default="tr_denoise")
    p.add_argument("--save_checkpoint", type=str, default="")
    p.add_argument("--load_checkpoint", type=str, default="")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--mixed_precision", choices=["no", "fp16", "bf16", "auto"], default="fp16")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve dataset geometry from a profile (explicit flags override profile values).
    from sdate.tr_diffusion.geometry import ANGLE_TIME_COND_CHANNELS, DEG_PER_FRAME, ROT_AXIS_COL
    from sdate.tr_diffusion.profiles import DatasetProfile
    prof = DatasetProfile.load(args.profile) if args.profile else None
    _pick = lambda v, pv, dv: v if v is not None else (pv if prof is not None else dv)
    args.mov = args.mov or (prof.mov_path if prof else None)
    if not args.mov:
        raise SystemExit("provide --mov or --profile")
    args.memmap = args.memmap or (prof.memmap_path if prof else None)
    args.crop_h = _pick(args.crop_h, prof.crop[0] if prof else None, 128)
    args.crop_w = _pick(args.crop_w, prof.crop[1] if prof else None, 512)
    args.frame_start = _pick(args.frame_start, prof.frame_start if prof else None, 400_000)
    args.frame_end = _pick(args.frame_end, prof.frame_end if prof else None, 600_000)
    args.deg_per_frame = _pick(args.deg_per_frame, prof.deg_per_frame if prof else None, DEG_PER_FRAME)
    args.axis_col = _pick(args.axis_col, prof.rot_axis_col if prof else None, ROT_AXIS_COL)
    print(f"geometry: mov={args.mov}\n  crop={(args.crop_h, args.crop_w)} range=[{args.frame_start},{args.frame_end}] "
          f"deg/frame={args.deg_per_frame} axis={args.axis_col}")

    if args.denoise_mode == "n2n" and args.extra_noise_dose is None:
        raise SystemExit("--denoise_mode n2n requires --extra_noise_dose (the fixed measurement dose to split)")
    norm_range = (args.norm_min, args.norm_max) if (args.norm_min is not None and args.norm_max is not None) else None
    dataset = TimeResolvedFrameDataset(
        mov_path=args.mov, memmap_path=args.memmap, k=args.k,
        frame_start=args.frame_start, frame_end=args.frame_end,
        crop=(args.crop_h, args.crop_w), include_mirror=args.include_mirror,
        neighborhoods=args.neighborhoods,
        norm_range=norm_range, extra_noise_dose=args.extra_noise_dose,
        max_samples=args.max_samples, seed=args.seed,
        axis_col=args.axis_col, deg_per_frame=args.deg_per_frame,
        n2n=(args.denoise_mode == "n2n"), p_range=(args.p_min, args.p_max), p_bins=args.p_bins,
        cond_angle_time=args.cond_angle_time, temporal_raw_pairs=args.temporal_raw_pairs,
    )
    n = len(dataset)
    test_size = max(1, int(args.test_fraction * n)) if n > 1 else 0
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_ds = torch.utils.data.Subset(dataset, idx[:-test_size] if test_size else idx)
    test_ds = torch.utils.data.Subset(dataset, idx[-test_size:] if test_size else idx)
    print(f"Dataset: {n} usable centres -> train={len(train_ds)}, test={len(test_ds)}")
    print(f"norm_min={dataset.norm_min:.3f}, norm_max={dataset.norm_max:.3f}, "
          f"in_channels(diffusion={dataset.in_channels_diffusion}, baseline={dataset.in_channels_baseline})")

    sample_size = (args.crop_h, args.crop_w)
    n2n = args.denoise_mode == "n2n"
    extra_cond_channels = ANGLE_TIME_COND_CHANNELS if args.cond_angle_time else 0
    if args.cond_angle_time and args.mode != "baseline":
        raise SystemExit("--cond_angle_time is currently only wired up for --mode baseline")
    if args.mode == "diffusion":
        model = create_diffusion_unet(k=args.k, sample_size=sample_size, include_mirror=args.include_mirror,
                                      neighborhoods=args.neighborhoods, temporal_raw_pairs=args.temporal_raw_pairs)
        noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
        loss_fn = (DiffusionN2NLoss(noise_scheduler, device, loss_type=args.loss_type,
                                    prediction_type=args.n2n_prediction_type,
                                    consistency_weight=args.n2n_consistency_weight,
                                    p_bins=args.p_bins) if n2n else
                   DiffusionN2VLoss(noise_scheduler, device, ratio=args.n2v_ratio, window=args.n2v_window,
                                    conditioning_probability=args.conditioning_probability, loss_type=args.loss_type))
        assert model.config.in_channels == dataset.in_channels_diffusion
    else:
        model = create_baseline_unet(k=args.k, sample_size=sample_size, include_mirror=args.include_mirror,
                                     neighborhoods=args.neighborhoods, extra_cond_channels=extra_cond_channels,
                                     temporal_raw_pairs=args.temporal_raw_pairs)
        loss_fn = (BaselineN2NLoss(device, loss_type=args.loss_type) if n2n else
                   BaselineN2VLoss(device, ratio=args.n2v_ratio, window=args.n2v_window,
                                   conditioning_probability=args.conditioning_probability, loss_type=args.loss_type,
                                   edge_weight=args.edge_weight))
        assert model.config.in_channels == dataset.in_channels_baseline
    print(f"mode={args.mode}  denoise={args.denoise_mode}  model params: {sum(p.numel() for p in model.parameters()):,}")

    if args.save_checkpoint:
        checkpoint_path = args.save_checkpoint
    else:
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"checkpoints/tr_denoise_{args.mode}_{args.exp_name}.pt"
    os.makedirs(Path(checkpoint_path).parent, exist_ok=True)

    with open(checkpoint_path.replace(".pt", "_config.json"), "w") as f:
        json.dump({
            "mode": args.mode, "denoise_mode": args.denoise_mode, "k": args.k,
            "crop": [args.crop_h, args.crop_w],
            "include_mirror": args.include_mirror, "neighborhoods": args.neighborhoods,
            "temporal_raw_pairs": args.temporal_raw_pairs,
            "in_channels": model.config.in_channels,
            "norm_min": dataset.norm_min, "norm_max": dataset.norm_max,
            "deg_per_frame": args.deg_per_frame, "axis_col": args.axis_col,
            "profile": args.profile,
            "frame_start": args.frame_start, "frame_end": args.frame_end,
            "extra_noise_dose": args.extra_noise_dose,
            "n2v_ratio": args.n2v_ratio, "n2v_window": args.n2v_window,
            "conditioning_probability": args.conditioning_probability,
            "cond_angle_time": args.cond_angle_time,
            "loss_type": args.loss_type, "edge_weight": args.edge_weight,
            "p_min": args.p_min, "p_max": args.p_max, "p_bins": args.p_bins,
            "n2n_prediction_type": args.n2n_prediction_type,
            "n2n_consistency_weight": args.n2n_consistency_weight,
            "mov": args.mov,
        }, f, indent=2)

    if args.load_checkpoint:
        try:
            ckpt = torch.load(args.load_checkpoint, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded weights from {args.load_checkpoint}")
        except Exception as exc:  # noqa: BLE001
            print(f"Could not load {args.load_checkpoint}: {exc}. Training from scratch.")

    model.to(device)
    exp = PyTorchExperiment(
        args=vars(args), train_dataset=train_ds, test_dataset=test_ds,
        batch_size=args.batch_size, model=model, loss_fn=loss_fn,
        checkpoint_path=checkpoint_path, experiment_name=args.exp_name,
        with_wandb=args.wandb, num_workers=args.num_workers, seed=args.seed,
        save_always=True, mixed_precision=args.mixed_precision,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=args.warmup_steps,
        num_training_steps=len(exp.train_loader) * args.epochs,
    )
    exp.train(args.epochs, optimizer, scheduler=lr_scheduler)


if __name__ == "__main__":
    main()
