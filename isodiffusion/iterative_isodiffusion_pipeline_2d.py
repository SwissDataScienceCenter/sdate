#!/usr/bin/env python3
"""Iterative 2D IsoDiffusion pipeline.

Each round:
  1. Train (or fine-tune) the conditional 2D diffusion model on the current volume.
  2. Run guided DDIM inference (DDIMPipeline2D + GuidedDDIMScheduler) on the current
     volume with Fourier consistency guidance from the original condition volume.
  3. The output reconstruction becomes the input for the next round.
"""

import multiprocessing
import subprocess
import sys
from pathlib import Path

import numpy as np
import wandb

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Iterative 2D IsoDiffusion Pipeline")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Condition/measured volume (starting point)")
    parser.add_argument("--ground_truth_path", type=str, default=None,
                        help="Optional ground truth for metrics")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for reconstructions and checkpoints")
    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--debug_crop", type=int, default=None,
                        help="Crop volumes to this cubic size for fast debugging")

    # --- Training params (forwarded to train_conditional_2d.py) ---
    parser.add_argument("--cone_width_deg", type=float, default=72.0)
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--volume_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Axial slice batch size for 2D diffusion training")
    parser.add_argument("--volume_batch_size", type=int, default=5,
                        help="Number of 3D volumes per dataloader step")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Epochs for round 0")
    parser.add_argument("--finetune_epochs", type=int, default=20,
                        help="Epochs for finetuning in rounds > 0")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--exp_name", type=str, default="isodiffusion2d_iterative")
    parser.add_argument("--no_rotate", action="store_true")
    parser.add_argument("--conditioning_probability", type=float, default=0.5)
    parser.add_argument("--loss_type", choices=["mae", "mse", "huber"], default="huber")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16", "auto"], default="fp16")
    parser.add_argument("--carve_center_angle_deg", type=float, default=0.0)
    parser.add_argument("--tilt_axis", type=int, default=0)
    parser.add_argument("--wandb_training", action="store_true",
                        help="Enable wandb logging inside training subprocesses")

    # --- Inference params ---
    parser.add_argument("--inference_patch_size", type=int, default=None,
                        help="Patch size for DDIM inference; defaults to --volume_size (must match UNet input size)")
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--slice_batch_size", type=int, default=100,
                        help="Number of 2D slices per UNet forward pass during inference")
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--start_step_frac", type=float, default=0.01,
                        help="Fraction of DDIM steps to skip at the start (truncated diffusion)")
    parser.add_argument("--inference_batch_size", type=int, default=10,
                        help="Volume replicas run in parallel; >1 enables ensemble averaging")
    parser.add_argument("--clip_sample_range", type=float, default=6.0)
    parser.add_argument("--no_axial_only", dest="axial_only", action="store_false",
                        help="Use multi-axial (axial+sagittal+coronal) instead of axial-only inference")
    parser.set_defaults(axial_only=True)

    return parser.parse_args()


def run_training_subprocess(data_path, checkpoint_path, epochs, args, is_finetuning=False):
    train_script = _PROJECT_ROOT / "isodiffusion" / "train_conditional_2d.py"
    cmd = [
        sys.executable, str(train_script),
        "--data_path", str(data_path),
        "--cone_width_deg", str(args.cone_width_deg),
        "--patch_size", str(args.patch_size),
        "--volume_size", str(args.volume_size),
        "--batch_size", str(args.batch_size),
        "--volume_batch_size", str(args.volume_batch_size),
        "--epochs", str(epochs),
        "--learning_rate", str(args.learning_rate),
        "--weight_decay", str(args.weight_decay),
        "--warmup_steps", str(args.warmup_steps),
        "--exp_name", args.exp_name,
        "--loss_type", args.loss_type,
        "--mixed_precision", args.mixed_precision,
        "--conditioning_probability", str(args.conditioning_probability),
        "--carve_center_angle_deg", str(args.carve_center_angle_deg),
        "--tilt_axis", str(args.tilt_axis),
        "--save_checkpoint", str(checkpoint_path),
    ]
    if args.no_rotate:
        cmd.append("--no_rotate")
    if args.wandb_training:
        cmd.append("--wandb")
    if is_finetuning:
        cmd.extend(["--load_checkpoint", str(checkpoint_path)])

    print(f"Running training: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("Training subprocess failed")


def run_inference_worker(checkpoint_path, condition_vol_path, current_vol_path, output_path, args):
    import torch
    from isodiffusion.fourier_wedge import enforce_known_fourier
    from isodiffusion.recon_utils import (
        load_norm_fns_from_checkpoint_sidecar,
        load_npy_volume,
        load_unet2d,
    )
    from isodiffusion.schedulers.pipeline_ddim_2d import DDIMPipeline2D
    from isodiffusion.schedulers.scheduling_ddim import GuidedDDIMScheduler

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_unet2d(checkpoint_path, device=device)
    normalize_fn, denormalize_fn, norm_config = load_norm_fns_from_checkpoint_sidecar(checkpoint_path)

    cone_width_deg = float(norm_config.get("cone_width_deg", args.cone_width_deg))
    angular_range_deg = 180.0 - cone_width_deg
    center = float(norm_config.get("carve_center_angle_deg", args.carve_center_angle_deg))
    start_angle_deg = (center + cone_width_deg / 2.0) % 180.0
    tilt_axis = int(norm_config.get("tilt_axis", args.tilt_axis))

    condition = load_npy_volume(condition_vol_path)   # original measured volume, fixed
    current = load_npy_volume(current_vol_path)       # current estimate, used as initial guess

    B = args.inference_batch_size
    condition_batch = condition.unsqueeze(0).expand(B, -1, -1, -1).contiguous()
    initial_batch = current.unsqueeze(0).expand(B, -1, -1, -1).contiguous()

    condition_batch_dev = condition_batch.to(device)

    def guidance(x_0_raw: torch.Tensor, timestep: int = 0) -> torch.Tensor:
        guided = torch.stack([
            enforce_known_fourier(
                estimate_volume=est,
                measured_volume=meas.to(est.device, dtype=est.dtype),
                angular_range_deg=angular_range_deg,
                start_angle_deg=start_angle_deg,
                tilt_axis=tilt_axis,
            )
            for est, meas in zip(x_0_raw, condition_batch_dev)
        ])
        # Consensus across ensemble replicas (no-op when B=1)
        return guided.mean(dim=0, keepdim=True).expand(B, -1, -1, -1).contiguous()

    noise_scheduler = GuidedDDIMScheduler(
        num_train_timesteps=1000,
        guidance_function=guidance,
        clip_sample_range=args.clip_sample_range,
    )

    inference_patch_size = args.inference_patch_size if args.inference_patch_size is not None else args.volume_size

    pipeline = DDIMPipeline2D(
        unet=model,
        scheduler=noise_scheduler,
        conditioning=condition_batch.to(device),
        normalize_fn=normalize_fn,
        denormalize_fn=denormalize_fn,
        slice_batch_size=args.slice_batch_size,
        overlap=args.overlap,
        patch_size=inference_patch_size,
        axial_only=args.axial_only,
    )

    num_inference_steps = args.num_inference_steps
    start_step = int(args.start_step_frac * num_inference_steps)

    result = pipeline.truncated_pipeline(
        initial_guess=initial_batch.float().to(device),
        start_step=start_step,
        num_inference_steps=num_inference_steps,
        use_clipped_model_output=True,
    )
    recon = result.images  # (B, D, H, W) on CPU, averaged across ensemble
    recon = recon.mean(dim=0)  # (D, H, W)
    recon = guidance(recon)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, recon.numpy().astype(np.float32))
    print(f"Saved reconstruction to {output_path}")


def compute_metrics(reconstruction_path, ground_truth_path):
    from skimage.metrics import mean_squared_error, peak_signal_noise_ratio, structural_similarity

    recon = np.load(reconstruction_path)
    gt = np.load(ground_truth_path)

    psnrs, mses, ssims = [], [], []
    for z in range(recon.shape[0]):
        slice_r = recon[z]
        slice_g = gt[z]
        data_range = slice_g.max() - slice_g.min()
        if data_range == 0:
            data_range = 1.0
        psnrs.append(peak_signal_noise_ratio(slice_g, slice_r, data_range=data_range))
        mses.append(mean_squared_error(slice_g, slice_r))
        ssims.append(structural_similarity(slice_g, slice_r, data_range=data_range))

    return {"psnr": np.mean(psnrs), "mse": np.mean(mses), "ssim": np.mean(ssims)}


def main():
    args = parse_args()

    wandb.init(project="isodiffusion2d_iterative", name=args.exp_name, config=vars(args))

    data_path = Path(args.data_path).resolve()
    gt_path = Path(args.ground_truth_path).resolve() if args.ground_truth_path else None
    output_dir = Path(args.output_dir).resolve() if args.output_dir else data_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.debug_crop:
        c = args.debug_crop
        print(f"Applying debug crop of size {c}^3...")
        crop_data_path = output_dir / f"cropped_condition_{c}.npy"
        if not crop_data_path.exists():
            vol = np.load(data_path)
            np.save(crop_data_path, vol[:c, :c, :c])
        data_path = crop_data_path

        if gt_path:
            crop_gt_path = output_dir / f"cropped_gt_{c}.npy"
            if not crop_gt_path.exists():
                gt_vol = np.load(gt_path)
                np.save(crop_gt_path, gt_vol[:c, :c, :c])
            gt_path = crop_gt_path

    checkpoint_path = output_dir / "isodiffusion2d_checkpoint_latest.pt"
    condition_vol = data_path   # original measured volume, never changes
    current_vol = data_path     # current reconstruction estimate, updated each round

    for rnd in range(args.num_rounds):
        print(f"\n{'='*50}\nStarting Round {rnd}\n{'='*50}")

        recon_output_path = output_dir / f"isodiffusion2d_reconstruction_round_{rnd}.npy"

        if recon_output_path.exists():
            print(f"Reconstruction for round {rnd} already exists. Skipping.")
            current_vol = recon_output_path
            continue

        epochs = args.epochs if rnd == 0 else args.finetune_epochs
        is_finetuning = rnd > 0
        print(f"Training phase: round={rnd}, epochs={epochs}, finetuning={is_finetuning}")
        run_training_subprocess(
            data_path=current_vol,
            checkpoint_path=checkpoint_path,
            epochs=epochs,
            args=args,
            is_finetuning=is_finetuning,
        )

        print(f"Inference phase: round={rnd}")
        p = multiprocessing.Process(
            target=run_inference_worker,
            args=(checkpoint_path, condition_vol, current_vol, recon_output_path, args),
        )
        p.start()
        p.join()

        if p.exitcode != 0:
            raise RuntimeError(f"Inference failed for round {rnd}")

        current_vol = recon_output_path

        if gt_path:
            metrics = compute_metrics(recon_output_path, gt_path)
            print(f"Round {rnd} metrics: {metrics}")
            wandb.log({"round": rnd, "val_psnr": metrics["psnr"], "val_mse": metrics["mse"], "val_ssim": metrics["ssim"]})
        else:
            wandb.log({"round": rnd})

    print("Pipeline completed successfully.")
    wandb.finish()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    main()
