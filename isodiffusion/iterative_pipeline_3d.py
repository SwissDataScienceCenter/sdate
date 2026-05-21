#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
import multiprocessing
import json
from pathlib import Path
import numpy as np
import wandb

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

def parse_args():
    parser = argparse.ArgumentParser(description="Iterative 3D IsoDiffusion Pipeline")
    parser.add_argument("--data_path", type=str, required=True, help="Starting condition volume (e.g. la_fourier_1.npy)")
    parser.add_argument("--ground_truth_path", type=str, default=None, help="Optional ground truth for metrics")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save reconstructions and checkpoints. Defaults to same directory as data_path")
    
    parser.add_argument("--num_rounds", type=int, default=5, help="Number of iterative rounds")
    parser.add_argument("--debug_crop", type=int, default=None, help="Crop volumes to this size (e.g. 96) for fast debugging")
    
    # Training params
    parser.add_argument("--cone_width_deg", type=float, default=72.0)
    parser.add_argument("--patch_size", type=int, default=167)
    parser.add_argument("--volume_size", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for round 0")
    parser.add_argument("--finetune_epochs", type=int, default=10, help="Epochs for finetuning in round > 0")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--exp_name", type=str, default="isodiffusion_iterative")
    parser.add_argument("--no_rotate", action="store_true", help="Disable rotation during training")
    
    # Inference params
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--subvolume_batch_size", type=int, default=2, help="Batch size for subvolume processing, this is for gpu memory")
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of diffusion steps for sampling")
    parser.add_argument("--start_step_frac", type=float, default=0.1, help="Fraction of diffusion steps to skip at the beginning, starts from conditioning")
    parser.add_argument("--reconstruction_batch_size", type=int, default=2, help="Batch size for reconstruction. If > 1, average over slices for guidance")
    
    return parser.parse_args()


def run_training_subprocess(data_path, checkpoint_path, epochs, args, is_finetuning=False):
    train_script = _PROJECT_ROOT / "isodiffusion" / "train_conditional_3d.py"
    cmd = [
        sys.executable, str(train_script),
        "--data_path", str(data_path),
        "--cone_width_deg", str(args.cone_width_deg),
        "--patch_size", str(args.patch_size),
        "--volume_size", str(args.volume_size),
        "--batch_size", str(args.batch_size),
        "--epochs", str(epochs),
        "--learning_rate", str(args.learning_rate),
        "--exp_name", args.exp_name,
        "--save_checkpoint", str(checkpoint_path)
    ]
    if args.no_rotate:
        cmd.append("--no_rotate")
    if is_finetuning and Path(checkpoint_path).exists():
        cmd.extend(["--load_checkpoint", str(checkpoint_path)])
        
    print(f"Running training: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("Training subprocess failed")


def run_inference_worker(checkpoint_path, condition_path, initial_path, output_path, args):
    import torch
    import numpy as np
    from isodiffusion.recon_utils import load_unet3d, load_norm_fns_from_checkpoint_sidecar, load_npy_volume
    from isodiffusion.schedulers.pipeline_ddim import DDIMPipeline
    from isodiffusion.schedulers.scheduling_ddim import GuidedDDIMScheduler
    from isodiffusion.fourier_wedge import enforce_known_fourier

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, config = load_unet3d(checkpoint_path, device=device)
    normalize_fn, denormalize_fn, norm_config = load_norm_fns_from_checkpoint_sidecar(checkpoint_path)
    
    cone_width_deg = float(norm_config.get('cone_width_deg', 72.0))
    angular_range_deg = 180.0 - cone_width_deg
    center = float(norm_config.get('carve_center_angle_deg', 0.0))
    start_angle_deg = (center + cone_width_deg / 2.0) % 180.0
    tilt_axis = int(norm_config.get('tilt_axis', 0))
    
    condition = load_npy_volume(condition_path)
    initial = load_npy_volume(initial_path)
    
    def make_volume_batch(volume: torch.Tensor, batch_size: int) -> torch.Tensor:
        if volume.ndim == 3:
            return volume.unsqueeze(0).repeat(batch_size, 1, 1, 1).contiguous()
        if volume.ndim == 4:
            return volume.contiguous()
        raise ValueError(f'expected (D, H, W) or (bs, D, H, W), got {tuple(volume.shape)}')
        
    condition_batch = make_volume_batch(condition, args.reconstruction_batch_size)
    initial_batch = make_volume_batch(initial, args.reconstruction_batch_size)
    
    def fourier_guidance(source_volume: torch.Tensor, vol_init: torch.Tensor, timestep: int = 0) -> torch.Tensor:
        if vol_init.ndim == 3:
            if source_volume.ndim == 4:
                source_volume = source_volume[0]
            return enforce_known_fourier(
                estimate_volume=vol_init,
                measured_volume=source_volume.to(vol_init.device, dtype=vol_init.dtype),
                angular_range_deg=angular_range_deg,
                start_angle_deg=start_angle_deg,
                tilt_axis=tilt_axis,
            )

        if vol_init.ndim != 4:
            raise ValueError(f'guidance expects (D, H, W) or (bs, D, H, W), got {tuple(vol_init.shape)}')

        if source_volume.ndim == 3:
            source_volume = source_volume.unsqueeze(0).expand(vol_init.shape[0], -1, -1, -1)
        
        guided = [
            enforce_known_fourier(
                estimate_volume=estimate,
                measured_volume=measured.to(estimate.device, dtype=estimate.dtype),
                angular_range_deg=angular_range_deg,
                start_angle_deg=start_angle_deg,
                tilt_axis=tilt_axis,
            )
            for estimate, measured in zip(vol_init, source_volume)
        ]
        return torch.stack(guided, dim=0)

    guidance_source_volume = condition_batch
    
    def guidance(x_0_raw: torch.Tensor, timestep: int) -> torch.Tensor:
        x =  fourier_guidance(
            source_volume=guidance_source_volume,
            vol_init=x_0_raw,
            timestep=int(timestep),
        )
        x = torch.mean(x, dim=0).repeat(x.shape[0], 1, 1, 1)  # average over slices
        return x

    noise_scheduler = GuidedDDIMScheduler(
        num_train_timesteps=1000,
        guidance_function=guidance,
        clip_sample_range=6
    )

    pipeline = DDIMPipeline(
        unet=model.to(device),
        scheduler=noise_scheduler,
        conditioning=condition_batch.to(device),
        normalize_fn=normalize_fn,
        denormalize_fn=denormalize_fn,
        subvolume_batch_size=args.subvolume_batch_size,
        overlap=args.overlap,
    )
    
    recon = initial_batch.float()
    
    print(f"Running truncated DDIM pipeline...")
    result = pipeline.truncated_pipeline(
        initial_guess=recon.to(device),
        start_step=int(args.start_step_frac * args.num_inference_steps),
        num_inference_steps=args.num_inference_steps,
        use_clipped_model_output=True,
        p_use_conditioning=1.0,
    )
    recon = result.images
    recon = guidance(recon.to(device), 0).cpu()
    if len(recon.shape) == 4:
        recon = recon.mean(dim=0)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, recon.numpy().astype(np.float32))
    print(f"Saved reconstruction to {output_path}")

def compute_metrics(reconstruction_path, ground_truth_path):
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
    recon = np.load(reconstruction_path)
    gt = np.load(ground_truth_path)

    if len(recon.shape) == 4: # if we produced a batch, we test the mean reconsruction
        recon = np.mean(recon, axis=0)
    
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
        
    return {
        "psnr": np.mean(psnrs),
        "mse": np.mean(mses),
        "ssim": np.mean(ssims),
    }

def main():
    args = parse_args()
    
    # Setup WandB
    wandb.init(project="isodiffusion_iterative", name=args.exp_name, config=vars(args))
    
    data_path = Path(args.data_path).resolve()
    gt_path = Path(args.ground_truth_path).resolve() if args.ground_truth_path else None
    
    output_dir = Path(args.output_dir).resolve() if args.output_dir else data_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.debug_crop:
        print(f"Applying debug crop of size {args.debug_crop}^3...")
        c = args.debug_crop
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
    
    checkpoint_path = output_dir / "checkpoint_latest.pt"
    
    current_condition = data_path
    
    for rnd in range(args.num_rounds):
        print(f"\\n{'='*50}\\nStarting Round {rnd}\\n{'='*50}")
        
        recon_output_path = output_dir / f"reconstruction_round_{rnd}.npy"
        
        if recon_output_path.exists():
            print(f"Reconstruction for round {rnd} already exists. Skipping to next round.")
            current_condition = recon_output_path
            continue
            
        # Training Phase
        epochs = args.epochs if rnd == 0 else args.finetune_epochs
        is_finetuning = (rnd > 0)
        print(f"Starting training phase for round {rnd} (epochs: {epochs}, finetuning: {is_finetuning})")
        run_training_subprocess(
            data_path=current_condition,
            checkpoint_path=checkpoint_path,
            epochs=epochs,
            args=args,
            is_finetuning=is_finetuning
        )
        
        # Inference Phase
        print(f"Starting inference phase for round {rnd}")
        # Initial guess is the current condition (which is reconstruction from previous round, or starting volume)
        initial_guess_path = current_condition 
        
        p = multiprocessing.Process(
            target=run_inference_worker, 
            args=(checkpoint_path, current_condition, initial_guess_path, recon_output_path, args)
        )
        p.start()
        p.join()
        
        if p.exitcode != 0:
            raise RuntimeError(f"Inference failed for round {rnd}")
            
        current_condition = recon_output_path
        
        # Metrics & Logging
        if gt_path:
            metrics = compute_metrics(recon_output_path, gt_path)
            print(f"Round {rnd} Metrics: {metrics}")
            wandb.log({
                "round": rnd,
                "val_psnr": metrics["psnr"],
                "val_mse": metrics["mse"],
                "val_ssim": metrics["ssim"]
            })
        else:
            wandb.log({"round": rnd})
            
    print("Pipeline completed successfully.")
    wandb.finish()

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    main()
