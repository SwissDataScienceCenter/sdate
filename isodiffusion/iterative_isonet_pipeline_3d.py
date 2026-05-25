#!/usr/bin/env python3
"""Iterative 3D IsoNet pipeline.

Each round:
  1. Train (or fine-tune) the IsoNet on the current volume.
  2. Carve the current volume, run it through the model with subvolume patching,
     aggregate patch predictions, and enforce Fourier consistency.
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
    parser = argparse.ArgumentParser(description="Iterative 3D IsoNet Pipeline")
    parser.add_argument("--data_path", type=str, required=True, help="Starting volume (measured/conditioned)")
    parser.add_argument("--target_path", type=str, default=None,
                        help="Frozen v1 target volume, same shape as --data_path. "
                             "Passed unchanged to every training round so the "
                             "supervision target does not drift.")
    parser.add_argument("--ground_truth_path", type=str, default=None, help="Optional ground truth for metrics")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory for reconstructions and checkpoints")

    parser.add_argument("--num_rounds", type=int, default=5)
    parser.add_argument("--debug_crop", type=int, default=None, help="Crop volumes to this size for fast debugging")

    # Training params
    parser.add_argument("--cone_width_deg", type=float, default=72.0)
    parser.add_argument("--patch_size", type=int, default=167)
    parser.add_argument("--volume_size", type=lambda s: (tuple(int(v) for v in s.split(",")) if "," in s else int(s)), default=96)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50, help="Epochs for round 0")
    parser.add_argument("--finetune_epochs", type=int, default=10, help="Epochs for finetuning in round > 0")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--exp_name", type=str, default="isonet_iterative")
    parser.add_argument("--no_rotate", action="store_true")
    parser.add_argument("--predict_residual", action="store_true",
                        help="Train to predict x - carved_x; add carved_x back at inference.")
    parser.add_argument("--fourier_loss_weight", type=float, default=1.0,
                        help="Weight for the Fourier-space log-magnitude loss in the missing wedge (0 to disable).")
    parser.add_argument("--fourier_mse_weight", type=float, default=1.0,
                        help="Weight for the phase MSE term relative to the log-magnitude term (0 to disable).")
    parser.add_argument("--scheduler_type", choices=["cosine_warmup", "cosine_restarts"],
                        default="cosine_warmup")
    parser.add_argument("--T_0", type=int, default=None,
                        help="Epochs per restart for cosine_restarts (default: epochs for that round).")
    parser.add_argument("--T_mult", type=int, default=1)
    parser.add_argument("--model_type", choices=["unet3d", "dynunet", "ddwunet"], default="unet3d",
                        help="Network architecture forwarded to train_isonet_3d.py.")
    parser.add_argument("--dynunet_filters", type=str, default="32,64,128,256,320",
                        help="Comma-separated DynUNet filter counts (used when --model_type dynunet).")
    parser.add_argument("--ddwunet_chans", type=int, default=32,
                        help="Base channel width for DDWUNet (used when --model_type ddwunet).")

    # Inference params
    parser.add_argument("--overlap", type=int, default=10)
    parser.add_argument("--subvolume_batch_size", type=int, default=2, help="Patches per GPU forward pass")

    return parser.parse_args()


def run_training_subprocess(data_path, checkpoint_path, epochs, args, is_finetuning=False):
    train_script = _PROJECT_ROOT / "isodiffusion" / "train_isonet_3d.py"
    cmd = [
        sys.executable, str(train_script),
        "--data_path", str(data_path),
        "--cone_width_deg", str(args.cone_width_deg),
        "--patch_size", str(args.patch_size),
        "--volume_size", ",".join(str(v) for v in args.volume_size) if isinstance(args.volume_size, (tuple, list)) else str(args.volume_size),
        "--batch_size", str(args.batch_size),
        "--epochs", str(epochs),
        "--learning_rate", str(args.learning_rate),
        "--exp_name", args.exp_name,
        "--save_checkpoint", str(checkpoint_path),
    ]
    if args.target_path:
        cmd.extend(["--target_path", str(args.target_path)])
    if args.no_rotate:
        cmd.append("--no_rotate")
    if args.predict_residual:
        cmd.append("--predict_residual")
    cmd.extend(["--scheduler_type", args.scheduler_type])
    if args.T_0 is not None:
        cmd.extend(["--T_0", str(args.T_0)])
    cmd.extend(["--T_mult", str(args.T_mult)])
    cmd.extend(["--model_type", args.model_type])
    if args.model_type == "dynunet":
        cmd.extend(["--dynunet_filters", args.dynunet_filters])
    if args.model_type == "ddwunet":
        cmd.extend(["--ddwunet_chans", str(args.ddwunet_chans)])
    cmd.extend(["--fourier_loss_weight", str(args.fourier_loss_weight)])
    cmd.extend(["--fourier_mse_weight", str(args.fourier_mse_weight)])
    try:
        cmd.extend(["--load_checkpoint", str(checkpoint_path)])
    except:
        print("model not found, starting from random initialization.")

    print(f"Running training: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError("Training subprocess failed")


def run_inference_worker(checkpoint_path, current_vol_path, output_path, args):
    import torch
    from isodiffusion.recon_utils import (
        load_isonet3d,
        load_norm_fns_from_checkpoint_sidecar,
        load_npy_volume,
        patch_starts,
    )
    from isodiffusion.fourier_wedge import apply_missing_wedge, enforce_known_fourier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_isonet3d(checkpoint_path, device=device)
    normalize_fn, denormalize_fn, norm_config = load_norm_fns_from_checkpoint_sidecar(checkpoint_path)

    predict_residual = bool(norm_config.get("predict_residual", False))
    cone_width_deg = float(norm_config.get("cone_width_deg", 72.0))
    angular_range_deg = 180.0 - cone_width_deg
    center = float(norm_config.get("carve_center_angle_deg", 0.0))
    start_angle_deg = (center + cone_width_deg / 2.0) % 180.0
    tilt_axis = int(norm_config.get("tilt_axis", 0))
    volume_size_raw = norm_config.get("volume_size", args.volume_size)
    if isinstance(volume_size_raw, (list, tuple)):
        patch_d, patch_h, patch_w = (int(v) for v in volume_size_raw)
    else:
        patch_d = patch_h = patch_w = int(volume_size_raw)
    cross_attention_dim = int(norm_config.get("cross_attention_dim", 128))

    current_vol = load_npy_volume(current_vol_path)  # (D, H, W), raw values

    # Carve the current volume to produce the model input, matching training
    # carved_vol = apply_missing_wedge(current_vol, angular_range_deg, start_angle_deg, tilt_axis)
    # carved_norm = normalize_fn(carved_vol)  # (D, H, W), normalized 
    
    # no missing wedge is imposed before feeding it to the inference model, so input does not have a missing wedge (except from the firt round)
    carved_norm = normalize_fn(current_vol)  # (D, H, W), normalized 

    d, h, w = carved_norm.shape
    d_starts = patch_starts(d, patch_d, args.overlap)
    h_starts = patch_starts(h, patch_h, args.overlap)
    w_starts = patch_starts(w, patch_w, args.overlap)

    positions = [
        (d0, h0, w0)
        for d0 in d_starts
        for h0 in h_starts
        for w0 in w_starts
    ]
    patches = [
        carved_norm[d0:d0 + patch_d, h0:h0 + patch_h, w0:w0 + patch_w]
        for d0, h0, w0 in positions
    ]

    output = torch.zeros(d, h, w)
    counts = torch.zeros(d, h, w)

    print(f"Running IsoNet inference over {len(patches)} patches...")
    with torch.no_grad():
        for batch_start in range(0, len(patches), args.subvolume_batch_size):
            batch_patches = patches[batch_start:batch_start + args.subvolume_batch_size]
            batch_pos = positions[batch_start:batch_start + args.subvolume_batch_size]

            batch = torch.stack(batch_patches).unsqueeze(1).to(device)  # (B, 1, D, H, W)
            bsz = batch.shape[0]
            timesteps = torch.zeros(bsz, device=device, dtype=torch.long)
            enc_states = torch.zeros(bsz, 1, cross_attention_dim, device=device, dtype=batch.dtype)

            pred = model(
                batch, timestep=timesteps, encoder_hidden_states=enc_states, return_dict=False
            )[0].squeeze(1).cpu()  # (B, D, H, W)

            for i, (d0, h0, w0) in enumerate(batch_pos):
                output[d0:d0 + patch_d, h0:h0 + patch_h, w0:w0 + patch_w] += pred[i]
                counts[d0:d0 + patch_d, h0:h0 + patch_h, w0:w0 + patch_w] += 1.0

    recon = output / counts.clamp_min(1.0)
    if predict_residual:
        recon = recon + torch.as_tensor(np.asarray(carved_norm), dtype=torch.float32)
    recon = denormalize_fn(recon)

    # Enforce the known (non-missing-cone) Fourier components from the current volume
    recon = enforce_known_fourier(
        estimate_volume=recon,
        measured_volume=current_vol,
        angular_range_deg=angular_range_deg,
        start_angle_deg=start_angle_deg,
        tilt_axis=tilt_axis,
    )

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

    wandb.init(project="isonet_iterative", name=args.exp_name, config=vars(args))

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

    checkpoint_path = output_dir / "isonet_checkpoint_latest.pt"
    current_vol = data_path

    for rnd in range(args.num_rounds):
        print(f"\n{'='*50}\nStarting Round {rnd}\n{'='*50}")

        recon_output_path = output_dir / f"isonet_reconstruction_round_{rnd}.npy"

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
            args=(checkpoint_path, current_vol, recon_output_path, args),
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
