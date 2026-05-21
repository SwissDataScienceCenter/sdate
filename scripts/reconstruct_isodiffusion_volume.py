#!/usr/bin/env python3
"""Patch-based 3D iso-diffusion reconstruction for full volumes.

The script loads a limited-angle/reconstruction volume, runs the 3D conditional
diffusion model over overlapping ``W^3`` patches, uniformly averages overlaps,
and saves a full reconstructed ``.npy`` volume.  ``--initial_volume`` can be
provided to start from a previous estimate; otherwise the conditioning volume is
used as the initial estimate.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from isodiffusion.recon_utils import (
    load_npy_volume,
    load_norm_fns_from_checkpoint_sidecar,
    load_unet3d,
    model_config_from_norm_json,
    reconstruct_volume_patches,
)
from isodiffusion.fourier_wedge import make_norm_fns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct a full volume with 3D iso-diffusion.")
    parser.add_argument("--condition_volume", type=str, required=True)
    parser.add_argument("--initial_volume", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--overlap", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_outer_iters", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--start_step_frac", type=float, default=0.8)
    parser.add_argument("--angular_range_deg", type=float, default=None)
    parser.add_argument("--start_angle_deg", type=float, default=None)
    parser.add_argument("--tilt_axis", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | PyTorch {torch.__version__}")

    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config) if args.config else None
    model, config = load_unet3d(checkpoint_path, device=device, config_path=config_path)
    if config_path is None:
        normalize_fn, denormalize_fn, norm_config = load_norm_fns_from_checkpoint_sidecar(checkpoint_path)
    else:
        norm_config = model_config_from_norm_json(config_path)
        normalize_fn, denormalize_fn = make_norm_fns(
            float(norm_config["norm_min"]),
            float(norm_config["norm_max"]),
        )

    condition = load_npy_volume(Path(args.condition_volume))
    if args.initial_volume is None:
        initial = condition.clone()
    else:
        initial = load_npy_volume(Path(args.initial_volume))

    angular_range_deg = args.angular_range_deg
    if angular_range_deg is None:
        cone_width_deg = float(norm_config.get("cone_width_deg", config.get("cone_width_deg", 72.0)))
        angular_range_deg = 180.0 - cone_width_deg
    start_angle_deg = args.start_angle_deg
    if start_angle_deg is None:
        cone_width_deg = float(norm_config.get("cone_width_deg", config.get("cone_width_deg", 72.0)))
        center = float(norm_config.get("carve_center_angle_deg", config.get("carve_center_angle_deg", 0.0)))
        start_angle_deg = (center + cone_width_deg / 2.0) % 180.0
    tilt_axis = args.tilt_axis
    if tilt_axis is None:
        tilt_axis = int(norm_config.get("tilt_axis", config.get("tilt_axis", 0)))

    print(f"Condition volume: {tuple(condition.shape)}")
    print(
        "Fourier guidance: "
        f"angular_range={angular_range_deg:.1f}, start={start_angle_deg:.1f}, axis={tilt_axis}"
    )
    print(
        "Patches: "
        f"size={args.patch_size}, overlap={args.overlap}, batch={args.batch_size}, "
        f"outer_iters={args.num_outer_iters}"
    )

    recon = initial
    for outer_idx in range(args.num_outer_iters):
        t0 = time.time()
        print(f"[{outer_idx + 1}/{args.num_outer_iters}] reconstructing patches...")
        recon = reconstruct_volume_patches(
            initial_volume=recon,
            condition_volume=condition,
            model=model,
            normalize_fn=normalize_fn,
            denormalize_fn=denormalize_fn,
            angular_range_deg=angular_range_deg,
            start_angle_deg=start_angle_deg,
            tilt_axis=tilt_axis,
            patch_size=args.patch_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            start_step_frac=args.start_step_frac,
            device=device,
        )
        print(f"  completed in {time.time() - t0:.1f}s")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, recon.numpy().astype(np.float32))
    print(f"Saved reconstruction to {output_path}")


if __name__ == "__main__":
    main()
