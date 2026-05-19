#!/usr/bin/env python3
"""Train a 2D conditional DDPM on axial slices from 3D missing-wedge volumes.

python /myhome/sdate/isodiffusion/train_conditional_2d.py --data_path /myhome/data/sdate/shared/compression_paper/file_1_extracted/reconstruction/la_fourier_1.npy --cone_width_deg 72 --patch_size 222 --volume_size 128 --batch_size 10 --epochs 300 --exp_name f1_2d --load_checkpoint=/myhome/iso_diffusion/checkpoints/ddpm_isodiffusion_f1_2d.pt --wandb

This script uses the same ``UNet2DModel`` architecture as
``ladiff/train_conditional_large_ladiff.py`` while loading 3D training pairs
from ``MissingConeVolumes``.  Each dataloader item is one or more complete
volumes, and ``--batch_size`` controls how many axial slices are sampled inside
``compute_loss`` for the 2D diffusion training batch.
"""

from __future__ import annotations

import json
import os
import random
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import lovely_tensors as lt
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from diffusers import DDPMScheduler
from diffusers.models import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup

lt.monkey_patch()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
for _candidate in (Path("/myhome/BaseTraining"), Path("/myhome/sdsc"), Path("/myhome/chip-project")):
    if _candidate.exists() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))

from pytorch_base.base_loss import BaseLoss
from pytorch_base.experiment import PyTorchExperiment

from isodiffusion.datasets import MissingConeVolumes


def build_datasets(args) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, float, float]:
    normalize_range = None
    if args.norm_min is not None and args.norm_max is not None:
        normalize_range = (args.norm_min, args.norm_max)

    ds_train = MissingConeVolumes(
        data_path=Path(args.data_path),
        cone_width_deg=args.cone_width_deg,
        patch_size=args.patch_size,
        target_size=args.volume_size,
        normalize_range=normalize_range,
        samples_per_volume=args.samples_per_volume,
        carve_center_angle_deg=args.carve_center_angle_deg,
        tilt_axis=args.tilt_axis,
        rotate=not args.no_rotate,
    )
    norm_min = ds_train.norm_min
    norm_max = ds_train.norm_max

    ds_test = MissingConeVolumes(
        data_path=Path(args.data_path),
        cone_width_deg=args.cone_width_deg,
        patch_size=args.patch_size,
        target_size=args.volume_size,
        normalize_range=(norm_min, norm_max),
        samples_per_volume=args.samples_per_volume,
        carve_center_angle_deg=args.carve_center_angle_deg,
        tilt_axis=args.tilt_axis,
        rotate=not args.no_rotate,
    )

    n = len(ds_train)
    test_size = max(1, int(args.test_fraction * n)) if n > 1 else 0
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_indices = indices[:-test_size] if test_size else indices
    test_indices = indices[-test_size:] if test_size else indices

    train_ds = torch.utils.data.Subset(ds_train, train_indices)
    test_ds = torch.utils.data.Subset(ds_test, test_indices)

    print(f"Dataset: {n} total 3D patches -> train={len(train_ds)}, test={len(test_ds)}")
    print(f"norm_min={norm_min:.4f}, norm_max={norm_max:.4f}")
    print(f"cone_width_deg={args.cone_width_deg:.1f}, target_size={args.volume_size}")
    return train_ds, test_ds, norm_min, norm_max


def _add_bool_arg(
    parser: ArgumentParser,
    name: str,
    default: bool,
    help_text: str,
    disable_help: str,
) -> None:
    dashed_name = name.replace("_", "-")
    enabled_flags = [f"--{name}"]
    disabled_flags = [f"--no_{name}", f"--no-{dashed_name}"]
    if dashed_name != name:
        enabled_flags.append(f"--{dashed_name}")

    group = parser.add_mutually_exclusive_group()
    group.add_argument(*enabled_flags, dest=name, action="store_true", help=help_text)
    group.add_argument(*disabled_flags, dest=name, action="store_false", help=disable_help)
    parser.set_defaults(**{name: default})


def create_model(args) -> UNet2DModel:
    """Create the same conditional 2D UNet used by large LADiff training."""
    channels = (64, 64, 128, 128, 256, 256)
    return UNet2DModel(
        sample_size=args.volume_size,
        in_channels=2,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=channels,
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        class_embed_type="timestep",
    )


def enable_optional_attention_acceleration(model: nn.Module, enabled: bool) -> None:
    if not enabled:
        return
    try:
        model.enable_xformers_memory_efficient_attention()
        print("xFormers memory-efficient attention enabled.")
    except Exception as exc:
        print(f"xFormers was requested but could not be enabled: {exc}")


class ConditionalIsoDiffusion2DLoss(BaseLoss):
    """Noise-prediction loss for axial slices sampled from 3D volume pairs."""

    def __init__(
        self,
        noise_scheduler: DDPMScheduler,
        device: torch.device,
        slice_batch_size: int,
        conditioning_probability: float = 0.5,
        loss_type: str = "huber",
    ) -> None:
        super().__init__(["loss"])
        if slice_batch_size < 1:
            raise ValueError("slice_batch_size must be >= 1")
        if not 0.0 <= conditioning_probability <= 1.0:
            raise ValueError("conditioning_probability must be between 0 and 1")

        self.noise_scheduler = noise_scheduler
        self.device = device
        self.slice_batch_size = int(slice_batch_size)
        self.conditioning_probability = float(conditioning_probability)

        loss_type = loss_type.lower()
        if loss_type == "mae":
            self.loss = nn.L1Loss()
        elif loss_type == "mse":
            self.loss = nn.MSELoss()
        elif loss_type == "huber":
            self.loss = nn.HuberLoss()
        else:
            raise ValueError("loss_type must be one of: mae, mse, huber")

    def _sample_axial_slices(
        self,
        carved_x: torch.Tensor,
        x_0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x_0.dim() == 3:
            x_0 = x_0.unsqueeze(0)
        if carved_x.dim() == 3:
            carved_x = carved_x.unsqueeze(0)
        if x_0.dim() == 5 and x_0.shape[1] == 1:
            x_0 = x_0.squeeze(1)
        if carved_x.dim() == 5 and carved_x.shape[1] == 1:
            carved_x = carved_x.squeeze(1)
        if x_0.dim() != 4 or carved_x.dim() != 4:
            raise ValueError(
                "Expected volume tensors shaped (V, D, H, W) or (D, H, W), "
                f"got x_0={tuple(x_0.shape)}, carved_x={tuple(carved_x.shape)}"
            )
        if x_0.shape != carved_x.shape:
            raise ValueError(
                f"x_0 and carved_x must have the same shape, got {tuple(x_0.shape)} "
                f"and {tuple(carved_x.shape)}"
            )

        volume_bsz, depth, height, width = x_0.shape
        if self.slice_batch_size <= depth:
            slice_indices = torch.stack(
                [
                    torch.randperm(depth, device=x_0.device)[: self.slice_batch_size]
                    for _ in range(volume_bsz)
                ],
                dim=0,
            )
        else:
            slice_indices = torch.randint(
                0,
                depth,
                (volume_bsz, self.slice_batch_size),
                device=x_0.device,
            )

        volume_indices = torch.arange(volume_bsz, device=x_0.device).unsqueeze(1)
        x_slices = x_0[volume_indices, slice_indices].reshape(-1, height, width)
        carved_slices = carved_x[volume_indices, slice_indices].reshape(-1, height, width)
        return carved_slices.contiguous(), x_slices.contiguous()

    def compute_loss(self, instance, model: UNet2DModel):
        carved_x, x_0 = instance
        carved_x, x_0 = self._sample_axial_slices(carved_x, x_0)

        x_0 = x_0.float().unsqueeze(1).to(self.device, non_blocking=True)
        carved_x = carved_x.float().unsqueeze(1).to(self.device, non_blocking=True)

        noise = torch.randn_like(x_0)
        bsz = x_0.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=self.device,
        ).long()
        x_t = self.noise_scheduler.add_noise(x_0, noise, timesteps)

        use_conditioning = (
            torch.rand((bsz,), device=self.device) < self.conditioning_probability
        ).long()
        carved_x = carved_x * use_conditioning.view(bsz, 1, 1, 1).to(carved_x.dtype)
        model_input = torch.cat([x_t, carved_x], dim=1)

        noise_pred = model(
            model_input,
            timestep=timesteps,
            class_labels=use_conditioning,
            return_dict=False,
        )[0]

        loss = self.loss(noise_pred, noise)
        return loss, {"loss": loss.item()}


def parse_args():
    parser = ArgumentParser(description="Train 2D conditional iso-diffusion on axial volume slices.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--cone_width_deg", type=float, required=True)
    parser.add_argument("--norm_min", type=float, default=None)
    parser.add_argument("--norm_max", type=float, default=None)
    parser.add_argument("--patch_size", type=int, default=112)
    parser.add_argument("--volume_size", type=int, default=64)
    parser.add_argument("--samples_per_volume", type=int, default=None)
    parser.add_argument("--carve_center_angle_deg", type=float, default=0.0)
    parser.add_argument("--tilt_axis", type=int, default=0)
    parser.add_argument("--no_rotate", action="store_true")
    parser.add_argument("--test_fraction", type=float, default=0.1)

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of axial slices sampled from each loaded volume for each 2D training batch.",
    )
    parser.add_argument(
        "--volume_batch_size",
        type=int,
        default=1,
        help="Number of full MissingConeVolumes items loaded by the dataloader per step.",
    )
    parser.add_argument(
        "--conditioning_probability",
        type=float,
        default=0.5,
        help="Probability of keeping the carved conditioning channel for each sampled slice.",
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--scheduler", type=str, default="[500]")
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="isodiffusion2d")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--loss_type", choices=["mae", "mse", "huber"], default="huber")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument(
        "--mixed_precision",
        choices=["no", "fp16", "bf16", "auto"],
        default="fp16",
        help="Use AMP for training. Use 'no' to restore full-float32 training.",
    )
    _add_bool_arg(
        parser,
        "allow_tf32",
        True,
        "Allow TF32 Tensor Core kernels for remaining float32 matmul/convolution ops.",
        "Disable TF32 Tensor Core kernels.",
    )
    _add_bool_arg(
        parser,
        "cudnn_benchmark",
        True,
        "Let cuDNN autotune kernels for the fixed 2D slice shape.",
        "Disable cuDNN autotuning.",
    )
    parser.add_argument(
        "--compile_model",
        action="store_true",
        help="Wrap the model with torch.compile. Useful to try after AMP is stable.",
    )
    parser.add_argument(
        "--enable_xformers",
        action="store_true",
        help="Try to enable xFormers memory-efficient attention kernels.",
    )
    _add_bool_arg(
        parser,
        "pin_memory",
        torch.cuda.is_available(),
        "Pin DataLoader batches so CUDA transfers can be non-blocking.",
        "Disable pinned DataLoader memory.",
    )
    _add_bool_arg(
        parser,
        "persistent_workers",
        True,
        "Keep DataLoader workers alive between epochs when num_workers > 0.",
        "Restart DataLoader workers every epoch.",
    )
    parser.add_argument(
        "--prefetch_factor",
        type=int,
        default=2,
        help="Number of batches each DataLoader worker prefetches.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    milestones = [int(x) for x in args.scheduler.replace("[", "").replace("]", "").split(",") if x.strip()]

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_ds, test_ds, norm_min, norm_max = build_datasets(args)

    os.makedirs("samples", exist_ok=True)
    carved_sample, x_sample = train_ds[0]
    mid = x_sample.shape[0] // 2
    TF.to_pil_image(x_sample[mid].clamp(0, 1)).save(f"samples/sample_x_{args.exp_name}.png")
    TF.to_pil_image(carved_sample[mid].clamp(0, 1)).save(
        f"samples/sample_carved_x_{args.exp_name}.png"
    )
    print(
        f"Sample axial images saved to samples/sample_x_{args.exp_name}.png "
        f"and samples/sample_carved_x_{args.exp_name}.png"
    )

    model = create_model(args)
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/ddpm_isodiffusion2d_{args.exp_name}.pt"
    norm_config_path = checkpoint_path.replace(".pt", "_norm.json")
    with open(norm_config_path, "w") as f:
        json.dump(
            {
                "norm_min": norm_min,
                "norm_max": norm_max,
                "cone_width_deg": args.cone_width_deg,
                "patch_size": args.patch_size,
                "volume_size": args.volume_size,
                "carve_center_angle_deg": args.carve_center_angle_deg,
                "tilt_axis": args.tilt_axis,
                "batch_size": args.batch_size,
                "volume_batch_size": args.volume_batch_size,
                "conditioning_probability": args.conditioning_probability,
                "architecture": "ladiff.train_conditional_large_ladiff.UNet2DModel",
                "mixed_precision": args.mixed_precision,
                "allow_tf32": args.allow_tf32,
                "cudnn_benchmark": args.cudnn_benchmark,
                "compile_model": args.compile_model,
                "enable_xformers": args.enable_xformers,
            },
            f,
            indent=2,
        )
    print(f"Normalisation + model config saved to {norm_config_path}")

    if args.load_checkpoint:
        try:
            ckpt = torch.load(args.load_checkpoint, map_location=torch.device("cpu"))
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Model loaded from checkpoint {args.load_checkpoint}")
        except Exception as exc:
            print(f"Could not load checkpoint {args.load_checkpoint}: {exc}. Training from scratch.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    enable_optional_attention_acceleration(model, args.enable_xformers)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    loss_fn = ConditionalIsoDiffusion2DLoss(
        noise_scheduler,
        device,
        slice_batch_size=args.batch_size,
        conditioning_probability=args.conditioning_probability,
        loss_type=args.loss_type,
    )

    exp = PyTorchExperiment(
        args=vars(args),
        train_dataset=train_ds,
        test_dataset=test_ds,
        batch_size=args.volume_batch_size,
        model=model,
        loss_fn=loss_fn,
        checkpoint_path=checkpoint_path,
        experiment_name=args.exp_name,
        with_wandb=args.wandb,
        num_workers=args.num_workers,
        seed=args.seed,
        save_always=True,
        mixed_precision=args.mixed_precision,
        enable_tf32=args.allow_tf32,
        cudnn_benchmark=args.cudnn_benchmark,
        compile_model=args.compile_model,
        pin_memory=args.pin_memory,
        persistent_workers=args.persistent_workers,
        prefetch_factor=args.prefetch_factor,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    total_steps = len(exp.train_loader) * args.epochs
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )

    exp.train(
        args.epochs,
        optimizer,
        milestones=milestones,
        gamma=args.lr_decay,
        scheduler=lr_scheduler,
    )


if __name__ == "__main__":
    main()
