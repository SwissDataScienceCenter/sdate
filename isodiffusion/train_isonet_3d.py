#!/usr/bin/env python3
"""Train a 3D IsoNet for missing-wedge recovery via direct regression.

The model takes a single-channel normalized carved volume patch as input and
directly predicts the normalized full volume patch. Loss: MSE(normalize(x), model(normalize(carved_x))).

Unlike the diffusion counterpart, there is no noise schedule — the UNet3DConditionModel
is reused with in_channels=1, timestep=0, and zero encoder_hidden_states.
"""

from __future__ import annotations

import json
import os
import random
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from diffusers.models import UNet3DConditionModel
from diffusers.optimization import get_cosine_schedule_with_warmup
import lovely_tensors as lt
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


def _parse_channels(value: str) -> Tuple[int, ...]:
    channels = tuple(int(v.strip()) for v in value.split(",") if v.strip())
    if not channels:
        raise ValueError("--channels must contain at least one integer")
    return channels


def _parse_volume_size(value: str):
    parts = [int(v.strip()) for v in value.split(",") if v.strip()]
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 3:
        return tuple(parts)
    raise ValueError("--volume_size must be a single int (e.g. 96) or D,H,W (e.g. 15,96,96)")


def _add_bool_arg(parser, name, default, help_text, disable_help):
    dashed_name = name.replace("_", "-")
    enabled_flags = [f"--{name}"]
    disabled_flags = [f"--no_{name}", f"--no-{dashed_name}"]
    if dashed_name != name:
        enabled_flags.append(f"--{dashed_name}")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(*enabled_flags, dest=name, action="store_true", help=help_text)
    group.add_argument(*disabled_flags, dest=name, action="store_false", help=disable_help)
    parser.set_defaults(**{name: default})


def create_model(args):
    if args.model_type == "dynunet":
        from isodiffusion.dynunet_wrapper import DynUNetIsoNet, parse_dynunet_filters
        filters = parse_dynunet_filters(args.dynunet_filters)
        return DynUNetIsoNet(in_channels=1, out_channels=1, filters=filters)

    return UNet3DConditionModel(
        sample_size=list(args.volume_size)[-2:] if isinstance(args.volume_size, tuple) else args.volume_size,
        in_channels=1,
        out_channels=1,
        down_block_types=("DownBlock3D", "DownBlock3D", "CrossAttnDownBlock3D"),
        up_block_types=("CrossAttnUpBlock3D", "UpBlock3D", "UpBlock3D"),
        block_out_channels=args.channels,
        layers_per_block=args.layers_per_block,
        cross_attention_dim=args.cross_attention_dim,
        attention_head_dim=args.attention_head_dim,
        norm_num_groups=args.norm_num_groups,
    )


class IsoNetLoss(BaseLoss):
    """Direct regression loss: MSE(normalize(x), model(normalize(carved_x)))."""

    def __init__(self, device: torch.device, cross_attention_dim: int, loss_type: str = "huber") -> None:
        super().__init__(["loss"])
        self.device = device
        self.cross_attention_dim = int(cross_attention_dim)
        loss_type = loss_type.lower()
        if loss_type == "mae":
            self.loss_fn = nn.L1Loss()
        elif loss_type == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_type == "huber":
            self.loss_fn = nn.HuberLoss()
        else:
            raise ValueError("loss_type must be one of: mae, mse, huber")

    def compute_loss(self, instance, model: UNet3DConditionModel):
        carved_x, x = instance

        if x.dim() == 4:
            x = x.unsqueeze(1)
        if carved_x.dim() == 4:
            carved_x = carved_x.unsqueeze(1)

        x = x.float().to(self.device, non_blocking=True)
        carved_x = carved_x.float().to(self.device, non_blocking=True)

        bsz = x.shape[0]
        timesteps = torch.zeros(bsz, device=self.device, dtype=torch.long)
        encoder_hidden_states = torch.zeros(
            bsz, 1, self.cross_attention_dim, device=self.device, dtype=carved_x.dtype
        )

        pred = model(
            carved_x,
            timestep=timesteps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]

        loss = self.loss_fn(pred, x)
        return loss, {"loss": loss.detach()}


def parse_args():
    parser = ArgumentParser(description="Train 3D IsoNet (direct regression) for missing-wedge recovery.")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--cone_width_deg", type=float, required=True)
    parser.add_argument("--norm_min", type=float, default=None)
    parser.add_argument("--norm_max", type=float, default=None)
    parser.add_argument("--patch_size", type=int, default=112)
    parser.add_argument("--volume_size", type=_parse_volume_size, default=64)
    parser.add_argument("--samples_per_volume", type=int, default=None)
    parser.add_argument("--carve_center_angle_deg", type=float, default=0.0)
    parser.add_argument("--tilt_axis", type=int, default=0)
    parser.add_argument("--no_rotate", action="store_true")
    parser.add_argument("--test_fraction", type=float, default=0.1)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--scheduler", type=str, default="[50000000]")
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="isonet3d")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--loss_type", choices=["mae", "mse", "huber"], default="mae")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--model_type", choices=["unet3d", "dynunet"], default="unet3d",
                        help="Network architecture: unet3d (diffusers) or dynunet (MONAI).")
    parser.add_argument("--channels", type=_parse_channels, default=(32, 64, 128))
    parser.add_argument("--layers_per_block", type=int, default=1)
    parser.add_argument("--norm_num_groups", type=int, default=16)
    parser.add_argument("--cross_attention_dim", type=int, default=128)
    parser.add_argument("--attention_head_dim", type=int, default=8)
    parser.add_argument("--dynunet_filters", type=str, default="32,64,128,256,320",
                        help="Comma-separated filter counts per DynUNet level (used when --model_type dynunet).")
    parser.add_argument("--load_checkpoint", type=str, default="")
    parser.add_argument("--save_checkpoint", type=str, default="", help="Absolute path to save the checkpoint.")
    parser.add_argument("--mixed_precision", choices=["no", "fp16", "bf16", "auto"], default="fp16")
    _add_bool_arg(parser, "allow_tf32", True, "Allow TF32 Tensor Core kernels.", "Disable TF32.")
    _add_bool_arg(parser, "cudnn_benchmark", True, "Let cuDNN autotune kernels.", "Disable cuDNN autotuning.")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--enable_xformers", action="store_true")
    _add_bool_arg(parser, "pin_memory", torch.cuda.is_available(), "Pin DataLoader batches.", "Disable pinned memory.")
    _add_bool_arg(parser, "persistent_workers", True, "Keep DataLoader workers alive.", "Restart workers every epoch.")
    parser.add_argument("--prefetch_factor", type=int, default=2)
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
    TF.to_pil_image(carved_sample[mid].clamp(0, 1)).save(f"samples/sample_carved_x_{args.exp_name}.png")

    if args.save_checkpoint:
        checkpoint_path = args.save_checkpoint
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    else:
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"checkpoints/isonet_{args.exp_name}.pt"

    model = create_model(args)

    norm_config_path = checkpoint_path.replace(".pt", "_norm.json")
    with open(norm_config_path, "w") as f:
        from isodiffusion.dynunet_wrapper import parse_dynunet_filters
        json.dump(
            {
                "model_type": "isonet",
                "arch": args.model_type,
                "in_channels": 1,
                "norm_min": norm_min,
                "norm_max": norm_max,
                "cone_width_deg": args.cone_width_deg,
                "patch_size": args.patch_size,
                "volume_size": list(args.volume_size) if isinstance(args.volume_size, tuple) else args.volume_size,
                "carve_center_angle_deg": args.carve_center_angle_deg,
                "tilt_axis": args.tilt_axis,
                # unet3d params
                "channels": list(args.channels),
                "layers_per_block": args.layers_per_block,
                "norm_num_groups": args.norm_num_groups,
                "cross_attention_dim": model.config.cross_attention_dim,
                "attention_head_dim": args.attention_head_dim,
                # dynunet params
                "dynunet_filters": parse_dynunet_filters(args.dynunet_filters) if args.model_type == "dynunet" else None,
            },
            f,
            indent=2,
        )
    print(f"Model config saved to {norm_config_path}")

    if args.load_checkpoint:
        try:
            ckpt = torch.load(args.load_checkpoint, map_location=torch.device("cpu"))
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded checkpoint from {args.load_checkpoint}")
        except Exception as exc:
            print(f"Could not load checkpoint {args.load_checkpoint}: {exc}. Training from scratch.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model.to(device)

    if args.enable_xformers:
        try:
            model.enable_xformers_memory_efficient_attention()
            print("xFormers enabled.")
        except Exception as exc:
            print(f"xFormers could not be enabled: {exc}")

    loss_fn = IsoNetLoss(device, cross_attention_dim=model.config.cross_attention_dim, loss_type=args.loss_type)

    exp = PyTorchExperiment(
        args=vars(args),
        train_dataset=train_ds,
        test_dataset=test_ds,
        batch_size=args.batch_size,
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

    exp.train(args.epochs, optimizer, milestones=milestones, gamma=args.lr_decay, scheduler=lr_scheduler)


if __name__ == "__main__":
    main()
