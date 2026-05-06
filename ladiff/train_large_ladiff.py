"""Train a DDPM diffusion model on 2D slices from pre-computed la_fbp .npy volumes.

Dataset: NpyVolumeSliceDataset (ladiff.datasets)
Each item is a single 2D slice (H, W) float32 tensor normalised to [0, 1].
The UNet operates with in_channels=out_channels=1.

Example
-------
Single file::

python /myhome/sdate/ladiff/train_large_ladiff.py --data_path /myhome/data/sdate/shared/compression_paper/file_1_extracted/reconstruction/training --image_size 640 --epochs 40 --batch_size 10 --exp_name large_ladiff --load_checkpoint=/myhome/sdate/checkpoints/ddpm_large_ladiff_file_1.pt --wandb

Directory with multiple .npy files::

    python train_large_ladiff.py \\
        --data_path /myhome/data/sdate/shared/reconstruction/ \\
        --image_size 640 \\
        --epochs 30 \\
        --batch_size 10 \\
        --exp_name large_ladiff_multi \\
        --load_checkpoint=/myhome/sdate/checkpoints/ddpm_large_ladiff_file_10.pt \\
        --wandb
"""

import os
import sys
import json
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../sdate/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.models import UNet2DModel

from pytorch_base.experiment import PyTorchExperiment
from pytorch_base.base_loss import BaseLoss

from ladiff.datasets import NpyVolumeSliceDataset


def build_datasets(
    args,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, float, float]:
    """Build train/test datasets from NpyVolumeSliceDataset.

    Normalisation is resolved in priority order:
    1. ``--norm_min`` / ``--norm_max`` CLI arguments (explicit override).
    2. ``<stem>_norm.json`` sidecar next to the ``.npy`` file (or ``norm.json``
       inside the directory), saved by the Large_LA_Dataset notebook.
    3. 1st–99th percentile computed from all loaded slices (automatic fallback).

    Returns
    -------
    train_ds, test_ds, norm_min, norm_max
    """
    normalize_range = None
    if args.norm_min is not None and args.norm_max is not None:
        normalize_range = (args.norm_min, args.norm_max)

    # Full dataset with augmentation for statistics + train split.
    ds_train = NpyVolumeSliceDataset(
        data_path=Path(args.data_path),
        normalize_range=normalize_range,
        augment=not args.no_augment,
        scale_range=(1.0 - args.scale_jitter, 1.0 + args.scale_jitter),
        rotation_deg=args.rotation_deg,
        shift_fraction=args.shift_fraction,
    )
    norm_min = ds_train.norm_min
    norm_max = ds_train.norm_max

    # Test split uses the same normalisation but no augmentation.
    ds_test = NpyVolumeSliceDataset(
        data_path=Path(args.data_path),
        normalize_range=(norm_min, norm_max),
        augment=False,
    )

    n = len(ds_train)
    train_size = int(1.0 * n)
    test_size = int(0.1 * n)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_indices = indices[:train_size]
    test_indices = indices[-test_size:]

    train_ds = torch.utils.data.Subset(ds_train, train_indices)
    test_ds = torch.utils.data.Subset(ds_test, test_indices)

    print(
        f"Dataset: {n} total slices  →  "
        f"train={len(train_ds)}, test={len(test_ds)}"
    )
    print(f"norm_min={norm_min:.4f}, norm_max={norm_max:.4f}")
    return train_ds, test_ds, norm_min, norm_max


def create_model(args) -> nn.Module:
    channels = (32, 32, 32, 32, 64, 64) if args.tiny else (64, 64, 128, 128, 256, 256)
    model = UNet2DModel(
        sample_size=args.image_size,
        in_channels=1,
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
    )
    return model


class LargeLA_DiffLoss(BaseLoss):
    """DDPM-style noise prediction loss on 2D slices from NpyVolumeSliceDataset.

    Each item from the dataloader is a batch of 2D slices of shape ``(B, H, W)``.
    A channel dimension is added to give ``(B, 1, H, W)`` before passing through
    the UNet.
    """

    def __init__(self, noise_scheduler: DDPMScheduler, device: torch.device, rotation_shift: float = 0.0):
        super().__init__(["loss"])
        self.mse = nn.MSELoss()
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.rotation_shift = rotation_shift

    def compute_loss(self, instance, model: UNet2DModel):
        x_0 = instance
        if x_0.dim() == 3:
            x_0 = x_0.unsqueeze(1)  # (B, H, W) -> (B, 1, H, W)

        x_0 = x_0.float().to(self.device)
        if self.rotation_shift != 0.0:
            x_0 = TF.rotate(x_0, angle=-self.rotation_shift, interpolation=InterpolationMode.BILINEAR)
        noise = torch.randn_like(x_0)
        bsz = x_0.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=self.device,
        ).long()
        x_t = self.noise_scheduler.add_noise(x_0, noise, timesteps)

        model.zero_grad()
        noise_pred = model(x_t, timestep=timesteps, return_dict=False)[0]
        loss = self.mse(noise_pred, noise)
        return loss, {"loss": loss.item()}


def main():
    parser = ArgumentParser(
        description="Train a DDPM diffusion model on 2D slices from la_fbp .npy volumes"
    )

    # Dataset / paths
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to a single la_fbp .npy file or a directory containing .npy files",
    )
    parser.add_argument(
        "--norm_min",
        type=float,
        default=None,
        help="Minimum value for normalisation (default: auto from data)",
    )
    parser.add_argument(
        "--norm_max",
        type=float,
        default=None,
        help="Maximum value for normalisation (default: auto from data)",
    )

    # Augmentation
    parser.add_argument(
        "--no_augment",
        action="store_true",
        help="Disable data augmentation",
    )
    parser.add_argument(
        "--scale_jitter",
        type=float,
        default=0.10,
        help="Scale jitter half-range (default 0.10 → scale in [0.9, 1.1])",
    )
    parser.add_argument(
        "--rotation_deg",
        type=float,
        default=5.0,
        help="Max rotation in degrees for augmentation (default 5.0)",
    )
    parser.add_argument(
        "--shift_fraction",
        type=float,
        default=0.05,
        help="Max shift as fraction of image size (default 0.05)",
    )

    # Training hyperparams
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--scheduler", type=str, default="[500]")
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="large_ladiff")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--rotation_shift",
        type=float,
        default=90.0,
        help="Rotate all batch images clockwise by this many degrees in compute_loss (default 90.0)",
    )

    # Model
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Use a smaller channel configuration",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=640,
        help="Expected spatial size of the reconstruction slices",
    )
    parser.add_argument(
        "--load_checkpoint",
        type=str,
        default="",
        help="Optional checkpoint path to resume training",
    )

    args = parser.parse_args()

    milestones = (
        args.scheduler.replace("[", "").replace("]", "").replace(" ", "")
    )
    milestones = [int(x) for x in milestones.split(",") if x]

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_ds, test_ds, norm_min, norm_max = build_datasets(args)

    # save the first image of the dataset for sanity checking and visualisation in TensorBoard/WandB
    sample_image = train_ds[0]  # Get the first slice from the first volume
    os.makedirs("samples", exist_ok=True)
    sample_image_path = f"samples/sample_image_{args.exp_name}.png"
    TF.to_pil_image(sample_image).save(sample_image_path)
    print(f"Sample image saved to {sample_image_path}")

    model = create_model(args)
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = (
        f"checkpoints/ddpm_{args.exp_name}{'_tiny' if args.tiny else ''}.pt"
    )

    # Save normalisation config next to the checkpoint for inference use.
    norm_config_path = checkpoint_path.replace(".pt", "_norm.json")
    with open(norm_config_path, "w") as f:
        json.dump({"norm_min": norm_min, "norm_max": norm_max}, f, indent=2)
    print(f"Normalisation saved to {norm_config_path}")

    if args.load_checkpoint:
        try:
            ckpt = torch.load(
                args.load_checkpoint, map_location=torch.device("cpu")
            )
            model.load_state_dict(ckpt["model_state_dict"])
            print(f"Model loaded from checkpoint {args.load_checkpoint}")
        except Exception as e:
            print(
                f"Could not load checkpoint {args.load_checkpoint}: {e}. "
                "Training from scratch."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    loss_fn = LargeLA_DiffLoss(noise_scheduler, device, rotation_shift=args.rotation_shift)

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
        num_workers=4,
        seed=args.seed,
        save_always=True,
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
