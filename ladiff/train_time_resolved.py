"""Train a DDPM diffusion model on 2D slices from time-resolved 4D TIFF volumes.

Dataset: TifVolumeSliceDataset (ladiff.datasets)
Each item is a single 2D slice (H, W) float32 tensor normalised to [0, 1].
The UNet operates with in_channels=out_channels=1.

Example
-------
python train_time_resolved.py \\
    --data_path /myhome/data/sdate/shared/time_resolved/212_Wunderkerze2/timesteps/212_Wunderkerze2_rotate_04001.tif \\
    --file_range 10 \\
    --image_size 528 \\
    --epochs 30 \\
    --batch_size 8 \\
    --exp_name ladiff_time_resolved
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
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.models import UNet2DModel

from pytorch_base.experiment import PyTorchExperiment
from pytorch_base.base_loss import BaseLoss

from ladiff.datasets import TifVolumeSliceDataset


def build_datasets(args) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, float, float]:
    """Build train/test datasets from TifVolumeSliceDataset.

    A single dataset is constructed (optionally with auto normalisation) and
    split 90/10 into train/test.

    Returns
    -------
    train_ds, test_ds, norm_min, norm_max
    """
    normalize_range = None
    if args.norm_min is not None and args.norm_max is not None:
        normalize_range = (args.norm_min, args.norm_max)

    resize = args.image_size if args.image_size > 0 else None

    ds = TifVolumeSliceDataset(
        data_path=Path(args.data_path),
        file_range=args.file_range if args.file_range > 0 else None,
        resize=resize,
        normalize_range=normalize_range,
        augment=True
    )

    norm_min = ds.norm_min
    norm_max = ds.norm_max

    train_size = int(0.9 * len(ds))
    test_size = len(ds) - train_size
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_ds, _ = torch.utils.data.random_split(ds, [1.0, 0.0])
    _, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])
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


class TimeResolvedDiffusionLoss(BaseLoss):
    """DDPM noise-prediction loss on 2D slices from TifVolumeSliceDataset.

    Each item from the dataloader is a (H, W) float32 tensor.
    A channel dimension is added to produce (B, 1, H, W) for the UNet.
    """

    def __init__(self, noise_scheduler: DDPMScheduler, device: torch.device):
        super().__init__(["loss"])
        self.mse = nn.MSELoss()
        self.noise_scheduler = noise_scheduler
        self.device = device

    def compute_loss(self, instance, model: UNet2DModel):
        x_0 = instance.float().to(self.device)  # (B, H, W)
        x_0 = x_0.unsqueeze(1)                 # (B, 1, H, W)

        noise = torch.randn_like(x_0)
        bsz = x_0.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=self.device,
        ).long()
        x_t = self.noise_scheduler.add_noise(x_0, noise, timesteps)

        model.zero_grad()
        noise_pred = model(x_t, timestep=timesteps, return_dict=False)[0]
        loss = self.mse(noise_pred, noise)
        return loss, {"loss": loss.item()}


def main():
    parser = ArgumentParser(
        description="Train a DDPM diffusion model on 2D slices from time-resolved 4D TIFF volumes"
    )

    # Dataset / paths
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the starting .tif file')
    parser.add_argument('--file_range', type=int, default=0,
                        help='Number of consecutive tif files to load (0 = single file only)')
    parser.add_argument('--norm_min', type=float, default=None,
                        help='Manual normalisation minimum (default: auto from data)')
    parser.add_argument('--norm_max', type=float, default=None,
                        help='Manual normalisation maximum (default: auto from data)')

    # Training hyperparams
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--scheduler', type=str, default='[500]')
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='ladiff_time_resolved')
    parser.add_argument('--wandb', action='store_true')

    # Model
    parser.add_argument('--tiny', action='store_true',
                        help='Use a smaller channel configuration')
    parser.add_argument('--image_size', type=int, default=-1,
                        help='Square image size expected by the UNet (use --resize to downsample)')
    parser.add_argument('--load_checkpoint', type=str, default='',
                        help='Optional checkpoint path to resume training')

    args = parser.parse_args()

    milestones = args.scheduler.replace('[', '').replace(']', '').replace(' ', '')
    milestones = [int(x) for x in milestones.split(',') if x]

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_ds, test_ds, norm_min, norm_max = build_datasets(args)

    print(f"Train size : {len(train_ds)}")
    print(f"Test  size : {len(test_ds)}")
    print(f"Norm range : [{norm_min:.4g}, {norm_max:.4g}]")

    model = create_model(args)
    os.makedirs('checkpoints', exist_ok=True)
    if args.load_checkpoint:
        try:
            checkpoint_path = args.load_checkpoint
            ckpt = torch.load(args.load_checkpoint, map_location='cpu')
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"Model loaded from {args.load_checkpoint}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Training from scratch.")
    else:
        checkpoint_path = f"checkpoints/ddpm_{args.exp_name}{'_tiny' if args.tiny else ''}.pt"

    # Save normalisation config next to the checkpoint for inference
    norm_config_path = checkpoint_path.replace('.pt', '_norm.json')
    with open(norm_config_path, 'w') as f:
        json.dump({"norm_min": norm_min, "norm_max": norm_max}, f, indent=2)
    print(f"Normalisation config saved to {norm_config_path}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    loss_fn = TimeResolvedDiffusionLoss(noise_scheduler, device)

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

    exp.train(args.epochs, optimizer, milestones=milestones, gamma=args.lr_decay, scheduler=lr_scheduler)


if __name__ == '__main__':
    main()
