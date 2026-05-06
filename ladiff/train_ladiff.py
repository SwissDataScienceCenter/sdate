import os
import sys
import json
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple

# Ensure the sdate project root is importable regardless of how the script is invoked
_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../sdate/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup

from pytorch_base.experiment import PyTorchExperiment
from pytorch_base.base_loss import BaseLoss

from diffusers.models import UNet2DModel

from ladiff import BaseLimitedAngleReconstructions


def build_datasets(args) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, float]:
    """Build train/test datasets from BaseLimitedAngleReconstructions.

    A single dataset is constructed and split 90/10 into train/test.
    Each item is a (ny, nx) FBP reconstruction tensor, already normalized
    by the dataset's calibrated norm_scale.
    Note: num_workers=0 is required by the ASTRA-based FBP backend.

    Returns
    -------
    train_ds, test_ds, norm_scale
    """
    ds = BaseLimitedAngleReconstructions(
        data_path=args.data_path,
        num_projections=args.num_projections,
        target_size=(args.target_height, args.target_width),
        n_slices=args.n_slices,
        k_angles=args.k_angles,
        angular_range_deg=(args.angle_start, args.angle_end),
        height_skip=args.height_skip,
        det_spacing_mm=args.det_spacing_mm,
        filter_type=args.filter_type,
        gap=args.gap,
        num_frames=args.num_frames,
    )

    norm_scale = ds.norm_scale

    train_size = int(0.9 * len(ds))
    test_size = len(ds) - train_size
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_ds, test_ds = torch.utils.data.random_split(ds, [train_size, test_size])
    return train_ds, test_ds, norm_scale


def create_model(args) -> nn.Module:
    channels = (32, 32, 32, 32, 64, 64) if args.tiny else (64, 64, 128, 128, 256, 256)
    model = UNet2DModel(
        sample_size=args.image_size,
        in_channels=args.num_frames,
        out_channels=args.num_frames,
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


class LaDiffLoss(BaseLoss):
    """DDPM-style noise prediction loss on limited-angle FBP reconstructions.

    Each item from the dataloader is a batch of FBP reconstructions used as
    x_0 for diffusion training.  When ``num_frames=1`` the batch has shape
    ``(B, ny, nx)`` and a channel dim is added.  When ``num_frames > 1`` the
    frames are treated as channels: shape ``(B, num_frames, ny, nx)``.
    """

    def __init__(self, noise_scheduler: DDPMScheduler, device: torch.device):
        super().__init__(["loss"])
        self.mse = nn.MSELoss()
        self.noise_scheduler = noise_scheduler
        self.device = device

    def compute_loss(self, instance, model: UNet2DModel):
        x_0 = instance
        if x_0.dim() == 3:
            x_0 = x_0.unsqueeze(1)  # (B, ny, nx) -> (B, 1, ny, nx)
        if x_0.dim() == 5:
            x_0 = x_0.squeeze(2)  # (B, num_frames, 1, ny, nx) -> (B, num_frames, ny, nx)
        # num_frames > 1: already (B, num_frames, ny, nx) = (B, C, H, W)

        x_0 = x_0.float().to(self.device)
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
    parser = ArgumentParser(description="Train ladiff diffusion model on limited-angle FBP reconstructions")

    # Dataset / paths
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the extracted tomography projection folder')
    parser.add_argument('--num_projections', type=int, default=1501,
                        help='Total number of projections in the dataset')
    parser.add_argument('--target_height', type=int, default=640,
                        help='Projection resize height')
    parser.add_argument('--target_width', type=int, default=540,
                        help='Projection resize width (= reconstruction size W; must match --image_size)')
    parser.add_argument('--n_slices', type=int, default=20,
                        help='Number of time slices to extract from the volume')
    parser.add_argument('--k_angles', type=int, default=400,
                        help='Contiguous projections assigned per time slice')
    parser.add_argument('--angle_start', type=float, default=0.0,
                        help='Start of the angular sweep in degrees')
    parser.add_argument('--angle_end', type=float, default=180.0,
                        help='End of the angular sweep in degrees')
    parser.add_argument('--height_skip', type=int, default=0,
                        help='Rows skipped between consecutive slices')
    parser.add_argument('--det_spacing_mm', type=float, default=1.0,
                        help='Detector pixel spacing in mm')
    parser.add_argument('--filter_type', type=str, default='hann',
                        help='Ramp-filter variant for FBP')
    parser.add_argument('--gap', type=int, default=1,
                        help='Stride between consecutive sliding-window start indices')
    parser.add_argument('--num_frames', type=int, default=5,
                        help='Number of consecutive temporal frames per sample (1=2D, >1=3D diffusion)')

    # Training hyperparams
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--scheduler', type=str, default='[500]')
    parser.add_argument('--lr_decay', type=float, default=0.1)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--exp_name', type=str, default='ladiff3d')
    parser.add_argument('--wandb', action='store_true')

    # Model
    parser.add_argument('--tiny', action='store_true',
                        help='Use a smaller channel configuration')
    parser.add_argument('--compression', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=540,
                        help='Square reconstruction size (should equal --target_width)')
    parser.add_argument('--cross_attention_dim', type=int, default=32,
                        help='(unused, kept for backwards compatibility)')
    parser.add_argument('--load_checkpoint', type=str, default='',
                        help='Optional checkpoint path to resume training')

    args = parser.parse_args()

    milestones = args.scheduler.replace('[', '').replace(']', '').replace(' ', '')
    milestones = [int(x) for x in milestones.split(',') if x]

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_ds, test_ds, norm_scale = build_datasets(args)

    model = create_model(args)
    os.makedirs('checkpoints', exist_ok=True)
    checkpoint_path = f"checkpoints/ddpm_{args.exp_name}{'_tiny' if args.tiny else ''}.pt"

    # Save normalization config next to the checkpoint for inference use
    norm_config_path = checkpoint_path.replace('.pt', '_norm.json')
    with open(norm_config_path, 'w') as f:
        json.dump({"norm_scale": norm_scale}, f, indent=2)
    print(f"Normalization scale = {norm_scale:.4f} saved to {norm_config_path}")

    if args.load_checkpoint:
        checkpoint_path = args.load_checkpoint
        try:
            ckpt = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            model.load_state_dict(ckpt['model_state_dict'])
            print(f"Model loaded from checkpoint {checkpoint_path}")
        except Exception as e:
            print(f"Could not load checkpoint {checkpoint_path}: {e}. Training from scratch.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    loss_fn = LaDiffLoss(noise_scheduler, device)

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
        num_workers=0,  # ASTRA FBP must run in the main process
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
