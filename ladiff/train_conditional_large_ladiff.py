"""Train a conditional DDPM diffusion model for missing-cone recovery in Fourier space.

Dataset: MissingConeDataset (ladiff.datasets)
Each item is a 5-tuple ``(carved_x, x, beta_int, alpha_int, slice_type)`` where:
  - ``x``        – target 2D slice (normalised, existing missing cone at random alpha)
  - ``carved_x`` – conditioning 2D slice (x with a second cone of cone_width_deg
                   removed at angle beta)
  - ``beta_int`` – integer angle [0, 179] of the carved cone.
  - ``alpha_int`` – integer angle encoding the relative offset between alpha and beta.
  - ``slice_type`` – string describing the source slice type that produced this sample.

The UNet has ``in_channels=2`` (``x_t`` concatenated with ``carved_x``) and
``out_channels=1``.  The class embedding type is "timestep" so the binary
conditioning flag is encoded with the same sinusoidal + MLP embedding as the
diffusion timestep.

Example
-------
Single .npy file::

    python /myhome/sdate/ladiff/train_conditional_large_ladiff.py --data_path /myhome/data/sdate/shared/compression_paper/file_10_extracted/reconstruction/la_fbp_1.npy --cone_width_deg 72 --image_size 640 --epochs 100 --batch_size 10 --exp_name cond_ladiff_f10 --wandb

    python /myhome/sdate/ladiff/train_conditional_large_ladiff.py --data_path /myhome/data/sdate/shared/em_data_npy/neuron/synapse-bin4-5i-demo.npy --cone_width_deg 60 --gamma -6.0 --image_size 256 --epochs 100 --batch_size 20 --exp_name cond_ladiff_neuron --wandb

Directory with multiple .npy files::

    python /myhome/sdate/ladiff/train_conditional_large_ladiff.py \\
        --data_path /myhome/data/sdate/shared/reconstruction/ \\
        --cone_width_deg 72 \\
        --image_size 640 \\
        --epochs 100 \\
        --batch_size 10 \\
        --exp_name cond_ladiff_multi \\
        --wandb

To include only axial and random slices::

    python /myhome/sdate/ladiff/train_conditional_large_ladiff.py \\
        --data_path /myhome/data/sdate/shared/reconstruction/ \\
        --cone_width_deg 72 \\
        --slice_types axis0,rand \\
        --image_size 640 \\
        --epochs 100 \\
        --batch_size 10 \\
        --exp_name cond_ladiff_axial_rand \\
        --wandb
"""

import os
import sys
import json
import random
from argparse import ArgumentParser
from pathlib import Path
from typing import Tuple
import lovely_tensors  as lt
lt.monkey_patch()  # for nicer tensor printing in debug logs

_PROJECT_ROOT = Path(__file__).resolve().parent.parent  # .../sdate/
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from diffusers import DDPMScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.models import UNet2DModel

from pytorch_base.experiment import PyTorchExperiment
from pytorch_base.base_loss import BaseLoss

from ladiff.datasets import MissingConeDataset


def build_datasets(
    args,
) -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset, float, float]:
    """Build train/test datasets from MissingConeDataset.

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

    slice_types = None
    if args.slice_types:
        slice_types = tuple(args.slice_types.split(','))

    ds_train = MissingConeDataset(
        data_path=Path(args.data_path),
        cone_width_deg=args.cone_width_deg,
        normalize_range=normalize_range,
        augment=not args.no_augment,
        scale_range=(1.0 - args.scale_jitter, 1.0 + args.scale_jitter),
        shift_fraction=args.shift_fraction,
        gamma=args.gamma,
        slice_types=slice_types,
    )
    norm_min = ds_train.norm_min
    norm_max = ds_train.norm_max

    # Test split: same normalisation, no augmentation.
    ds_test = MissingConeDataset(
        data_path=Path(args.data_path),
        cone_width_deg=args.cone_width_deg,
        normalize_range=(norm_min, norm_max),
        augment=False,
        gamma=args.gamma,
        slice_types=slice_types,
    )

    n = len(ds_train)
    test_size = max(1, int(0.1 * n))

    torch.manual_seed(args.seed)
    indices = torch.randperm(
        n, generator=torch.Generator().manual_seed(args.seed)
    ).tolist()

    test_indices = indices[-test_size:]

    train_ds = torch.utils.data.Subset(ds_train, indices)
    test_ds = torch.utils.data.Subset(ds_test, test_indices)

    print(
        f"Dataset: {n} total slices  →  "
        f"train={len(train_ds)}, test={len(test_ds)}"
    )
    print(f"norm_min={norm_min:.4f}, norm_max={norm_max:.4f}")
    print(f"cone_width_deg={args.cone_width_deg:.1f}")
    return train_ds, test_ds, norm_min, norm_max


def create_model(args) -> nn.Module:
    """Create a conditional UNet2DModel.

    - ``in_channels=2``: noisy target ``x_t`` concatenated with conditioner ``carved_x``.
    - ``out_channels=1``: predicted noise for the target channel only.
    - ``class_embed_type="timestep"``: the binary conditioning flag is embedded
      with the same sinusoidal + MLP as the diffusion timestep.
    """
    channels = (64, 64, 128, 128, 256, 256)
    model = UNet2DModel(
        sample_size=args.image_size,
        in_channels=2,   # x_t + carved_x
        out_channels=1,  # noise prediction for x only
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
        class_embed_type="timestep",  # conditioning flag encoded like a timestep
    )
    return model


class ConditionalLADiffLoss(BaseLoss):
    """DDPM-style noise prediction loss for the conditional missing-cone model.

    Each batch from the dataloader is a list ``[carved_x, x_0, beta, alpha, slice_type]`` produced
    by :class:`~ladiff.datasets.MissingConeDataset`:

    * ``carved_x`` – ``(B, H, W)`` float32 conditioner (cone removed at beta)
    * ``x_0``      – ``(B, H, W)`` float32 target (original rotated slice)
    * ``beta``     – ``(B,)`` long tensor of integer angles ``[0, 179]``
    * ``alpha``    – ``(B,)`` long tensor of integer relative offsets.
    * ``slice_type`` – metadata string describing the source slice type.

    The forward pass:
    1. Add noise to ``x_0`` to get ``x_t``.
    2. Concatenate ``x_t`` (``B, 1, H, W``) with ``carved_x`` (``B, 1, H, W``)
       along the channel dim → ``model_input`` (``B, 2, H, W``).
    3. Pass ``model_input``, ``timesteps``, and conditioning coin flips as
       ``class_labels`` to the UNet.
    4. Compute MAE or MSE between predicted noise and actual noise.
    """

    def __init__(
        self,
        noise_scheduler: DDPMScheduler,
        device: torch.device,
        loss_type: str = "huber",
    ):
        super().__init__(["loss"])
        self.noise_scheduler = noise_scheduler
        self.device = device
        self.loss_type = loss_type.lower()
        if self.loss_type == "mae":
            self.loss = nn.L1Loss()
        elif self.loss_type == "mse":
            self.loss = nn.MSELoss()
        elif self.loss_type == "huber":
            self.loss = nn.HuberLoss()
        else:
            raise ValueError(
                f"Unsupported loss_type={loss_type!r}. Choose 'mae', 'mse', or 'huber'."
            )

    def compute_loss(self, instance, model: UNet2DModel):
        carved_x, x_0, _beta, _alpha, slice_type = instance

        # Add channel dimension: (B, H, W) → (B, 1, H, W)
        if x_0.dim() == 3:
            x_0 = x_0.unsqueeze(1)
        if carved_x.dim() == 3:
            carved_x = carved_x.unsqueeze(1)

        x_0 = x_0.float().to(self.device)
        carved_x = carved_x.float().to(self.device)

        noise = torch.randn_like(x_0)
        bsz = x_0.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=self.device,
        ).long()

        # Forward diffusion: add noise to target x_0
        x_t = self.noise_scheduler.add_noise(x_0, noise, timesteps)

        if isinstance(slice_type, str):
            slice_type = (slice_type,)
        use_conditioning = torch.ones(bsz, dtype=torch.long, device=self.device)
        non_axis0 = torch.tensor(
            [stype != "axis0" for stype in slice_type],
            dtype=torch.bool,
            device=self.device,
        )
        use_conditioning[non_axis0] = torch.randint(
            0,
            2,
            (int(non_axis0.sum().item()),),
            device=self.device,
            dtype=torch.long,
        )
        carved_x = carved_x * use_conditioning.view(bsz, 1, 1, 1).to(carved_x.dtype)

        # Concatenate noisy target with conditioning (carved_x) along channel dim
        model_input = torch.cat([x_t, carved_x], dim=1)  # (B, 2, H, W)

        model.zero_grad()
        noise_pred = model(
            model_input,
            timestep=timesteps,
            class_labels=use_conditioning,
            return_dict=False,
        )[0]  # (B, 1, H, W)

        loss = self.loss(noise_pred, noise)
        return loss, {"loss": loss.item()}


def main():
    parser = ArgumentParser(
        description=(
            "Train a conditional DDPM diffusion model for missing-cone recovery.  "
            "Input: carved_x (conditioner) + x_t (noisy target).  "
            "Conditioning: binary conditioning flag as class label."
        )
    )

    # ── Dataset / paths ──────────────────────────────────────────────────────
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help=(
            "Path to a single la_fbp .npy file or a directory containing .npy files"
        ),
    )
    parser.add_argument(
        "--cone_width_deg",
        type=float,
        required=True,
        help=(
            "Width in degrees of the Fourier cone that is carved out.  "
            "Should match the missing-cone width of the LA reconstruction.  "
            "Example: 72 for a 60%%-coverage acquisition (0.6 * 180 = 108 kept, "
            "180 - 108 = 72 missing)."
        ),
    )
    parser.add_argument(
        "--norm_min",
        type=float,
        default=None,
        help="Minimum value for normalisation (default: auto from data or sidecar)",
    )
    parser.add_argument(
        "--norm_max",
        type=float,
        default=None,
        help="Maximum value for normalisation (default: auto from data or sidecar)",
    )

    # ── Augmentation ─────────────────────────────────────────────────────────
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help=(
            "Orientation correction in degrees.  Set to the angle of the existing "
            "missing cone in the dataset when it deviates from 0°.  Each slice is "
            "rotated by -gamma before the random alpha rotation so the pipeline "
            "convention (cone at 0°) is restored.  Default: 0.0 (no correction)."
        ),
    )
    parser.add_argument(
        "--slice_types",
        type=str,
        default="axis0,axis1,axis2,rand",
        help=(
            "Comma-separated list of slice types to include. Valid: 'axis0', 'axis1', 'axis2', 'rand'. "
            "Default: None (includes all: axis0, axis1, axis2, rand)."
        ),
    )
    parser.add_argument(
        "--no_augment",
        action="store_true",
        help="Disable scale / translation augmentation",
    )
    parser.add_argument(
        "--scale_jitter",
        type=float,
        default=0.10,
        help="Scale jitter half-range (default 0.10 → scale in [0.9, 1.1])",
    )
    parser.add_argument(
        "--shift_fraction",
        type=float,
        default=0.05,
        help="Max shift as fraction of image size (default 0.05)",
    )

    # ── Training hyperparameters ─────────────────────────────────────────────
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--scheduler", type=str, default="[500]")
    parser.add_argument("--lr_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exp_name", type=str, default="cond_ladiff")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument(
        "--loss_type",
        type=str,
        default="huber",
        choices=["mae", "mse", "huber"],
        help="Loss type for noise prediction: 'mae', 'mse', or 'huber'. Default: huber.",
    )
    # ── Model ────────────────────────────────────────────────────────────────
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

    # ── Save sample images for sanity-checking ───────────────────────────────
    os.makedirs("samples", exist_ok=True)
    sample = train_ds[0]  # (carved_x, x, beta_int, alpha_int, slice_type)
    carved_sample, x_sample, beta_sample, alpha_sample, slice_type_sample = sample
    TF.to_pil_image(x_sample.clamp(0, 1)).save(
        f"samples/sample_x_{args.exp_name}.png"
    )
    TF.to_pil_image(carved_sample.clamp(0, 1)).save(
        f"samples/sample_carved_x_{args.exp_name}.png"
    )
    print(
        f"Sample images saved to samples/sample_x_{args.exp_name}.png "
        f"and samples/sample_carved_x_{args.exp_name}.png  "
        f"(beta={beta_sample}°, alpha={alpha_sample}°, slice_type={slice_type_sample})"
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = create_model(args)
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = f"checkpoints/ddpm_ladiff_{args.exp_name}.pt"

    # Save normalisation config next to the checkpoint for inference use.
    norm_config_path = checkpoint_path.replace(".pt", "_norm.json")
    with open(norm_config_path, "w") as f:
        json.dump(
            {
                "norm_min": norm_min,
                "norm_max": norm_max,
                "cone_width_deg": args.cone_width_deg,
                "gamma": args.gamma,
            },
            f,
            indent=2,
        )
    print(f"Normalisation + cone config saved to {norm_config_path}")

    if args.load_checkpoint:
        checkpoint_path = args.load_checkpoint
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
    loss_fn = ConditionalLADiffLoss(noise_scheduler, device, loss_type=args.loss_type)

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
