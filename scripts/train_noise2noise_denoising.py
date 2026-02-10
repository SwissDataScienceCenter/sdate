#!/usr/bin/env python
"""
Training script for Noise2Noise Denoising of Tomographic Projections.

This script trains a UNet model using the Noise2Noise paradigm:
Given projections P_{i-1} and P_{i+1}, predict P_i.

Usage:
    python train_noise2noise_denoising.py --data_path /path/to/ct_files --epochs 100

    # With multiple folders
    python train_noise2noise_denoising.py --data_path /path/to/folder1 /path/to/folder2 --epochs 100

    # Resume from checkpoint
    python train_noise2noise_denoising.py --data_path /path/to/ct_files --load_checkpoint checkpoints/model.pt
"""

import os
import sys
import random
from argparse import ArgumentParser
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from diffusers import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup

from sdate.datasets.projection_triplet_dataset import (
    ProjectionTripletDataset,
    create_train_val_split
)
from sdate.losses.noise2noise_loss import Noise2NoiseLoss

# Optional: wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def create_model(
    sample_size: int = 512,
    in_channels: int = 2,
    out_channels: int = 1,
    model_size: str = "medium"
) -> UNet2DModel:
    """
    Create the UNet2DModel for denoising.
    
    Args:
        sample_size: Input image size
        in_channels: Number of input channels (2 for prev+next projections)
        out_channels: Number of output channels (1 for middle projection)
        model_size: One of "small", "medium", "large"
        
    Returns:
        UNet2DModel instance
    """
    if model_size == "small":
        block_out_channels = (32, 32, 64, 64, 128, 128)
    elif model_size == "medium":
        block_out_channels = (64, 64, 128, 128, 256, 256)
    elif model_size == "large":
        block_out_channels = (64, 128, 256, 256, 512, 512)
    else:
        raise ValueError(f"Unknown model_size: {model_size}")
    
    model = UNet2DModel(
        sample_size=sample_size,
        in_channels=in_channels,
        out_channels=out_channels,
        layers_per_block=2,
        block_out_channels=block_out_channels,
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


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    best_val_loss: float,
    checkpoint_path: str
):
    """Save training checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_val_loss': best_val_loss,
    }, checkpoint_path)
    print(f"💾 Checkpoint saved to {checkpoint_path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    checkpoint_path: str,
    device: torch.device
) -> tuple:
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint['epoch']
    best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    print(f"📂 Loaded checkpoint from {checkpoint_path} (epoch {epoch})")
    return epoch, best_val_loss


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: Noise2NoiseLoss,
    device: torch.device,
    epoch: int,
    use_wandb: bool = False
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        optimizer.zero_grad()
        
        # Compute loss
        loss, loss_dict = loss_fn.compute_loss(batch, model)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })
        
        # Log to wandb
        if use_wandb and WANDB_AVAILABLE:
            wandb.log({
                'train/loss': loss.item(),
                'train/mse_loss': loss_dict['mse_loss'].item(),
                'learning_rate': optimizer.param_groups[0]['lr']
            })
    
    return total_loss / num_batches


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: Noise2NoiseLoss,
    device: torch.device,
    epoch: int,
    use_wandb: bool = False
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
    for batch in pbar:
        loss, loss_dict = loss_fn.compute_loss(batch, model)
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / num_batches
    
    if use_wandb and WANDB_AVAILABLE:
        wandb.log({'val/loss': avg_loss})
    
    return avg_loss


def main():
    parser = ArgumentParser(description="Train Noise2Noise Denoising for Tomographic Projections")
    
    # Data arguments
    parser.add_argument("--data_path", nargs='+', default=[f'/myhome/data/sdate/shared/compression_paper/file_{i}_extracted' for i in range(1, 13)],
                        help="Path(s) to folders containing TIFF files")
    parser.add_argument("--target_size", type=int, default=512,
                        help="Target image size (default: 512)")
    parser.add_argument("--num_darks", type=int, default=None,
                        help="Number of dark field images per folder (default: auto-detect from log file)")
    parser.add_argument("--num_flats", type=int, default=None,
                        help="Number of flat field images per folder (default: auto-detect from log file)")
    parser.add_argument("--use_attenuation", action='store_true', default=True,
                        help="Apply dark/flat correction and compute attenuation (default: True)")
    parser.add_argument("--cache_in_memory", action='store_true',
                        help="Cache projections in memory for faster training")
    parser.add_argument("--preload", action='store_true',
                        help="Preload all projections at startup")
    parser.add_argument("--num_input_projections", type=int, default=3,
                        help="Number of input projections (k). Target is the (k+1)-th projection (default: 3)")
    
    # Model arguments
    parser.add_argument("--model_size", type=str, default="medium",
                        choices=["small", "medium", "large"],
                        help="Model size (default: medium)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size (default: 2)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs (default: 100)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay (default: 0.01)")
    parser.add_argument("--warmup_steps", type=int, default=500,
                        help="Number of warmup steps (default: 500)")
    parser.add_argument("--val_ratio", type=float, default=0.05,
                        help="Validation set ratio (default: 0.1)")
    parser.add_argument("--augment", action='store_true',
                        help="Apply data augmentation")
    
    # Loss arguments
    parser.add_argument("--use_l1", action='store_true',
                        help="Use L1 loss instead of MSE")
    parser.add_argument("--use_gradient_loss", action='store_true',
                        help="Add gradient consistency loss")
    parser.add_argument("--gradient_weight", type=float, default=0.1,
                        help="Weight for gradient loss (default: 0.1)")
    
    # Checkpoint arguments
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                        help="Directory to save checkpoints (default: checkpoints)")
    parser.add_argument("--load_checkpoint", type=str, default="",
                        help="Path to checkpoint to resume from")
    parser.add_argument("--save_every", type=int, default=10,
                        help="Save checkpoint every N epochs (default: 10)")
    
    # Experiment tracking
    parser.add_argument("--exp_name", type=str, default="noise2noise_denoising",
                        help="Experiment name (default: noise2noise_denoising)")
    parser.add_argument("--wandb", action='store_true',
                        help="Use wandb for experiment tracking")
    parser.add_argument("--wandb_project", type=str, default="tomography_denoising",
                        help="Wandb project name")
    
    # Other
    parser.add_argument("--seed", type=int, default=-1,
                        help="Random seed (-1 for random)")
    parser.add_argument("--num_workers", type=int, default=torch.cpu.device_count(),
                        help="Number of data loader workers (default: 4)")
    
    args = parser.parse_args()
    
    # Set seed
    seed = args.seed if args.seed >= 0 else random.randint(0, 20000)
    torch.manual_seed(seed)
    random.seed(seed)
    print(f"🎲 Random seed: {seed}")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    
    # Determine whether to use attenuation
    use_attenuation = args.use_attenuation
    print(f"📊 Using {'attenuation-corrected' if use_attenuation else 'raw'} projections")
    
    # Create dataset
    print("\n📚 Loading dataset...")
    dataset = ProjectionTripletDataset(
        folder_paths=args.data_path,
        target_size=args.target_size,
        cache_in_memory=args.cache_in_memory,
        preload=args.preload,
        augment=args.augment,
        verbose=True,
        use_attenuation=use_attenuation,
        num_input_projections=args.num_input_projections
    )
    
    # Split into train/val
    train_dataset, val_dataset = create_train_val_split(
        dataset, val_ratio=args.val_ratio, seed=seed
    )
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0 if args.preload else args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if args.preload else args.num_workers,
        pin_memory=True
    )
    
    # Create model
    print("\n🏗️  Creating model...")
    model = create_model(
        sample_size=args.target_size,
        in_channels=args.num_input_projections, 
        out_channels=1,
        model_size=args.model_size
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Model parameters: {num_params:,}")
    
    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    num_training_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Create loss function
    loss_fn = Noise2NoiseLoss(
        device=device,
        use_l1=args.use_l1,
        use_gradient_loss=args.use_gradient_loss,
        gradient_weight=args.gradient_weight
    )
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    if args.load_checkpoint:
        start_epoch, best_val_loss = load_checkpoint(
            model, optimizer, scheduler, args.load_checkpoint, device
        )
    
    # Initialize wandb
    use_wandb = args.wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.exp_name,
            config=vars(args)
        )
        wandb.watch(model)
    
    # Training loop
    print("\n🚀 Starting training...")
    print("=" * 60)
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            loss_fn, device, epoch, use_wandb
        )
        
        # Validate
        val_loss = validate(model, val_loader, loss_fn, device, epoch, use_wandb)
        
        print(f"\n📊 Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_loss,
                checkpoint_dir / "best_model.pt"
            )
            print(f"   ✨ New best model! Val Loss = {val_loss:.4f}")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_loss,
                checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, scheduler, args.epochs - 1, best_val_loss,
        checkpoint_dir / "final_model.pt"
    )
    
    print("\n✅ Training completed!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Checkpoints saved to: {checkpoint_dir}")
    
    if use_wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
