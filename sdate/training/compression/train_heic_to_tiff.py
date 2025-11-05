"""
Training script for HEIC to TIFF translation using UNet3DConditionModel.

This script trains a 3D UNet to translate HEIC compressed volumes back to original TIFF quality,
using positional encodings as conditioning information.
"""

import os
import sys
import argparse
import math
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

import numpy as np
from tqdm.auto import tqdm
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed

# Diffusers imports
from diffusers import UNet3DConditionModel
from diffusers.models.embeddings import get_3d_sincos_pos_embed

# Local imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from sdate.datasets import TiffVolumeDataset


logger = get_logger(__name__)


class PositionalEncoder(nn.Module):
    """
    Positional encoder for 3D coordinates using sinusoidal embeddings.
    
    Args:
        d_model: Dimension of the embedding
        max_position: Maximum position value expected
    """
    
    def __init__(self, d_model: int = 768, max_position: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_position = max_position
        
        # Create positional encoding lookup table
        pe = torch.zeros(max_position, d_model)
        position = torch.arange(0, max_position, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe)
        
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: Tensor of shape (batch_size, 3) containing [d, h, w] positions
            
        Returns:
            Positional embeddings of shape (batch_size, d_model)
        """
        batch_size = positions.shape[0]
        
        # Encode each coordinate separately and sum them
        embeddings = torch.zeros(batch_size, self.d_model, device=positions.device)
        
        for i in range(3):  # d, h, w coordinates
            coords = positions[:, i].long().clamp(0, self.max_position - 1)
            embeddings += self.pe[coords]
            
        return embeddings


class HeicToTiffLoss(nn.Module):
    """
    Combined loss function for HEIC to TIFF translation.
    """
    
    def __init__(self, l1_weight: float = 1.0, l2_weight: float = 0.5, perceptual_weight: float = 0.0):
        super().__init__()
        self.l1_weight = l1_weight
        self.l2_weight = l2_weight
        self.perceptual_weight = perceptual_weight
        
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss.
        
        Args:
            pred: Predicted TIFF volume of shape (B, 1, D, H, W)
            target: Target TIFF volume of shape (B, 1, D, H, W)
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses = {}
        
        # L1 loss
        l1_loss = self.l1_loss(pred, target)
        losses['l1'] = l1_loss
        
        # L2 loss
        l2_loss = self.l2_loss(pred, target)
        losses['l2'] = l2_loss
        
        # Combined loss
        total_loss = self.l1_weight * l1_loss + self.l2_weight * l2_loss
        losses['total'] = total_loss
        
        return losses


class HeicToTiffTrainer:
    """
    Trainer class for HEIC to TIFF translation using UNet3DConditionModel.
    """
    
    def __init__(
        self,
        data_path: str,
        output_dir: str,
        volume_size: int = 64,
        stride: int = 32,
        num_frames: int = 100,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        num_epochs: int = 100,
        validation_split: float = 0.2,
        heic_quality: int = 85,
        max_workers: int = 8,
        mixed_precision: str = "fp16",
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        warmup_steps: int = 1000,
        logging_steps: int = 100,
        save_steps: int = 1000,
        eval_steps: int = 500,
        seed: int = 42,
        use_wandb: bool = False,
        wandb_project: str = "heic-to-tiff-translation",
    ):
        """
        Initialize the trainer.
        
        Args:
            data_path: Path to TIFF data directory
            output_dir: Directory to save model checkpoints and logs
            volume_size: Size of sub-volumes for training
            stride: Stride for sub-volume extraction
            num_frames: Number of TIFF frames to load
            batch_size: Training batch size
            learning_rate: Learning rate for optimizer
            num_epochs: Number of training epochs
            validation_split: Fraction of data to use for validation
            heic_quality: HEIC compression quality
            max_workers: Number of workers for data loading
            mixed_precision: Mixed precision training ("fp16", "bf16", or None)
            gradient_accumulation_steps: Steps to accumulate gradients
            max_grad_norm: Maximum gradient norm for clipping
            warmup_steps: Number of warmup steps for learning rate
            logging_steps: Steps between logging
            save_steps: Steps between model saves
            eval_steps: Steps between evaluations
            seed: Random seed
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
        """
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.volume_size = volume_size
        self.stride = stride
        self.num_frames = num_frames
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.validation_split = validation_split
        self.heic_quality = heic_quality
        self.max_workers = max_workers
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.warmup_steps = warmup_steps
        self.logging_steps = logging_steps
        self.save_steps = save_steps
        self.eval_steps = eval_steps
        self.seed = seed
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize accelerator
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with="wandb" if use_wandb else None,
            project_dir=str(self.output_dir),
        )
        
        # Set seed
        set_seed(seed)
        
        # Setup logging
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(f"Accelerator state: {self.accelerator.state}")
        
        # Initialize models and dataset
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.positional_encoder = None
        self.train_dataset = None
        self.val_dataset = None
        self.train_dataloader = None
        self.val_dataloader = None
        
    def setup_dataset(self):
        """Setup training and validation datasets."""
        logger.info("Setting up datasets...")
        
        # Create full dataset with dual-channel loading (TIFF + HEIC)
        full_dataset = TiffVolumeDataset(
            data_path=self.data_path,
            volume_size=self.volume_size,
            stride=self.stride,
            num_frames=self.num_frames,
            start_offset=0,
            normalize=True,
            global_normalize=True,
            use_heic_compression=True,
            heic_quality=self.heic_quality,
            dual_channel=True,  # Load both TIFF and HEIC
            max_workers=self.max_workers,
        )
        
        # Split into train/validation
        dataset_size = len(full_dataset)
        val_size = int(self.validation_split * dataset_size)
        train_size = dataset_size - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        
        logger.info(f"Dataset sizes - Train: {len(self.train_dataset)}, Validation: {len(self.val_dataset)}")
        
        # Custom collate function for dual-channel data
        def collate_fn(batch):
            """
            Collate function that separates HEIC and TIFF channels.
            
            Input batch: List of (dual_channel_volume, positions)
            - dual_channel_volume: (2, D, H, W) where dim 0 is [TIFF, HEIC]
            - positions: (3,) tensor with [d_start, h_start, w_start]
            
            Returns:
            - heic_volumes: (B, 1, D, H, W) - input to model
            - tiff_volumes: (B, 1, D, H, W) - target for model
            - positions: (B, 3) - positional information
            """
            dual_volumes = torch.stack([item[0] for item in batch])  # (B, 2, D, H, W)
            positions = torch.stack([item[1] for item in batch])     # (B, 3)
            
            # Split channels
            tiff_volumes = dual_volumes[:, 0:1]  # (B, 1, D, H, W) - target
            heic_volumes = dual_volumes[:, 1:2]  # (B, 1, D, H, W) - input
            
            return heic_volumes, tiff_volumes, positions
        
        # Create data loaders
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.max_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.max_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )
        
        logger.info(f"Data loaders created - Train batches: {len(self.train_dataloader)}, Val batches: {len(self.val_dataloader)}")
    
    def setup_model(self):
        """Setup the UNet3D model and related components."""
        logger.info("Setting up model...")

        # Initialize UNet3D model
        self.model = UNet3DConditionModel(
            sample_size=self.volume_size,
            in_channels=2,  # HEIC input
            out_channels=1, # TIFF output
            layers_per_block=2,
            block_out_channels=(64, 128, 128, 256),
            down_block_types=(
                "DownBlock3D",
                "DownBlock3D", 
                "CrossAttnDownBlock3D",
                "DownBlock3D",
            ),
            up_block_types=(
                "UpBlock3D",
                "CrossAttnUpBlock3D",
                "UpBlock3D",
                "UpBlock3D",
            ),
            cross_attention_dim=768,  # Dimension for positional encoding
            attention_head_dim=64,
        )
        
        # Initialize positional encoder
        self.positional_encoder = PositionalEncoder(
            d_model=768,  # Match cross_attention_dim
            max_position=max(1000, self.volume_size * 4)  # Allow for larger volumes
        )
        
        # Initialize loss function
        self.loss_fn = HeicToTiffLoss(
            l1_weight=.0,
            l2_weight=1.,
        )
        
        logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        logger.info("Setting up optimizer and scheduler...")
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8,
        )
        
        # Learning rate scheduler with warmup
        total_steps = len(self.train_dataloader) * self.num_epochs
        
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps - self.warmup_steps,
            eta_min=self.learning_rate * 0.01
        )
        
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps]
        )
        
        logger.info(f"Optimizer and scheduler set up for {total_steps} total steps")
    
    def setup_accelerator(self):
        """Prepare models and optimizers with accelerator."""
        logger.info("Preparing models with accelerator...")
        
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.scheduler
        )
        
        self.positional_encoder = self.positional_encoder.to(self.accelerator.device)
        
        # Initialize W&B if requested
        if self.use_wandb and self.accelerator.is_main_process:
            self.accelerator.init_trackers(
                project_name=self.wandb_project,
                config={
                    "volume_size": self.volume_size,
                    "stride": self.stride,
                    "num_frames": self.num_frames,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "num_epochs": self.num_epochs,
                    "heic_quality": self.heic_quality,
                    "mixed_precision": self.mixed_precision,
                    "gradient_accumulation_steps": self.gradient_accumulation_steps,
                }
            )
    
    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Tuple of (heic_volumes, tiff_volumes, positions)
            
        Returns:
            Dictionary of losses
        """
        heic_volumes, tiff_volumes, positions = batch

        inputs = torch.cat([heic_volumes, torch.zeros_like(heic_volumes)], dim=1)
        
        # Generate positional encodings
        encoder_hidden_states = self.positional_encoder(positions)  # (B, 768)
        encoder_hidden_states = encoder_hidden_states.unsqueeze(1)  # (B, 1, 768)
        
        # Forward pass through UNet
        with self.accelerator.accumulate(self.model):
            # Predict TIFF from HEIC
            predicted_tiff = self.model(
                sample=inputs,
                timestep=torch.zeros(heic_volumes.shape[0], device=heic_volumes.device),  # No timestep for direct translation
                encoder_hidden_states=encoder_hidden_states,
            ).sample
            
            # Calculate losses
            losses = self.loss_fn(predicted_tiff, tiff_volumes)
            
            # Backward pass
            self.accelerator.backward(losses['total'])
            
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
        
        return {k: v.item() for k, v in losses.items()}
    
    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single validation step.
        
        Args:
            batch: Tuple of (heic_volumes, tiff_volumes, positions)
            
        Returns:
            Dictionary of losses
        """
        heic_volumes, tiff_volumes, positions = batch
        
        # Generate positional encodings
        encoder_hidden_states = self.positional_encoder(positions)  # (B, 768)
        encoder_hidden_states = encoder_hidden_states.unsqueeze(1)  # (B, 1, 768)
        
        with torch.no_grad():
            # Predict TIFF from HEIC
            predicted_tiff = self.model(
                sample=heic_volumes,
                timestep=torch.zeros(heic_volumes.shape[0], device=heic_volumes.device),
                encoder_hidden_states=encoder_hidden_states,
            ).sample
            
            # Calculate losses
            losses = self.loss_fn(predicted_tiff, tiff_volumes)
        
        return {k: v.item() for k, v in losses.items()}
    
    def evaluate(self) -> Dict[str, float]:
        """Run validation and return average losses."""
        self.model.eval()
        
        total_losses = {}
        num_batches = 0
        
        for batch in tqdm(self.val_dataloader, desc="Validation", disable=not self.accelerator.is_local_main_process):
            losses = self.validation_step(batch)
            
            # Accumulate losses
            for key, value in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0
                total_losses[key] += value
            
            num_batches += 1
        
        # Average losses
        avg_losses = {f"val_{key}": total / num_batches for key, total in total_losses.items()}
        
        self.model.train()
        return avg_losses
    
    def save_checkpoint(self, step: int):
        """Save model checkpoint."""
        if self.accelerator.is_main_process:
            checkpoint_dir = self.output_dir / f"checkpoint-{step}"
            checkpoint_dir.mkdir(exist_ok=True)
            
            # Save model
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            unwrapped_model.save_pretrained(checkpoint_dir / "unet")
            
            # Save positional encoder
            torch.save(
                self.positional_encoder.state_dict(),
                checkpoint_dir / "positional_encoder.pth"
            )
            
            # Save training state
            torch.save({
                'step': step,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'config': {
                    'volume_size': self.volume_size,
                    'stride': self.stride,
                    'num_frames': self.num_frames,
                    'heic_quality': self.heic_quality,
                }
            }, checkpoint_dir / "training_state.pth")
            
            logger.info(f"Checkpoint saved at step {step}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Setup everything
        self.setup_dataset()
        self.setup_model()
        self.setup_optimizer()
        self.setup_accelerator()
        
        # Training loop
        global_step = 0
        self.model.train()
        
        for epoch in range(self.num_epochs):
            logger.info(f"Starting epoch {epoch + 1}/{self.num_epochs}")
            
            epoch_losses = {}
            num_batches = 0
            
            progress_bar = tqdm(
                self.train_dataloader,
                desc=f"Epoch {epoch + 1}",
                disable=not self.accelerator.is_local_main_process
            )
            
            for batch in progress_bar:
                # Training step
                losses = self.train_step(batch)
                
                # Accumulate losses for logging
                for key, value in losses.items():
                    if key not in epoch_losses:
                        epoch_losses[key] = 0
                    epoch_losses[key] += value
                
                num_batches += 1
                global_step += 1
                
                # Update progress bar
                if self.accelerator.is_local_main_process:
                    progress_bar.set_postfix({
                        "loss": f"{losses['total']:.4f}",
                        "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })
                
                # Logging
                if global_step % self.logging_steps == 0:
                    avg_losses = {key: total / min(num_batches, self.logging_steps) 
                                for key, total in epoch_losses.items()}
                    avg_losses['learning_rate'] = self.scheduler.get_last_lr()[0]
                    
                    if self.accelerator.is_main_process:
                        self.accelerator.log(avg_losses, step=global_step)
                        logger.info(f"Step {global_step}: {avg_losses}")
                    
                    # Reset loss accumulation
                    epoch_losses = {}
                    num_batches = 0
                
                # Evaluation
                if global_step % self.eval_steps == 0:
                    val_losses = self.evaluate()
                    if self.accelerator.is_main_process:
                        self.accelerator.log(val_losses, step=global_step)
                        logger.info(f"Validation at step {global_step}: {val_losses}")
                
                # Save checkpoint
                if global_step % self.save_steps == 0:
                    self.save_checkpoint(global_step)
        
        # Save final checkpoint
        self.save_checkpoint(global_step)
        
        if self.use_wandb and self.accelerator.is_main_process:
            self.accelerator.end_training()
        
        logger.info("Training completed!")


def main():
    """Main function for command-line training."""
    parser = argparse.ArgumentParser(description="Train HEIC to TIFF translation model")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, required=True,
                      help="Path to TIFF data directory")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save model and logs")
    
    # Model arguments
    parser.add_argument("--volume_size", type=int, default=64,
                      help="Size of sub-volumes for training")
    parser.add_argument("--stride", type=int, default=32,
                      help="Stride for sub-volume extraction")
    parser.add_argument("--num_frames", type=int, default=32,
                      help="Number of TIFF frames to load")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4,
                      help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                      help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=100,
                      help="Number of training epochs")
    parser.add_argument("--validation_split", type=float, default=0.01,
                      help="Fraction of data for validation")
    
    # HEIC arguments
    parser.add_argument("--heic_quality", type=int, default=50,
                      help="HEIC compression quality")
    
    # System arguments
    parser.add_argument("--max_workers", type=int, default=8,
                      help="Number of data loading workers")
    parser.add_argument("--mixed_precision", type=str, default="fp16",
                      choices=["fp16", "bf16", "no"],
                      help="Mixed precision training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                      help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                      help="Maximum gradient norm for clipping")
    
    # Logging arguments
    parser.add_argument("--warmup_steps", type=int, default=480,
                      help="Number of warmup steps")
    parser.add_argument("--logging_steps", type=int, default=480,
                      help="Steps between logging")
    parser.add_argument("--save_steps", type=int, default=480,
                      help="Steps between model saves")
    parser.add_argument("--eval_steps", type=int, default=500,
                      help="Steps between evaluations")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed")
    
    # W&B arguments
    parser.add_argument("--use_wandb", action="store_true",
                      help="Use Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="heic-to-tiff-translation",
                      help="W&B project name")
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = HeicToTiffTrainer(
        data_path=args.data_path,
        output_dir=args.output_dir,
        volume_size=args.volume_size,
        stride=args.stride,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        validation_split=args.validation_split,
        heic_quality=args.heic_quality,
        max_workers=args.max_workers,
        mixed_precision=args.mixed_precision if args.mixed_precision != "no" else None,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        seed=args.seed,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )
    
    trainer.train()


if __name__ == "__main__":
    main()