#!/usr/bin/env python
"""
Training script for INCT (Instant Neural Compression for Tomography).

This script trains an Instant NGP model on tomographic projection data.

Usage:
    python train_inct.py --data_path /path/to/data --num_projections 10
    
    # Quick test with fewer projections
    python train_inct.py --data_path /path/to/data --num_projections 5 --num_epochs 50
    
    # Full training with more capacity
    python train_inct.py --data_path /path/to/data --n_levels 16 --table_size 524288
"""

import argparse
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from torch.utils.data import DataLoader

from inct import (
    InstantNGPModel,
    ProjectionVolumeDataset,
    BatchVoxelDataset,
    Trainer,
    TrainingConfig,
)
from inct.utils import psnr, get_model_size_bytes, compute_compression_ratio


def parse_args():
    parser = argparse.ArgumentParser(description="Train INCT model")
    
    # Data arguments
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to tomography data folder')
    parser.add_argument('--num_projections', type=int, default=10,
                        help='Number of projections to load')
    parser.add_argument('--start_projection', type=int, default=0,
                        help='Index of first projection')
    parser.add_argument('--target_size', type=int, default=256,
                        help='Resize projections to this size')
    
    # Model arguments
    parser.add_argument('--n_dims', type=int, default=3,
                        help='Number of input dimensions')
    parser.add_argument('--n_levels', type=int, default=16,
                        help='Number of hash encoding levels')
    parser.add_argument('--n_features', type=int, default=2,
                        help='Features per level')
    parser.add_argument('--base_resolution', type=int, default=16,
                        help='Coarsest resolution')
    parser.add_argument('--max_resolution', type=int, default=512,
                        help='Finest resolution')
    parser.add_argument('--table_size', type=int, default=2**19,
                        help='Hash table size')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[64, 64],
                        help='MLP hidden dimensions')
    
    # Training arguments
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=65536,
                        help='Batch size (number of voxels)')
    parser.add_argument('--n_batches', type=int, default=500,
                        help='Number of batches per epoch')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6,
                        help='Weight decay')
    parser.add_argument('--loss', type=str, default='mse',
                        choices=['mse', 'l1', 'huber'],
                        help='Loss function')
    
    # Output arguments
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_inct',
                        help='Directory for checkpoints')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='inct',
                        help='W&B project name')
    
    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None,
                        help='Device (cuda/cpu)')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print("=" * 60)
    
    # Load data
    print(f"\n📂 Loading data from: {args.data_path}")
    print(f"   Projections: {args.start_projection} to {args.start_projection + args.num_projections - 1}")
    print(f"   Target size: {args.target_size}x{args.target_size}")
    
    dataset = ProjectionVolumeDataset(
        folder_path=args.data_path,
        num_projections=args.num_projections,
        target_size=args.target_size,
        batch_size=args.batch_size,
        n_batches=args.n_batches,
        start_projection=args.start_projection,
        normalize_values=True,
        verbose=True,
    )
    
    # Create dataloader
    train_loader = DataLoader(
        dataset,
        batch_size=None,  # Dataset already returns batches
        shuffle=False,
        num_workers=0,
    )
    
    # For validation, use the same dataset but different batches
    val_loader = train_loader  # In practice, could split the data
    
    # Create model
    print(f"\n🏗️  Creating model...")
    model = InstantNGPModel(
        n_dims=args.n_dims,
        n_levels=args.n_levels,
        n_features_per_level=args.n_features,
        base_resolution=args.base_resolution,
        max_resolution=args.max_resolution,
        table_size=args.table_size,
        hidden_dims=args.hidden_dims,
        output_dim=1,
        output_activation='sigmoid',
    )
    
    print(model.get_model_info())
    
    # Compute compression ratio
    model_size = get_model_size_bytes(model)
    original_shape = dataset.shape
    compression_ratio = compute_compression_ratio(original_shape, model_size, dtype_bits=16)
    
    print(f"\n📊 Compression Statistics:")
    print(f"   Original volume shape: {original_shape}")
    print(f"   Original size: {(torch.prod(torch.tensor(original_shape)) * 2 / 1024**2):.2f} MB")
    print(f"   Model size: {model_size / 1024:.2f} KB")
    print(f"   Compression ratio: {compression_ratio:.1f}x")
    
    # Create training config
    config = TrainingConfig(
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        log_interval=50,
        eval_interval=10,
        checkpoint_interval=25,
        checkpoint_dir=args.checkpoint_dir,
        loss_fn=args.loss,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
    )
    
    # Create trainer
    trainer = Trainer(model, config, device=device)
    
    # Train
    print(f"\n🚀 Starting training...")
    print("=" * 60)
    
    history = trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        verbose=True,
    )
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("📈 Final Evaluation")
    print("=" * 60)
    
    final_metrics = trainer.evaluate(val_loader, n_batches=50)
    print(f"   Final Loss: {final_metrics['loss']:.6f}")
    print(f"   Final PSNR: {final_metrics['psnr']:.2f} dB")
    print(f"   Final MSE: {final_metrics['mse']:.8f}")
    
    # Save final model with metadata
    final_save_path = Path(args.checkpoint_dir) / 'final_model_with_metadata.pt'
    torch.save({
        'config': model.config,
        'state_dict': model.state_dict(),
        'data_config': {
            'data_path': str(args.data_path),
            'num_projections': args.num_projections,
            'start_projection': args.start_projection,
            'target_size': args.target_size,
            'original_shape': original_shape,
            'value_min': dataset.value_min,
            'value_max': dataset.value_max,
        },
        'metrics': final_metrics,
        'compression_ratio': compression_ratio,
    }, final_save_path)
    
    print(f"\n✅ Model saved to: {final_save_path}")
    print(f"   Compression ratio: {compression_ratio:.1f}x")
    print(f"   PSNR: {final_metrics['psnr']:.2f} dB")


if __name__ == '__main__':
    main()
