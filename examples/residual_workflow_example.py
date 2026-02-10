#!/usr/bin/env python3
"""
Example: Complete workflow for computing and using residuals

This script demonstrates:
1. Computing residuals from a trained model
2. Loading dataset with residuals
3. Training a simple residual correction model

Usage:
    python examples/residual_workflow_example.py \
        --data_path=/path/to/data \
        --checkpoint_path=/path/to/checkpoint
"""

import sys
import argparse
import torch
import torch.nn as nn
from pathlib import Path

sys.path.append('/myhome/sdate')

from sdate.datasets import TiffVolumeDataset
from torch.utils.data import DataLoader


class SimpleResidualCorrector(nn.Module):
    """
    Simple 3D CNN to predict correction from HEIC and residuals.
    
    Input: 2 channels (HEIC + Residual)
    Output: 1 channel (Correction)
    Final: Corrected TIFF = HEIC + Correction
    """
    def __init__(self, in_channels=2, hidden_dim=32):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, hidden_dim*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim*2, hidden_dim*4, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv3d(hidden_dim*4, hidden_dim*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim*2, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_dim, 1, 3, padding=1),
        )
    
    def forward(self, heic, residual):
        """
        Args:
            heic: (B, 1, D, H, W) - compressed input
            residual: (B, 1, D, H, W) - model residuals
        
        Returns:
            correction: (B, 1, D, H, W) - correction to apply
        """
        x = torch.cat([heic, residual], dim=1)  # (B, 2, D, H, W)
        features = self.encoder(x)
        correction = self.decoder(features)
        return correction


def compute_residuals_step(args):
    """Step 1: Compute residuals from trained model"""
    print("\n" + "="*80)
    print("STEP 1: Computing Residuals")
    print("="*80)
    
    from scripts.compute_residuals import compute_residuals
    
    # Check if checkpoint exists
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("   Please train a model first with run_heic_to_tiff_training.py")
        return False
    
    # Compute residuals
    output_path = args.output_dir / "residuals.npy"
    
    print(f"\nComputing residuals:")
    print(f"  Data: {args.data_path}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Output: {output_path}")
    
    compute_residuals(
        data_path=args.data_path,
        checkpoint_path=checkpoint_path,
        output_path=str(output_path),
        volume_size=args.volume_size,
        stride=args.stride,
        num_frames=args.num_frames,
        heic_quality=args.heic_quality,
        batch_size=args.batch_size,
        device=args.device,
    )
    
    print("\n✅ Residuals computed successfully!")
    return True


def load_dataset_with_residuals(args):
    """Step 2: Load dataset with residuals"""
    print("\n" + "="*80)
    print("STEP 2: Loading Dataset with Residuals")
    print("="*80)
    
    residuals_path = args.output_dir / "residuals_residuals.npy"
    
    if not residuals_path.exists():
        print(f"❌ Residuals not found: {residuals_path}")
        print("   Run Step 1 first to compute residuals")
        return None
    
    print(f"\nLoading dataset:")
    print(f"  Data: {args.data_path}")
    print(f"  Residuals: {residuals_path}")
    
    dataset = TiffVolumeDataset(
        data_path=args.data_path,
        volume_size=args.volume_size,
        stride=args.stride,
        num_frames=args.num_frames,
        use_heic_compression=True,
        heic_quality=args.heic_quality,
        dual_channel=True,
        use_residuals=True,
        residuals_path=str(residuals_path),
        normalize=True,
        global_normalize=True,
    )
    
    print(f"\n✅ Dataset loaded: {len(dataset)} sub-volumes")
    print(f"   Shape per sample: (3, {args.volume_size}, {args.volume_size}, {args.volume_size})")
    
    # Show sample statistics
    sample, _ = dataset[0]
    print(f"\n   Channel 0 (TIFF):     range=[{sample[0].min():.3f}, {sample[0].max():.3f}]")
    print(f"   Channel 1 (HEIC):     range=[{sample[1].min():.3f}, {sample[1].max():.3f}]")
    print(f"   Channel 2 (Residual): range=[{sample[2].min():.3f}, {sample[2].max():.3f}]")
    
    return dataset


def train_residual_correction(args, dataset):
    """Step 3: Train a simple residual correction model"""
    print("\n" + "="*80)
    print("STEP 3: Training Residual Correction Model")
    print("="*80)
    
    if dataset is None:
        print("❌ No dataset provided")
        return
    
    # Create model
    model = SimpleResidualCorrector(in_channels=2, hidden_dim=16)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"\nModel: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Device: {device}")
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )
    
    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    print(f"\nTraining for {args.num_epochs} epochs...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: 1e-4")
    
    # Training loop
    model.train()
    for epoch in range(args.num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (batch_volumes, batch_positions) in enumerate(loader):
            # Extract channels
            tiff = batch_volumes[:, 0:1].to(device)       # (B, 1, D, H, W)
            heic = batch_volumes[:, 1:2].to(device)       # (B, 1, D, H, W)
            residual = batch_volumes[:, 2:3].to(device)   # (B, 1, D, H, W)
            
            # Forward pass: predict correction
            correction = model(heic, residual)
            
            # Apply correction
            corrected_tiff = heic + correction
            
            # Compute loss
            loss = criterion(corrected_tiff, tiff)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Epoch {epoch+1}/{args.num_epochs}, "
                      f"Batch {batch_idx}/{len(loader)}, "
                      f"Loss: {loss.item():.6f}")
        
        avg_loss = total_loss / num_batches
        print(f"\n  Epoch {epoch+1} completed: Avg Loss = {avg_loss:.6f}\n")
    
    print("✅ Training completed!")
    
    # Save model
    output_path = args.output_dir / "residual_corrector.pth"
    torch.save(model.state_dict(), output_path)
    print(f"   Model saved to: {output_path}")
    
    return model


def evaluate_model(model, dataset, args):
    """Step 4: Evaluate the trained model"""
    print("\n" + "="*80)
    print("STEP 4: Evaluating Model")
    print("="*80)
    
    if model is None or dataset is None:
        print("❌ Model or dataset not available")
        return
    
    model.eval()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Evaluate on a few samples
    num_samples = min(10, len(dataset))
    
    total_heic_error = 0
    total_corrected_error = 0
    
    print(f"\nEvaluating on {num_samples} samples...")
    
    with torch.no_grad():
        for idx in range(num_samples):
            sample, _ = dataset[idx]
            
            tiff = sample[0:1].unsqueeze(0).to(device)
            heic = sample[1:2].unsqueeze(0).to(device)
            residual = sample[2:3].unsqueeze(0).to(device)
            
            # Compute correction
            correction = model(heic, residual)
            corrected = heic + correction
            
            # Compute errors
            heic_error = torch.mean((heic - tiff) ** 2).item()
            corrected_error = torch.mean((corrected - tiff) ** 2).item()
            
            total_heic_error += heic_error
            total_corrected_error += corrected_error
    
    avg_heic_error = total_heic_error / num_samples
    avg_corrected_error = total_corrected_error / num_samples
    improvement = (avg_heic_error - avg_corrected_error) / avg_heic_error * 100
    
    print(f"\nResults (MSE on {num_samples} samples):")
    print(f"  HEIC error:       {avg_heic_error:.6f}")
    print(f"  Corrected error:  {avg_corrected_error:.6f}")
    print(f"  Improvement:      {improvement:.2f}%")
    
    if improvement > 0:
        print("\n✅ Model successfully learned to correct residuals!")
    else:
        print("\n⚠️  Model needs more training or different architecture")


def main():
    parser = argparse.ArgumentParser(
        description="Complete workflow for residual computation and training"
    )
    
    # Data parameters
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to TIFF data directory")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to trained HEIC-to-TIFF model checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/residual_workflow",
                       help="Output directory for residuals and trained model")
    
    # Volume parameters
    parser.add_argument("--volume_size", type=int, default=64,
                       help="Size of sub-volumes")
    parser.add_argument("--stride", type=int, default=64,
                       help="Stride for sub-volume extraction")
    parser.add_argument("--num_frames", type=int, default=100,
                       help="Number of frames to process")
    parser.add_argument("--heic_quality", type=int, default=85,
                       help="HEIC compression quality")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use (cuda/cpu)")
    
    # Workflow control
    parser.add_argument("--skip_compute", action="store_true",
                       help="Skip residual computation (use existing)")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training (only compute residuals)")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir = Path(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("RESIDUAL CORRECTION WORKFLOW")
    print("="*80)
    print(f"Data: {args.data_path}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Output: {args.output_dir}")
    
    # Step 1: Compute residuals
    if not args.skip_compute:
        success = compute_residuals_step(args)
        if not success:
            return
    
    # Step 2: Load dataset with residuals
    dataset = load_dataset_with_residuals(args)
    
    if args.skip_training:
        print("\n✅ Residuals computed. Training skipped.")
        return
    
    if dataset is None:
        return
    
    # Step 3: Train correction model
    model = train_residual_correction(args, dataset)
    
    # Step 4: Evaluate
    evaluate_model(model, dataset, args)
    
    print("\n" + "="*80)
    print("🎉 WORKFLOW COMPLETED!")
    print("="*80)
    print(f"\nFiles created:")
    print(f"  {args.output_dir / 'residuals_residuals.npy'}")
    print(f"  {args.output_dir / 'residuals_positions.npy'}")
    print(f"  {args.output_dir / 'residuals_metadata.npz'}")
    print(f"  {args.output_dir / 'residual_corrector.pth'}")


if __name__ == "__main__":
    main()
