#!/usr/bin/env python3
"""
Compute and save residuals from trained HEIC to TIFF model.

This script:
1. Loads the trained model
2. Runs inference on all sub-volumes
3. Computes residuals (predicted - target)
4. Saves residuals in a format compatible with TiffVolumeDataset

The residuals are saved as a memory-mapped numpy array for efficient access.
"""

import sys
import os
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm.auto import tqdm

# Add project root to path
sys.path.append('/myhome/sdate')

from sdate.datasets import TiffVolumeDataset
from sdate.training.compression.train_heic_to_tiff import PositionalEncoder
from diffusers import UNet3DConditionModel


def compute_residuals(
    data_path: str,
    checkpoint_path: str,
    output_path: str,
    volume_size: int = 64,
    stride: int = 64,
    num_frames: int = 100,
    heic_quality: int = 85,
    batch_size: int = 4,
    start_offset: int = 0,
    device: str = None,
):
    """
    Compute residuals for all sub-volumes and save to disk.
    
    Args:
        data_path: Path to TIFF data directory
        checkpoint_path: Path to trained model checkpoint
        output_path: Path to save residuals (will create .npy file)
        volume_size: Size of sub-volumes
        stride: Stride for sub-volume extraction
        num_frames: Number of frames to process
        heic_quality: HEIC compression quality
        batch_size: Batch size for inference
        device: Device to use (cuda/cpu)
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("="*80)
    print("RESIDUAL COMPUTATION")
    print("="*80)
    print(f"Data path: {data_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print(f"Device: {device}")
    print()
    
    # 1. Load dataset
    print("Loading dataset...")
    dataset = TiffVolumeDataset(
        data_path=data_path,
        volume_size=volume_size,
        stride=stride,
        num_frames=num_frames,
        start_offset=start_offset,
        normalize=True,
        global_normalize=True,
        use_heic_compression=True,
        heic_quality=heic_quality,
        dual_channel=True,
        max_workers=8,
    )
    
    print(f"Dataset loaded: {len(dataset)} sub-volumes")
    print(f"Volume shape: {dataset.volume.shape}")
    print()
    
    # 2. Load model
    print("Loading trained model...")
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        # Try to find latest checkpoint
        output_dir = checkpoint_path.parent
        if output_dir.exists():
            checkpoints = sorted(output_dir.glob('checkpoint-*'))
            if checkpoints:
                checkpoint_path = checkpoints[-1]
                print(f"Using latest checkpoint: {checkpoint_path}")
    
    model = UNet3DConditionModel.from_pretrained(checkpoint_path / "unet")
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load positional encoder
    positional_encoder = PositionalEncoder(
        d_model=768,
        max_position=max(1000, volume_size * 4)
    )
    
    pos_encoder_path = checkpoint_path / "positional_encoder.pth"
    if pos_encoder_path.exists():
        positional_encoder.load_state_dict(torch.load(pos_encoder_path))
    
    positional_encoder = positional_encoder.to(device)
    positional_encoder.eval()
    print("Positional encoder loaded")
    print()
    
    # 3. Prepare output array
    print("Preparing output arrays...")
    num_subvolumes = len(dataset)
    
    # Create memory-mapped array for residuals
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if '.npy' not in output_path.suffix:
        output_path = output_path / ".npy"
    
    residuals_file = str(output_path).replace('.npy', '_residuals.npy')
    positions_file = str(output_path).replace('.npy', '_positions.npy')
    metadata_file = str(output_path).replace('.npy', '_metadata.npz')
    
    # Shape: (num_subvolumes, volume_size, volume_size, volume_size)
    residuals = np.lib.format.open_memmap(
        residuals_file,
        mode='w+',
        dtype=np.float32,
        shape=(num_subvolumes, volume_size, volume_size, volume_size)
    )
    
    # Store positions for each sub-volume
    positions = np.zeros((num_subvolumes, 3), dtype=np.int32)
    
    print(f"Output files:")
    print(f"  Residuals: {residuals_file}")
    print(f"  Positions: {positions_file}")
    print(f"  Metadata: {metadata_file}")
    print()
    
    # 4. Compute residuals
    print("Computing residuals...")
    num_batches = (num_subvolumes + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_subvolumes)
            
            # Load batch
            batch_volumes = []
            batch_positions = []
            
            for idx in range(start_idx, end_idx):
                dual_volume, position = dataset[idx]
                batch_volumes.append(dual_volume)
                batch_positions.append(position)
                
                # Store position
                positions[idx] = position.numpy()
            
            # Stack into batch
            batch_volumes = torch.stack(batch_volumes)  # (B, 2, D, H, W)
            batch_positions = torch.stack(batch_positions)  # (B, 3)
            
            # Split channels
            tiff_volumes = batch_volumes[:, 0:1]  # (B, 1, D, H, W) - target
            heic_volumes = batch_volumes[:, 1:2]  # (B, 1, D, H, W) - input
            
            # Move to device
            heic_volumes = heic_volumes.to(device)
            tiff_volumes = tiff_volumes.to(device)
            batch_positions = batch_positions.to(device)
            
            # Generate positional encodings
            encoder_hidden_states = positional_encoder(batch_positions)
            encoder_hidden_states = encoder_hidden_states.unsqueeze(1)

            combined_inputs = torch.cat([heic_volumes, torch.zeros_like(heic_volumes)], dim=1)
            
            # Run model
            predicted_tiff = model(
                sample=combined_inputs,
                timestep=torch.zeros(heic_volumes.shape[0], device=device),
                encoder_hidden_states=encoder_hidden_states,
            ).sample
            
            # Compute residuals: predicted - target
            batch_residuals = (predicted_tiff - tiff_volumes).squeeze(1).cpu().numpy()
            
            # Store residuals
            for i, idx in enumerate(range(start_idx, end_idx)):
                residuals[idx] = batch_residuals[i]
    
    # 5. Save metadata
    print("\nSaving metadata...")
    np.savez(
        metadata_file,
        volume_size=volume_size,
        stride=stride,
        num_frames=num_frames,
        heic_quality=heic_quality,
        num_subvolumes=num_subvolumes,
        full_volume_shape=dataset.volume.shape,
        global_min=dataset.global_min,
        global_max=dataset.global_max,
        data_path=str(data_path),
        checkpoint_path=str(checkpoint_path),
    )
    
    # Save positions
    np.save(positions_file, positions)
    
    # Flush residuals to disk
    del residuals  # Close memmap
    
    print("\n" + "="*80)
    print("RESIDUAL COMPUTATION COMPLETE!")
    print("="*80)
    print(f"\nFiles created:")
    print(f"  {residuals_file}")
    print(f"  {positions_file}")
    print(f"  {metadata_file}")
    print(f"\nStatistics:")
    
    # Load and compute statistics
    residuals_loaded = np.load(residuals_file, mmap_mode='r')
    print(f"  Residual shape: {residuals_loaded.shape}")
    print(f"  Residual mean: {residuals_loaded.mean():.6f}")
    print(f"  Residual std: {residuals_loaded.std():.6f}")
    print(f"  Residual min: {residuals_loaded.min():.6f}")
    print(f"  Residual max: {residuals_loaded.max():.6f}")
    
    print(f"\nTo use residuals with TiffVolumeDataset, set:")
    print(f"  use_residuals=True")
    print(f"  residuals_path='{residuals_file}'")


def main():
    parser = argparse.ArgumentParser(description="Compute and save residuals from trained model")
    
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to TIFF data directory")
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save residuals (.npy)")
    
    parser.add_argument("--volume_size", type=int, default=64,
                       help="Size of sub-volumes")
    parser.add_argument("--stride", type=int, default=64,
                       help="Stride for sub-volume extraction")
    parser.add_argument("--num_frames", type=int, default=100,
                       help="Number of frames to process")
    parser.add_argument("--heic_quality", type=int, default=50,
                       help="HEIC compression quality")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size for inference")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (cuda/cpu)")
    parser.add_argument("--start_offset", type=int, default=150,
                       help="Start offset for frame selection")
    
    args = parser.parse_args()
    
    
    compute_residuals(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint_path,
        output_path=args.output_path,
        volume_size=args.volume_size,
        stride=args.stride,
        start_offset=args.start_offset,
        num_frames=args.num_frames,
        heic_quality=args.heic_quality,
        batch_size=args.batch_size,
        device=args.device,
    )


if __name__ == "__main__":
    main()
