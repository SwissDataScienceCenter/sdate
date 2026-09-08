"""
Example usage of TiffVolumeDataset

This script demonstrates how to use the TiffVolumeDataset class to:
1. Load a sequence of TIFF files as a 3D volume
2. Split the volume into overlapping sub-volumes
3. Use the dataset with PyTorch DataLoader
"""

import torch
from torch.utils.data import DataLoader
from sdate.datasets import TiffVolumeDataset

# Example 1: Basic usage with default settings
print("Example 1: Basic usage")
print("=" * 60)

# Create dataset
dataset = TiffVolumeDataset(
    data_path='./data/ct_files/file_1_extracted',  # Path to TIFF folder
    num_frames=100,                                # Load 100 frames
    volume_size=64,                                # Create 64x64x64 sub-volumes
    stride=64,                                     # No overlap (stride = volume_size)
)

print(f"\nDataset info:")
print(f"  Total sub-volumes: {len(dataset)}")
print(f"  Full volume shape: {dataset.volume.shape}")
print(f"  Sub-volume shape: {dataset[0].shape}")

# Example 2: Overlapping volumes
print("\n\nExample 2: Overlapping volumes (50% overlap)")
print("=" * 60)

dataset_overlap = TiffVolumeDataset(
    data_path='./data/ct_files/file_1_extracted',
    num_frames=100,
    volume_size=64,
    stride=32,  # 50% overlap (stride = volume_size / 2)
    normalize=True,
    global_normalize=True
)

print(f"\nDataset with overlap:")
print(f"  Total sub-volumes: {len(dataset_overlap)}")
print(f"  Overlap: {((dataset_overlap.volume_size - dataset_overlap.stride) / dataset_overlap.volume_size * 100):.0f}%")

# Get info about a specific sub-volume
info = dataset_overlap.get_volume_info(0)
print(f"\nSub-volume 0 info:")
print(f"  Position: {info['position']}")
print(f"  End position: {info['end_position']}")
print(f"  Shape: {info['shape']}")

# Example 3: Use with DataLoader
print("\n\nExample 3: Using with PyTorch DataLoader")
print("=" * 60)

loader = DataLoader(
    dataset_overlap,
    batch_size=4,
    shuffle=True,
    num_workers=0
)

print(f"\nDataLoader created:")
print(f"  Batch size: {loader.batch_size}")
print(f"  Total batches: {len(loader)}")

# Process one batch
for batch_idx, batch in enumerate(loader):
    print(f"\nBatch {batch_idx}:")
    print(f"  Shape: {batch.shape}")  # Should be (4, 64, 64, 64)
    print(f"  Min value: {batch.min():.4f}")
    print(f"  Max value: {batch.max():.4f}")
    print(f"  Mean value: {batch.mean():.4f}")
    break  # Just show first batch

# Example 4: Advanced usage with custom parameters
print("\n\nExample 4: Advanced usage")
print("=" * 60)

dataset_advanced = TiffVolumeDataset(
    data_path='./data/ct_files/file_1_extracted',
    num_frames=50,              # Load fewer frames
    volume_size=32,             # Smaller sub-volumes
    stride=16,                  # 50% overlap
    start_offset=10,            # Start from frame 10
    clip_range=(0, 1000),       # Clip values
    normalize=True,             # Normalize to [0, 1]
    global_normalize=True,      # Use global min/max
    dtype=torch.float32         # Data type
)

print(f"\nAdvanced dataset:")
print(f"  Frames loaded: {dataset_advanced.num_frames}")
print(f"  Start offset: {dataset_advanced.start_offset}")
print(f"  Volume shape: {dataset_advanced.volume.shape}")
print(f"  Sub-volumes: {len(dataset_advanced)}")
print(f"  Clip range: {dataset_advanced.clip_range}")

# Example 5: Volume reconstruction
print("\n\nExample 5: Volume reconstruction from sub-volumes")
print("=" * 60)

# Simulate processing all sub-volumes (e.g., with a model)
predictions = []
for idx in range(len(dataset_overlap)):
    # In practice, you would pass dataset_overlap[idx] through a model
    # Here we just use the original sub-volume as "prediction"
    predictions.append(dataset_overlap[idx])

# Reconstruct the full volume
reconstructed = dataset_overlap.reconstruct_volume(predictions, aggregation='mean')

print(f"\nReconstruction:")
print(f"  Original volume shape: {dataset_overlap.volume.shape}")
print(f"  Reconstructed shape: {reconstructed.shape}")
print(f"  Max difference: {torch.abs(dataset_overlap.volume - reconstructed).max():.6f}")

print("\n" + "=" * 60)
print("Examples completed!")
