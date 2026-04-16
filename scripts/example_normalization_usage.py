#!/usr/bin/env python3
"""
Example: Loading and Using Normalization Metadata for Reconstruction

This script demonstrates how to load normalization metadata saved during
compression and use it to reconstruct original pixel values from compressed data.
"""

from pathlib import Path
import numpy as np
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sdate.utils.normalization import (
    load_normalization_metadata,
    denormalize_frame,
    get_normalization_info,
    find_normalization_file
)


def example_1_load_and_display():
    """Example 1: Load normalization metadata and display information."""
    print("=" * 80)
    print("EXAMPLE 1: Load and Display Normalization Metadata")
    print("=" * 80)
    
    # Example paths (adjust to your actual files)
    npz_path = "data/streaming_output/file_3_extracted_q90/file_3_extracted_q90_projections.npz"
    
    if not Path(npz_path).exists():
        print(f"\n⚠️  File not found: {npz_path}")
        print("Please adjust the path to point to an actual normalization file.")
        return
    
    # Load metadata
    print(f"\nLoading: {npz_path}\n")
    metadata = load_normalization_metadata(npz_path)
    
    # Display formatted info
    print(get_normalization_info(npz_path))
    
    print("\n✅ Metadata loaded successfully!")
    print(f"Use per-frame normalization: {metadata['use_per_frame']}")


def example_2_denormalize_frames():
    """Example 2: Denormalize compressed frames to original values."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Denormalize Compressed Frames")
    print("=" * 80)
    
    # Simulate normalized frames (in practice, these come from decompressed video)
    num_frames = 10
    height, width = 100, 100
    
    # Create synthetic normalized frames in [0, 1]
    normalized_frames = np.random.rand(num_frames, height, width).astype(np.float32)
    
    # Create example metadata (simulating per-frame normalization)
    metadata = {
        'use_per_frame': True,
        'global_min': 100.0,
        'per_frame_max': np.linspace(5000, 6000, num_frames),  # Varying per-frame max
        'percentile': 99.0
    }
    
    print(f"\nDenormalizing {num_frames} frames...")
    print(f"Global min: {metadata['global_min']}")
    print(f"Per-frame max range: [{metadata['per_frame_max'].min():.1f}, {metadata['per_frame_max'].max():.1f}]")
    
    # Denormalize each frame
    denormalized_frames = []
    for i in range(num_frames):
        denormalized = denormalize_frame(normalized_frames[i], metadata, frame_idx=i)
        denormalized_frames.append(denormalized)
        
        if i < 3:  # Show first 3 frames
            print(f"\nFrame {i}:")
            print(f"  Normalized range: [{normalized_frames[i].min():.3f}, {normalized_frames[i].max():.3f}]")
            print(f"  Denormalized range: [{denormalized.min():.1f}, {denormalized.max():.1f}]")
            print(f"  Expected max: {metadata['per_frame_max'][i]:.1f}")
    
    print(f"\n✅ Successfully denormalized {num_frames} frames!")


def example_3_auto_find_normalization():
    """Example 3: Automatically find normalization file for a video."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Auto-Find Normalization File")
    print("=" * 80)
    
    # Example video path
    video_path = "data/streaming_output/file_3_extracted_q90/file_3_extracted_q90_projections.mov"
    
    print(f"\nVideo file: {video_path}")
    
    # Find corresponding normalization file
    npz_path = find_normalization_file(video_path)
    
    if npz_path:
        print(f"✅ Found normalization file: {npz_path}")
        
        # Load and display info
        if npz_path.exists():
            print("\n" + get_normalization_info(npz_path))
    else:
        print("⚠️  No normalization file found for this video.")


def example_4_reconstruction_workflow():
    """Example 4: Complete reconstruction workflow."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Complete Reconstruction Workflow")
    print("=" * 80)
    
    print("""
Typical reconstruction workflow:

1. Decompress the HEVC video file to get normalized frames (0-1 range)
   
2. Load the corresponding normalization metadata:
   >>> metadata = load_normalization_metadata("video_file.npz")
   
3. For each frame, denormalize to original pixel values:
   >>> for i, normalized_frame in enumerate(decompressed_frames):
   ...     original = denormalize_frame(normalized_frame, metadata, frame_idx=i)
   
4. Save or process the reconstructed frames

Key considerations:
- Per-frame normalization provides better dynamic range preservation
- The 99th percentile normalization reduces impact of outliers
- Normalization metadata is essential for accurate reconstruction
- Each data type (darks, flats, projections) has its own normalization file
""")
    
    print("✅ See the examples above for implementation details.")


def main():
    """Run all examples."""
    print("NORMALIZATION METADATA USAGE EXAMPLES")
    print("=" * 80)
    print()
    
    # Run examples
    example_1_load_and_display()
    example_2_denormalize_frames()
    example_3_auto_find_normalization()
    example_4_reconstruction_workflow()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
