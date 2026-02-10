#!/usr/bin/env python3
"""
Test script for residual computation and loading functionality.
"""

import sys
sys.path.append('/myhome/sdate')

import torch
import numpy as np
from pathlib import Path
from sdate.datasets import TiffVolumeDataset


def test_dataset_with_residuals():
    """Test loading dataset with pre-computed residuals."""
    
    print("="*80)
    print("TEST: TiffVolumeDataset with Residuals")
    print("="*80)
    
    # This test assumes you have:
    # 1. TIFF data in /myhome/data/sdate/shared/compression_paper/file_3_extracted
    # 2. Pre-computed residuals file
    
    data_path = "/myhome/data/sdate/shared/compression_paper/file_3_extracted"
    residuals_path = "/myhome/sdate/outputs/residuals/file_3_residuals.npy"
    
    # Check if files exist
    if not Path(data_path).exists():
        print(f"❌ Data path not found: {data_path}")
        print("   Skipping test.")
        return False
    
    if not Path(residuals_path).exists():
        print(f"⚠️  Residuals not found: {residuals_path}")
        print("   Compute residuals first with:")
        print(f"   python scripts/compute_residuals.py \\")
        print(f"       --data_path={data_path} \\")
        print(f"       --checkpoint_path=outputs/heic_to_tiff/checkpoint-final \\")
        print(f"       --output_path={residuals_path}")
        return False
    
    try:
        # Load dataset with residuals
        print("\n1. Loading dataset with residuals...")
        dataset = TiffVolumeDataset(
            data_path=data_path,
            volume_size=64,
            stride=64,
            num_frames=100,
            use_heic_compression=True,
            heic_quality=85,
            dual_channel=True,
            use_residuals=True,
            residuals_path=residuals_path,
            normalize=True,
            global_normalize=True,
        )
        print(f"   ✓ Dataset loaded: {len(dataset)} sub-volumes")
        
        # Get a sample
        print("\n2. Testing data access...")
        sub_volume, position = dataset[0]
        print(f"   ✓ Sub-volume shape: {sub_volume.shape}")
        print(f"   ✓ Position: {position.tolist()}")
        
        # Verify channel structure
        print("\n3. Verifying channel structure...")
        assert sub_volume.shape[0] == 3, f"Expected 3 channels, got {sub_volume.shape[0]}"
        assert sub_volume.shape[1:] == (64, 64, 64), f"Expected (64,64,64) spatial dims, got {sub_volume.shape[1:]}"
        
        tiff_channel = sub_volume[0]
        heic_channel = sub_volume[1]
        residual_channel = sub_volume[2]
        
        print(f"   ✓ TIFF channel shape: {tiff_channel.shape}")
        print(f"   ✓ HEIC channel shape: {heic_channel.shape}")
        print(f"   ✓ Residual channel shape: {residual_channel.shape}")
        
        # Compute statistics
        print("\n4. Channel statistics:")
        print(f"   TIFF    - range: [{tiff_channel.min():.4f}, {tiff_channel.max():.4f}], "
              f"mean: {tiff_channel.mean():.4f}, std: {tiff_channel.std():.4f}")
        print(f"   HEIC    - range: [{heic_channel.min():.4f}, {heic_channel.max():.4f}], "
              f"mean: {heic_channel.mean():.4f}, std: {heic_channel.std():.4f}")
        print(f"   Residual - range: [{residual_channel.min():.4f}, {residual_channel.max():.4f}], "
              f"mean: {residual_channel.mean():.4f}, std: {residual_channel.std():.4f}")
        
        # Test multiple samples
        print("\n5. Testing multiple samples...")
        for idx in [0, len(dataset)//2, len(dataset)-1]:
            sub_vol, pos = dataset[idx]
            assert sub_vol.shape == (3, 64, 64, 64), f"Wrong shape at index {idx}"
        print(f"   ✓ Tested samples at indices: 0, {len(dataset)//2}, {len(dataset)-1}")
        
        # Test DataLoader
        print("\n6. Testing with DataLoader...")
        from torch.utils.data import DataLoader
        
        loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
        batch_volumes, batch_positions = next(iter(loader))
        
        print(f"   ✓ Batch volumes shape: {batch_volumes.shape}")
        print(f"   ✓ Batch positions shape: {batch_positions.shape}")
        assert batch_volumes.shape == (4, 3, 64, 64, 64), "Wrong batch shape"
        
        print("\n" + "="*80)
        print("✅ ALL TESTS PASSED!")
        print("="*80)
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_without_residuals():
    """Test that dataset works without residuals (backward compatibility)."""
    
    print("\n" + "="*80)
    print("TEST: TiffVolumeDataset WITHOUT Residuals (backward compatibility)")
    print("="*80)
    
    data_path = "/myhome/data/sdate/shared/compression_paper/file_3_extracted"
    
    if not Path(data_path).exists():
        print(f"❌ Data path not found: {data_path}")
        return False
    
    try:
        # Load dataset without residuals
        print("\n1. Loading dataset without residuals...")
        dataset = TiffVolumeDataset(
            data_path=data_path,
            volume_size=64,
            stride=64,
            num_frames=100,
            use_heic_compression=True,
            heic_quality=85,
            dual_channel=True,
            use_residuals=False,  # No residuals
            normalize=True,
        )
        print(f"   ✓ Dataset loaded: {len(dataset)} sub-volumes")
        
        # Get a sample
        print("\n2. Testing data access...")
        sub_volume, position = dataset[0]
        print(f"   ✓ Sub-volume shape: {sub_volume.shape}")
        
        # Verify only 2 channels
        assert sub_volume.shape[0] == 2, f"Expected 2 channels, got {sub_volume.shape[0]}"
        print("   ✓ Correct number of channels (2: TIFF + HEIC)")
        
        print("\n✅ BACKWARD COMPATIBILITY TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = True
    
    # Test with residuals
    success &= test_dataset_with_residuals()
    
    # Test backward compatibility
    success &= test_dataset_without_residuals()
    
    if success:
        print("\n" + "="*80)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("="*80)
        sys.exit(0)
    else:
        print("\n" + "="*80)
        print("⚠️  SOME TESTS FAILED")
        print("="*80)
        sys.exit(1)
