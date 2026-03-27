#!/usr/bin/env python
"""
Simple tests for INCT library (no pytest required).
Run with: python tests/test_inct_simple.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch


def test_hash_encoding():
    """Test hash encoding modules."""
    print("Testing hash encoding...")
    from inct import HashEncoding, MultiResolutionHashEncoding
    
    # Single level
    encoding = HashEncoding(n_dims=3, n_features=2, resolution=16, table_size=2**14)
    coords = torch.rand(100, 3)
    features = encoding(coords)
    assert features.shape == (100, 2), f"Expected (100, 2), got {features.shape}"
    assert not torch.isnan(features).any(), "NaN values in features"
    print("  ✅ Single-level encoding works")
    
    # Multi-resolution
    mr_encoding = MultiResolutionHashEncoding(
        n_dims=3, n_levels=8, n_features_per_level=2,
        base_resolution=16, max_resolution=256, table_size=2**14
    )
    mr_features = mr_encoding(coords)
    expected_dim = 8 * 2 + 3  # levels * features + input coords
    assert mr_features.shape == (100, expected_dim), f"Expected (100, {expected_dim}), got {mr_features.shape}"
    print("  ✅ Multi-resolution encoding works")
    
    # 4D encoding
    enc_4d = HashEncoding(n_dims=4, n_features=2, resolution=16, table_size=2**14)
    coords_4d = torch.rand(50, 4)
    features_4d = enc_4d(coords_4d)
    assert features_4d.shape == (50, 2), f"Expected (50, 2), got {features_4d.shape}"
    print("  ✅ 4D encoding works")
    
    print("  ✅ Hash encoding tests passed!")


def test_model():
    """Test InstantNGP model."""
    print("\nTesting model...")
    from inct import InstantNGPModel
    
    model = InstantNGPModel(
        n_dims=3, n_levels=4, n_features_per_level=2,
        base_resolution=16, max_resolution=64, table_size=2**10,
        hidden_dims=[32, 32]
    )
    
    # Forward pass
    coords = torch.rand(100, 3)
    output = model(coords)
    assert output.shape == (100, 1), f"Expected (100, 1), got {output.shape}"
    assert (output >= 0).all() and (output <= 1).all(), "Output not in [0, 1]"
    print("  ✅ Forward pass works")
    
    # Volume prediction
    volume = model.predict_volume(shape=(16, 16, 4), batch_size=256)
    assert volume.shape == (16, 16, 4), f"Expected (16, 16, 4), got {volume.shape}"
    print("  ✅ Volume prediction works")
    
    # Save and load
    import tempfile
    import os
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "model.pt")
        model.save(save_path)
        
        loaded_model = InstantNGPModel.load(save_path)
        
        with torch.no_grad():
            out1 = model(coords)
            out2 = loaded_model(coords)
        
        assert torch.allclose(out1, out2), "Loaded model gives different output"
    print("  ✅ Save/load works")
    
    print("  ✅ Model tests passed!")


def test_dataset():
    """Test dataset classes."""
    print("\nTesting datasets...")
    from inct import VoxelDataset, BatchVoxelDataset
    
    volume = torch.rand(32, 32, 8)
    
    # VoxelDataset
    dataset = VoxelDataset(volume)
    sample = dataset[0]
    assert 'coords' in sample and 'values' in sample
    assert sample['coords'].shape == (3,)
    print("  ✅ VoxelDataset works")
    
    # BatchVoxelDataset
    batch_dataset = BatchVoxelDataset(volume, batch_size=256, n_batches=10)
    assert len(batch_dataset) == 10
    batch = batch_dataset[0]
    assert batch['coords'].shape == (256, 3)
    assert batch['values'].shape == (256, 1)
    print("  ✅ BatchVoxelDataset works")
    
    # Normalization
    volume2 = torch.rand(32, 32, 8) * 100 + 50  # Range [50, 150]
    norm_dataset = BatchVoxelDataset(volume2, batch_size=256, n_batches=1, normalize_values=True)
    batch = norm_dataset[0]
    assert batch['values'].min() >= 0 and batch['values'].max() <= 1
    print("  ✅ Normalization works")
    
    print("  ✅ Dataset tests passed!")


def test_trainer():
    """Test training infrastructure."""
    print("\nTesting trainer...")
    from inct import InstantNGPModel, BatchVoxelDataset, Trainer, TrainingConfig
    from torch.utils.data import DataLoader
    
    model = InstantNGPModel(
        n_dims=3, n_levels=4, n_features_per_level=2,
        base_resolution=16, max_resolution=64, table_size=2**10,
        hidden_dims=[32, 32]
    )
    
    config = TrainingConfig(num_epochs=1, checkpoint_dir='/tmp/inct_test')
    trainer = Trainer(model, config, device=torch.device('cpu'))
    
    # Training step
    batch = {'coords': torch.rand(64, 3), 'values': torch.rand(64, 1)}
    loss = trainer.train_step(batch)
    assert isinstance(loss, float) and loss >= 0
    print("  ✅ Training step works")
    
    # Evaluation
    volume = torch.rand(16, 16, 4)
    dataset = BatchVoxelDataset(volume, batch_size=64, n_batches=5)
    loader = DataLoader(dataset, batch_size=None)
    
    metrics = trainer.evaluate(loader, n_batches=2)
    assert 'loss' in metrics and 'psnr' in metrics
    print("  ✅ Evaluation works")
    
    print("  ✅ Trainer tests passed!")


def test_utils():
    """Test utility functions."""
    print("\nTesting utils...")
    from inct import psnr, mse
    
    # PSNR
    assert psnr(0.01) > psnr(0.1)  # Lower MSE = higher PSNR
    assert psnr(0.0) == float('inf')
    print("  ✅ PSNR works")
    
    # MSE
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 3.0])
    assert mse(pred, target) == 0.0
    
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([2.0, 3.0, 4.0])
    assert mse(pred, target) == 1.0
    print("  ✅ MSE works")
    
    print("  ✅ Utils tests passed!")


def test_end_to_end():
    """Test end-to-end training."""
    print("\nTesting end-to-end training...")
    from inct import InstantNGPModel, BatchVoxelDataset, Trainer, TrainingConfig, psnr
    from torch.utils.data import DataLoader
    
    # Create test volume
    volume = torch.rand(32, 32, 4)
    
    # Normalize
    vmin, vmax = volume.min(), volume.max()
    volume_norm = (volume - vmin) / (vmax - vmin + 1e-8)
    
    # Create dataset and model
    dataset = BatchVoxelDataset(volume_norm, batch_size=1024, n_batches=50)
    loader = DataLoader(dataset, batch_size=None)
    
    model = InstantNGPModel(
        n_dims=3, n_levels=6, n_features_per_level=2,
        base_resolution=16, max_resolution=64, table_size=2**12,
        hidden_dims=[32, 32]
    )
    
    # Train briefly
    config = TrainingConfig(
        num_epochs=5,
        log_interval=100,
        eval_interval=10,
        checkpoint_dir='/tmp/inct_e2e_test'
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(model, config, device=device)
    
    history = trainer.train(loader, loader, verbose=False)
    
    # Reconstruct
    model.eval()
    recon = model.predict_volume(volume_norm.shape, device=device)
    
    # Check quality
    mse_val = ((recon.cpu() - volume_norm) ** 2).mean().item()
    psnr_val = psnr(mse_val)
    
    print(f"  Final PSNR: {psnr_val:.2f} dB")
    assert psnr_val > 20, f"PSNR too low: {psnr_val}"
    print("  ✅ End-to-end test passed!")


def main():
    print("=" * 60)
    print("INCT Library Tests")
    print("=" * 60)
    
    try:
        test_hash_encoding()
        test_model()
        test_dataset()
        test_trainer()
        test_utils()
        test_end_to_end()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("=" * 60)
        return 0
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
