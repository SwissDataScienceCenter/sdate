"""
Tests for INCT library.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from inct import (
    HashEncoding,
    MultiResolutionHashEncoding,
    InstantNGPModel,
    TinyMLP,
    BatchVoxelDataset,
    VoxelDataset,
    Trainer,
    TrainingConfig,
    psnr,
    mse,
)


class TestHashEncoding:
    """Tests for hash encoding modules."""
    
    def test_single_level_encoding(self):
        """Test single-level hash encoding."""
        encoding = HashEncoding(n_dims=3, n_features=2, resolution=16, table_size=2**14)
        coords = torch.rand(100, 3)
        features = encoding(coords)
        
        assert features.shape == (100, 2)
        assert not torch.isnan(features).any()
    
    def test_multi_resolution_encoding(self):
        """Test multi-resolution hash encoding."""
        encoding = MultiResolutionHashEncoding(
            n_dims=3, n_levels=8, n_features_per_level=2,
            base_resolution=16, max_resolution=256, table_size=2**14
        )
        coords = torch.rand(100, 3)
        features = encoding(coords)
        
        # Output dim = n_levels * n_features + n_dims (if include_input)
        expected_dim = 8 * 2 + 3
        assert features.shape == (100, expected_dim)
        assert encoding.output_dim == expected_dim
    
    def test_resolution_progression(self):
        """Test that resolutions follow geometric progression."""
        encoding = MultiResolutionHashEncoding(
            n_dims=3, n_levels=4, n_features_per_level=2,
            base_resolution=16, max_resolution=128, table_size=2**14
        )
        
        resolutions = encoding.resolutions
        assert resolutions[0] == 16
        assert resolutions[-1] <= 128
        
        # Check geometric progression
        ratios = [resolutions[i+1] / resolutions[i] for i in range(len(resolutions)-1)]
        assert all(r > 1 for r in ratios)  # Increasing
    
    def test_4d_encoding(self):
        """Test encoding with 4D coordinates."""
        encoding = HashEncoding(n_dims=4, n_features=2, resolution=16, table_size=2**14)
        coords = torch.rand(50, 4)
        features = encoding(coords)
        
        assert features.shape == (50, 2)


class TestTinyMLP:
    """Tests for MLP module."""
    
    def test_forward_pass(self):
        """Test MLP forward pass."""
        mlp = TinyMLP(input_dim=32, output_dim=1, hidden_dims=[64, 64])
        x = torch.rand(100, 32)
        y = mlp(x)
        
        assert y.shape == (100, 1)
    
    def test_output_activation(self):
        """Test output activation."""
        mlp = TinyMLP(input_dim=32, output_dim=1, hidden_dims=[64], output_activation='sigmoid')
        x = torch.rand(100, 32)
        y = mlp(x)
        
        assert (y >= 0).all() and (y <= 1).all()


class TestInstantNGPModel:
    """Tests for the complete model."""
    
    def test_forward_pass(self):
        """Test model forward pass."""
        model = InstantNGPModel(
            n_dims=3, n_levels=4, n_features_per_level=2,
            base_resolution=16, max_resolution=64, table_size=2**10,
            hidden_dims=[32, 32]
        )
        coords = torch.rand(100, 3)
        output = model(coords)
        
        assert output.shape == (100, 1)
        assert (output >= 0).all() and (output <= 1).all()  # Sigmoid output
    
    def test_volume_prediction(self):
        """Test volume reconstruction."""
        model = InstantNGPModel(
            n_dims=3, n_levels=4, n_features_per_level=2,
            base_resolution=16, max_resolution=64, table_size=2**10,
            hidden_dims=[32, 32]
        )
        
        volume = model.predict_volume(shape=(16, 16, 4), batch_size=256)
        assert volume.shape == (16, 16, 4)
    
    def test_save_load(self, tmp_path):
        """Test model save and load."""
        model = InstantNGPModel(
            n_dims=3, n_levels=4, n_features_per_level=2,
            base_resolution=16, max_resolution=64, table_size=2**10,
        )
        
        # Save
        save_path = tmp_path / "model.pt"
        model.save(str(save_path))
        
        # Load
        loaded_model = InstantNGPModel.load(str(save_path))
        
        # Test same output
        coords = torch.rand(10, 3)
        with torch.no_grad():
            out1 = model(coords)
            out2 = loaded_model(coords)
        
        assert torch.allclose(out1, out2)


class TestDataset:
    """Tests for dataset classes."""
    
    def test_voxel_dataset(self):
        """Test VoxelDataset."""
        volume = torch.rand(32, 32, 8)
        dataset = VoxelDataset(volume)
        
        sample = dataset[0]
        assert 'coords' in sample
        assert 'values' in sample
        assert sample['coords'].shape == (3,)
        assert sample['values'].shape == (1,)
    
    def test_batch_voxel_dataset(self):
        """Test BatchVoxelDataset."""
        volume = torch.rand(32, 32, 8)
        dataset = BatchVoxelDataset(volume, batch_size=256, n_batches=10)
        
        assert len(dataset) == 10
        
        batch = dataset[0]
        assert batch['coords'].shape == (256, 3)
        assert batch['values'].shape == (256, 1)
    
    def test_normalization(self):
        """Test value normalization."""
        volume = torch.rand(32, 32, 8) * 100 + 50  # Range [50, 150]
        dataset = BatchVoxelDataset(volume, batch_size=256, n_batches=1, normalize_values=True)
        
        batch = dataset[0]
        values = batch['values']
        
        # Should be normalized to [0, 1]
        assert values.min() >= 0
        assert values.max() <= 1


class TestTrainer:
    """Tests for training infrastructure."""
    
    def test_training_step(self):
        """Test single training step."""
        model = InstantNGPModel(
            n_dims=3, n_levels=4, n_features_per_level=2,
            base_resolution=16, max_resolution=64, table_size=2**10,
            hidden_dims=[32, 32]
        )
        
        config = TrainingConfig(num_epochs=1)
        trainer = Trainer(model, config, device=torch.device('cpu'))
        
        batch = {
            'coords': torch.rand(64, 3),
            'values': torch.rand(64, 1),
        }
        
        loss = trainer.train_step(batch)
        assert isinstance(loss, float)
        assert loss >= 0
    
    def test_evaluation(self):
        """Test model evaluation."""
        model = InstantNGPModel(
            n_dims=3, n_levels=4, n_features_per_level=2,
            base_resolution=16, max_resolution=64, table_size=2**10,
        )
        
        config = TrainingConfig(num_epochs=1)
        trainer = Trainer(model, config, device=torch.device('cpu'))
        
        volume = torch.rand(16, 16, 4)
        dataset = BatchVoxelDataset(volume, batch_size=64, n_batches=5)
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=None)
        
        metrics = trainer.evaluate(loader, n_batches=2)
        
        assert 'loss' in metrics
        assert 'psnr' in metrics
        assert 'mse' in metrics


class TestUtils:
    """Tests for utility functions."""
    
    def test_psnr(self):
        """Test PSNR calculation."""
        assert psnr(0.01) > psnr(0.1)  # Lower MSE = higher PSNR
        assert psnr(0.0) == float('inf')
    
    def test_mse(self):
        """Test MSE calculation."""
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        assert mse(pred, target) == 0.0
        
        pred = torch.tensor([1.0, 2.0, 3.0])
        target = torch.tensor([2.0, 3.0, 4.0])
        assert mse(pred, target) == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
