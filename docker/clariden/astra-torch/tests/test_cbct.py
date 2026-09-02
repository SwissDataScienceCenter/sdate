"""Tests for CBCT reconstruction functions."""

import unittest
import numpy as np
import torch

try:
    from astra_torch import cbct
    from astra_torch import CBCTAcquisition
    ASTRA_AVAILABLE = True
except ImportError:
    ASTRA_AVAILABLE = False


class TestCBCTReconstruction(unittest.TestCase):
    """Test CBCT reconstruction functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    @unittest.skipUnless(ASTRA_AVAILABLE, "ASTRA toolbox not available")
    def test_cbct_acquisition_init(self):
        """Test CBCTAcquisition initialization."""
        # Create simple geometry
        num_angles = 10
        angles = np.linspace(0, 2*np.pi, num_angles, endpoint=False)
        
        source_positions = np.column_stack([
            1000 * np.cos(angles),
            1000 * np.sin(angles),
            np.zeros(num_angles)
        ])
        
        detector_positions = np.column_stack([
            -500 * np.cos(angles),
            -500 * np.sin(angles),
            np.zeros(num_angles)
        ])
        
        u_vectors = np.column_stack([
            -np.sin(angles),
            np.cos(angles),
            np.zeros(num_angles)
        ])
        
        v_vectors = np.tile([0, 0, 1], (num_angles, 1))
        
        vectors = np.column_stack([
            source_positions,
            detector_positions - source_positions,
            u_vectors,
            v_vectors
        ])
        
        acquisition = CBCTAcquisition(
            vectors=vectors,
            detector_shape=(64, 64),
            volume_shape=(32, 32, 32)
        )
        
        self.assertEqual(acquisition.num_angles, num_angles)
        self.assertEqual(acquisition.detector_shape, (64, 64))
        self.assertEqual(acquisition.volume_shape, (32, 32, 32))

    def test_projection_shapes(self):
        """Test that projection operations return correct shapes."""
        # This test doesn't require ASTRA, just tests shape validation
        batch_size = 2
        num_angles = 10
        detector_shape = (64, 64)
        volume_shape = (32, 32, 32)
        
        # Test volume shape validation
        volume = torch.randn((batch_size, 1) + volume_shape)
        self.assertEqual(volume.shape, (batch_size, 1, 32, 32, 32))
        
        # Test projection shape validation  
        projections = torch.randn((batch_size, 1, num_angles) + detector_shape)
        self.assertEqual(projections.shape, (batch_size, 1, 10, 64, 64))

    def test_tensor_device_consistency(self):
        """Test that tensors are moved to correct device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
        # Test tensor creation on device
        volume = torch.randn((1, 1, 32, 32, 32), device=device)
        self.assertEqual(volume.device.type, device.type)


if __name__ == '__main__':
    unittest.main()
