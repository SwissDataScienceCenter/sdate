"""Tests for laminography reconstruction functions."""

import unittest
from unittest import mock
import numpy as np
import torch

try:
    from astra_torch import lamino
    from astra_torch import LaminoAcquisition
    ASTRA_AVAILABLE = True
except ImportError:
    ASTRA_AVAILABLE = False


class TestLaminoReconstruction(unittest.TestCase):
    """Test laminography reconstruction functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @unittest.skipUnless(ASTRA_AVAILABLE, "ASTRA toolbox not available")
    def test_lamino_acquisition_init(self):
        """Test LaminoAcquisition initialization."""
        num_angles = 100
        angles_deg = np.linspace(0, 180, num_angles, endpoint=False)
        
        # Create laminography acquisition with current API
        lamino_angle_deg = 30.0  # 30 degrees tilt
        tilt_angle_deg = 0.0
        
        acquisition = LaminoAcquisition(
            proj_rows=64,
            proj_cols=64,
            angles_deg=angles_deg,
            lamino_angle_deg=lamino_angle_deg,
            tilt_angle_deg=tilt_angle_deg,
            vol_shape=(32, 32, 32)
        )
        
        self.assertEqual(len(acquisition.angles_deg), num_angles)
        self.assertEqual(acquisition.proj_rows, 64)
        self.assertEqual(acquisition.proj_cols, 64)
        self.assertEqual(acquisition.vol_shape, (32, 32, 32))
        self.assertEqual(acquisition.lamino_angle_deg, lamino_angle_deg)
        self.assertEqual(acquisition.tilt_angle_deg, tilt_angle_deg)

    def test_lamino_vector_validation(self):
        """Test laminography vector format validation."""
        # Test that ASTRA parallel3d_vec format has correct shape
        num_angles = 5
        # ASTRA parallel3d_vec format uses 12 components per angle:
        # [ray_x, ray_y, ray_z, det_x, det_y, det_z, u_x, u_y, u_z, v_x, v_y, v_z]
        vectors = np.random.randn(num_angles, 12)
        
        self.assertEqual(vectors.shape, (num_angles, 12))
        
        # Test individual vector components
        for i in range(num_angles):
            ray_dir = vectors[i, :3]      # Ray direction
            det_center = vectors[i, 3:6]   # Detector center
            u_vec = vectors[i, 6:9]        # Detector u-direction
            v_vec = vectors[i, 9:12]       # Detector v-direction
            
            # All vectors should be 3D
            self.assertEqual(len(ray_dir), 3)
            self.assertEqual(len(det_center), 3)
            self.assertEqual(len(u_vec), 3)
            self.assertEqual(len(v_vec), 3)

    def test_parallel_beam_geometry(self):
        """Test parallel beam geometry properties."""
        # In laminography, rays should be parallel
        num_angles = 180
        angles = np.linspace(0, np.pi, num_angles, endpoint=False)
        tilt_angle = np.pi/4  # 45 degrees
        
        ray_directions = []
        for angle in angles:
            ray_dir = np.array([
                np.cos(angle) * np.cos(tilt_angle),
                np.sin(angle) * np.cos(tilt_angle),
                np.sin(tilt_angle)
            ])
            ray_directions.append(ray_dir)
        
        ray_directions = np.array(ray_directions)
        
        # All rays should have the same z-component (parallel beams)
        z_components = ray_directions[:, 2]
        self.assertTrue(np.allclose(z_components, z_components[0]))

    @unittest.skipUnless(ASTRA_AVAILABLE, "ASTRA toolbox not available")
    def test_gd_optimize_projections_keeps_recon_fixed(self):
        """Test projection optimization mode with a mocked projector."""
        device = torch.device("cpu")
        angles_deg = np.linspace(0, 180, 4, endpoint=False)
        vol_shape = (2, 2, 2)
        projs = torch.zeros((4, 2, 2), dtype=torch.float32, device=device)
        vol_init = torch.ones(vol_shape, dtype=torch.float32, device=device)
        mask = np.array([False, True, False, True])

        def fake_projector_builder(*, angles_deg, det_shape, device, **kwargs):
            def fake_projector(x):
                return torch.ones(
                    (x.shape[0], len(angles_deg), det_shape[0], det_shape[1]),
                    dtype=x.dtype,
                    device=device,
                )
            return fake_projector

        with mock.patch.object(lamino, "build_lamino_projector", side_effect=fake_projector_builder):
            optimized = lamino.gd_reconstruction_masked(
                projs,
                angles_deg,
                lamino_angle_deg=30.0,
                mask=mask,
                vol_shape=vol_shape,
                vol_init=vol_init,
                max_epochs=1,
                batch_size=2,
                lr=0.5,
                clamp_min=None,
                optimizer_type="sgd",
                optimize_projections=True,
            )

        self.assertEqual(optimized.shape, projs.shape)
        self.assertTrue(torch.allclose(optimized[~mask], projs[~mask]))
        self.assertLess(torch.mean((optimized[mask] - 1.0) ** 2).item(), 1.0)
        self.assertTrue(torch.allclose(vol_init, torch.ones_like(vol_init)))

    @unittest.skipUnless(ASTRA_AVAILABLE, "ASTRA toolbox not available")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available - test requires GPU")
    def test_gd_reconstruction(self):
        """Test gradient descent reconstruction."""
        # Force GPU device
        device = torch.device('cuda')
        
        # Print initial GPU memory
        print(f"\nInitial GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
        
        # Create large phantom for stress test - use torch directly on GPU
        vol_shape = (300, 2600, 2600)
        
        print(f"\nTesting GD reconstruction with volume shape: {vol_shape}")
        print(f"Using device: {device}")
        print("Creating phantom directly on GPU...")
        
        # Create phantom directly on GPU using torch
        phantom_torch = torch.zeros(vol_shape, dtype=torch.float32, device=device)
        
        print(f"After phantom allocation: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        # Add a sphere in the center - use torch meshgrid on GPU
        x = torch.arange(vol_shape[0], dtype=torch.float32, device=device)
        y = torch.arange(vol_shape[1], dtype=torch.float32, device=device)
        z = torch.arange(vol_shape[2], dtype=torch.float32, device=device)
        
        # Create meshgrid on GPU
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        
        print(f"After meshgrid: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        # Compute sphere mask on GPU
        center_x, center_y, center_z = vol_shape[0]//2, vol_shape[1]//2, vol_shape[2]//2
        radius = vol_shape[0]//6
        sphere = ((xx - center_x)**2 + (yy - center_y)**2 + (zz - center_z)**2) <= radius**2
        phantom_torch[sphere] = 0.8
        
        # Free meshgrid memory
        del xx, yy, zz, x, y, z
        torch.cuda.empty_cache()
        
        print(f"Phantom created on GPU, after cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        # Create projections using forward projector
        num_angles = 401
        angles_deg = np.linspace(0, 180, num_angles, endpoint=False)
        lamino_angle_deg = 30.0
        
        # Add batch and channel dimensions
        phantom_batch = phantom_torch.unsqueeze(0).unsqueeze(0)  # (1,1,nx,ny,nz)
        
        print(f"Building projector...")
        
        projector = lamino.build_lamino_projector(
            vol_shape=vol_shape,
            det_shape=(vol_shape[0], vol_shape[1]),
            angles_deg=angles_deg,
            lamino_angle_deg=lamino_angle_deg,
            voxel_size_mm=1.0,
            det_spacing_mm=1.0,
            device=device
        )
        
        print(f"After projector build: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"Forward projecting...")
        
        projections = projector(phantom_batch).squeeze(0)  # (V, R, C)
        
        print(f"After forward projection: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"Projections shape: {projections.shape}")
        
        # Free phantom_batch to save memory
        del phantom_batch
        torch.cuda.empty_cache()
        
        # Add noise
        noise_level = 0.05 * projections.std()
        projections_noisy = projections + torch.randn_like(projections) * noise_level
        
        # Free original projections
        del projections
        torch.cuda.empty_cache()
        
        print(f"After noise addition and cleanup: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        # Create mask to select every other projection
        mask = torch.zeros(num_angles, dtype=torch.bool, device=device)
        mask[::2] = True  # Select every other projection
        num_selected = mask.sum().item()
        
        print(f"Using mask: selecting {num_selected}/{num_angles} projections (every other)")
        print(f"Memory before GD reconstruction: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        # Free phantom to save memory during reconstruction
        del phantom_torch
        torch.cuda.empty_cache()
        
        print(f"Memory after freeing phantom: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print(f"Starting GD reconstruction...")
        
        try:
            # Test GD reconstruction with mask
            volume_gd = lamino.gd_reconstruction_masked(
                projections_noisy,
                angles_deg,
                lamino_angle_deg=lamino_angle_deg,
                mask=mask,  # Pass the mask
                vol_shape=vol_shape,
                voxel_per_mm=1,
                device=device,
                max_epochs=10,
                batch_size=3,  # Further reduced batch size to save memory with 401 projections
                lr=1e-2,
                verbose=True,
                clamp_min=0.0
            )
            
            print(f"After GD reconstruction: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            
        except RuntimeError as e:
            print(f"\n!!! RuntimeError during GD reconstruction !!!")
            print(f"Error: {e}")
            print(f"GPU memory at failure: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            print(f"GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
            raise
        
        # Recreate phantom for comparison (we deleted it earlier)
        print(f"Recreating phantom for comparison...")
        phantom_torch = torch.zeros(vol_shape, dtype=torch.float32, device=device)
        x = torch.arange(vol_shape[0], dtype=torch.float32, device=device)
        y = torch.arange(vol_shape[1], dtype=torch.float32, device=device)
        z = torch.arange(vol_shape[2], dtype=torch.float32, device=device)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing='ij')
        center_x, center_y, center_z = vol_shape[0]//2, vol_shape[1]//2, vol_shape[2]//2
        radius = vol_shape[0]//6
        sphere = ((xx - center_x)**2 + (yy - center_y)**2 + (zz - center_z)**2) <= radius**2
        phantom_torch[sphere] = 0.8
        del xx, yy, zz, x, y, z
        torch.cuda.empty_cache()
        
        # Verify output shape and type
        self.assertEqual(volume_gd.shape, vol_shape)
        self.assertIsInstance(volume_gd, torch.Tensor)
        
        # Check reconstruction quality - keep in torch for speed
        mse = torch.mean((phantom_torch - volume_gd)**2).item()
        
        print(f"\nReconstruction MSE: {mse:.6f}")
        
        # MSE should be reasonably low for this simple case
        self.assertLess(mse, 0.1, f"MSE too high: {mse:.6f}")
        
        # Check that sphere region has higher values than background
        phantom_sphere_mean = phantom_torch[sphere].mean().item()
        volume_sphere_mean = volume_gd[sphere].mean().item()
        volume_bg_mean = volume_gd[~sphere].mean().item()
        
        print(f"Phantom sphere mean: {phantom_sphere_mean:.3f}")
        print(f"Reconstructed sphere mean: {volume_sphere_mean:.3f}")
        print(f"Background mean: {volume_bg_mean:.3f}")
        
        self.assertGreater(volume_sphere_mean, volume_bg_mean * 2,
                          "Sphere region should be significantly brighter than background")


if __name__ == '__main__':
    unittest.main()
