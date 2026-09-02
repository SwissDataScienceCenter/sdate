"""Tests for 2D parallel beam reconstruction functions."""

import unittest
import numpy as np
import torch

try:
    from astra_torch import parallel2D
    from astra_torch.parallel2D import Parallel2DAcquisition
    ASTRA_AVAILABLE = True
except ImportError:
    ASTRA_AVAILABLE = False


class TestParallel2DReconstruction(unittest.TestCase):
    """Test 2D parallel beam reconstruction functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @unittest.skipUnless(ASTRA_AVAILABLE, "ASTRA toolbox not available")
    def test_parallel2d_acquisition_init(self):
        """Test Parallel2DAcquisition initialization."""
        num_angles = 100
        angles_deg = np.linspace(0, 180, num_angles, endpoint=False)
        
        # Create 2D parallel beam acquisition
        acquisition = Parallel2DAcquisition(
            proj_cols=128,
            angles_deg=angles_deg,
            vol_shape=(64, 64)
        )
        
        self.assertEqual(len(acquisition.angles_deg), num_angles)
        self.assertEqual(acquisition.proj_cols, 128)
        self.assertEqual(acquisition.vol_shape, (64, 64))

    @unittest.skipUnless(ASTRA_AVAILABLE, "ASTRA toolbox not available")
    def test_parallel2d_geometry_creation(self):
        """Test 2D parallel beam geometry creation."""
        vol_shape = (64, 64)
        det_cols = 128
        angles_deg = np.linspace(0, 180, 100, endpoint=False)
        
        vol_geom, proj_geom = parallel2D._create_parallel2d_geometry(
            vol_shape=vol_shape,
            det_cols=det_cols,
            angles_deg=angles_deg,
            voxel_size_mm=1.0,
            det_spacing_mm=1.0,
        )
        
        # Check volume geometry
        self.assertIn('GridRowCount', vol_geom)
        self.assertIn('GridColCount', vol_geom)
        self.assertEqual(vol_geom['GridRowCount'], 64)
        self.assertEqual(vol_geom['GridColCount'], 64)
        
        # Check projection geometry
        self.assertEqual(proj_geom['type'], 'parallel')
        self.assertEqual(proj_geom['DetectorCount'], det_cols)

    @unittest.skipUnless(ASTRA_AVAILABLE, "ASTRA toolbox not available")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available - test requires GPU")
    def test_fbp_reconstruction_simple(self):
        """Test FBP reconstruction with a simple phantom."""
        device = torch.device('cuda')
        
        # Create a simple circular phantom
        size = 64
        phantom = torch.zeros((size, size), dtype=torch.float32, device=device)
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, size, device=device),
            torch.linspace(-1, 1, size, device=device),
            indexing='ij'
        )
        phantom[x**2 + y**2 < 0.5] = 1.0
        
        # Generate projections
        num_angles = 180
        angles_deg = np.linspace(0, 180, num_angles, endpoint=False)
        det_cols = int(size * 1.5)  # Detector larger than object
        
        # Forward project
        projector = parallel2D.build_parallel2d_projector(
            vol_shape=(size, size),
            det_cols=det_cols,
            angles_deg=angles_deg,
            voxel_size_mm=1.0,
            det_spacing_mm=1.0,
            device=device
        )
        
        phantom_batch = phantom.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        projs = projector(phantom_batch)  # (1, V, C)
        projs_vc = projs[0]  # (V, C)
        
        # Reconstruct
        recon = parallel2D.fbp_reconstruction_masked(
            projs_vc=projs_vc,
            angles_deg=angles_deg,
            vol_shape=(size, size),
            det_spacing_mm=1.0,
            filter_type='hann',
            device=device,
        )
        
        # Check shape
        self.assertEqual(recon.shape, (size, size))
        
        # Check that reconstruction is reasonable (correlation with phantom)
        # Center region should have high correlation
        center_slice = slice(size//4, 3*size//4)
        phantom_center = phantom[center_slice, center_slice]
        recon_center = recon[center_slice, center_slice]
        
        # Normalize both
        phantom_norm = (phantom_center - phantom_center.mean()) / phantom_center.std()
        recon_norm = (recon_center - recon_center.mean()) / recon_center.std()
        
        correlation = torch.sum(phantom_norm * recon_norm) / phantom_norm.numel()
        self.assertGreater(correlation.item(), 0.5, "Reconstruction correlation too low")

    @unittest.skipUnless(ASTRA_AVAILABLE, "ASTRA toolbox not available")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available - test requires GPU")
    def test_sirt_reconstruction_simple(self):
        """Test SIRT reconstruction with a simple phantom."""
        device = torch.device('cuda')
        
        # Create a simple square phantom
        size = 32  # Smaller for faster test
        phantom = torch.zeros((size, size), dtype=torch.float32, device=device)
        phantom[size//4:3*size//4, size//4:3*size//4] = 1.0
        
        # Generate projections
        num_angles = 60
        angles_deg = np.linspace(0, 180, num_angles, endpoint=False)
        det_cols = size
        
        # Forward project
        projector = parallel2D.build_parallel2d_projector(
            vol_shape=(size, size),
            det_cols=det_cols,
            angles_deg=angles_deg,
            voxel_size_mm=1.0,
            det_spacing_mm=1.0,
            device=device
        )
        
        phantom_batch = phantom.unsqueeze(0).unsqueeze(0)
        projs = projector(phantom_batch)
        projs_vc = projs[0]
        
        # Reconstruct with SIRT (few iterations for speed)
        recon = parallel2D.sirt_reconstruction_masked(
            projs_vc=projs_vc,
            angles_deg=angles_deg,
            vol_shape=(size, size),
            det_spacing_mm=1.0,
            num_iterations=10,
            min_constraint=0.0,
            device=device,
        )
        
        # Check shape
        self.assertEqual(recon.shape, (size, size))
        
        # Check non-negativity constraint
        self.assertGreaterEqual(recon.min().item(), 0.0)
        
        # Check that reconstruction has reasonable values
        self.assertGreater(recon.max().item(), 0.1)

    @unittest.skipUnless(ASTRA_AVAILABLE, "ASTRA toolbox not available")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available - test requires GPU")
    def test_gd_reconstruction_simple(self):
        """Test gradient descent reconstruction with a simple phantom."""
        device = torch.device('cuda')
        
        # Create a simple phantom
        size = 32  # Small for fast test
        phantom = torch.zeros((size, size), dtype=torch.float32, device=device)
        y, x = torch.meshgrid(
            torch.linspace(-1, 1, size, device=device),
            torch.linspace(-1, 1, size, device=device),
            indexing='ij'
        )
        phantom[x**2 + y**2 < 0.6] = 1.0
        
        # Generate projections
        num_angles = 60
        angles_deg = np.linspace(0, 180, num_angles, endpoint=False)
        det_cols = size
        
        # Forward project
        projector = parallel2D.build_parallel2d_projector(
            vol_shape=(size, size),
            det_cols=det_cols,
            angles_deg=angles_deg,
            device=device
        )
        
        phantom_batch = phantom.unsqueeze(0).unsqueeze(0)
        projs = projector(phantom_batch)
        projs_vc = projs[0]
        
        # Reconstruct with GD (few epochs for speed)
        recon = parallel2D.gd_reconstruction_masked(
            projs_vc=projs_vc,
            angles_deg=angles_deg,
            vol_shape=(size, size),
            max_epochs=5,
            batch_size=10,
            lr=1e-2,
            clamp_min=0.0,
            optimizer_type="adam",
            device=device,
            verbose=False,
        )
        
        # Check shape
        self.assertEqual(recon.shape, (size, size))
        
        # Check non-negativity
        self.assertGreaterEqual(recon.min().item(), 0.0)

    @unittest.skipUnless(ASTRA_AVAILABLE, "ASTRA toolbox not available")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available - test requires GPU")
    def test_projector_gradients(self):
        """Test that gradients flow through the projector."""
        device = torch.device('cuda')
        
        size = 16  # Very small for fast test
        vol_shape = (size, size)
        det_cols = size
        num_angles = 30
        angles_deg = np.linspace(0, 180, num_angles, endpoint=False)
        
        # Create projector
        projector = parallel2D.build_parallel2d_projector(
            vol_shape=vol_shape,
            det_cols=det_cols,
            angles_deg=angles_deg,
            device=device
        )
        
        # Create input volume with gradient tracking
        vol = torch.randn((1, 1, size, size), device=device, dtype=torch.float32, requires_grad=True)
        
        # Forward pass
        projs = projector(vol)
        
        # Compute loss and backprop
        loss = projs.sum()
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(vol.grad)
        self.assertEqual(vol.grad.shape, vol.shape)
        
        # Check that gradients are non-zero
        self.assertGreater(torch.abs(vol.grad).sum().item(), 0)

    @unittest.skipUnless(ASTRA_AVAILABLE, "ASTRA toolbox not available")
    def test_masked_reconstruction(self):
        """Test reconstruction with masked views."""
        device = self.device
        
        size = 32
        num_angles = 60
        angles_deg = np.linspace(0, 180, num_angles, endpoint=False)
        det_cols = size
        
        # Create dummy projections
        projs_vc = torch.randn((num_angles, det_cols), device=device, dtype=torch.float32)
        
        # Create mask (use only half the views)
        mask = np.zeros(num_angles, dtype=bool)
        mask[::2] = True  # Every other view
        
        # Reconstruct with mask
        try:
            recon = parallel2D.fbp_reconstruction_masked(
                projs_vc=projs_vc,
                angles_deg=angles_deg,
                mask=mask,
                vol_shape=(size, size),
                device=device,
            )
            
            # Check shape
            self.assertEqual(recon.shape, (size, size))
        except RuntimeError as e:
            if "CuPy" in str(e) and not device.type == 'cuda':
                self.skipTest("GPU not available for masked reconstruction test")
            else:
                raise

    @unittest.skipUnless(ASTRA_AVAILABLE, "ASTRA toolbox not available")
    def test_ramp_filter_cpu(self):
        """Test ramp filter on CPU."""
        size = 64
        num_angles = 30
        sino = np.random.randn(num_angles, size).astype(np.float32)
        
        # Apply filter
        filtered = parallel2D._apply_ramp_filter(
            sino,
            use_gpu=False,
            filter_type='ram-lak',
            det_spacing_mm=1.0,
        )
        
        # Check shape
        self.assertEqual(filtered.shape, sino.shape)
        
        # Check that it's not all zeros
        self.assertGreater(np.abs(filtered).sum(), 0)

    @unittest.skipUnless(ASTRA_AVAILABLE, "ASTRA toolbox not available")
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_ramp_filter_gpu(self):
        """Test ramp filter on GPU."""
        try:
            import cupy as cp
        except ImportError:
            self.skipTest("CuPy not available")
        
        size = 64
        num_angles = 30
        sino = cp.random.randn(num_angles, size).astype(cp.float32)
        
        # Apply filter
        filtered = parallel2D._apply_ramp_filter(
            sino,
            use_gpu=True,
            filter_type='hann',
            det_spacing_mm=1.0,
        )
        
        # Check shape
        self.assertEqual(filtered.shape, sino.shape)
        
        # Check that it's not all zeros
        self.assertGreater(float(cp.abs(filtered).sum()), 0)

    @unittest.skipUnless(ASTRA_AVAILABLE, "ASTRA toolbox not available")
    def test_filter_types(self):
        """Test different filter types."""
        size = 32
        num_angles = 20
        sino = np.random.randn(num_angles, size).astype(np.float32)
        
        filter_types = ['ram-lak', 'shepp-logan', 'cosine', 'hamming', 'hann']
        
        for ftype in filter_types:
            with self.subTest(filter_type=ftype):
                filtered = parallel2D._apply_ramp_filter(
                    sino,
                    use_gpu=False,
                    filter_type=ftype,
                    det_spacing_mm=1.0,
                )
                
                self.assertEqual(filtered.shape, sino.shape)
                self.assertGreater(np.abs(filtered).sum(), 0)

    def test_invalid_filter_type(self):
        """Test that invalid filter type raises error."""
        size = 32
        num_angles = 20
        sino = np.random.randn(num_angles, size).astype(np.float32)
        
        with self.assertRaises(ValueError):
            parallel2D._apply_ramp_filter(
                sino,
                use_gpu=False,
                filter_type='invalid_filter',
                det_spacing_mm=1.0,
            )


if __name__ == '__main__':
    unittest.main()
