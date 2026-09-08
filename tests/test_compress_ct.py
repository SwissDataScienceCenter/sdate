"""
Tests for the compress_ct package.

These tests use small synthetic frames (256×256 or 512×512) so they run fast
and do not require real tomographic data or a trained model checkpoint.
A lightweight "dummy" UNet stand-in is used so that the full compress →
decompress round-trip can be verified without loading a large model.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Ensure project root is on the path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from compress_ct.entropy import ResidualEncoder, ResidualDecoder
from compress_ct.predictor import BlockPredictor
from compress_ct.compressor import CTCompressor, _encode_jpeg, _decode_jpeg
from compress_ct.decompressor import CTDecompressor


# -----------------------------------------------------------------------
# Tiny stand-in model for testing (no diffusers dependency required)
# -----------------------------------------------------------------------

class _DummyUNet(nn.Module):
    """
    Minimal model that simply averages the k input channels and adds a
    small learned bias — good enough to produce a *reasonable* prediction
    that exercises the full pipeline.
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        # Initialize to average the inputs
        nn.init.constant_(self.conv.weight, 1.0 / (in_channels * 9))
        nn.init.constant_(self.conv.bias, 0.0)

    def forward(self, x, timesteps=None, return_dict=False):
        out = self.conv(x)
        if return_dict:
            return {"sample": out}
        return (out,)


# -----------------------------------------------------------------------
# Test suite
# -----------------------------------------------------------------------

class TestJpegHelpers(unittest.TestCase):
    """Test the internal JPEG encode / decode helpers."""

    def test_roundtrip(self):
        patch = np.random.rand(256, 256).astype(np.float32)
        blob = _encode_jpeg(patch, quality=95)
        decoded = _decode_jpeg(blob)
        self.assertEqual(decoded.shape, (256, 256))
        # JPEG is lossy so tolerance is generous
        np.testing.assert_allclose(decoded, patch, atol=0.05)

    def test_output_is_bytes(self):
        patch = np.zeros((64, 64), dtype=np.float32)
        blob = _encode_jpeg(patch)
        self.assertIsInstance(blob, bytes)


class TestResidualEntropyRoundtrip(unittest.TestCase):
    """Test DCT + Huffman residual encode → decode round-trip."""

    def _roundtrip(self, H, W, quality):
        encoder = ResidualEncoder(block_size=8, quality=quality)
        decoder = ResidualDecoder(block_size=8, quality=quality)

        residual = (np.random.randn(H, W) * 0.02).astype(np.float32)
        blob = encoder.encode(residual)
        self.assertIsInstance(blob, bytes)

        reconstructed = decoder.decode(blob)
        self.assertEqual(reconstructed.shape, (H, W))

        # High quality should keep error small
        if quality >= 80:
            np.testing.assert_allclose(reconstructed, residual, atol=0.12)

    def test_256x256_quality80(self):
        self._roundtrip(256, 256, 80)

    def test_256x256_quality50(self):
        self._roundtrip(256, 256, 50)

    def test_128x128(self):
        self._roundtrip(128, 128, 90)

    def test_zero_residual(self):
        """All-zero residual (edge case for Huffman — single symbol)."""
        encoder = ResidualEncoder(block_size=8, quality=80)
        decoder = ResidualDecoder(block_size=8, quality=80)
        residual = np.zeros((64, 64), dtype=np.float32)
        blob = encoder.encode(residual)
        recon = decoder.decode(blob)
        np.testing.assert_allclose(recon, residual, atol=1e-6)


class TestBlockPredictor(unittest.TestCase):
    """Test the block predictor with a dummy model."""

    def setUp(self):
        self.model = _DummyUNet(in_channels=3)
        self.predictor = BlockPredictor(
            model=self.model,
            block_size=256,
            num_input_projections=3,
            device=torch.device("cpu"),
            overlap=32,
        )

    def test_predict_frame_shape(self):
        frames = [np.random.rand(512, 512).astype(np.float32) for _ in range(3)]
        pred = self.predictor.predict_frame(frames)
        self.assertEqual(pred.shape, (512, 512))

    def test_predict_frame_non_power_of_two(self):
        """Non power-of-two frame dimensions should still work."""
        frames = [np.random.rand(300, 500).astype(np.float32) for _ in range(3)]
        pred = self.predictor.predict_frame(frames)
        self.assertEqual(pred.shape, (300, 500))

    def test_predict_frame_values_finite(self):
        frames = [np.random.rand(256, 256).astype(np.float32) for _ in range(3)]
        pred = self.predictor.predict_frame(frames)
        self.assertTrue(np.all(np.isfinite(pred)))


class TestCTCompressorDecompressor(unittest.TestCase):
    """Full round-trip: compress → decompress and verify reconstruction."""

    def setUp(self):
        self.model = _DummyUNet(in_channels=3)
        self.predictor = BlockPredictor(
            model=self.model,
            block_size=256,
            num_input_projections=3,
            device=torch.device("cpu"),
            overlap=32,
        )

    def _make_frames(self, n_frames, H, W, smooth=True):
        """Generate synthetic frames with slow spatial variation."""
        frames = []
        base = np.random.rand(H, W).astype(np.float32) * 0.8 + 0.1
        for i in range(n_frames):
            noise = np.random.randn(H, W).astype(np.float32) * 0.01
            frames.append(np.clip(base + noise + i * 0.002, 0, 1))
        return frames

    def test_roundtrip_small(self):
        """Compress 6 frames of 256×256 and decompress."""
        frames = self._make_frames(6, 256, 256)
        compressor = CTCompressor(
            predictor=self.predictor,
            patch_size=256,
            residual_quality=90,
            jpeg_quality=95,
            verbose=False,
        )
        decompressor = CTDecompressor(
            predictor=self.predictor, verbose=False
        )

        with tempfile.NamedTemporaryFile(suffix=".ctc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            stats = compressor.compress(frames, tmp_path)
            self.assertTrue(tmp_path.exists())
            self.assertGreater(stats["total_bytes"], 0)

            recovered = decompressor.decompress(tmp_path)
            self.assertEqual(len(recovered), len(frames))

            for i, (orig, rec) in enumerate(zip(frames, recovered)):
                self.assertEqual(rec.shape, orig.shape, f"Frame {i} shape mismatch")
                # Lossy pipeline — generous tolerance
                mse = np.mean((orig - rec) ** 2)
                # Lossy pipeline with dummy model — tolerance must be generous
                self.assertLess(mse, 0.05, f"Frame {i} MSE too large: {mse:.6f}")
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_roundtrip_non_square(self):
        """Frames that are not a multiple of patch_size."""
        frames = self._make_frames(5, 300, 500)
        compressor = CTCompressor(
            predictor=self.predictor,
            patch_size=256,
            residual_quality=80,
            jpeg_quality=95,
            verbose=False,
        )
        decompressor = CTDecompressor(
            predictor=self.predictor, verbose=False
        )

        with tempfile.NamedTemporaryFile(suffix=".ctc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            stats = compressor.compress(frames, tmp_path)
            recovered = decompressor.decompress(tmp_path)
            self.assertEqual(len(recovered), len(frames))
            for i, (orig, rec) in enumerate(zip(frames, recovered)):
                self.assertEqual(rec.shape, orig.shape, f"Frame {i} shape mismatch")
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_fallback_threshold(self):
        """With fallback_threshold=0 all patches should fall back to JPEG."""
        frames = self._make_frames(5, 256, 256)
        compressor = CTCompressor(
            predictor=self.predictor,
            patch_size=256,
            residual_quality=80,
            jpeg_quality=95,
            fallback_threshold=0.0,  # force all to JPEG
            verbose=False,
        )

        with tempfile.NamedTemporaryFile(suffix=".ctc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            stats = compressor.compress(frames, tmp_path)
            # All predicted frames should have used fallback
            for i in range(3, len(stats["fallback_counts"])):
                self.assertGreater(
                    stats["fallback_counts"][i], 0,
                    f"Frame {i} should have used JPEG fallback",
                )
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_compression_ratio(self):
        """Verify we get a meaningful compression ratio."""
        frames = self._make_frames(8, 256, 256)
        compressor = CTCompressor(
            predictor=self.predictor,
            patch_size=256,
            residual_quality=50,
            jpeg_quality=80,
            verbose=False,
        )

        with tempfile.NamedTemporaryFile(suffix=".ctc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            stats = compressor.compress(frames, tmp_path)
            # With smooth synthetic data we should get reasonable compression
            self.assertGreater(stats["compression_ratio"], 1.0)
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_stats_structure(self):
        """Check that stats dict has expected keys."""
        frames = self._make_frames(5, 256, 256)
        compressor = CTCompressor(
            predictor=self.predictor,
            patch_size=256,
            verbose=False,
        )

        with tempfile.NamedTemporaryFile(suffix=".ctc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            stats = compressor.compress(frames, tmp_path)
            for key in [
                "num_frames", "height", "width", "total_bytes",
                "compression_ratio", "frame_bytes",
                "fallback_counts", "residual_counts",
            ]:
                self.assertIn(key, stats, f"Missing key: {key}")
            self.assertEqual(len(stats["frame_bytes"]), 5)
        finally:
            tmp_path.unlink(missing_ok=True)


class TestEndToEndWithDataset(unittest.TestCase):
    """
    Integration test simulating the notebook workflow: pull frames from a
    mock dataset-style structure, compress, decompress, and compare.
    """

    def test_simulated_projection_sequence(self):
        """Simulate a CT projection sequence (smoothly varying frames)."""
        H, W = 512, 512
        k = 3
        n_frames = 8

        # Create frames that slowly rotate a gradient pattern
        frames = []
        for i in range(n_frames):
            y, x = np.mgrid[0:H, 0:W].astype(np.float32)
            angle = i * 0.05  # slow rotation
            pattern = 0.5 + 0.3 * np.sin(
                (x * np.cos(angle) + y * np.sin(angle)) / 40
            )
            noise = np.random.randn(H, W).astype(np.float32) * 0.005
            frames.append(np.clip(pattern + noise, 0, 1).astype(np.float32))

        model = _DummyUNet(in_channels=k)
        predictor = BlockPredictor(
            model=model, block_size=256, num_input_projections=k,
            device=torch.device("cpu"), overlap=32,
        )
        compressor = CTCompressor(
            predictor=predictor, patch_size=256,
            residual_quality=80, jpeg_quality=95, verbose=False,
        )
        decompressor = CTDecompressor(predictor=predictor, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".ctc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            stats = compressor.compress(frames, tmp_path)
            recovered = decompressor.decompress(tmp_path)

            self.assertEqual(len(recovered), n_frames)
            for i in range(n_frames):
                self.assertEqual(recovered[i].shape, (H, W))
                psnr = 10 * np.log10(
                    1.0 / max(np.mean((frames[i] - recovered[i]) ** 2), 1e-10)
                )
                # Expect at least ~25 dB with this lossy pipeline
                self.assertGreater(psnr, 20, f"Frame {i} PSNR too low: {psnr:.1f} dB")
        finally:
            tmp_path.unlink(missing_ok=True)


class TestConditioningRoundtrip(unittest.TestCase):
    """Test compress → decompress with use_conditioning=True."""

    def test_roundtrip_with_conditioning(self):
        """Full pipeline with per-patch center_x conditioning."""
        k = 3
        n_frames = 6
        H, W = 256, 256

        model = _DummyUNet(in_channels=k)
        predictor = BlockPredictor(
            model=model, block_size=256, num_input_projections=k,
            device=torch.device("cpu"), overlap=32,
            use_conditioning=True,
        )

        frames = []
        base = np.random.rand(H, W).astype(np.float32) * 0.8 + 0.1
        for i in range(n_frames):
            noise = np.random.randn(H, W).astype(np.float32) * 0.01
            frames.append(np.clip(base + noise + i * 0.002, 0, 1))

        center_x_coords = [0.5] * n_frames  # simulate centred projection

        compressor = CTCompressor(
            predictor=predictor, patch_size=256,
            residual_quality=80, jpeg_quality=95, verbose=False,
        )
        decompressor = CTDecompressor(predictor=predictor, verbose=False)

        with tempfile.NamedTemporaryFile(suffix=".ctc", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            stats = compressor.compress(
                frames, tmp_path, center_x_coords=center_x_coords
            )
            recovered = decompressor.decompress(tmp_path)

            self.assertEqual(len(recovered), n_frames)
            for i, (orig, rec) in enumerate(zip(frames, recovered)):
                self.assertEqual(rec.shape, orig.shape)
                mse = np.mean((orig - rec) ** 2)
                self.assertLess(mse, 0.05, f"Frame {i} MSE too large: {mse:.6f}")
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_conditioning_requires_center_x(self):
        """Compressor should raise when center_x is missing."""
        k = 3
        model = _DummyUNet(in_channels=k)
        predictor = BlockPredictor(
            model=model, block_size=256, num_input_projections=k,
            device=torch.device("cpu"), overlap=32,
            use_conditioning=True,
        )
        compressor = CTCompressor(
            predictor=predictor, patch_size=256, verbose=False,
        )
        frames = [np.random.rand(256, 256).astype(np.float32) for _ in range(5)]

        with tempfile.NamedTemporaryFile(suffix=".ctc", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with self.assertRaises(ValueError):
                compressor.compress(frames, tmp_path)  # no center_x_coords
        finally:
            tmp_path.unlink(missing_ok=True)

    def test_predictor_requires_center_x(self):
        """predict_frame should raise when center_x is missing."""
        model = _DummyUNet(in_channels=3)
        predictor = BlockPredictor(
            model=model, block_size=256, num_input_projections=3,
            device=torch.device("cpu"), overlap=32,
            use_conditioning=True,
        )
        frames = [np.random.rand(256, 256).astype(np.float32) for _ in range(3)]
        with self.assertRaises(ValueError):
            predictor.predict_frame(frames)  # no center_x


if __name__ == "__main__":
    unittest.main()
