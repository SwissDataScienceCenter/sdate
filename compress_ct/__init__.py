"""
compress_ct — Neural CT Projection Compression.

Uses a trained Noise2Noise UNet model to predict future projections
from previous ones, stores compact residuals with entropy coding, and
falls back to JPEG when the prediction is poor.

Main entry points
-----------------
- ``CTCompressor``  – compress a sequence of projection frames
- ``CTDecompressor`` – reconstruct the sequence from the compressed archive
"""

from .compressor import CTCompressor
from .decompressor import CTDecompressor
from .predictor import BlockPredictor
from .drift_predictor import DriftPredictor, DriftMode
from .entropy import ResidualEncoder, ResidualDecoder

__all__ = [
    "CTCompressor",
    "CTDecompressor",
    "BlockPredictor",
    "DriftPredictor",
    "DriftMode",
    "ResidualEncoder",
    "ResidualDecoder",
]
