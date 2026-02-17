"""
INCT: Instant Neural Compression for Tomography

A library implementing Instant Neural Graphics Primitives (NGP) style
hash encoding for compressing 4D tomographic projection data.

Based on: "Instant Neural Graphics Primitives with a Multiresolution Hash Encoding"
https://arxiv.org/abs/2201.05989

Instead of learning 2D images, this library learns 4D tensors representing
tomographic projections using multi-resolution hash encoding.
"""

from .hash_encoding import HashEncoding, MultiResolutionHashEncoding
from .model import InstantNGPModel, TinyMLP
from .dataset import VoxelDataset, BatchVoxelDataset, ProjectionVolumeDataset, RandomCoordDataset, TensorVolumeDataset
from .trainer import Trainer, TrainingConfig
from .utils import psnr, mse, normalize_coords, denormalize_coords

__version__ = "0.1.0"
__all__ = [
    "HashEncoding",
    "MultiResolutionHashEncoding", 
    "InstantNGPModel",
    "TinyMLP",
    "VoxelDataset",
    "BatchVoxelDataset",
    "ProjectionVolumeDataset",
    "TensorVolumeDataset",
    "RandomCoordDataset",
    "Trainer",
    "TrainingConfig",
    "psnr",
    "mse",
    "normalize_coords",
    "denormalize_coords",
]
