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
from .dataset_slices import ProjectionSliceDataset, TensorSliceDataset
from .dataset_chunked import ChunkedProjectionDataset, ChunkedSliceDataset, create_zarr_from_projections, load_zarr_metadata
from .trainer import Trainer, TrainingConfig
from .naf import NeuralAttenuationField, NAFTrainer, ParallelBeamGeometry, DifferentiableRayTracer, create_sinogram_dataset, ParallelBeamGeometry3D, DifferentiableRayTracer3D, NAFTrainer3D

# Aliases for backward compatibility
SliceDataset = ProjectionSliceDataset
BatchSliceDataset = TensorSliceDataset
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
    "ProjectionSliceDataset",
    "TensorVolumeDataset",
    "TensorSliceDataset",
    "ChunkedProjectionDataset",
    "ChunkedSliceDataset",
    "create_zarr_from_projections",
    "load_zarr_metadata",
    "RandomCoordDataset",
    "Trainer",
    "TrainingConfig",
    "psnr",
    "mse",
    "normalize_coords",
    "denormalize_coords",
    # Neural Attenuation Field
    "NeuralAttenuationField",
    "NAFTrainer",
    "ParallelBeamGeometry",
    "DifferentiableRayTracer",
    "create_sinogram_dataset",
    # Backward compatibility aliases
    "SliceDataset",
    "BatchSliceDataset",
]
