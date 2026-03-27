"""
Datasets for Instant NGP Training on Tomographic Data.

Provides efficient sampling of voxel coordinates and values
from tomographic projection volumes.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict
from PIL import Image
import sys

# Add parent to path to import from sdate
sys.path.insert(0, str(Path(__file__).parent.parent))


class VoxelDataset(Dataset):
    """
    Dataset that samples voxel coordinates and values from a volume.
    
    Stores the full volume in memory and returns (coordinate, value) pairs.
    Coordinates are normalized to [0, 1].
    """
    
    def __init__(
        self,
        volume: torch.Tensor,
        n_samples: Optional[int] = None,
        normalize_values: bool = True,
    ):
        """
        Args:
            volume: Tensor of shape (D1, D2, ..., Dn) containing the volume data
            n_samples: Number of samples per epoch. If None, use all voxels.
            normalize_values: Whether to normalize values to [0, 1]
        """
        super().__init__()
        
        self.volume = volume
        self.shape = volume.shape
        self.n_dims = len(self.shape)
        self.n_voxels = volume.numel()
        self.n_samples = n_samples if n_samples else self.n_voxels
        
        # Store normalization parameters
        self.normalize_values = normalize_values
        if normalize_values:
            self.value_min = volume.min().item()
            self.value_max = volume.max().item()
            self.value_range = max(self.value_max - self.value_min, 1e-6)
        else:
            self.value_min = 0.0
            self.value_max = 1.0
            self.value_range = 1.0
        
        # Precompute all coordinates (flattened)
        grids = [torch.arange(s, dtype=torch.float32) / max(s - 1, 1) for s in self.shape]
        self.all_coords = torch.stack(
            torch.meshgrid(*grids, indexing='ij'), dim=-1
        ).reshape(-1, self.n_dims)
        
        # Precompute all values (flattened and normalized)
        self.all_values = volume.flatten()
        if normalize_values:
            self.all_values = (self.all_values - self.value_min) / self.value_range
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Random sampling if n_samples != n_voxels
        if self.n_samples != self.n_voxels:
            idx = torch.randint(0, self.n_voxels, (1,)).item()
        
        return {
            'coords': self.all_coords[idx],
            'values': self.all_values[idx:idx+1],
        }
    
    def denormalize_values(self, values: torch.Tensor) -> torch.Tensor:
        """Convert normalized values back to original range."""
        if self.normalize_values:
            return values * self.value_range + self.value_min
        return values


class BatchVoxelDataset(Dataset):
    """
    Dataset that returns batches of voxels for efficient training.
    
    Each __getitem__ call returns a batch of random voxels.
    """
    
    def __init__(
        self,
        volume: torch.Tensor,
        batch_size: int = 65536,
        n_batches: int = 1000,
        normalize_values: bool = True,
    ):
        """
        Args:
            volume: Tensor of shape (D1, D2, ..., Dn) containing the volume data
            batch_size: Number of voxels per batch
            n_batches: Number of batches per epoch
            normalize_values: Whether to normalize values to [0, 1]
        """
        super().__init__()
        
        self.volume = volume
        self.shape = volume.shape
        self.n_dims = len(self.shape)
        self.n_voxels = volume.numel()
        self.batch_size = batch_size
        self.n_batches = n_batches
        
        # Store normalization parameters
        self.normalize_values = normalize_values
        if normalize_values:
            self.value_min = volume.min().item()
            self.value_max = volume.max().item()
            self.value_range = max(self.value_max - self.value_min, 1e-6)
        else:
            self.value_min = 0.0
            self.value_max = 1.0
            self.value_range = 1.0
        
        # Precompute all coordinates (flattened)
        grids = [torch.arange(s, dtype=torch.float32) / max(s - 1, 1) for s in self.shape]
        self.all_coords = torch.stack(
            torch.meshgrid(*grids, indexing='ij'), dim=-1
        ).reshape(-1, self.n_dims)
        
        # Precompute all values (flattened and normalized)
        self.all_values = volume.flatten().float()
        if normalize_values:
            self.all_values = (self.all_values - self.value_min) / self.value_range
    
    def __len__(self) -> int:
        return self.n_batches
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Sample random voxel indices
        indices = torch.randint(0, self.n_voxels, (self.batch_size,))
        
        return {
            'coords': self.all_coords[indices],
            'values': self.all_values[indices].unsqueeze(-1),
        }
    
    def denormalize_values(self, values: torch.Tensor) -> torch.Tensor:
        """Convert normalized values back to original range."""
        if self.normalize_values:
            return values * self.value_range + self.value_min
        return values


class TensorVolumeDataset(Dataset):
    """
    Dataset that creates batches from a pre-computed tensor volume.
    
    Similar to BatchVoxelDataset but designed for use with residual volumes
    or any pre-computed tensor. Useful for training a second INCT model
    on residuals from a first model.
    """
    
    def __init__(
        self,
        volume: torch.Tensor,
        batch_size: int = 65536,
        n_batches: int = 1000,
        normalize_values: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            volume: Pre-computed tensor volume of shape (H, W, N) or any shape
            batch_size: Number of voxels per batch
            n_batches: Number of batches per epoch
            normalize_values: Whether to normalize values to [0, 1]
            verbose: Print dataset info
        """
        super().__init__()
        
        self.volume = volume.float()
        self.shape = tuple(volume.shape)
        self.n_dims = len(self.shape)
        self.n_voxels = volume.numel()
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.normalize_values = normalize_values
        
        # Store normalization parameters
        if normalize_values:
            self.value_min = volume.min().item()
            self.value_max = volume.max().item()
            self.value_range = max(self.value_max - self.value_min, 1e-6)
        else:
            self.value_min = 0.0
            self.value_max = 1.0
            self.value_range = 1.0
        
        # Precompute all coordinates (flattened, normalized to [0, 1])
        grids = [torch.arange(s, dtype=torch.float32) / max(s - 1, 1) for s in self.shape]
        self.all_coords = torch.stack(
            torch.meshgrid(*grids, indexing='ij'), dim=-1
        ).reshape(-1, self.n_dims)
        
        # Precompute all values (flattened and normalized)
        self.all_values = volume.flatten().float()
        if normalize_values:
            self.all_values = (self.all_values - self.value_min) / self.value_range
        
        if verbose:
            print(f"TensorVolumeDataset created:")
            print(f"  Shape: {self.shape}")
            print(f"  Total voxels: {self.n_voxels:,}")
            print(f"  Batch size: {batch_size}")
            print(f"  Batches per epoch: {n_batches}")
            print(f"  Value range: [{self.value_min:.4f}, {self.value_max:.4f}]")
    
    def __len__(self) -> int:
        return self.n_batches
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a batch of random voxels."""
        indices = torch.randint(0, self.n_voxels, (self.batch_size,))
        
        return {
            'coords': self.all_coords[indices],
            'values': self.all_values[indices].unsqueeze(-1),
        }
    
    def denormalize_values(self, values: torch.Tensor) -> torch.Tensor:
        """Convert normalized values back to original range."""
        if self.normalize_values:
            return values * self.value_range + self.value_min
        return values
    
    def get_full_volume(self) -> torch.Tensor:
        """Return the full volume tensor."""
        return self.volume


class ProjectionVolumeDataset(Dataset):
    """
    Dataset for loading projection volumes from TomographyFolderProcessor.
    
    Creates a 3D volume from projection images:
    - Dimension 0: Height
    - Dimension 1: Width  
    - Dimension 2: Projection index
    """
    
    def __init__(
        self,
        folder_path: Union[str, Path],
        num_projections: Optional[int] = None,
        target_size: Optional[int] = None,
        batch_size: int = 65536,
        use_attenuation: bool = False,
        n_batches: int = 1000,
        normalize_values: bool = True,
        start_projection: int = 0,
        cache_volume: bool = True,
        verbose: bool = True,
    ):
        """
        Args:
            folder_path: Path to folder with tomographic data
            num_projections: Number of projections to load (None = all)
            target_size: Resize projections to this size (None = original)
            batch_size: Number of voxels per batch
            n_batches: Number of batches per epoch
            normalize_values: Whether to normalize values to [0, 1]
            start_projection: Index of first projection to load
            use_attenuation: Whether to use attenuation correction when loading projections
            cache_volume: Whether to cache the full volume in memory
            verbose: Print loading progress
        """
        super().__init__()
        
        self.folder_path = Path(folder_path)
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.normalize_values = normalize_values
        self.verbose = verbose
        
        # Import TomographyFolderProcessor
        try:
            from sdate.datasets.projection_triplet_dataset import (
                TomographyFolderProcessor,
                load_tomography_params
            )
        except ImportError:
            raise ImportError(
                "Could not import TomographyFolderProcessor. "
                "Make sure sdate is in your Python path."
            )
        
        # Create processor
        self.processor = TomographyFolderProcessor(
            folder_path=folder_path,
            num_darks=None,  # Auto-detect
            num_flats=None,  # Auto-detect
            cache_in_memory=False,
            verbose=verbose,
            use_attenuation=use_attenuation,
        )
        
        # Determine number of projections to load
        available = self.processor.num_projections
        if num_projections is None:
            num_projections = available - start_projection
        else:
            num_projections = min(num_projections, available - start_projection)
        
        self.num_projections = num_projections
        self.start_projection = start_projection
        self.target_size = target_size
        
        # Load volume
        if cache_volume:
            self._load_volume()
        else:
            self.volume = None
            self._setup_lazy_loading()
    
    def _load_volume(self):
        """Load all projections into a 3D volume."""
        if self.verbose:
            print(f"Loading {self.num_projections} projections...")
        
        projections = []
        for i in range(self.num_projections):
            proj_idx = self.start_projection + i
            proj = self.processor.get_projection(proj_idx, normalize=False)
            
            # Resize if needed
            if self.target_size is not None:
                proj = self._resize_projection(proj)
            
            projections.append(torch.from_numpy(proj).float())
            
            if self.verbose and (i + 1) % 50 == 0:
                print(f"  Loaded {i + 1}/{self.num_projections}")
        
        # Stack into volume: (H, W, N_projections)
        self.volume = torch.stack(projections, dim=-1)
        self.original_volume = self.volume.clone()  # Keep original for reference

        self.shape = self.volume.shape
        self.n_dims = 3
        self.n_voxels = self.volume.numel()
        
        if self.verbose:
            print(f"Volume shape: {self.shape}")
            print(f"Total voxels: {self.n_voxels:,}")
        
        # Compute normalization
        if self.normalize_values:
            self.value_min = self.volume.min().item()
            self.value_max = self.volume.max().item()
            self.value_range = max(self.value_max - self.value_min, 1e-6)
        else:
            self.value_min = 0.0
            self.value_max = 1.0
            self.value_range = 1.0
        
        # Precompute coordinates
        grids = [torch.arange(s, dtype=torch.float32) / max(s - 1, 1) for s in self.shape]
        self.all_coords = torch.stack(
            torch.meshgrid(*grids, indexing='ij'), dim=-1
        ).reshape(-1, self.n_dims)
        
        # Precompute normalized values
        self.all_values = self.volume.flatten()
        if self.normalize_values:
            self.all_values = (self.all_values - self.value_min) / self.value_range
    
    def _resize_projection(self, proj: np.ndarray) -> np.ndarray:
        """Resize a projection to target size."""
        from PIL import Image
        
        img = Image.fromarray(proj)
        img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
        return np.array(img, dtype=np.float32)
    
    def _setup_lazy_loading(self):
        """Setup for lazy loading (not caching full volume)."""
        # Get dimensions from first projection
        proj = self.processor.get_projection(self.start_projection, normalize=False)
        if self.target_size is not None:
            proj = self._resize_projection(proj)
        
        h, w = proj.shape
        self.shape = (h, w, self.num_projections)
        self.n_dims = 3
        self.n_voxels = h * w * self.num_projections
        
        # Will compute normalization from samples
        self.value_min = 0.0
        self.value_max = 1.0
        self.value_range = 1.0
    
    def __len__(self) -> int:
        return self.n_batches
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return a batch of random voxels."""
        if self.volume is None:
            raise RuntimeError("Lazy loading not implemented. Set cache_volume=True.")
        
        # Sample random voxel indices
        indices = torch.randint(0, self.n_voxels, (self.batch_size,))
        
        return {
            'coords': self.all_coords[indices],
            'values': self.all_values[indices].unsqueeze(-1),
        }
    
    def denormalize_values(self, values: torch.Tensor) -> torch.Tensor:
        """Convert normalized values back to original range."""
        if self.normalize_values:
            return values * self.value_range + self.value_min
        return values
    
    def get_full_volume(self) -> torch.Tensor:
        """Return the full volume tensor."""
        if self.volume is None:
            self._load_volume()
        return self.volume
    
    def get_slice(self, dim: int, idx: int) -> torch.Tensor:
        """Get a 2D slice along a dimension."""
        if self.volume is None:
            self._load_volume()
        
        if dim == 0:
            return self.volume[idx, :, :]
        elif dim == 1:
            return self.volume[:, idx, :]
        elif dim == 2:
            return self.volume[:, :, idx]
        else:
            raise ValueError(f"Invalid dimension: {dim}")


class RandomCoordDataset(IterableDataset):
    """
    Iterable dataset that generates random coordinates on-the-fly.
    
    Useful for very large volumes where we don't want to precompute all coordinates.
    """
    
    def __init__(
        self,
        volume: torch.Tensor,
        batch_size: int = 65536,
        normalize_values: bool = True,
    ):
        """
        Args:
            volume: Volume tensor to sample from
            batch_size: Number of coordinates per iteration
            normalize_values: Whether to normalize values
        """
        super().__init__()
        
        self.volume = volume
        self.shape = volume.shape
        self.n_dims = len(self.shape)
        self.batch_size = batch_size
        
        if normalize_values:
            self.value_min = volume.min().item()
            self.value_max = volume.max().item()
            self.value_range = max(self.value_max - self.value_min, 1e-6)
            self.volume_normalized = (volume - self.value_min) / self.value_range
        else:
            self.value_min = 0.0
            self.value_max = 1.0
            self.value_range = 1.0
            self.volume_normalized = volume
    
    def __iter__(self):
        while True:
            # Generate random indices for each dimension
            indices = [torch.randint(0, s, (self.batch_size,)) for s in self.shape]
            
            # Convert to normalized coordinates
            coords = torch.stack([
                idx.float() / max(s - 1, 1) 
                for idx, s in zip(indices, self.shape)
            ], dim=-1)
            
            # Get values at these locations
            values = self.volume_normalized[tuple(indices)].unsqueeze(-1)
            
            yield {
                'coords': coords,
                'values': values,
            }
