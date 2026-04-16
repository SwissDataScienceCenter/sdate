"""
Slice-based Dataset for Instant NGP Training on Tomographic Data.

Provides efficient sampling of entire slices (projections) from tomographic 
projection volumes, rather than randomly sampling individual voxels.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Union, Dict


class ProjectionSliceDataset(Dataset):
    """
    Dataset for loading projection volumes that samples entire slices.

    Unlike ProjectionVolumeDataset which samples random voxels uniformly,
    this dataset samples entire slices (projections) uniformly at random,
    returning all voxels from the selected slices.

    Creates a 3D volume from projection images:
    - Dimension 0: Height
    - Dimension 1: Width  
    - Dimension 2: Projection index (slice dimension)

    When __getitem__ is called, instead of returning a batch of random voxels,
    it returns all voxels from one or more randomly sampled slices.
    """

    def __init__(
        self,
        folder_path: Union[str, Path],
        num_projections: Optional[int] = None,
        target_size: Optional[int] = None,
        slices_per_batch: int = 1,
        batch_size: Optional[int] = None,
        n_batches: int = 1000,
        normalize_values: bool = True,
        start_projection: int = 0,
        cache_volume: bool = True,
        slice_dim: int = 2,
        verbose: bool = True,
        use_attenuation: bool = False,
    ):
        """
        Args:
            folder_path: Path to folder with tomographic data
            num_projections: Number of projections to load (None = all)
            target_size: Resize projections to this size (None = original)
            slices_per_batch: Number of slices to sample per __getitem__ call
            batch_size: Number of voxels to return per batch. If None, returns all voxels
                        from the sampled slices. If specified, subsamples voxels uniformly
                        at random from the selected slices to match this size.
            n_batches: Number of batches per epoch
            normalize_values: Whether to normalize values to [0, 1]
            start_projection: Index of first projection to load
            cache_volume: Whether to cache the full volume in memory.
                            If False, slices are loaded on-demand.
            slice_dim: Dimension along which to sample slices (default=2 for projections)
            verbose: Print loading progress
            use_attenuation: Whether to use attenuation correction when loading projections
        """
        super().__init__()
    
        self.folder_path = Path(folder_path)
        self.slices_per_batch = slices_per_batch
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.normalize_values = normalize_values
        self.slice_dim = slice_dim
        self.verbose = verbose
        self.cache_volume = cache_volume
    
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
    
        # Get dimensions from first projection
        proj = self.processor.get_projection(self.start_projection, normalize=False)
        if self.target_size is not None:
            proj = self._resize_projection(proj)
        self.proj_height, self.proj_width = proj.shape
    
        # Volume shape: (H, W, N_projections)
        self.shape = (self.proj_height, self.proj_width, self.num_projections)
        self.n_dims = 3
        self.n_voxels = self.proj_height * self.proj_width * self.num_projections
    
        # Number of slices along each dimension
        self.n_slices_per_dim = self.shape
    
        # Voxels per slice along each dimension
        self.voxels_per_slice = [
            self.shape[1] * self.shape[2],  # dim 0: W * N
            self.shape[0] * self.shape[2],  # dim 1: H * N
            self.shape[0] * self.shape[1],  # dim 2: H * W
        ]
    
        if verbose:
            print(f"ProjectionSliceDataset:")
            print(f"  Volume shape: {self.shape}")
            print(f"  Slice dimension: {self.slice_dim}")
            print(f"  Slices per batch: {self.slices_per_batch}")
            print(f"  Voxels per slice (dim={self.slice_dim}): {self.voxels_per_slice[self.slice_dim]:,}")
    
        # Load volume or setup lazy loading
        if cache_volume:
            self._load_volume()
        else:
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
    
        if self.verbose:
            print(f"Value range: [{self.value_min:.4f}, {self.value_max:.4f}]")
    
        # Precompute coordinates for each slice dimension
        self._precompute_slice_coords()

    def _setup_lazy_loading(self):
        """Setup for lazy loading (not caching full volume)."""
        # Will compute normalization from samples or use defaults
        self.volume = None
        self.value_min = 0.0
        self.value_max = 1.0
        self.value_range = 1.0
    
        # For lazy loading, we'll estimate normalization from first few projections
        if self.normalize_values:
            self._estimate_normalization()
    
        # Precompute coordinate grids for slices
        self._precompute_slice_coords()

    def _estimate_normalization(self, n_samples: int = 5):
        """Estimate normalization from a few sample projections."""
        value_min = float('inf')
        value_max = float('-inf')
    
        sample_indices = np.linspace(0, self.num_projections - 1, n_samples, dtype=int)
        for i in sample_indices:
            proj_idx = self.start_projection + i
            proj = self.processor.get_projection(proj_idx, normalize=False)
            if self.target_size is not None:
                proj = self._resize_projection(proj)
            value_min = min(value_min, proj.min())
            value_max = max(value_max, proj.max())
    
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.value_range = max(self.value_max - self.value_min, 1e-6)
    
        if self.verbose:
            print(f"Estimated value range: [{self.value_min:.4f}, {self.value_max:.4f}]")

    def _precompute_slice_coords(self):
        """Precompute normalized coordinates for slices along slice_dim."""
        # Create coordinate grids for each dimension
        grids = [
            torch.arange(s, dtype=torch.float32) / max(s - 1, 1) 
            for s in self.shape
        ]
    
        # For slice_dim, we'll substitute the slice index when sampling
        # Store the non-slice dimension grids for meshgrid
        self.coord_grids = grids
    
        # Precompute the 2D mesh for the non-slice dimensions
        if self.slice_dim == 2:
            # Slicing along projection dimension: coords are (H, W)
            h_grid, w_grid = torch.meshgrid(grids[0], grids[1], indexing='ij')
            self.slice_h_coords = h_grid.flatten()  # Shape: (H*W,)
            self.slice_w_coords = w_grid.flatten()  # Shape: (H*W,)
        elif self.slice_dim == 0:
            # Slicing along height dimension: coords are (W, N)
            w_grid, n_grid = torch.meshgrid(grids[1], grids[2], indexing='ij')
            self.slice_w_coords = w_grid.flatten()
            self.slice_n_coords = n_grid.flatten()
        elif self.slice_dim == 1:
            # Slicing along width dimension: coords are (H, N)
            h_grid, n_grid = torch.meshgrid(grids[0], grids[2], indexing='ij')
            self.slice_h_coords = h_grid.flatten()
            self.slice_n_coords = n_grid.flatten()

    def _resize_projection(self, proj: np.ndarray) -> np.ndarray:
        """Resize a projection to target size."""
        from PIL import Image
    
        img = Image.fromarray(proj)
        # if target_size is an int, make it a square
        if isinstance(self.target_size, int):
            size = (self.target_size, self.target_size)
        else:
            size = self.target_size
        img = img.resize(size, Image.BILINEAR)
        return np.array(img, dtype=np.float32)

    def _get_slice_data(self, slice_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get coordinates and values for the given slice indices.
    
        Args:
            slice_indices: Tensor of slice indices to sample
        
        Returns:
            Dict with 'coords' and 'values' tensors
        """
        n_slices = len(slice_indices)
        voxels_per_slice = self.voxels_per_slice[self.slice_dim]
        total_voxels = n_slices * voxels_per_slice
    
        if self.cache_volume and self.volume is not None:
            return self._get_slice_data_cached(slice_indices)
        else:
            return self._get_slice_data_lazy(slice_indices)

    def _get_slice_data_cached(self, slice_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get slice data when volume is cached in memory."""
        n_slices = len(slice_indices)
        voxels_per_slice = self.voxels_per_slice[self.slice_dim]
        total_voxels = n_slices * voxels_per_slice
    
        all_coords = []
        all_values = []
    
        for slice_idx in slice_indices:
            slice_idx = slice_idx.item()
        
            # Get the slice values
            if self.slice_dim == 2:
                # Slice along projection dimension
                slice_values = self.volume[:, :, slice_idx].flatten()
                slice_coord_norm = slice_idx / max(self.shape[2] - 1, 1)
            
                # Build coords: (H*W, 3) with [h_coord, w_coord, slice_coord]
                coords = torch.stack([
                    self.slice_h_coords,
                    self.slice_w_coords,
                    torch.full((voxels_per_slice,), slice_coord_norm),
                ], dim=-1)
            
            elif self.slice_dim == 0:
                # Slice along height dimension
                slice_values = self.volume[slice_idx, :, :].flatten()
                slice_coord_norm = slice_idx / max(self.shape[0] - 1, 1)
            
                coords = torch.stack([
                    torch.full((voxels_per_slice,), slice_coord_norm),
                    self.slice_w_coords,
                    self.slice_n_coords,
                ], dim=-1)
            
            elif self.slice_dim == 1:
                # Slice along width dimension
                slice_values = self.volume[:, slice_idx, :].flatten()
                slice_coord_norm = slice_idx / max(self.shape[1] - 1, 1)
            
                coords = torch.stack([
                    self.slice_h_coords,
                    torch.full((voxels_per_slice,), slice_coord_norm),
                    self.slice_n_coords,
                ], dim=-1)
        
            all_coords.append(coords)
            all_values.append(slice_values)
    
        # Concatenate all slices
        coords = torch.cat(all_coords, dim=0)
        values = torch.cat(all_values, dim=0)
    
        # Normalize values
        if self.normalize_values:
            values = (values - self.value_min) / self.value_range
    
        return {
            'coords': coords,
            'values': values.unsqueeze(-1),
        }

    def _get_slice_data_lazy(self, slice_indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get slice data by loading slices on demand (lazy loading)."""
        # Currently only supports slice_dim=2 (projection dimension) for lazy loading
        if self.slice_dim != 2:
            raise RuntimeError(
                f"Lazy loading only supports slice_dim=2, got {self.slice_dim}. "
                "Set cache_volume=True for other slice dimensions."
            )
    
        n_slices = len(slice_indices)
        voxels_per_slice = self.voxels_per_slice[self.slice_dim]
    
        all_coords = []
        all_values = []
    
        for slice_idx in slice_indices:
            slice_idx = slice_idx.item()
        
            # Load the projection
            proj_idx = self.start_projection + slice_idx
            proj = self.processor.get_projection(proj_idx, normalize=False)
            if self.target_size is not None:
                proj = self._resize_projection(proj)
        
            slice_values = torch.from_numpy(proj).float().flatten()
            slice_coord_norm = slice_idx / max(self.shape[2] - 1, 1)
        
            # Build coords: (H*W, 3) with [h_coord, w_coord, slice_coord]
            coords = torch.stack([
                self.slice_h_coords,
                self.slice_w_coords,
                torch.full((voxels_per_slice,), slice_coord_norm),
            ], dim=-1)
        
            all_coords.append(coords)
            all_values.append(slice_values)
    
        # Concatenate all slices
        coords = torch.cat(all_coords, dim=0)
        values = torch.cat(all_values, dim=0)
    
        # Normalize values
        if self.normalize_values:
            values = (values - self.value_min) / self.value_range
    
        return {
            'coords': coords,
            'values': values.unsqueeze(-1),
        }

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return voxels from randomly sampled slices.
    
        Returns:
            Dict with:
                - 'coords': Tensor of shape (batch_size, 3) or (slices_per_batch * voxels_per_slice, 3)
                            with normalized coordinates in [0, 1]
                - 'values': Tensor of shape (batch_size, 1) or (slices_per_batch * voxels_per_slice, 1)
                            with normalized values
        """
        # Sample random slice indices uniformly
        n_slices = self.shape[self.slice_dim]
        slice_indices = torch.randint(0, n_slices, (self.slices_per_batch,))
    
        data = self._get_slice_data(slice_indices)
    
        # Subsample if batch_size is specified
        if self.batch_size is not None:
            total_voxels = data['coords'].shape[0]
            if total_voxels > self.batch_size:
                # Randomly subsample to match batch_size
                indices = torch.randperm(total_voxels)[:self.batch_size]
                data['coords'] = data['coords'][indices]
                data['values'] = data['values'][indices]
    
        return data

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
            if dim == 2:
                # Can load on demand for projection dimension
                proj_idx = self.start_projection + idx
                proj = self.processor.get_projection(proj_idx, normalize=False)
                if self.target_size is not None:
                    proj = self._resize_projection(proj)
                return torch.from_numpy(proj).float()
            else:
                self._load_volume()
    
        if dim == 0:
            return self.volume[idx, :, :]
        elif dim == 1:
            return self.volume[:, idx, :]
        elif dim == 2:
            return self.volume[:, :, idx]
        else:
            raise ValueError(f"Invalid dimension: {dim}")

    def get_batch_size(self) -> int:
        """Return the effective batch size (voxels per __getitem__ call)."""
        if self.batch_size is not None:
            return self.batch_size
        return self.slices_per_batch * self.voxels_per_slice[self.slice_dim]


class TensorSliceDataset(Dataset):
    """
    Dataset that samples entire slices from a pre-computed tensor volume.

    Similar to ProjectionSliceDataset but for any pre-computed tensor,
    such as residual volumes from a first INCT model.
    """

    def __init__(
        self,
        volume: torch.Tensor,
        slices_per_batch: int = 1,
        batch_size: Optional[int] = None,
        n_batches: int = 1000,
        normalize_values: bool = True,
        slice_dim: int = 2,
        verbose: bool = True,
    ):
        """
        Args:
            volume: Pre-computed tensor volume of shape (H, W, N) or any 3D shape
            slices_per_batch: Number of slices to sample per __getitem__ call
            batch_size: Number of voxels to return per batch. If None, returns all voxels
                        from the sampled slices. If specified, subsamples voxels uniformly
                        at random from the selected slices to match this size.
            n_batches: Number of batches per epoch
            normalize_values: Whether to normalize values to [0, 1]
            slice_dim: Dimension along which to sample slices
            verbose: Print dataset info
        """
        super().__init__()
    
        self.volume = volume.float()
        self.shape = tuple(volume.shape)
        self.n_dims = len(self.shape)
        self.n_voxels = volume.numel()
        self.slices_per_batch = slices_per_batch
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.slice_dim = slice_dim
        self.normalize_values = normalize_values
    
        if self.n_dims != 3:
            raise ValueError(f"TensorSliceDataset requires 3D volume, got {self.n_dims}D")
    
        # Voxels per slice along each dimension
        self.voxels_per_slice = [
            self.shape[1] * self.shape[2],  # dim 0
            self.shape[0] * self.shape[2],  # dim 1
            self.shape[0] * self.shape[1],  # dim 2
        ]
    
        # Store normalization parameters
        if normalize_values:
            self.value_min = volume.min().item()
            self.value_max = volume.max().item()
            self.value_range = max(self.value_max - self.value_min, 1e-6)
        else:
            self.value_min = 0.0
            self.value_max = 1.0
            self.value_range = 1.0
    
        # Precompute coordinate grids
        grids = [
            torch.arange(s, dtype=torch.float32) / max(s - 1, 1) 
            for s in self.shape
        ]
    
        # Precompute the 2D mesh for the non-slice dimensions
        if self.slice_dim == 2:
            h_grid, w_grid = torch.meshgrid(grids[0], grids[1], indexing='ij')
            self.slice_h_coords = h_grid.flatten()
            self.slice_w_coords = w_grid.flatten()
        elif self.slice_dim == 0:
            w_grid, n_grid = torch.meshgrid(grids[1], grids[2], indexing='ij')
            self.slice_w_coords = w_grid.flatten()
            self.slice_n_coords = n_grid.flatten()
        elif self.slice_dim == 1:
            h_grid, n_grid = torch.meshgrid(grids[0], grids[2], indexing='ij')
            self.slice_h_coords = h_grid.flatten()
            self.slice_n_coords = n_grid.flatten()
    
        if verbose:
            print(f"TensorSliceDataset created:")
            print(f"  Shape: {self.shape}")
            print(f"  Total voxels: {self.n_voxels:,}")
            print(f"  Slice dimension: {self.slice_dim}")
            print(f"  Slices per batch: {slices_per_batch}")
            print(f"  Voxels per slice: {self.voxels_per_slice[slice_dim]:,}")
            print(f"  Value range: [{self.value_min:.4f}, {self.value_max:.4f}]")

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return voxels from randomly sampled slices."""
        n_slices = self.shape[self.slice_dim]
        slice_indices = torch.randint(0, n_slices, (self.slices_per_batch,))
    
        voxels_per_slice = self.voxels_per_slice[self.slice_dim]
    
        all_coords = []
        all_values = []
    
        for slice_idx in slice_indices:
            slice_idx = slice_idx.item()
        
            if self.slice_dim == 2:
                slice_values = self.volume[:, :, slice_idx].flatten()
                slice_coord_norm = slice_idx / max(self.shape[2] - 1, 1)
                coords = torch.stack([
                    self.slice_h_coords,
                    self.slice_w_coords,
                    torch.full((voxels_per_slice,), slice_coord_norm),
                ], dim=-1)
            
            elif self.slice_dim == 0:
                slice_values = self.volume[slice_idx, :, :].flatten()
                slice_coord_norm = slice_idx / max(self.shape[0] - 1, 1)
                coords = torch.stack([
                    torch.full((voxels_per_slice,), slice_coord_norm),
                    self.slice_w_coords,
                    self.slice_n_coords,
                ], dim=-1)
            
            elif self.slice_dim == 1:
                slice_values = self.volume[:, slice_idx, :].flatten()
                slice_coord_norm = slice_idx / max(self.shape[1] - 1, 1)
                coords = torch.stack([
                    self.slice_h_coords,
                    torch.full((voxels_per_slice,), slice_coord_norm),
                    self.slice_n_coords,
                ], dim=-1)
        
            all_coords.append(coords)
            all_values.append(slice_values)
    
        coords = torch.cat(all_coords, dim=0)
        values = torch.cat(all_values, dim=0)
    
        if self.normalize_values:
            values = (values - self.value_min) / self.value_range
    
        data = {
            'coords': coords,
            'values': values.unsqueeze(-1),
        }
    
        # Subsample if batch_size is specified
        if self.batch_size is not None:
            total_voxels = coords.shape[0]
            if total_voxels > self.batch_size:
                # Randomly subsample to match batch_size
                indices = torch.randperm(total_voxels)[:self.batch_size]
                data['coords'] = data['coords'][indices]
                data['values'] = data['values'][indices]
    
        return data

    def denormalize_values(self, values: torch.Tensor) -> torch.Tensor:
        """Convert normalized values back to original range."""
        if self.normalize_values:
            return values * self.value_range + self.value_min
        return values

    def get_full_volume(self) -> torch.Tensor:
        """Return the full volume tensor."""
        return self.volume

    def get_batch_size(self) -> int:
        """Return the effective batch size (voxels per __getitem__ call)."""
        if self.batch_size is not None:
            return self.batch_size
        return self.slices_per_batch * self.voxels_per_slice[self.slice_dim]
