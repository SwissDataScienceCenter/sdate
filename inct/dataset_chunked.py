"""
Chunked Dataset for Instant NGP Training on Tomographic Data.

Uses Zarr format with chunked storage for efficient random access
to voxels without loading the entire volume into memory.
"""

import torch
import numpy as np
import zarr
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Union, Dict, Tuple
import json
import shutil


def create_zarr_from_projections(
    folder_path: Union[str, Path],
    zarr_path: Union[str, Path],
    num_projections: Optional[int] = None,
    start_projection: int = 0,
    target_size: Optional[int] = None,
    chunk_size: int = 128,
    compressor: Optional[str] = 'lz4',
    compression_level: int = 3,
    verbose: bool = True,
    overwrite: bool = False,
) -> Dict:
    """
    Convert projection TIFF files to a chunked Zarr volume.
    
    Processes projections in batches of chunk_size to minimize memory usage.
    
    Args:
        folder_path: Path to folder with tomographic data (TIFF projections)
        zarr_path: Path where the Zarr store will be created
        num_projections: Number of projections to process (None = all)
        start_projection: Index of first projection to load
        target_size: Resize projections to this size (None = original)
        chunk_size: Size of chunks in each dimension (default 128)
        compressor: Compression algorithm ('zstd', 'lz4', 'blosc', None)
        compression_level: Compression level (1-9, higher = more compression)
        verbose: Print progress information
        overwrite: If True, overwrite existing Zarr store
        
    Returns:
        Dict with metadata about the created Zarr store
    """
    folder_path = Path(folder_path)
    zarr_path = Path(zarr_path)
    
    # Check if Zarr already exists
    if zarr_path.exists():
        if overwrite:
            if verbose:
                print(f"Removing existing Zarr store: {zarr_path}")
            shutil.rmtree(zarr_path)
        else:
            raise FileExistsError(
                f"Zarr store already exists at {zarr_path}. "
                "Use overwrite=True to replace it."
            )
    
    # Import TomographyFolderProcessor
    try:
        from sdate.datasets.projection_triplet_dataset import (
            TomographyFolderProcessor,
        )
    except ImportError:
        raise ImportError(
            "Could not import TomographyFolderProcessor. "
            "Make sure sdate is in your Python path."
        )
    
    # Create processor
    processor = TomographyFolderProcessor(
        folder_path=folder_path,
        num_darks=None,
        num_flats=None,
        cache_in_memory=False,
        verbose=verbose,
        use_attenuation=False,
    )
    
    # Determine number of projections
    available = processor.num_projections
    if num_projections is None:
        num_projections = available - start_projection
    else:
        num_projections = min(num_projections, available - start_projection)
    
    # Get dimensions from first projection
    first_proj = processor.get_projection(start_projection, normalize=False)
    if target_size is not None:
        first_proj = _resize_projection(first_proj, target_size)
    
    proj_height, proj_width = first_proj.shape
    
    # Volume shape: (H, W, N_projections)
    volume_shape = (proj_height, proj_width, num_projections)
    
    if verbose:
        print(f"Creating Zarr store:")
        print(f"  Source: {folder_path}")
        print(f"  Destination: {zarr_path}")
        print(f"  Volume shape: {volume_shape}")
        print(f"  Chunk size: {chunk_size}")
    
    # Setup compressor (zarr 2.x API)
    if compressor == 'zstd':
        comp = zarr.codecs.Zstd(level=compression_level)
    elif compressor == 'lz4':
        comp = zarr.codecs.LZ4()
    elif compressor == 'blosc':
        comp = zarr.codecs.Blosc(cname='zstd', clevel=compression_level)
    else:
        comp = None
    
    # Determine chunk shape
    chunk_shape = (
        min(chunk_size, proj_height),
        min(chunk_size, proj_width),
        min(chunk_size, num_projections),
    )
    
    if verbose:
        print(f"  Chunk shape: {chunk_shape}")
        print(f"  Compressor: {compressor}")
    
    # Create Zarr store
    store = zarr.open_group(zarr_path, mode='w')
    
    # Create the volume array (zarr 2.x uses create_dataset)
    volume = store.create_dataset(
        'volume',
        shape=volume_shape,
        chunks=chunk_shape,
        dtype=np.float32,
        compressor=comp,
    )
    
    # Statistics for metadata
    global_min = float('inf')
    global_max = float('-inf')
    all_values_for_percentile = []
    sample_interval = max(1, num_projections // 20)  # Sample ~20 projections for percentiles
    
    # Process projections in batches
    batch_size = chunk_size  # Process chunk_size projections at a time
    n_batches = (num_projections + batch_size - 1) // batch_size
    
    if verbose:
        print(f"Processing {num_projections} projections in {n_batches} batches...")
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_projections)
        actual_batch_size = batch_end - batch_start
        
        # Load batch of projections
        batch_projections = np.zeros((proj_height, proj_width, actual_batch_size), dtype=np.float32)
        
        for i in range(actual_batch_size):
            proj_idx = start_projection + batch_start + i
            proj = processor.get_projection(proj_idx, normalize=False)
            if target_size is not None:
                proj = _resize_projection(proj, target_size)
            batch_projections[:, :, i] = proj.astype(np.float32)
            
            # Update statistics
            global_min = min(global_min, proj.min())
            global_max = max(global_max, proj.max())
            
            # Sample values for percentile calculation
            if (batch_start + i) % sample_interval == 0:
                # Sample a subset of values to avoid memory issues
                flat = proj.flatten()
                sample_size = min(10000, len(flat))
                sample_indices = np.random.choice(len(flat), sample_size, replace=False)
                all_values_for_percentile.extend(flat[sample_indices].tolist())
        
        # Write batch to Zarr
        volume[:, :, batch_start:batch_end] = batch_projections
        
        if verbose:
            progress = (batch_idx + 1) / n_batches * 100
            print(f"  Batch {batch_idx + 1}/{n_batches} ({progress:.1f}%)")
        
        # Free memory
        del batch_projections
    
    # Compute percentiles from sampled values
    all_values_for_percentile = np.array(all_values_for_percentile)
    percentile_1 = float(np.percentile(all_values_for_percentile, 1))
    percentile_99 = float(np.percentile(all_values_for_percentile, 99))
    percentile_5 = float(np.percentile(all_values_for_percentile, 5))
    percentile_95 = float(np.percentile(all_values_for_percentile, 95))
    mean_value = float(np.mean(all_values_for_percentile))
    std_value = float(np.std(all_values_for_percentile))
    
    # Create metadata
    metadata = {
        'source_folder': str(folder_path),
        'volume_shape': list(volume_shape),
        'chunk_shape': list(chunk_shape),
        'dtype': 'float32',
        'num_projections': num_projections,
        'start_projection': start_projection,
        'target_size': target_size,
        'compressor': compressor,
        'compression_level': compression_level,
        'statistics': {
            'min': float(global_min),
            'max': float(global_max),
            'percentile_1': percentile_1,
            'percentile_5': percentile_5,
            'percentile_95': percentile_95,
            'percentile_99': percentile_99,
            'mean': mean_value,
            'std': std_value,
            'value_range': float(global_max - global_min),
        }
    }
    
    # Store metadata in Zarr attrs
    store.attrs.update(metadata)
    
    # Also save as JSON for easy inspection
    metadata_path = zarr_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    if verbose:
        print(f"\n✅ Zarr store created successfully!")
        print(f"  Total voxels: {np.prod(volume_shape):,}")
        print(f"  Value range: [{global_min:.4f}, {global_max:.4f}]")
        print(f"  1st-99th percentile: [{percentile_1:.4f}, {percentile_99:.4f}]")
    
    return metadata


def _resize_projection(proj: np.ndarray, target_size: int) -> np.ndarray:
    """Resize a projection to target size."""
    from PIL import Image
    img = Image.fromarray(proj)
    img = img.resize((target_size, target_size), Image.BILINEAR)
    return np.array(img, dtype=np.float32)


def load_zarr_metadata(zarr_path: Union[str, Path]) -> Dict:
    """Load metadata from a Zarr store."""
    zarr_path = Path(zarr_path)
    
    # Try JSON first (faster)
    metadata_path = zarr_path / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    # Fall back to Zarr attrs
    store = zarr.open_group(zarr_path, mode='r')
    return dict(store.attrs)


class ChunkedProjectionDataset(Dataset):
    """
    Dataset for efficient random access to projection volumes stored in Zarr format.

    Instead of sampling random individual voxels, this dataset samples random
    *chunks* from the Zarr store, reads each chunk in a single contiguous I/O
    operation, and returns all voxels contained in those chunks. This is the
    fastest possible read pattern for Zarr: one syscall per chunk, zero wasted
    reads, fully sequential within each chunk.

    The interface is drop-in compatible with ProjectionVolumeDataset:
    both return {'coords': ..., 'values': ...} dicts with the same normalization.

    If the Zarr store does not exist it is created automatically from the source
    projection TIFF files.
    """

    def __init__(
        self,
        folder_path: Union[str, Path],
        zarr_path: Optional[Union[str, Path]] = None,
        num_projections: Optional[int] = None,
        target_size: Optional[int] = None,
        chunks_per_batch: int = 8,
        n_batches: int = 1000,
        normalize_values: bool = True,
        start_projection: int = 0,
        cache_volume: bool = False,
        chunk_size: int = 128,
        compressor: Optional[str] = 'zstd',
        compression_level: int = 3,
        verbose: bool = True,
        return_chunk_shape: bool = False,
    ):
        """
        Args:
            folder_path: Path to folder with tomographic data (TIFF projections)
            zarr_path: Path for the Zarr store. Defaults to <folder>_chunked.zarr
            num_projections: Number of projections to use (None = all)
            target_size: Resize projections to this size (None = original)
            chunks_per_batch: Number of random chunks to read per __getitem__ call.
                              Each chunk contains chunk_size^3 voxels (at most), so
                              the effective batch size is chunks_per_batch * voxels_per_chunk.
            n_batches: Number of batches per epoch
            normalize_values: Whether to normalize values to [0, 1]
            start_projection: Index of first projection to load
            cache_volume: If True, load entire volume into memory (fast path)
            chunk_size: Chunk edge length used when creating the Zarr store
            compressor: Compression algorithm ('zstd', 'lz4', 'blosc', None)
            compression_level: Compression level
            verbose: Print progress information
            return_chunk_shape: If True, return 'chunk_shape' in batch dict for DCT loss.
                                When True, only returns a single chunk per batch.
        """
        super().__init__()

        self.folder_path = Path(folder_path)
        self.chunks_per_batch = chunks_per_batch
        self.n_batches = n_batches
        self.normalize_values = normalize_values
        self.verbose = verbose
        self.cache_volume = cache_volume
        self.return_chunk_shape = return_chunk_shape

        # Determine Zarr path
        if zarr_path is None:
            zarr_path = self.folder_path.parent / f"{self.folder_path.name}_chunked.zarr"
        self.zarr_path = Path(zarr_path)

        # Create or load Zarr store
        if not self.zarr_path.exists():
            if verbose:
                print(f"Zarr store not found. Creating from projections...")
            self.metadata = create_zarr_from_projections(
                folder_path=folder_path,
                zarr_path=self.zarr_path,
                num_projections=num_projections,
                start_projection=start_projection,
                target_size=target_size,
                chunk_size=chunk_size,
                compressor=compressor,
                compression_level=compression_level,
                verbose=verbose,
            )
        else:
            if verbose:
                print(f"Loading existing Zarr store: {self.zarr_path}")
            self.metadata = load_zarr_metadata(self.zarr_path)

        # Open Zarr store
        self.store = zarr.open_group(self.zarr_path, mode='r')
        self.zarr_volume = self.store['volume']

        # Shape / chunk info from metadata
        self.shape = tuple(self.metadata['volume_shape'])
        self.n_dims = 3
        self.n_voxels = int(np.prod(self.shape))
        self.chunk_shape = tuple(self.metadata['chunk_shape'])

        # Total number of chunks along each axis
        self._n_chunks = tuple(
            (self.shape[i] + self.chunk_shape[i] - 1) // self.chunk_shape[i]
            for i in range(3)
        )
        self._total_chunks = int(np.prod(self._n_chunks))

        # Normalization parameters
        stats = self.metadata['statistics']
        self.value_min = stats['min']
        self.value_max = stats['max']
        self.value_range = max(stats['value_range'], 1e-6)
        self.percentile_1 = stats['percentile_1']
        self.percentile_99 = stats['percentile_99']

        # Effective voxels per chunk (worst-case full chunk)
        self._voxels_per_full_chunk = int(np.prod(self.chunk_shape))

        if verbose:
            print(f"\nChunkedProjectionDataset initialized:")
            print(f"  Volume shape:     {self.shape}")
            print(f"  Chunk shape:      {self.chunk_shape}")
            print(f"  Total chunks:     {self._total_chunks:,}")
            print(f"  Chunks per batch: {chunks_per_batch}")
            print(f"  ~Voxels/batch:    {chunks_per_batch * self._voxels_per_full_chunk:,}")
            print(f"  Value range:      [{self.value_min:.4f}, {self.value_max:.4f}]")

        # Precompute coordinate scaling factors
        self.coord_scales = torch.tensor([
            1.0 / max(self.shape[0] - 1, 1),
            1.0 / max(self.shape[1] - 1, 1),
            1.0 / max(self.shape[2] - 1, 1),
        ], dtype=torch.float32)

        # Optionally cache the full volume
        self.volume = None
        if cache_volume:
            self._load_full_volume()

    # ------------------------------------------------------------------
    # Convenience property to keep API parity with ProjectionVolumeDataset
    # ------------------------------------------------------------------
    @property
    def batch_size(self) -> int:
        """Approximate voxels per batch (chunks_per_batch × voxels_per_chunk)."""
        return self.chunks_per_batch * self._voxels_per_full_chunk

    def _load_full_volume(self):
        """Load the entire volume into memory (cache_volume=True path)."""
        if self.verbose:
            print("Loading full volume into memory...")
        self.volume = torch.from_numpy(self.zarr_volume[:]).float()

        grids = [torch.arange(s, dtype=torch.float32) / max(s - 1, 1) for s in self.shape]
        self.all_coords = torch.stack(
            torch.meshgrid(*grids, indexing='ij'), dim=-1
        ).reshape(-1, self.n_dims)

        self.all_values = self.volume.flatten()
        if self.normalize_values:
            self.all_values = (self.all_values - self.value_min) / self.value_range

        if self.verbose:
            print("  Volume cached in memory.")

    def __len__(self) -> int:
        return self.n_batches

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return all voxels from randomly sampled chunks."""
        # When return_chunk_shape=True, return single chunk with full 3D structure preserved
        # This allows the Trainer to properly reshape for DCT operations
        if self.return_chunk_shape:
            return self._get_chunk_with_structure()
        
        if self.volume is not None:
            return self._get_batch_cached()
        return self._get_batch_chunks()

    def _get_chunk_with_structure(self) -> Dict[str, torch.Tensor]:
        """
        Sample a single random chunk and return with full 3D structure preserved.
        
        Instead of flattening the chunk into a 1D list of voxels, this method
        returns the chunk as a 3D tensor along with coordinates in 3D grid form.
        This makes it easy for the Trainer to know the exact chunk dimensions
        without needing to pass chunk_shape separately.
        
        Returns:
            dict with:
            - 'coords': (H, W, N, 3) coordinate tensor
            - 'values': (H, W, N, 1) value tensor  
            - 'chunk_shape': (3,) tensor with [H, W, N]
        """
        ch, cw, cn = self.chunk_shape
        nch_h, nch_w, nch_n = self._n_chunks
        
        # Sample a single random chunk
        flat_id = np.random.randint(0, self._total_chunks)
        
        # Decode flat chunk id → (chunk_h, chunk_w, chunk_n)
        ci_h = flat_id // (nch_w * nch_n)
        remainder = flat_id % (nch_w * nch_n)
        ci_w = remainder // nch_n
        ci_n = remainder % nch_n
        
        # Volume-space boundaries for this chunk
        h0 = ci_h * ch;  h1 = min(h0 + ch, self.shape[0])
        w0 = ci_w * cw;  w1 = min(w0 + cw, self.shape[1])
        n0 = ci_n * cn;  n1 = min(n0 + cn, self.shape[2])
        
        # Single contiguous Zarr read
        chunk_data = self.zarr_volume[h0:h1, w0:w1, n0:n1]  # (dh, dw, dn) float32
        
        dh, dw, dn = chunk_data.shape
        
        # Build coordinate grid (H, W, N, 3) - preserves 3D structure
        h_idx = torch.arange(h0, h1, dtype=torch.float32)
        w_idx = torch.arange(w0, w1, dtype=torch.float32)
        n_idx = torch.arange(n0, n1, dtype=torch.float32)
        
        hh, ww, nn = torch.meshgrid(h_idx, w_idx, n_idx, indexing='ij')
        coords = torch.stack([
            hh * self.coord_scales[0],
            ww * self.coord_scales[1],
            nn * self.coord_scales[2],
        ], dim=-1)  # (dh, dw, dn, 3)
        
        values = torch.from_numpy(chunk_data).float()  # (dh, dw, dn)
        
        if self.normalize_values:
            values = (values - self.value_min) / self.value_range
        
        return {
            'coords': coords,  # (H, W, N, 3) - preserves structure!
            'values': values.unsqueeze(-1),  # (H, W, N, 1)
            'chunk_shape': torch.tensor([dh, dw, dn], dtype=torch.int32),
        }

    def _get_batch_cached(self) -> Dict[str, torch.Tensor]:
        """Fast path when volume is fully cached."""
        indices = torch.randint(0, self.n_voxels, (self.batch_size,))
        return {
            'coords': self.all_coords[indices],
            'values': self.all_values[indices].unsqueeze(-1),
        }

    def _get_batch_chunks(self) -> Dict[str, torch.Tensor]:
        """
        Sample random chunks and return every voxel inside them.

        One Zarr read per chunk — the optimal I/O pattern.
        """
        ch, cw, cn = self.chunk_shape
        nch_h, nch_w, nch_n = self._n_chunks

        # Sample chunk indices (flat), without replacement when possible
        n_sample = min(self.chunks_per_batch, self._total_chunks)
        flat_chunk_ids = np.random.choice(self._total_chunks, size=n_sample, replace=False)

        all_coords = []
        all_values = []

        for flat_id in flat_chunk_ids:
            # Decode flat chunk id → (chunk_h, chunk_w, chunk_n)
            ci_h = flat_id // (nch_w * nch_n)
            remainder = flat_id % (nch_w * nch_n)
            ci_w = remainder // nch_n
            ci_n = remainder % nch_n

            # Volume-space boundaries for this chunk
            h0 = ci_h * ch;  h1 = min(h0 + ch, self.shape[0])
            w0 = ci_w * cw;  w1 = min(w0 + cw, self.shape[1])
            n0 = ci_n * cn;  n1 = min(n0 + cn, self.shape[2])

            # Single contiguous Zarr read
            chunk_data = self.zarr_volume[h0:h1, w0:w1, n0:n1]  # (dh, dw, dn) float32

            dh, dw, dn = chunk_data.shape
            n_vox = dh * dw * dn

            # Build coordinates for every voxel in the chunk
            h_idx = torch.arange(h0, h1, dtype=torch.float32)
            w_idx = torch.arange(w0, w1, dtype=torch.float32)
            n_idx = torch.arange(n0, n1, dtype=torch.float32)

            hh, ww, nn = torch.meshgrid(h_idx, w_idx, n_idx, indexing='ij')
            coords = torch.stack([
                hh.flatten() * self.coord_scales[0],
                ww.flatten() * self.coord_scales[1],
                nn.flatten() * self.coord_scales[2],
            ], dim=-1)  # (n_vox, 3)

            values = torch.from_numpy(chunk_data.reshape(-1))  # (n_vox,)

            all_coords.append(coords)
            all_values.append(values)

        coords = torch.cat(all_coords, dim=0)
        values = torch.cat(all_values, dim=0).float()

        if self.normalize_values:
            values = (values - self.value_min) / self.value_range

        return {
            'coords': coords,
            'values': values.unsqueeze(-1),
        }

    def denormalize_values(self, values: torch.Tensor) -> torch.Tensor:
        """Convert normalized values back to original range."""
        if self.normalize_values:
            return values * self.value_range + self.value_min
        return values

    def get_full_volume(self) -> torch.Tensor:
        """Return the full volume tensor."""
        if self.volume is None:
            self._load_full_volume()
        return self.volume

    def get_slice(self, dim: int, idx: int) -> torch.Tensor:
        """Get a 2D slice along a dimension."""
        if dim == 0:
            data = self.zarr_volume[idx, :, :]
        elif dim == 1:
            data = self.zarr_volume[:, idx, :]
        elif dim == 2:
            data = self.zarr_volume[:, :, idx]
        else:
            raise ValueError(f"Invalid dimension: {dim}")
        return torch.from_numpy(data).float()

    def get_statistics(self) -> Dict:
        """Return the volume statistics from metadata."""
        return self.metadata['statistics']

    def get_metadata(self) -> Dict:
        """Return full metadata."""
        return self.metadata


class ChunkedSliceDataset(Dataset):
    """
    Slice-based dataset that reads from Zarr store.
    
    Combines the benefits of slice-based sampling with efficient Zarr storage.
    Samples entire slices but can subsample voxels if batch_size is specified.
    """
    
    def __init__(
        self,
        folder_path: Union[str, Path],
        zarr_path: Optional[Union[str, Path]] = None,
        num_projections: Optional[int] = None,
        target_size: Optional[int] = None,
        slices_per_batch: int = 1,
        batch_size: Optional[int] = None,
        n_batches: int = 1000,
        normalize_values: bool = True,
        start_projection: int = 0,
        slice_dim: int = 2,
        chunk_size: int = 128,
        compressor: Optional[str] = 'zstd',
        compression_level: int = 3,
        verbose: bool = True,
    ):
        """
        Args:
            folder_path: Path to folder with tomographic data
            zarr_path: Path for the Zarr store
            num_projections: Number of projections to use
            target_size: Resize projections to this size
            slices_per_batch: Number of slices to sample per batch
            batch_size: Max voxels per batch (None = all voxels from slices)
            n_batches: Number of batches per epoch
            normalize_values: Whether to normalize values
            start_projection: Index of first projection
            slice_dim: Dimension to slice along
            chunk_size: Chunk size for Zarr
            compressor: Compression for Zarr
            compression_level: Compression level
            verbose: Print progress
        """
        super().__init__()
        
        self.folder_path = Path(folder_path)
        self.slices_per_batch = slices_per_batch
        self.batch_size = batch_size
        self.n_batches = n_batches
        self.normalize_values = normalize_values
        self.slice_dim = slice_dim
        self.verbose = verbose
        
        # Determine Zarr path
        if zarr_path is None:
            zarr_path = self.folder_path.parent / f"{self.folder_path.name}_chunked.zarr"
        self.zarr_path = Path(zarr_path)
        
        # Create or load Zarr store
        if not self.zarr_path.exists():
            if verbose:
                print(f"Zarr store not found. Creating from projections...")
            self.metadata = create_zarr_from_projections(
                folder_path=folder_path,
                zarr_path=self.zarr_path,
                num_projections=num_projections,
                start_projection=start_projection,
                target_size=target_size,
                chunk_size=chunk_size,
                compressor=compressor,
                compression_level=compression_level,
                verbose=verbose,
            )
        else:
            if verbose:
                print(f"Loading existing Zarr store: {self.zarr_path}")
            self.metadata = load_zarr_metadata(self.zarr_path)
        
        # Open Zarr store
        self.store = zarr.open_group(self.zarr_path, mode='r')
        self.zarr_volume = self.store['volume']
        
        # Extract shape and stats
        self.shape = tuple(self.metadata['volume_shape'])
        self.n_dims = 3
        self.n_voxels = int(np.prod(self.shape))
        
        # Normalization
        stats = self.metadata['statistics']
        self.value_min = stats['min']
        self.value_max = stats['max']
        self.value_range = max(stats['value_range'], 1e-6)
        
        # Voxels per slice
        self.voxels_per_slice = [
            self.shape[1] * self.shape[2],
            self.shape[0] * self.shape[2],
            self.shape[0] * self.shape[1],
        ]
        
        # Precompute coordinate grids for slices
        self._precompute_slice_coords()
        
        if verbose:
            print(f"\nChunkedSliceDataset initialized:")
            print(f"  Volume shape: {self.shape}")
            print(f"  Slice dimension: {slice_dim}")
            print(f"  Slices per batch: {slices_per_batch}")
            print(f"  Voxels per slice: {self.voxels_per_slice[slice_dim]:,}")
            if batch_size:
                print(f"  Batch size limit: {batch_size:,}")
    
    def _precompute_slice_coords(self):
        """Precompute normalized coordinates for slices."""
        grids = [
            torch.arange(s, dtype=torch.float32) / max(s - 1, 1) 
            for s in self.shape
        ]
        
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
            
            # Read slice from Zarr
            if self.slice_dim == 2:
                slice_data = self.zarr_volume[:, :, slice_idx]
                slice_values = torch.from_numpy(slice_data).float().flatten()
                slice_coord_norm = slice_idx / max(self.shape[2] - 1, 1)
                coords = torch.stack([
                    self.slice_h_coords,
                    self.slice_w_coords,
                    torch.full((voxels_per_slice,), slice_coord_norm),
                ], dim=-1)
            elif self.slice_dim == 0:
                slice_data = self.zarr_volume[slice_idx, :, :]
                slice_values = torch.from_numpy(slice_data).float().flatten()
                slice_coord_norm = slice_idx / max(self.shape[0] - 1, 1)
                coords = torch.stack([
                    torch.full((voxels_per_slice,), slice_coord_norm),
                    self.slice_w_coords,
                    self.slice_n_coords,
                ], dim=-1)
            elif self.slice_dim == 1:
                slice_data = self.zarr_volume[:, slice_idx, :]
                slice_values = torch.from_numpy(slice_data).float().flatten()
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
                indices = torch.randperm(total_voxels)[:self.batch_size]
                data['coords'] = data['coords'][indices]
                data['values'] = data['values'][indices]
        
        return data
    
    def denormalize_values(self, values: torch.Tensor) -> torch.Tensor:
        """Convert normalized values back to original range."""
        if self.normalize_values:
            return values * self.value_range + self.value_min
        return values
    
    def get_slice(self, dim: int, idx: int) -> torch.Tensor:
        """Get a 2D slice along a dimension."""
        if dim == 0:
            data = self.zarr_volume[idx, :, :]
        elif dim == 1:
            data = self.zarr_volume[:, idx, :]
        elif dim == 2:
            data = self.zarr_volume[:, :, idx]
        else:
            raise ValueError(f"Invalid dimension: {dim}")
        return torch.from_numpy(data).float()
    
    def get_batch_size(self) -> int:
        """Return the effective batch size."""
        if self.batch_size is not None:
            return self.batch_size
        return self.slices_per_batch * self.voxels_per_slice[self.slice_dim]
