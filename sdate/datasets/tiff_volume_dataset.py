import os
import glob
import torch
from torch.utils.data import Dataset
import tifffile
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

# HEIC support
try:
    import pillow_heif
    from PIL import Image
    pillow_heif.register_heif_opener()
    HEIC_AVAILABLE = True
except ImportError:
    HEIC_AVAILABLE = False
    print("Warning: pillow-heif not available. HEIC compression disabled.")
    print("Install with: pip install pillow-heif")


class TiffVolumeDataset(Dataset):
    """
    A PyTorch Dataset for loading a sequence of TIFF files as a 3D volume and splitting it
    into overlapping sub-volumes. Features parallel TIFF loading with progress bars and
    returns both sub-volumes and their position indices.
    
    The dataset loads NUM_FRAMES TIFF images from a folder to form a 3D tensor of shape 
    (NUM_FRAMES, Height, Width), then splits this volume into smaller cubic volumes of size
    (w, w, w) with configurable overlap controlled by a stride parameter.
    
    Args:
        data_path (str or Path): Path to the directory containing TIFF files.
        num_frames (int): Number of TIFF frames to load from the sequence.
        volume_size (int): Size of the cubic sub-volumes (w x w x w). Default: 64.
        stride (int): Stride for sliding window. Controls overlap between volumes.
                     For example, stride=32 with volume_size=64 gives 50% overlap.
                     stride=volume_size means no overlap. Default: None (same as volume_size).
        start_offset (int): Frame offset to start loading from. Default: 0.
        clip_range (tuple, optional): Tuple (min, max) to clip tensor values.
        normalize (bool): Whether to normalize each volume to [0, 1]. Default: False.
        global_normalize (bool): Whether to normalize using global min/max from all frames.
                                Default: False.
        transform (callable, optional): Optional transform to apply to each sub-volume.
        max_workers (int): Number of threads for parallel TIFF loading. Default: 8.
        dtype (torch.dtype): Data type for the tensors. Default: torch.float32.
        use_heic_compression (bool): Whether to create/use HEIC compressed versions. Default: False.
        heic_quality (int): HEIC compression quality (0-100, higher = better). Default: 85.
        heic_subfolder (str): Name of subfolder for HEIC files. Default: 'heic'.
        dual_channel (bool): Whether to return dual-channel data (TIFF + HEIC). Default: True when use_heic_compression=True.
        use_residuals (bool): Whether to load and include residuals as a third channel. Default: False.
        residuals_path (str, optional): Path to residuals .npy file. Required if use_residuals=True.
        
    Attributes:
        volume (torch.Tensor): The full 3D volume. Shape depends on dual_channel:
                              - Single channel: (NUM_FRAMES, Height, Width)  
                              - Dual channel: (NUM_FRAMES, 2, Height, Width) where dim 1 is [TIFF, HEIC]
        volume_indices (list): List of (d_start, h_start, w_start) tuples for each sub-volume.
        residuals (np.ndarray, optional): Memory-mapped residuals array of shape (num_subvolumes, volume_size, volume_size, volume_size).
        residuals_positions (np.ndarray, optional): Position indices for residuals array of shape (num_subvolumes, 3).
        
    Returns:
        __getitem__ returns a tuple of (sub_volume, indices_info) where:
        - sub_volume: torch.Tensor of shape:
                     - Single channel: (volume_size, volume_size, volume_size)
                     - Dual channel: (2, volume_size, volume_size, volume_size) where dim 0 is [TIFF, HEIC]
                     - With residuals: (3, volume_size, volume_size, volume_size) where dim 0 is [TIFF, HEIC, RESIDUAL]
        - indices_info: Dict with position, shape, and metadata information
        
    Example:
        >>> dataset = TiffVolumeDataset(
        ...     data_path='./data/tiff_sequence/',
        ...     num_frames=100,
        ...     volume_size=64,
        ...     stride=32
        ... )
        >>> print(f"Dataset size: {len(dataset)}")
        >>> sub_volume, info = dataset[0]  # Returns volume + position info
        >>> print(f"Volume shape: {sub_volume.shape}")
        >>> print(f"Position: {info['position']}")
    """
    
    def __init__(
        self,
        data_path: Union[str, Path],
        num_frames: int,
        volume_size: int = 64,
        stride: Optional[int] = None,
        start_offset: int = 0,
        clip_range: Optional[Tuple[float, float]] = None,
        normalize: bool = False,
        global_normalize: bool = False,
        transform: Optional[callable] = None,
        max_workers: int = 8,
        dtype: torch.dtype = torch.float32,
        use_heic_compression: bool = False,
        heic_quality: int = 85,
        heic_subfolder: str = 'heic',
        dual_channel: Optional[bool] = None,
        use_residuals: bool = False,
        residuals_path: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        
        self.data_path = Path(data_path)
        self.num_frames = num_frames
        self.volume_size = volume_size
        self.stride = stride if stride is not None else volume_size
        self.start_offset = start_offset
        self.clip_range = clip_range
        self.normalize = normalize
        self.global_normalize = global_normalize
        self.transform = transform
        self.max_workers = max_workers
        self.dtype = dtype
        
        # HEIC compression parameters
        self.use_heic_compression = use_heic_compression
        self.heic_quality = heic_quality
        self.heic_subfolder = heic_subfolder
        self.dual_channel = dual_channel if dual_channel is not None else use_heic_compression
        self.heic_path = self.data_path / self.heic_subfolder
        
        # Residuals parameters
        self.use_residuals = use_residuals
        self.residuals_path = Path(residuals_path) if residuals_path else None
        self.residuals = None
        self.residuals_positions = None
        
        # Validate residuals configuration
        if self.use_residuals:
            if not self.residuals_path:
                raise ValueError("residuals_path must be provided when use_residuals=True")
            if not self.residuals_path.exists():
                raise ValueError(f"Residuals file not found: {self.residuals_path}")
            if not self.dual_channel:
                raise ValueError("use_residuals=True requires dual_channel=True (need HEIC channel)")

        
        # Validate HEIC availability
        if self.use_heic_compression and not HEIC_AVAILABLE:
            raise ImportError(
                "HEIC compression requested but pillow-heif not available. "
                "Install with: pip install pillow-heif"
            )
        
        # Validate inputs
        if not self.data_path.exists():
            raise ValueError(f"Data path does not exist: {self.data_path}")
        
        if self.stride <= 0 or self.stride > self.volume_size:
            raise ValueError(f"Stride must be positive and <= volume_size. Got stride={self.stride}, volume_size={self.volume_size}")
        
        # Load the 3D volume (with HEIC compression if requested)
        print(f"Loading {self.num_frames} TIFF frames from {self.data_path}")
        if self.use_heic_compression:
            print(f"HEIC compression enabled (quality={self.heic_quality}, dual_channel={self.dual_channel})")
        
        self.volume, self.global_min, self.global_max = self._load_volume()
        
        # Calculate sub-volume indices
        self.volume_indices = self._calculate_volume_indices()
        
        # Load residuals if requested
        if self.use_residuals:
            self._load_residuals()

        
        print(f"Created TiffVolumeDataset:")
        print(f"  Volume shape: {self.volume.shape}")
        print(f"  Sub-volume size: {self.volume_size}x{self.volume_size}x{self.volume_size}")
        print(f"  Stride: {self.stride}")
        print(f"  Number of sub-volumes: {len(self.volume_indices)}")
        print(f"  Dataset length: {len(self)}")
        if self.use_residuals:
            print(f"  Residuals loaded: shape={self.residuals.shape}")
    
    def _load_residuals(self):
        """Load residuals from disk using memory-mapping for efficiency."""
        print(f"Loading residuals from {self.residuals_path}")
        
        # Load residuals (memory-mapped for efficiency)
        self.residuals = np.load(self.residuals_path, mmap_mode='r')
        
        # Load positions
        positions_file = str(self.residuals_path).replace('_residuals.npy', '_positions.npy')
        if Path(positions_file).exists():
            self.residuals_positions = np.load(positions_file)
        else:
            raise ValueError(f"Residuals positions file not found: {positions_file}")
        
        # Load and validate metadata
        metadata_file = str(self.residuals_path).replace('_residuals.npy', '_metadata.npz')
        if Path(metadata_file).exists():
            metadata = np.load(metadata_file)
            
            # Validate compatibility
            if metadata['volume_size'] != self.volume_size:
                raise ValueError(
                    f"Residuals volume_size ({metadata['volume_size']}) does not match "
                    f"dataset volume_size ({self.volume_size})"
                )
            if metadata['stride'] != self.stride:
                print(f"Warning: Residuals stride ({metadata['stride']}) differs from dataset stride ({self.stride})")
            if metadata['num_frames'] != self.num_frames:
                raise ValueError(
                    f"Residuals num_frames ({metadata['num_frames']}) does not match "
                    f"dataset num_frames ({self.num_frames})"
                )
        
        # Validate shapes
        expected_num_subvolumes = len(self.volume_indices)
        if self.residuals.shape[0] != expected_num_subvolumes:
            raise ValueError(
                f"Number of residual sub-volumes ({self.residuals.shape[0]}) does not match "
                f"expected number ({expected_num_subvolumes})"
            )
        
        print(f"  Residuals shape: {self.residuals.shape}")
        print(f"  Residuals range: [{self.residuals.min():.4f}, {self.residuals.max():.4f}]")
        print(f"  Residuals mean: {self.residuals.mean():.4f}, std: {self.residuals.std():.4f}")
        
    def _load_volume(self) -> Tuple[torch.Tensor, float, float]:
        """
        Load TIFF files and optionally create/load HEIC compressed versions.
        Returns single-channel or dual-channel volume based on configuration.
        
        Returns:
            Tuple of (volume tensor, global_min, global_max)
        """
        # Collect all TIFF file paths in the directory
        tiff_files = sorted(
            glob.glob(os.path.join(self.data_path, '*.tif')) +
            glob.glob(os.path.join(self.data_path, '*.tiff'))
        )
        
        if len(tiff_files) == 0:
            raise ValueError(f"No TIFF files found in {self.data_path}")
        
        # Select the frames to load
        end_idx = self.start_offset + self.num_frames
        if end_idx > len(tiff_files):
            raise ValueError(
                f"Not enough TIFF files. Requested frames {self.start_offset} to {end_idx}, "
                f"but only {len(tiff_files)} files available."
            )
        
        selected_files = tiff_files[self.start_offset:end_idx]
        print(f"  Loading frames {self.start_offset} to {end_idx-1} ({len(selected_files)} files)")
        
        # Load TIFF data
        tiff_frames = self._load_tiff_frames(selected_files)
        
        # Handle HEIC compression if enabled
        if self.use_heic_compression:
            heic_frames = self._create_or_load_heic_frames(selected_files, tiff_frames)
            
            if self.dual_channel:
                # Create dual-channel volume: (num_frames, 2, height, width)
                volume_np = np.stack([tiff_frames, heic_frames], axis=1)
                print(f"  Created dual-channel volume: TIFF + HEIC")
            else:
                # Use only HEIC frames
                volume_np = heic_frames
                print(f"  Using HEIC compressed frames only")
        else:
            # Use only TIFF frames
            volume_np = tiff_frames
        
        # Calculate global min/max for normalization across all channels
        global_min = float(volume_np.min())
        global_max = float(volume_np.max())
        
        # Convert to torch tensor
        volume = torch.from_numpy(volume_np).to(self.dtype)
        
        print(f"  Volume loaded: shape={volume.shape}, dtype={volume.dtype}")
        print(f"  Value range: [{global_min:.2f}, {global_max:.2f}]")
        
        return volume, global_min, global_max
    
    def _load_tiff_frames(self, tiff_files: list) -> np.ndarray:
        """Load TIFF frames in parallel with progress bar."""
        def load_tiff_with_index(args):
            idx, path = args
            image = tifffile.imread(path)
            if not isinstance(image, np.ndarray):
                image = np.array(image)
            # If multi-frame TIFF, take the first frame
            if image.ndim > 2:
                image = image[0]
            return idx, image.astype(np.float32)
        
        # Prepare arguments for parallel loading
        loading_args = [(i, path) for i, path in enumerate(tiff_files)]
        frames = [None] * len(tiff_files)
        
        # Load TIFF files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(load_tiff_with_index, args): args[0] 
                for args in loading_args
            }
            
            with tqdm(total=len(tiff_files), desc="Loading TIFF files", unit="files") as pbar:
                for future in as_completed(future_to_idx):
                    try:
                        idx, frame = future.result()
                        frames[idx] = frame
                        pbar.update(1)
                    except Exception as e:
                        pbar.write(f"Error loading TIFF file {tiff_files[future_to_idx[future]]}: {e}")
                        raise
        
        if any(frame is None for frame in frames):
            raise RuntimeError("Some TIFF files failed to load")
        
        return np.stack(frames, axis=0)  # Shape: (num_frames, height, width)
    
    def _create_or_load_heic_frames(self, tiff_files: list, tiff_frames: np.ndarray) -> np.ndarray:
        """Create HEIC compressed versions or load existing ones."""
        # Create HEIC directory if it doesn't exist
        self.heic_path.mkdir(exist_ok=True)
        
        # Check which HEIC files already exist
        heic_files = []
        missing_indices = []
        
        for i, tiff_file in enumerate(tiff_files):
            tiff_path = Path(tiff_file)
            heic_file = self.heic_path / f"{tiff_path.stem}.heic"
            heic_files.append(heic_file)
            
            if not heic_file.exists():
                missing_indices.append(i)
        
        # Create missing HEIC files
        if missing_indices:
            print(f"  Creating {len(missing_indices)} HEIC compressed files...")
            self._create_heic_files(missing_indices, tiff_frames, heic_files)
        else:
            print(f"  All HEIC files already exist, loading...")
        
        # Load HEIC frames
        return self._load_heic_frames(heic_files)
    
    def _create_heic_files(self, indices: list, tiff_frames: np.ndarray, heic_files: list):
        """Create HEIC compressed files from TIFF frames."""
        def create_heic_with_index(args):
            idx, frame, output_path = args
            try:
                # Normalize frame to 0-255 range for PIL
                frame_normalized = frame.copy()
                frame_min, frame_max = frame_normalized.min(), frame_normalized.max()
                if frame_max > frame_min:
                    frame_normalized = (frame_normalized - frame_min) / (frame_max - frame_min) * 255
                frame_normalized = frame_normalized.astype(np.uint8)
                
                # Convert to PIL Image and save as HEIC
                pil_image = Image.fromarray(frame_normalized, mode='L')
                pil_image.save(output_path, 'HEIF', quality=self.heic_quality)
                return idx, True
            except Exception as e:
                return idx, f"Error creating HEIC file {output_path}: {e}"
        
        # Prepare arguments for parallel HEIC creation
        creation_args = [
            (idx, tiff_frames[idx], heic_files[idx]) 
            for idx in indices
        ]
        
        # Create HEIC files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(create_heic_with_index, args): args[0] 
                for args in creation_args
            }
            
            with tqdm(total=len(indices), desc="Creating HEIC files", unit="files") as pbar:
                for future in as_completed(future_to_idx):
                    idx, result = future.result()
                    if result is not True:
                        pbar.write(f"Warning: {result}")
                    pbar.update(1)
    
    def _load_heic_frames(self, heic_files: list) -> np.ndarray:
        """Load HEIC frames in parallel."""
        def load_heic_with_index(args):
            idx, path = args
            try:
                with Image.open(path) as img:
                    # Convert to grayscale if needed and then to array
                    if img.mode != 'L':
                        img = img.convert('L')
                    return idx, np.array(img).astype(np.float32)
            except Exception as e:
                raise RuntimeError(f"Error loading HEIC file {path}: {e}")
        
        # Prepare arguments for parallel loading
        loading_args = [(i, path) for i, path in enumerate(heic_files)]
        frames = [None] * len(heic_files)
        
        # Load HEIC files in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(load_heic_with_index, args): args[0] 
                for args in loading_args
            }
            
            with tqdm(total=len(heic_files), desc="Loading HEIC files", unit="files") as pbar:
                for future in as_completed(future_to_idx):
                    try:
                        idx, frame = future.result()
                        frames[idx] = frame
                        pbar.update(1)
                    except Exception as e:
                        pbar.write(f"Error: {e}")
                        raise
        
        if any(frame is None for frame in frames):
            raise RuntimeError("Some HEIC files failed to load")
        
        return np.stack(frames, axis=0)  # Shape: (num_frames, height, width)
    
    def _calculate_volume_indices(self) -> list:
        """
        Calculate the starting indices for all sub-volumes using a sliding window approach.
        
        Returns:
            List of tuples (d_start, h_start, w_start) for each sub-volume.
        """
        # Handle different volume shapes based on channel configuration
        if self.dual_channel:
            # Volume shape: (num_frames, 2, height, width)
            depth, channels, height, width = self.volume.shape
        else:
            # Volume shape: (num_frames, height, width)
            depth, height, width = self.volume.shape
        
        # Calculate how many sub-volumes fit in each dimension
        depth_positions = self._get_positions(depth, self.volume_size, self.stride)
        height_positions = self._get_positions(height, self.volume_size, self.stride)
        width_positions = self._get_positions(width, self.volume_size, self.stride)
        
        # Generate all combinations of starting positions
        indices = []
        for d_start in depth_positions:
            for h_start in height_positions:
                for w_start in width_positions:
                    indices.append((d_start, h_start, w_start))
        
        print(f"  Sub-volumes per dimension: D={len(depth_positions)}, H={len(height_positions)}, W={len(width_positions)}")
        
        return indices
    
    @staticmethod
    def _get_positions(dimension_size: int, volume_size: int, stride: int) -> list:
        """
        Calculate starting positions for sliding window along one dimension.
        
        Args:
            dimension_size: Size of the dimension (e.g., depth, height, or width).
            volume_size: Size of the sub-volume in this dimension.
            stride: Step size for the sliding window.
            
        Returns:
            List of starting positions.
        """
        if dimension_size < volume_size:
            raise ValueError(
                f"Dimension size ({dimension_size}) is smaller than volume size ({volume_size}). "
                f"Cannot extract sub-volumes."
            )
        
        positions = []
        pos = 0
        while pos + volume_size <= dimension_size:
            positions.append(pos)
            pos += stride
        
        # If we didn't reach the end, add one more position to cover the remainder
        if positions[-1] + volume_size < dimension_size:
            positions.append(dimension_size - volume_size)
        
        return positions
    
    def __len__(self) -> int:
        """Return the number of sub-volumes in the dataset."""
        return len(self.volume_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sub-volume by index along with its position information.
        
        Args:
            idx: Index of the sub-volume.
            
        Returns:
            Tuple of (sub_volume, indices) where:
            - sub_volume: torch.Tensor of shape (C, D, H, W) where:
                         C=1 for single-channel
                         C=2 for dual-channel (TIFF, HEIC)
                         C=3 with residuals (TIFF, HEIC, RESIDUAL)
            - indices: torch.Tensor of shape (3,) containing [d_start, h_start, w_start]
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        # Get starting indices for this sub-volume
        d_start, h_start, w_start = self.volume_indices[idx]
        
        # Extract the sub-volume
        if self.dual_channel:
            # Handle dual-channel data: volume shape is (D, 2, H, W)
            sub_volume = self.volume[
                d_start:d_start + self.volume_size,
                :,  # Keep both channels
                h_start:h_start + self.volume_size,
                w_start:w_start + self.volume_size
            ].clone()
            # Rearrange to (C, D, H, W) format
            sub_volume = sub_volume.permute(1, 0, 2, 3)  # (2, D, H, W)
        else:
            # Handle single-channel data: volume shape is (D, H, W)
            sub_volume = self.volume[
                d_start:d_start + self.volume_size,
                h_start:h_start + self.volume_size,
                w_start:w_start + self.volume_size
            ].clone()
            # Add channel dimension: (D, H, W) -> (1, D, H, W)
            sub_volume = sub_volume.unsqueeze(0)
        
        # Add residuals as third channel if available
        if self.use_residuals:
            # Load residual for this sub-volume from memory-mapped array
            residual = self.residuals[idx]  # Shape: (D, H, W)
            residual_tensor = torch.from_numpy(residual.copy()).to(self.dtype)
            residual_tensor = residual_tensor.unsqueeze(0)  # Shape: (1, D, H, W)
            
            # Concatenate: (2, D, H, W) + (1, D, H, W) -> (3, D, H, W)
            sub_volume = torch.cat([sub_volume, residual_tensor], dim=0)
        
        # Apply clipping if specified
        if self.clip_range is not None:
            sub_volume = torch.clamp(sub_volume, self.clip_range[0], self.clip_range[1])
        
        # Apply normalization
        if self.normalize:
            if self.global_normalize:
                # Normalize using global min/max from the entire volume
                sub_volume = (sub_volume - self.global_min) / (self.global_max - self.global_min + 1e-8)
            else:
                # Normalize using local min/max from this sub-volume
                vol_min = sub_volume.min()
                vol_max = sub_volume.max()
                sub_volume = (sub_volume - vol_min) / (vol_max - vol_min + 1e-8)
        
        # Apply optional transform
        if self.transform is not None:
            sub_volume = self.transform(sub_volume)

        return sub_volume, torch.tensor([d_start, h_start, w_start])

    def get_volume_info(self, idx: int) -> dict:
        """
        Get information about a specific sub-volume.
        
        Args:
            idx: Index of the sub-volume.
            
        Returns:
            Dictionary containing sub-volume position and metadata.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        d_start, h_start, w_start = self.volume_indices[idx]
        
        return {
            'index': idx,
            'position': (d_start, h_start, w_start),
            'end_position': (
                d_start + self.volume_size,
                h_start + self.volume_size,
                w_start + self.volume_size
            ),
            'shape': (self.volume_size, self.volume_size, self.volume_size),
            'full_volume_shape': tuple(self.volume.shape),
        }
    
    def reconstruct_volume(self, predictions: list, aggregation: str = 'mean') -> torch.Tensor:
        """
        Reconstruct the full volume from sub-volume predictions.
        Useful for inference when you process the volume in chunks.
        
        Args:
            predictions: List of tensors, one for each sub-volume (in dataset order).
            aggregation: How to aggregate overlapping regions. Options: 'mean', 'max'.
            
        Returns:
            Reconstructed volume tensor of shape matching self.volume.shape.
        """
        if len(predictions) != len(self):
            raise ValueError(f"Expected {len(self)} predictions, got {len(predictions)}")
        
        # Initialize accumulator and count tensors
        full_shape = self.volume.shape
        reconstructed = torch.zeros(full_shape, dtype=self.dtype)
        counts = torch.zeros(full_shape, dtype=torch.int32)
        
        # Accumulate predictions
        for idx, pred in enumerate(predictions):
            d_start, h_start, w_start = self.volume_indices[idx]
            
            if aggregation == 'mean':
                reconstructed[
                    d_start:d_start + self.volume_size,
                    h_start:h_start + self.volume_size,
                    w_start:w_start + self.volume_size
                ] += pred
                
                counts[
                    d_start:d_start + self.volume_size,
                    h_start:h_start + self.volume_size,
                    w_start:w_start + self.volume_size
                ] += 1
            elif aggregation == 'max':
                reconstructed[
                    d_start:d_start + self.volume_size,
                    h_start:h_start + self.volume_size,
                    w_start:w_start + self.volume_size
                ] = torch.max(
                    reconstructed[
                        d_start:d_start + self.volume_size,
                        h_start:h_start + self.volume_size,
                        w_start:w_start + self.volume_size
                    ],
                    pred
                )
            else:
                raise ValueError(f"Unknown aggregation method: {aggregation}")
        
        # Average the overlapping regions (for mean aggregation)
        if aggregation == 'mean':
            # Avoid division by zero
            counts = torch.where(counts == 0, torch.ones_like(counts), counts)
            reconstructed = reconstructed / counts.float()
        
        return reconstructed
    
    def get_sub_volume_only(self, idx: int) -> torch.Tensor:
        """
        Get only the sub-volume tensor (for backward compatibility).
        
        Args:
            idx: Index of the sub-volume.
            
        Returns:
            torch.Tensor: Sub-volume of shape (volume_size, volume_size, volume_size).
        """
        sub_volume, _ = self.__getitem__(idx)
        return sub_volume


# Example usage
if __name__ == '__main__':
    # Example: Create a dataset from a folder of TIFF files
    dataset = TiffVolumeDataset(
        data_path='./data/tiff_sequence/',
        num_frames=100,
        volume_size=64,
        stride=32,  # 50% overlap
        normalize=True,
        global_normalize=True
    )
    
    print(f"\nDataset created with {len(dataset)} sub-volumes")
    
    # Get a sample sub-volume with indices
    sample_volume, sample_info = dataset[0]
    print(f"Sample sub-volume shape: {sample_volume.shape}")
    print(f"Sample position info: {sample_info['position']}")
    
    # Get only the sub-volume (backward compatibility)
    volume_only = dataset.get_sub_volume_only(0)
    print(f"Volume only shape: {volume_only.shape}")
    
    # Get information about the sub-volume (deprecated - use dataset[idx][1] instead)
    info = dataset.get_volume_info(0)
    print(f"Sub-volume info: {info}")
    
    # Example: Use with DataLoader
    from torch.utils.data import DataLoader
    
    def collate_fn(batch):
        """Custom collate function to handle (volume, info) tuples."""
        volumes = torch.stack([item[0] for item in batch])
        infos = [item[1] for item in batch]
        return volumes, infos
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)
    for batch_idx, (batch_volumes, batch_infos) in enumerate(loader):
        print(f"Batch {batch_idx}: volumes shape={batch_volumes.shape}, {len(batch_infos)} info dicts")
        print(f"  First volume position: {batch_infos[0]['position']}")
        if batch_idx >= 2:  # Just show a few batches
            break
