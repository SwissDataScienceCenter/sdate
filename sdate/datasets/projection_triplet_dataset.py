"""
Projection Triplet Dataset for Noise2Noise Denoising in Tomography.

This dataset loads consecutive projection sequences from tomographic scans
for training a denoising network using the Noise2Noise paradigm.

The idea: given k+1 consecutive projections (P_i, P_{i+1}, ..., P_{i+k}),
use (P_i, P_{i+1}, ..., P_{i+k-1}) as input (k channels) to predict P_{i+k}
using MSE loss. When k=2 this reduces to the original triplet formulation.
"""

import os
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Optional, Union
from tqdm import tqdm
import re


def load_tomography_params(data_folder: Union[str, Path], verbose: bool = True) -> Dict[str, int]:
    """
    Load tomography parameters from log file or use defaults.
    
    Searches for .log, .txt, or config files in the data folder and
    extracts num_darks, num_flats, and num_projections parameters.
    
    Parameters:
    -----------
    data_folder : str or Path
        Path to folder containing TIFF files and possibly log files
    verbose : bool
        Whether to print status messages
        
    Returns:
    --------
    params : dict
        Dictionary containing 'num_darks', 'num_flats', 'num_projections'
    """
    data_path = Path(data_folder)
    
    # Look for common log file patterns
    log_patterns = ['*.log', '*.txt', '*param*', '*config*']
    log_files = []
    for pattern in log_patterns:
        log_files.extend(list(data_path.glob(pattern)))
    
    params = {}
    
    # Try to read from log file first
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
            # Try different patterns for each parameter
            patterns = {
                'num_darks': [r'num_darks\s*[=:]\s*(\d+)', r'dark.*?(\d+)', r'(\d+).*?dark'],
                'num_flats': [r'num_flats\s*[=:]\s*(\d+)', r'flat.*?(\d+)', r'(\d+).*?flat'],
                'num_projections': [r'num_projections\s*[=:]\s*(\d+)', r'proj.*?(\d+)', r'(\d+).*?proj']
            }
            
            for param_name, pattern_list in patterns.items():
                for pattern in pattern_list:
                    match = re.search(pattern, content, re.IGNORECASE)
                    if match:
                        params[param_name] = int(match.group(1))
                        break
                if param_name in params:
                    continue
                    
            if len(params) >= 2:  # At least darks and flats found
                if verbose:
                    print(f"✅ Found parameters in {log_file.name}: {params}")
                return params
                
        except Exception as e:
            continue
    
    # If no log file found or parameters not extracted, use defaults
    if verbose:
        print("⚠️  Could not find/parse log file with tomography parameters")
    
    # Count TIFF files to estimate
    tiff_files = sorted(list(data_path.glob('*.tif*')))
    total_files = len(tiff_files)
    
    # Common defaults for tomography scans
    params = {
        'num_darks': 10,        # Typical number of dark images
        'num_flats': 10,        # Typical number of flat images  
        'num_projections': total_files - 20  # Remaining files are projections
    }
    
    if verbose:
        print(f"🔧 Using default parameters: {params}")
        print(f"   Total TIFF files: {total_files}")
    
    return params


class TomographyFolderProcessor:
    """
    Handles loading and preprocessing of tomographic projection data from a folder.
    
    Manages dark/flat correction and caching of preprocessed projections.
    """
    
    def __init__(
        self, 
        folder_path: Union[str, Path],
        num_darks: Optional[int] = None,
        num_flats: Optional[int] = None,
        epsilon: float = 1e-6,
        cache_in_memory: bool = False,
        cache_path: Optional[str] = None,
        verbose: bool = True,
        use_attenuation: bool = False
    ):
        """
        Initialize the folder processor.
        
        Args:
            folder_path: Path to folder containing TIFF files
            num_darks: Number of dark field images at the start. If None, auto-detect from log file.
            num_flats: Number of flat field images after darks. If None, auto-detect from log file.
            epsilon: Small value to avoid log(0) in attenuation calculation
            cache_in_memory: If True, load all projections into memory
            cache_path: Optional path to store/load preprocessed data
            verbose: Whether to print progress messages
            use_attenuation: If True, apply dark/flat correction and compute attenuation.
                           If False, use raw projection values (darks/flats are still skipped).
        """
        self.folder_path = Path(folder_path)
        self.epsilon = epsilon
        self.cache_in_memory = cache_in_memory
        self.cache_path = cache_path
        self.verbose = verbose
        self.use_attenuation = use_attenuation
        
        # Find and sort TIFF files
        self.tiff_files = self._find_tiff_files()
        
        # Auto-detect num_darks and num_flats from log file if not provided
        if num_darks is None or num_flats is None:
            params = load_tomography_params(self.folder_path, verbose=verbose)
            if num_darks is None:
                num_darks = params.get('num_darks', 10)
            if num_flats is None:
                num_flats = params.get('num_flats', 10)
            num_projections = params.get('num_projections', len(self.tiff_files) - num_darks - num_flats)
                
        else:
            num_projections = len(self.tiff_files) - num_darks - num_flats
        
        self.num_darks = num_darks
        self.num_flats = num_flats
        self.num_projections = num_projections
        
        if self.num_projections <= 0:
            raise ValueError(
                f"Not enough TIFF files. Found {len(self.tiff_files)}, "
                f"need at least {num_darks + num_flats + 1} for darks, flats, and projections."
            )
        
        # Get image dimensions from first file
        first_img = Image.open(self.tiff_files[0])
        first_arr = np.array(first_img)
        self.original_height, self.original_width = first_arr.shape[:2]
        self.dtype = first_arr.dtype
        
        # Initialize dark/flat averages (computed lazily)
        self._dark_avg = None
        self._flat_avg = None
        self._denominator = None
        
        # In-memory cache for projections
        self._projection_cache = {}
        
        # Attenuation range (computed lazily)
        self._global_min = None
        self._global_max = None
        
        if self.verbose:
            print(f"📂 Loaded folder: {self.folder_path.name}")
            print(f"   Total TIFF files: {len(self.tiff_files)}")
            print(f"   Darks: {num_darks}, Flats: {num_flats}, Projections: {self.num_projections}")
            print(f"   Image dimensions: {self.original_width} x {self.original_height}")
            print(f"   Use attenuation: {use_attenuation}")
        
        # Try to load cached data
        self._load_cache()
    
    def _find_tiff_files(self) -> List[Path]:
        """Find and sort all TIFF files in the folder."""
        tiff_files = sorted(
            list(self.folder_path.glob('*.tif')) + 
            list(self.folder_path.glob('*.tiff'))
        )
        return tiff_files
    
    def _get_cache_path(self) -> Path:
        """Get the path to the cache file."""
        cache_filename = f".tomography_cache_{'atten' if self.use_attenuation else 'raw'}.npz"
        return self.folder_path / cache_filename
    
    def _load_cache(self):
        """Load cached data from disk if available."""
        cache_path = self._get_cache_path()
        
        if not cache_path.exists():
            return
        
        try:
            cache_data = np.load(cache_path, allow_pickle=False)
            
            # Validate cache matches current configuration
            if (cache_data['num_darks'] != self.num_darks or
                cache_data['num_flats'] != self.num_flats or
                cache_data['num_projections'] != self.num_projections):
                if self.verbose:
                    print("⚠️  Cache file exists but parameters don't match, ignoring cache")
                return
            
            # Load cached data
            if self.use_attenuation:
                self._dark_avg = cache_data['dark_avg']
                self._flat_avg = cache_data['flat_avg']
                self._denominator = cache_data['denominator']
            
            self._global_min = float(cache_data['global_min'])
            self._global_max = float(cache_data['global_max'])
            
            if self.verbose:
                range_type = "attenuation" if self.use_attenuation else "raw"
                print(f"✅ Loaded cached {range_type} data from {cache_path.name}")
                print(f"   Range: [{self._global_min:.4f}, {self._global_max:.4f}]")
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to load cache: {e}")
            # Reset cached values
            self._dark_avg = None
            self._flat_avg = None
            self._denominator = None
            self._global_min = None
            self._global_max = None
    
    def _save_cache(self):
        """Save computed data to cache file."""
        cache_path = self._get_cache_path()
        
        try:
            cache_data = {
                'num_darks': self.num_darks,
                'num_flats': self.num_flats,
                'num_projections': self.num_projections,
                'global_min': self._global_min,
                'global_max': self._global_max,
            }
            
            if self.use_attenuation and self._dark_avg is not None:
                cache_data['dark_avg'] = self._dark_avg
                cache_data['flat_avg'] = self._flat_avg
                cache_data['denominator'] = self._denominator
            
            np.savez_compressed(cache_path, **cache_data)
            
            if self.verbose:
                print(f"💾 Saved cache to {cache_path.name}")
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Failed to save cache: {e}")
    
    def _load_darks_flats(self):
        """Load and compute average dark and flat field images."""
        if self._dark_avg is not None:
            return
        
        if self.verbose:
            print("🔬 Loading dark and flat field images...")
        
        # Load dark images
        dark_files = self.tiff_files[:self.num_darks]
        darks = []
        for dark_file in tqdm(dark_files, desc="Loading darks", disable=not self.verbose):
            img = Image.open(dark_file)
            arr = np.array(img, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            darks.append(arr)
        darks = np.stack(darks, axis=0)
        self._dark_avg = np.median(darks, axis=0)
        del darks
        
        # Load flat images
        flat_files = self.tiff_files[self.num_darks:self.num_darks + self.num_flats]
        flats = []
        for flat_file in tqdm(flat_files, desc="Loading flats", disable=not self.verbose):
            img = Image.open(flat_file)
            arr = np.array(img, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[:, :, 0]
            flats.append(arr)
        flats = np.stack(flats, axis=0)
        self._flat_avg = np.median(flats, axis=0)
        del flats
        
        # Precompute denominator
        self._denominator = self._flat_avg - self._dark_avg
        self._denominator = np.where(self._denominator == 0, 1e-6, self._denominator)
        
        if self.verbose:
            print(f"   Dark average range: [{self._dark_avg.min():.2f}, {self._dark_avg.max():.2f}]")
            print(f"   Flat average range: [{self._flat_avg.min():.2f}, {self._flat_avg.max():.2f}]")
    
    def compute_range(self, sample_ratio: float = 0.1, max_samples: int = int(1e6)) -> Tuple[float, float]:
        """
        Estimate the global min/max values by sampling projections.
        
        Checks cache first. If not available, computes and saves to cache.
        
        Args:
            sample_ratio: Fraction of projections to sample
            max_samples: Maximum number of projections to sample
            
        Returns:
            Tuple of (global_min, global_max)
        """
        # Return cached values if available
        if self._global_min is not None:
            return self._global_min, self._global_max
        
        # Need to compute range
        if self.use_attenuation:
            self._load_darks_flats()
        
        num_samples = min(max_samples, max(10, int(self.num_projections * sample_ratio)))
        sample_indices = np.linspace(0, self.num_projections - 1, num_samples, dtype=int)
        
        global_min = float('inf')
        global_max = float('-inf')
        
        range_type = "attenuation" if self.use_attenuation else "raw projection"
        if self.verbose:
            print(f"🔍 Estimating {range_type} range from {num_samples} samples...")
        
        for idx in tqdm(sample_indices, desc="Sampling projections", disable=not self.verbose):
            projection = self._load_projection_data(idx)
            global_min = min(global_min, float(projection.min()))
            global_max = max(global_max, float(projection.max()))
        
        self._global_min = global_min
        self._global_max = global_max
        
        if self.verbose:
            print(f"   {range_type.capitalize()} range: [{global_min:.4f}, {global_max:.4f}]")
        
        # Save to cache
        self._save_cache()
        
        return global_min, global_max
    
    # Alias for backward compatibility
    def compute_attenuation_range(self, sample_ratio: float = 0.1, max_samples: int = None) -> Tuple[float, float]:
        """Alias for compute_range() for backward compatibility."""
        if max_samples is None:
            max_samples = self.num_projections
        return self.compute_range(sample_ratio, max_samples)
    
    def _load_projection_data(self, proj_idx: int) -> np.ndarray:
        """
        Load a single projection, optionally computing attenuation.
        
        Args:
            proj_idx: Index of the projection (0-based, excludes darks/flats)
            
        Returns:
            Projection data as numpy array (attenuation or raw depending on use_attenuation)
        """
        file_idx = self.num_darks + self.num_flats + proj_idx
        img = Image.open(self.tiff_files[file_idx])
        arr = np.array(img, dtype=np.float32)
        
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        
        if self.use_attenuation:
            # Load darks/flats if not already loaded
            self._load_darks_flats()
            
            # Compute attenuation: -log((projection - dark) / (flat - dark) + epsilon)
            normalized = (arr - self._dark_avg) / self._denominator
            normalized = np.clip(normalized, 0, None)
            arr = -np.log(normalized + self.epsilon)
        
        return arr
    
    # Alias for backward compatibility
    def _load_raw_attenuation(self, proj_idx: int) -> np.ndarray:
        """Alias for _load_projection_data() for backward compatibility."""
        return self._load_projection_data(proj_idx)
    
    def get_projection(self, proj_idx: int, normalize: bool = True) -> np.ndarray:
        """
        Get a preprocessed projection by index.
        
        Args:
            proj_idx: Index of the projection (0-based, excludes darks/flats)
            normalize: If True, normalize to [0, 1] using global range
            
        Returns:
            Preprocessed projection as numpy array
        """
        if proj_idx < 0 or proj_idx >= self.num_projections:
            raise IndexError(f"Projection index {proj_idx} out of range [0, {self.num_projections})")
        
        # Check cache first
        if self.cache_in_memory and proj_idx in self._projection_cache:
            return self._projection_cache[proj_idx]
        
        projection = self._load_projection_data(proj_idx)
        
        if normalize:
            if self._global_min is None:
                self.compute_range()
            projection = (projection - self._global_min) / (self._global_max - self._global_min)
            projection = np.clip(projection, 0, 1)
        
        if self.cache_in_memory:
            self._projection_cache[proj_idx] = projection
        
        return projection
    
    def preload_all_projections(self):
        """Preload all projections into memory cache."""
        if not self.cache_in_memory:
            print("⚠️  cache_in_memory is False, enabling it for preloading")
            self.cache_in_memory = True
        
        if self.verbose:
            print(f"📦 Preloading {self.num_projections} projections into memory...")
        
        for idx in tqdm(range(self.num_projections), desc="Preloading", disable=not self.verbose):
            self.get_projection(idx)
        
        if self.verbose:
            memory_mb = sum(p.nbytes for p in self._projection_cache.values()) / 1e6
            print(f"   Loaded {len(self._projection_cache)} projections ({memory_mb:.1f} MB)")


class ProjectionTripletDataset(Dataset):
    """
    Dataset for Noise2Noise training on tomographic projections.
    
    For each sample, returns:
    - input: Stack of k consecutive projections P_i, P_{i+1}, ..., P_{i+k-1} (k channels)
    - target: Next projection P_{i+k} (1 channel)
    - center_coords: (cx, cy) coordinates of original image center in cropped/padded image
    
    The number of input channels k is controlled by the `num_input_projections` parameter.
    When k=2, this is equivalent to the original triplet formulation.
    """
    
    def __init__(
        self,
        folder_paths: Union[str, Path, List[Union[str, Path]]],
        target_size: int = 2048,
        num_darks: Optional[int] = None,
        num_flats: Optional[int] = None,
        epsilon: float = 1e-6,
        cache_in_memory: bool = False,
        preload: bool = False,
        augment: bool = False,
        verbose: bool = True,
        use_attenuation: bool = True,
        num_input_projections: int = 2
    ):
        """
        Initialize the triplet dataset.
        
        Args:
            folder_paths: Path(s) to folder(s) containing TIFF files
            target_size: Target image size (will crop/pad to this size)
            num_darks: Number of dark field images at the start of each folder. 
                      If None, auto-detect from log file in each folder.
            num_flats: Number of flat field images after darks.
                      If None, auto-detect from log file in each folder.
            epsilon: Small value to avoid log(0)
            cache_in_memory: If True, cache projections in memory
            preload: If True, preload all projections at initialization
            augment: If True, apply data augmentation (random flips, rotations)
            verbose: Whether to print progress messages
            use_attenuation: If True, apply dark/flat correction and compute attenuation.
                           If False, use raw projection values (darks/flats files are still skipped).
            num_input_projections: Number of consecutive projections to use as input channels (k).
                                  The target will be the (k+1)-th projection. Default is 2.
        """
        self.target_size = target_size
        self.augment = augment
        self.verbose = verbose
        self.use_attenuation = use_attenuation
        self.num_input_projections = num_input_projections
        
        # Handle single path or list of paths
        if isinstance(folder_paths, (str, Path)):
            folder_paths = [folder_paths]
        
        # Initialize processors for each folder
        self.processors: List[TomographyFolderProcessor] = []
        self.folder_names: List[str] = []
        
        for path in folder_paths:
            path = Path(path)
            if not path.exists():
                if verbose:
                    print(f"⚠️  Skipping non-existent folder: {path}")
                continue
            
            try:
                processor = TomographyFolderProcessor(
                    folder_path=path,
                    num_darks=num_darks,
                    num_flats=num_flats,
                    epsilon=epsilon,
                    cache_in_memory=cache_in_memory,
                    verbose=verbose,
                    use_attenuation=use_attenuation
                )
                self.processors.append(processor)
                self.folder_names.append(path.name)
            except Exception as e:
                if verbose:
                    print(f"⚠️  Error loading folder {path}: {e}")
        
        if len(self.processors) == 0:
            raise ValueError("No valid folders found!")
        
        # Build index mapping: (global_idx) -> (processor_idx, start_idx)
        # Each window uses (num_input_projections + 1) consecutive projections:
        #   input: [start_idx, start_idx+1, ..., start_idx+k-1]  (k channels)
        #   target: start_idx + k
        self.index_map: List[Tuple[int, int]] = []
        window_size = self.num_input_projections + 1  # k input + 1 target
        for proc_idx, processor in enumerate(self.processors):
            num_windows = processor.num_projections - self.num_input_projections
            for start_idx in range(num_windows):
                self.index_map.append((proc_idx, start_idx))
        
        # Compute value ranges for all processors
        for processor in self.processors:
            processor.compute_range(sample_ratio=1.)
        
        # Preload if requested
        if preload:
            for processor in self.processors:
                processor.preload_all_projections()
        
        if verbose:
            print(f"\n📊 Dataset Summary:")
            print(f"   Total folders: {len(self.processors)}")
            print(f"   Total samples: {len(self.index_map)}")
            print(f"   Input projections per sample (k): {self.num_input_projections}")
            print(f"   Target size: {target_size} x {target_size}")
    
    def __len__(self) -> int:
        return len(self.index_map)
    
    def _crop_or_pad(
        self, 
        image: np.ndarray, 
        start_h: Optional[int] = None, 
        start_w: Optional[int] = None
    ) -> Tuple[np.ndarray, Tuple[float, float], Tuple[Optional[int], Optional[int]]]:
        """
        Crop or pad image to target_size x target_size.
        
        Uses random crops when image is larger than target size (covering any part of the image),
        and center padding when image is smaller. Can accept predetermined crop positions to
        ensure consistent cropping across multiple images.
        
        Args:
            image: Input image array
            start_h: Starting row index for cropping (optional, generates random if None)
            start_w: Starting column index for cropping (optional, generates random if None)
        
        Returns:
            Tuple of (processed_image, center_coordinates, (start_h_used, start_w_used))
            where center_coordinates are the (x, y) position of the original
            image center within the processed image, normalized to [0, 1],
            and (start_h_used, start_w_used) are the crop positions used (for consistency).
        """
        h, w = image.shape
        target = self.target_size
        
        # Calculate padding/cropping for each dimension
        if h >= target:
            # Crop height: random crop to cover any part of the image
            max_start_h = h - target
            if start_h is None:
                start_h = np.random.randint(0, max_start_h + 1)
            else:
                start_h = min(start_h, max_start_h)  # Clamp to valid range
            end_h = start_h + target
            result_h = image[start_h:end_h, :]
            # Original center (h/2) maps to (h/2 - start_h) in cropped image
            center_y = (h / 2 - start_h) / target
            start_h_used = start_h
        else:
            # Pad height: center padding
            pad_top = (target - h) // 2
            pad_bottom = target - h - pad_top
            result_h = np.pad(image, ((pad_top, pad_bottom), (0, 0)), mode='constant')
            # Original center (h/2) maps to (h/2 + pad_top) in padded image
            center_y = (h / 2 + pad_top) / target
            start_h_used = None  # No crop was applied
        
        if w >= target:
            # Crop width: random crop to cover any part of the image
            max_start_w = w - target
            if start_w is None:
                start_w = np.random.randint(0, max_start_w + 1)
            else:
                start_w = min(start_w, max_start_w)  # Clamp to valid range
            end_w = start_w + target
            result = result_h[:, start_w:end_w]
            # Original center (w/2) maps to (w/2 - start_w) in cropped image
            center_x = (w / 2 - start_w) / target
            start_w_used = start_w
        else:
            # Pad width: center padding
            pad_left = (target - w) // 2
            pad_right = target - w - pad_left
            result = np.pad(result_h, ((0, 0), (pad_left, pad_right)), mode='constant')
            # Original center (w/2) maps to (w/2 + pad_left) in padded image
            center_x = (w / 2 + pad_left) / target
            start_w_used = None  # No crop was applied
        
        return result, (center_x, center_y), (start_h_used, start_w_used)
    
    def _apply_augmentation(
        self, 
        projections: List[np.ndarray]
    ) -> List[np.ndarray]:
        """Apply consistent augmentation to all projections in the sequence."""
        if not self.augment:
            return projections
        
        # Random horizontal flip
        if np.random.random() > 0.5:
            projections = [np.flip(p, axis=1).copy() for p in projections]
        
        # Random vertical flip
        if np.random.random() > 0.5:
            projections = [np.flip(p, axis=0).copy() for p in projections]
        
        # Random 90-degree rotation
        k = np.random.randint(0, 4)
        if k > 0:
            projections = [np.rot90(p, k).copy() for p in projections]
        
        return projections
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample with k input projections and 1 target projection.
        
        Returns:
            Dictionary with:
            - 'input': Tensor of shape (k, target_size, target_size) where k = num_input_projections
            - 'target': Tensor of shape (1, target_size, target_size)
            - 'center_coords': Tensor of shape (2,) with (cx, cy) in [0, 1]
            - 'folder_name': Name of the source folder
            - 'projection_idx': Target projection index (i+k)
        """
        proc_idx, start_idx = self.index_map[idx]
        processor = self.processors[proc_idx]
        k = self.num_input_projections
        
        # Load k+1 consecutive projections: k inputs + 1 target
        projections = [
            processor.get_projection(start_idx + j)
            for j in range(k + 1)
        ]
        
        # Apply augmentation before cropping/padding
        projections = self._apply_augmentation(projections)
        
        # Crop/pad to target size - use the SAME random crop for all projections
        # First image determines the crop location
        processed = []
        start_h, start_w = None, None
        center_coords = None
        for i, proj in enumerate(projections):
            proj_processed, coords, (start_h, start_w) = self._crop_or_pad(proj, start_h, start_w)
            processed.append(proj_processed)
            if center_coords is None:
                center_coords = coords
        
        # Stack input channels: projections [0, 1, ..., k-1]
        input_tensor = torch.from_numpy(
            np.stack(processed[:k], axis=0)
        ).float()
        
        # Target is the last projection (index k)
        target_tensor = torch.from_numpy(processed[k]).unsqueeze(0).float()
        
        # Center coordinates as tensor
        center_coords_tensor = torch.tensor(center_coords, dtype=torch.float32)
        
        target_proj_idx = start_idx + k
        
        return {
            'input': input_tensor,
            'target': target_tensor,
            'center_coords': center_coords_tensor,
            'folder_name': self.folder_names[proc_idx],
            'projection_idx': target_proj_idx
        }
    
    def get_original_dimensions(self, idx: int) -> Tuple[int, int]:
        """Get the original dimensions for a sample."""
        proc_idx, _ = self.index_map[idx]
        processor = self.processors[proc_idx]
        return processor.original_height, processor.original_width


def create_train_val_split(
    dataset: ProjectionTripletDataset,
    val_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    """
    Split dataset into training and validation sets.
    
    Args:
        dataset: ProjectionTripletDataset instance
        val_ratio: Fraction of data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_subset, val_subset)
    """
    n = len(dataset)
    n_val = int(n * val_ratio)
    n_train = n - n_val
    
    generator = torch.Generator().manual_seed(seed)
    train_indices, val_indices = torch.utils.data.random_split(
        range(n), [n_train, n_val], generator=generator
    )
    
    train_subset = torch.utils.data.Subset(dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices.indices)
    
    return train_subset, val_subset


if __name__ == '__main__':
    # Test the dataset
    import lovely_tensors as lt
    lt.monkey_patch()
    
    # Example usage
    test_folder = Path('/das/home/barbaf_l/p22274/compression_paper/file_3_extracted')
    
    if test_folder.exists():
        # Test with attenuation (default)
        print("=" * 60)
        print("Testing with attenuation correction (use_attenuation=True)")
        print("=" * 60)
        dataset_atten = ProjectionTripletDataset(
            folder_paths=test_folder,
            target_size=2048,
            num_darks=10,
            num_flats=10,
            cache_in_memory=False,
            augment=True,
            verbose=True,
            use_attenuation=True
        )
        
        print(f"\nDataset length: {len(dataset_atten)}")
        sample = dataset_atten[0]
        print(f"\nSample:")
        print(f"  Input shape: {sample['input'].shape}")
        print(f"  Target shape: {sample['target'].shape}")
        print(f"  Center coords: {sample['center_coords']}")
        print(f"  Input range: [{sample['input'].min():.4f}, {sample['input'].max():.4f}]")
        
        # Test without attenuation (raw projections)
        print("\n" + "=" * 60)
        print("Testing with raw projections (use_attenuation=False)")
        print("=" * 60)
        dataset_raw = ProjectionTripletDataset(
            folder_paths=test_folder,
            target_size=2048,
            num_darks=10,
            num_flats=10,
            cache_in_memory=False,
            augment=True,
            verbose=True,
            use_attenuation=False
        )
        
        print(f"\nDataset length: {len(dataset_raw)}")
        sample_raw = dataset_raw[0]
        print(f"\nSample:")
        print(f"  Input shape: {sample_raw['input'].shape}")
        print(f"  Target shape: {sample_raw['target'].shape}")
        print(f"  Center coords: {sample_raw['center_coords']}")
        print(f"  Input range: [{sample_raw['input'].min():.4f}, {sample_raw['input'].max():.4f}]")
    else:
        print(f"Test folder not found: {test_folder}")
