"""
Batch Compression Pipeline for Tomographic Data
=================================================

This pipeline processes tomographic TIFF sequences by compressing darks, flats, 
and projections independently with separate dynamic range estimations for each type.

Each component (darks, flats, projections) is:
1. Analyzed independently for its dynamic range
2. Compressed into a separate HEVC video file
3. Documented with metadata (ranges, compression stats)

Correction Modes:
-----------------
- **Raw mode** (default): Compresses raw projection values directly.
- **Attenuation mode** (use_attenuation=True): Computes μ = -ln((I-dark)/(flat-dark))
  where transmission is clipped to avoid log of negative values. May lose information.
- **Transmission mode** (use_transmission=True): Computes T = (I-dark)/(flat-dark)
  without clipping. Values can be < 0 or > 1, preserving all information.

Percentile Normalization:
-------------------------
When use_per_frame_percentile=True, each frame is normalized using its own
[low_percentile, high_percentile] range (default: 1st and 99th percentile).

The pipeline processes multiple folders with configurable quality settings and
produces detailed CSV reports with all compression metrics and range metadata.

Author: Generated for tomographic data compression
Date: November 2025
"""

from pathlib import Path
import sys

# Resolve project root based on this file's location
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent.parent   # /das/home/barbaf_l/sdate

# Add project root so we can import the `sdate` package
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"Added {PROJECT_ROOT} to Python path")

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm.auto import tqdm

# Import video compression utilities
from sdate.stream_hvec import HevcGray10Streamer, EncoderParams
from sdate.utils.normalization import compute_and_apply_frame_cdf


def compute_transmission(projection: np.ndarray, dark_mean: np.ndarray, flat_mean: np.ndarray, 
                         epsilon: float = 1e-6) -> np.ndarray:
    """
    Compute transmission from a projection using flat-field and dark-field correction.
    
    Transmission formula: T = (I - dark) / (flat - dark)
    
    Unlike attenuation, transmission is NOT clipped to preserve all information.
    Values can be < 0 (darker than dark field) or > 1 (brighter than flat field).
    
    Parameters:
    -----------
    projection : ndarray
        Raw projection image
    dark_mean : ndarray
        Mean dark-field image
    flat_mean : ndarray
        Mean flat-field image
    epsilon : float
        Small value to avoid division by zero
        
    Returns:
    --------
    transmission : ndarray
        Transmission values (unclipped, can be negative or > 1)
    """
    # Compute the denominator (flat - dark)
    denominator = flat_mean - dark_mean
    # Avoid division by zero
    denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)
    
    # Compute the normalized transmission (unclipped)
    transmission = (projection.astype(np.float32) - dark_mean) / denominator
    
    return transmission


def compute_attenuation(projection: np.ndarray, dark_mean: np.ndarray, flat_mean: np.ndarray, 
                        epsilon: float = 1e-6) -> np.ndarray:
    """
    Compute attenuation from a projection using flat-field and dark-field correction.
    
    Attenuation formula: μ = -ln((I - dark) / (flat - dark))
    
    Parameters:
    -----------
    projection : ndarray
        Raw projection image
    dark_mean : ndarray
        Mean dark-field image
    flat_mean : ndarray
        Mean flat-field image
    epsilon : float
        Small value to avoid division by zero and log of zero
        
    Returns:
    --------
    attenuation : ndarray
        Attenuation-corrected image (μ values)
    """
    # Compute the denominator (flat - dark)
    denominator = flat_mean - dark_mean
    # Avoid division by zero
    denominator = np.maximum(denominator, epsilon)
    
    # Compute the normalized transmission
    transmission = (projection.astype(np.float32) - dark_mean) / denominator
    
    # Clip transmission to avoid log of zero or negative values
    transmission = np.clip(transmission, epsilon, None)
    
    # Compute attenuation (Beer-Lambert law)
    attenuation = -np.log(transmission)
    
    return attenuation


def compute_mean_dark_flat(tiff_files: List[Path], num_darks: int, num_flats: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean dark and flat field images from TIFF files.
    
    Parameters:
    -----------
    tiff_files : List[Path]
        List of all TIFF file paths (ordered: darks, flats, projections)
    num_darks : int
        Number of dark frames
    num_flats : int
        Number of flat frames
        
    Returns:
    --------
    dark_mean : ndarray
        Mean dark-field image (float32)
    flat_mean : ndarray
        Mean flat-field image (float32)
    """
    print(f"   📊 Computing mean dark and flat field images...")
    
    # Load dark frames and compute mean
    dark_files = tiff_files[:num_darks]
    dark_images = []
    for f in tqdm(dark_files, desc="      Loading darks", leave=False):
        img = Image.open(f)
        arr = np.array(img)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        dark_images.append(arr)
    dark_mean = np.mean(dark_images, axis=0).astype(np.float32)
    
    # Load flat frames and compute mean
    flat_files = tiff_files[num_darks:num_darks + num_flats]
    flat_images = []
    for f in tqdm(flat_files, desc="      Loading flats", leave=False):
        img = Image.open(f)
        arr = np.array(img)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        flat_images.append(arr)
    flat_mean = np.mean(flat_images, axis=0).astype(np.float32)
    
    print(f"      Dark mean range: [{dark_mean.min():.1f}, {dark_mean.max():.1f}]")
    print(f"      Flat mean range: [{flat_mean.min():.1f}, {flat_mean.max():.1f}]")
    
    return dark_mean, flat_mean


def load_tomography_params(data_folder: Path) -> Dict[str, int]:
    """
    Load tomography parameters from log file or estimate from TIFF files.
    
    Parameters:
    -----------
    data_folder : Path
        Path to folder containing TIFF files and possibly log files
        
    Returns:
    --------
    params : dict
        Dictionary containing num_darks, num_flats, num_projections
    """
    # Look for common log file patterns
    log_patterns = ['*.log', '*.txt', '*param*', '*config*']
    log_files = []
    for pattern in log_patterns:
        log_files.extend(list(data_folder.glob(pattern)))
    
    params = {}
    
    # Try to read from log file first
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
                
            # Look for common parameter names
            import re
            
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
                    
            if len(params) == 3:
                print(f"✅ Found parameters in {log_file.name}: {params}")
                return params
                
        except Exception:
            continue
    
    # If no log file found, use defaults
    tiff_files = sorted(list(data_folder.glob('*.tif*')))
    total_files = len(tiff_files)
    
    params = {
        'num_darks': 10,
        'num_flats': 10,
        'num_projections': total_files - 20
    }
    
    print(f"🔧 Using default parameters: {params}")
    print(f"   Total TIFF files: {total_files}")
    
    return params


def estimate_independent_ranges(
    tiff_files: List[Path],
    params: Dict[str, int],
    sample_ratio: float = 1.0,
    min_samples: int = 5,
    max_samples: int = 10000,
    create_histogram: bool = False,
    use_per_frame_percentile: bool = True,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    use_attenuation: bool = False,
    use_transmission: bool = False,
    dark_mean: Optional[np.ndarray] = None,
    flat_mean: Optional[np.ndarray] = None
) -> Dict[str, Dict[str, float]]:
    """
    Estimate dynamic ranges independently for darks, flats, and projections.
    
    Parameters:
    -----------
    tiff_files : List[Path]
        List of all TIFF file paths
    params : dict
        Tomography parameters with num_darks, num_flats, num_projections
    sample_ratio : float
        Fraction of files to sample from each type
    min_samples : int
        Minimum number of files to sample per type
    max_samples : int
        Maximum number of files to sample per type
    create_histogram : bool
        Whether to create histograms (useful for debugging)
    use_per_frame_percentile : bool
        If True, compute per-frame percentiles instead of global max
    low_percentile : float
        Lower percentile value for min (default 1.0)
    high_percentile : float
        Upper percentile value for max (default 99.0)
    use_attenuation : bool
        If True, compute ranges for attenuation-corrected projections instead of raw
    use_transmission : bool
        If True, compute ranges for unclipped transmission values instead of raw.
        Transmission is NOT clipped, preserving all information.
        Cannot be used together with use_attenuation.
    dark_mean : ndarray, optional
        Mean dark-field image (required if use_attenuation=True or use_transmission=True)
    flat_mean : ndarray, optional
        Mean flat-field image (required if use_attenuation=True or use_transmission=True)
    
    Returns:
    --------
    ranges : dict
        Dictionary with 'darks', 'flats', 'projections' keys, each containing
        'min', 'max', 'range', 'width', 'height', 'dtype' information
        If use_per_frame_percentile is True, also includes 'per_frame_max' array
        If use_attenuation is True, projections will have 'use_attenuation' flag
        If use_transmission is True, projections will have 'use_transmission' flag
    """
    num_darks = int(params['num_darks'])
    num_flats = int(params['num_flats'])
    num_projections = int(params['num_projections'])
    
    # Separate the files by type
    dark_files = tiff_files[:num_darks]
    flat_files = tiff_files[num_darks:num_darks + num_flats]
    projection_files = tiff_files[num_darks + num_flats:num_darks + num_flats + num_projections]
    
    # Validate that attenuation and transmission are not both enabled
    if use_attenuation and use_transmission:
        raise ValueError("Cannot use both attenuation and transmission modes simultaneously")
    
    # Determine if we're using any correction mode
    use_correction = use_attenuation or use_transmission
    
    print(f"🔍 Estimating independent dynamic ranges...")
    if use_attenuation:
        print(f"   📐 ATTENUATION MODE: Projections will be flat/dark corrected (clipped)")
    elif use_transmission:
        print(f"   📐 TRANSMISSION MODE: Projections will be flat/dark corrected (unclipped)")
    print(f"   Darks: {len(dark_files)} files")
    print(f"   Flats: {len(flat_files)} files")
    print(f"   Projections: {len(projection_files)} files")
    
    ranges = {}
    
    # Process each type independently
    for data_type, files in [('darks', dark_files), ('flats', flat_files), ('projections', projection_files)]:
        if len(files) == 0:
            print(f"   ⚠️  No {data_type} files found, skipping...")
            continue
        
        # Skip darks and flats if using correction mode (they are used for correction, not compressed)
        if use_correction and data_type in ['darks', 'flats']:
            print(f"   ⏭️  Skipping {data_type} (used for correction only)")
            continue
        
        width, height, dtype_str = None, None, None
        is_attenuation_projection = (use_attenuation and data_type == 'projections')
        is_transmission_projection = (use_transmission and data_type == 'projections')
        is_corrected_projection = is_attenuation_projection or is_transmission_projection
        
        if use_per_frame_percentile:
            # Compute per-frame percentiles for ALL files (not just samples)
            if is_attenuation_projection:
                mode_str = "attenuation"
            elif is_transmission_projection:
                mode_str = "transmission"
            else:
                mode_str = data_type
            print(f"   📊 {mode_str.capitalize()}: Computing per-frame [{low_percentile:.1f}th, {high_percentile}th] percentiles for {len(files)} files...")
            
            per_frame_min = np.zeros(len(files), dtype=np.float32)
            per_frame_max = np.zeros(len(files), dtype=np.float32)
            
            for idx, file_path in enumerate(tqdm(files, desc=f"  Analyzing {mode_str}", leave=False)):
                img = Image.open(file_path)
                arr = np.array(img)
                
                # Handle different image formats
                if arr.ndim == 3:  # Color image
                    arr = arr[:, :, 0]  # Take first channel
                
                # Apply correction if enabled for projections
                if is_attenuation_projection:
                    arr = compute_attenuation(arr, dark_mean, flat_mean)
                elif is_transmission_projection:
                    arr = compute_transmission(arr, dark_mean, flat_mean)
                
                # Get dimensions from first image
                if width is None:
                    height, width = arr.shape[:2]
                    dtype_str = str(arr.dtype) if not is_corrected_projection else 'float32'
                
                # Compute low and high percentiles for this frame
                valid_arr = arr[np.isfinite(arr)]  # Filter out inf/nan for attenuation/transmission
                if len(valid_arr) > 0:
                    per_frame_min[idx] = np.percentile(valid_arr, low_percentile)
                    per_frame_max[idx] = np.percentile(valid_arr, high_percentile)
                else:
                    per_frame_min[idx] = 0.0
                    per_frame_max[idx] = 1.0
            
            # Use global stats for range info reporting
            global_min = float(np.min(per_frame_min))
            global_max = float(np.max(per_frame_max))
            
            ranges[data_type] = {
                'min': global_min,
                'max': global_max,
                'range': global_max - global_min,
                'width': width,
                'height': height,
                'dtype': dtype_str,
                'num_files': len(files),
                'per_frame_min': per_frame_min,  # Array of per-frame min values (low percentile)
                'per_frame_max': per_frame_max,  # Array of per-frame max values (high percentile)
                'low_percentile': low_percentile,
                'high_percentile': high_percentile,
                'use_per_frame': True,
                'use_attenuation': is_attenuation_projection,
                'use_transmission': is_transmission_projection
            }
            
            print(f"      Per-frame {low_percentile:.1f}th percentile range: [{per_frame_min.min():.1f}, {per_frame_min.max():.1f}]")
            print(f"      Per-frame {high_percentile}th percentile range: [{per_frame_max.min():.1f}, {per_frame_max.max():.1f}]")
            print(f"      Global range: [{global_min:.1f}, {global_max:.1f}]")
        else:
            # Original behavior: use global min/max from samples
            # Calculate sampling
            num_samples = max(min_samples, min(max_samples, int(len(files) * sample_ratio)))
            num_samples = min(num_samples, len(files))  # Can't sample more than available
            
            sample_indices = np.linspace(0, len(files) - 1, num_samples, dtype=int)
            
            print(f"   📊 {data_type.capitalize()}: Sampling {num_samples}/{len(files)} files...")
            
            # Initialize min/max tracking
            global_min = float('inf')
            global_max = float('-inf')
            
            # Sample files to estimate range
            for idx in tqdm(sample_indices, desc=f"  Analyzing {data_type}", leave=False):
                img = Image.open(files[idx])
                arr = np.array(img)
                
                # Handle different image formats
                if arr.ndim == 3:  # Color image
                    arr = arr[:, :, 0]  # Take first channel
                
                # Get dimensions from first image
                if width is None:
                    height, width = arr.shape[:2]
                    dtype_str = str(arr.dtype)
                
                # Update global min/max
                current_min = float(arr.min())
                current_max = float(arr.max())
                global_min = min(global_min, current_min)
                global_max = max(global_max, current_max)
            
            
            ranges[data_type] = {
                'min': global_min,
                'max': global_max,
                'range': global_max - global_min,
                'width': width,
                'height': height,
                'dtype': dtype_str,
                'num_files': len(files),
                'use_per_frame': False,
                'use_attenuation': is_attenuation_projection,
                'use_transmission': is_transmission_projection
            }
            
            print(f"      Range: [{global_min:.1f}, {global_max:.1f}] (Δ={global_max - global_min:.1f})")
    
    return ranges


def stream_tomography_to_hevc(
    tiff_files: List[Path],
    params: Dict[str, int],
    ranges: Dict[str, Dict[str, float]],
    output_path: Path,
    folder_name: str,
    quality: int = 90,
    fps: int = 30,
    force_software: bool = False,
    preset_sw: str = "medium",
    use_attenuation: bool = False,
    use_transmission: bool = False,
    dark_mean: Optional[np.ndarray] = None,
    flat_mean: Optional[np.ndarray] = None,
    use_cdf_normalization: bool = False,
    cdf_num_bins: int = 1000
) -> Dict[str, any]:
    """
    Stream compress darks, flats, and projections independently to separate HEVC files.
    
    Parameters:
    -----------
    tiff_files : List[Path]
        List of all TIFF file paths
    params : dict
        Tomography parameters with num_darks, num_flats, num_projections
    ranges : dict
        Independent range information for each data type
    output_path : Path
        Base output directory
    folder_name : str
        Name of the dataset folder (for naming files)
    quality : int
        Compression quality (0-100, higher = better)
    fps : int
        Frames per second for output video
    force_software : bool
        Whether to force software encoding
    preset_sw : str
        Software encoding preset
    use_attenuation : bool
        If True, compress attenuation-corrected projections instead of raw
    use_transmission : bool
        If True, compress unclipped transmission values instead of raw.
        Cannot be used together with use_attenuation.
    dark_mean : ndarray, optional
        Mean dark-field image (required if use_attenuation=True or use_transmission=True)
    flat_mean : ndarray, optional
        Mean flat-field image (required if use_attenuation=True or use_transmission=True)
    use_cdf_normalization : bool
        If True, apply per-frame CDF-based histogram equalization after standard 
        normalization to create a more uniform pixel distribution before compression.
        Each frame gets its own CDF mapping which is saved for reconstruction.
    cdf_num_bins : int
        Number of bins for per-frame CDF computation (default: 1000)
    
    Returns:
    --------
    results : dict
        Compression results with paths, sizes, and metadata
    """
    # Validate that attenuation and transmission are not both enabled
    if use_attenuation and use_transmission:
        raise ValueError("Cannot use both attenuation and transmission modes simultaneously")
    
    use_correction = use_attenuation or use_transmission
    
    num_darks = int(params['num_darks'])
    num_flats = int(params['num_flats'])
    num_projections = int(params['num_projections'])
    
    # Separate the files by type
    dark_files = tiff_files[:num_darks]
    flat_files = tiff_files[num_darks:num_darks + num_flats]
    projection_files = tiff_files[num_darks + num_flats:num_darks + num_flats + num_projections]
    
    # Configure encoder parameters
    encoder_params = EncoderParams(
        fps=fps,
        cq_hw=quality,
        crf_sw=min(51, 51 - int(quality * 51 / 100)),
        preset_sw=preset_sw,
        force_software=force_software,
        threads=0
    )
    
    results = {
        'darks': {},
        'flats': {},
        'projections': {},
        'metadata': {
            'quality': quality,
            'fps': fps,
            'folder_name': folder_name,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    # Process each type independently
    # In correction mode, skip darks and flats (they were used for correction)
    types_to_process = [
        ('darks', dark_files, ranges.get('darks')),
        ('flats', flat_files, ranges.get('flats')),
        ('projections', projection_files, ranges.get('projections'))
    ]
    
    for data_type, files, type_range in types_to_process:
        # Skip darks and flats in correction mode
        if use_correction and data_type in ['darks', 'flats']:
            mode_name = "attenuation" if use_attenuation else "transmission"
            print(f"   ⏭️  Skipping {data_type} ({mode_name} mode - used for correction)")
            continue
            
        if type_range is None or len(files) == 0:
            print(f"   ⚠️  Skipping {data_type} (no data)")
            continue
        
        is_attenuation_projection = type_range.get('use_attenuation', False)
        is_transmission_projection = type_range.get('use_transmission', False)
        is_corrected_projection = is_attenuation_projection or is_transmission_projection
        
        if is_attenuation_projection:
            mode_str = "attenuation"
        elif is_transmission_projection:
            mode_str = "transmission"
        else:
            mode_str = data_type
        print(f"   🎬 Compressing {mode_str}: {len(files)} files...")
        if use_cdf_normalization:
            print(f"      📊 Per-frame CDF normalization enabled (bins={cdf_num_bins})")
        
        # Create output filename
        if is_attenuation_projection:
            suffix = "_attenuation"
        elif is_transmission_projection:
            suffix = "_transmission"
        else:
            suffix = ""
        cdf_suffix = "_cdf" if use_cdf_normalization else ""
        output_file = f"{folder_name}_q{quality}_{data_type}{suffix}{cdf_suffix}.mov"
        
        # Create streamer
        streamer = HevcGray10Streamer(
            base_path=output_path,
            segment_prefix=f"{folder_name}_{data_type}{suffix}{cdf_suffix}",
            params=encoder_params,
        )
        
        processed_frames = 0
        start_time = time.time()
        
        # Storage for per-frame CDF mappings (only bin_centers and cdf values needed for inversion)
        per_frame_cdf_bin_centers = [] if use_cdf_normalization else None
        per_frame_cdf_values = [] if use_cdf_normalization else None
        
        try:
            # Check if using per-frame normalization
            use_per_frame = type_range.get('use_per_frame', False)
            per_frame_min = type_range.get('per_frame_min', None)
            per_frame_max = type_range.get('per_frame_max', None)
            
            with streamer.start_segment(q=quality, outfile=output_file):
                # Process each file
                for frame_idx, tiff_file in enumerate(tqdm(files, desc=f"      {mode_str}", leave=False)):
                    # Load image
                    img = Image.open(tiff_file)
                    arr = np.array(img, np.int32)
                    
                    # Handle different formats
                    if arr.ndim == 3:
                        arr = arr[:, :, 0]
                    
                    # Apply correction if enabled for projections
                    if is_attenuation_projection:
                        arr = compute_attenuation(arr.astype(np.float32), dark_mean, flat_mean)
                    elif is_transmission_projection:
                        arr = compute_transmission(arr.astype(np.float32), dark_mean, flat_mean)
                    
                    # Convert to tensor and normalize
                    frame_tensor = torch.from_numpy(arr).float()
                    
                    if use_per_frame and per_frame_min is not None and per_frame_max is not None:
                        # Use per-frame percentile range for normalization
                        frame_min = per_frame_min[frame_idx]
                        frame_max = per_frame_max[frame_idx]
                        span = frame_max - frame_min
                        if span > 0:
                            frame_tensor = (frame_tensor - frame_min) / span
                        else:
                            frame_tensor = torch.zeros_like(frame_tensor)
                    else:
                        # Use global range for normalization (original behavior)
                        span = type_range['max'] - type_range['min']
                        if span > 0:
                            frame_tensor = (frame_tensor - type_range['min']) / span
                        else:
                            frame_tensor = torch.zeros_like(frame_tensor)
                    
                    frame_tensor = torch.clamp(frame_tensor, 0.0, 1.0)
                    
                    # Apply per-frame CDF normalization if enabled
                    if use_cdf_normalization:
                        frame_np = frame_tensor.numpy()
                        frame_np, cdf_mapping = compute_and_apply_frame_cdf(frame_np, num_bins=cdf_num_bins)
                        # Store CDF mapping for this frame (only what's needed for inversion)
                        per_frame_cdf_bin_centers.append(cdf_mapping['bin_centers'])
                        per_frame_cdf_values.append(cdf_mapping['cdf'])
                        frame_tensor = torch.from_numpy(frame_np).float()
                        frame_tensor = torch.clamp(frame_tensor, 0.0, 1.0)
                    
                    # Append to streamer
                    streamer.append_frame(frame_tensor)
                    processed_frames += 1
            
            compression_time = time.time() - start_time
            
            # Get output file info
            output_segments = streamer.segments
            if len(output_segments) > 0:
                output_file_path = output_segments[0]
                file_size_mb = output_file_path.stat().st_size / 1e6
                
                # Calculate metrics
                original_size_mb = (processed_frames * type_range['width'] * type_range['height'] * 2) / 1e6
                compression_ratio = original_size_mb / file_size_mb if file_size_mb > 0 else 0
                space_savings = (1 - file_size_mb / original_size_mb) * 100 if original_size_mb > 0 else 0
                bits_per_pixel = (file_size_mb * 8e6) / (processed_frames * type_range['width'] * type_range['height']) if processed_frames > 0 else 0
                
                results[data_type] = {
                    'output_file': str(output_file_path),
                    'processed_frames': processed_frames,
                    'original_size_mb': original_size_mb,
                    'compressed_size_mb': file_size_mb,
                    'compression_ratio': compression_ratio,
                    'space_savings_pct': space_savings,
                    'bits_per_pixel': bits_per_pixel,
                    'compression_time_s': compression_time,
                    'fps_processing': processed_frames / compression_time if compression_time > 0 else 0,
                    'range_min': type_range['min'],
                    'range_max': type_range['max'],
                    'range_delta': type_range['range'],
                    'use_attenuation': is_attenuation_projection,
                    'use_transmission': is_transmission_projection,
                    'use_cdf_normalization': use_cdf_normalization
                }
                
                # Save normalization metadata for reconstruction
                norm_file = output_file_path.with_suffix('.npz')
                if type_range.get('use_per_frame', False):
                    save_dict = {
                        'per_frame_min': type_range['per_frame_min'],
                        'per_frame_max': type_range['per_frame_max'],
                        'global_min': type_range['min'],
                        'global_max': type_range['max'],
                        'low_percentile': type_range.get('low_percentile', 1.0),
                        'high_percentile': type_range.get('high_percentile', 99.0),
                        'use_per_frame': True,
                        'use_attenuation': is_attenuation_projection,
                        'use_transmission': is_transmission_projection,
                        'use_cdf_normalization': use_cdf_normalization
                    }
                    # If using correction mode, also save dark and flat means for reconstruction
                    if is_corrected_projection and dark_mean is not None and flat_mean is not None:
                        save_dict['dark_mean'] = dark_mean
                        save_dict['flat_mean'] = flat_mean
                    # Save per-frame CDF mappings if enabled
                    if use_cdf_normalization and per_frame_cdf_bin_centers is not None:
                        # Stack per-frame CDF data into 2D arrays (num_frames x num_bins)
                        save_dict['per_frame_cdf_bin_centers'] = np.stack(per_frame_cdf_bin_centers, axis=0)
                        save_dict['per_frame_cdf_values'] = np.stack(per_frame_cdf_values, axis=0)
                        save_dict['cdf_num_bins'] = cdf_num_bins
                    np.savez(norm_file, **save_dict)
                    cdf_msg = " (with per-frame CDF)" if use_cdf_normalization else ""
                    print(f"      💾 Saved normalization values{cdf_msg} to {norm_file.name}")
                else:
                    save_dict = {
                        'global_min': type_range['min'],
                        'global_max': type_range['max'],
                        'use_per_frame': False,
                        'use_attenuation': is_attenuation_projection,
                        'use_transmission': is_transmission_projection,
                        'use_cdf_normalization': use_cdf_normalization
                    }
                    if is_corrected_projection and dark_mean is not None and flat_mean is not None:
                        save_dict['dark_mean'] = dark_mean
                        save_dict['flat_mean'] = flat_mean
                    # Save per-frame CDF mappings if enabled
                    if use_cdf_normalization and per_frame_cdf_bin_centers is not None:
                        save_dict['per_frame_cdf_bin_centers'] = np.stack(per_frame_cdf_bin_centers, axis=0)
                        save_dict['per_frame_cdf_values'] = np.stack(per_frame_cdf_values, axis=0)
                        save_dict['cdf_num_bins'] = cdf_num_bins
                    np.savez(norm_file, **save_dict)
                    cdf_msg = " (with per-frame CDF)" if use_cdf_normalization else ""
                    print(f"      💾 Saved normalization values{cdf_msg} to {norm_file.name}")
                
                print(f"      ✅ {processed_frames} frames → {file_size_mb:.1f} MB ({compression_ratio:.1f}:1)")
            
        except Exception as e:
            print(f"      ❌ Error: {e}")
            results[data_type] = {'error': str(e)}
    
    return results


def batch_compress_tomography(
    ct_files_base_path: Path,
    output_path: Path,
    quality_settings: List[int] = [100, 95, 90],
    sample_ratio: float = 1.0,
    min_samples: int = 5,
    max_samples: int = 10000,
    fps: int = 30,
    create_histogram: bool = False,
    force_software_encoding: bool = True,
    preset_sw: str = "slow",
    limit_num_folders: Optional[int] = None,
    folder_id: Optional[str] = None,
    use_per_frame_percentile: bool = True,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    use_attenuation: bool = False,
    use_transmission: bool = False,
    use_cdf_normalization: bool = False,
    cdf_num_bins: int = 1000,
    overwrite: bool = False
) -> pd.DataFrame:
    """
    Batch process all tomographic TIFF sequences with independent compression.
    
    This is the main entry point for the pipeline. It:
    1. Discovers all folders containing TIFF files
    2. For each folder and quality setting:
       - Estimates ranges independently for darks, flats, projections
       - Compresses each type to a separate HEVC file
       - Records all metadata and metrics
    3. Produces a comprehensive CSV report
    
    Parameters:
    -----------
    ct_files_base_path : Path
        Base directory containing folders with TIFF sequences
    output_path : Path
        Directory for compressed outputs and reports
    quality_settings : List[int]
        List of quality values to test (0-100)
    sample_ratio : float
        Fraction of files to sample for range estimation
    min_samples : int
        Minimum files to sample per type
    max_samples : int
        Maximum files to sample per type
    fps : int
        Output video frame rate
    create_histogram : bool
        Whether to create histograms during range estimation
    force_software_encoding : bool
        Force software encoding
    preset_sw : str
        Software encoding preset (ultrafast, fast, medium, slow, veryslow)
    folder_id : Optional[str]
        If provided, only process folder named 'file_{folder_id}_extracted'.
        If None, process all folders with '_extracted' in name.
    limit_num_folders : Optional[int]
        Limit the number of folders to process (for testing)
    use_per_frame_percentile : bool
        If True, use per-frame percentile normalization (default: True)
    low_percentile : float
        Lower percentile value for per-frame normalization (default: 1.0)
    high_percentile : float
        Upper percentile value for per-frame normalization (default: 99.0)
    use_attenuation : bool
        If True, compress attenuation-corrected projections instead of raw.
        In this mode, darks and flats are used for correction but not compressed.
        The metadata includes dark/flat means for reconstruction back to raw.
        Attenuation clips transmission values, potentially losing information.
    use_transmission : bool
        If True, compress unclipped transmission values instead of raw.
        Transmission is T = (I - dark) / (flat - dark) without clipping.
        This preserves all information including values < 0 or > 1.
        Cannot be used together with use_attenuation.
    use_cdf_normalization : bool
        If True, apply per-frame CDF-based histogram equalization after standard 
        normalization. Each frame gets its own CDF mapping which is saved to the 
        .npz file for reconstruction. This creates a more uniform pixel distribution 
        which may improve compression.
    cdf_num_bins : int
        Number of bins for per-frame CDF histogram computation (default: 1000)
    overwrite : bool
        If True, reprocess all folders even if results exist in CSV.
        If False (default), skip folders/quality combinations already in CSV.
    
    Returns:
    --------
    results_df : pd.DataFrame
        DataFrame with all compression results and metadata
    """
    # Validate that attenuation and transmission are not both enabled
    if use_attenuation and use_transmission:
        raise ValueError("Cannot use both attenuation and transmission modes simultaneously")
    
    use_correction = use_attenuation or use_transmission
    
    print("🚀 Tomographic Data Compression Pipeline")
    print("=" * 80)
    print(f"📁 Base path: {ct_files_base_path.absolute()}")
    print(f"💾 Output path: {output_path.absolute()}")
    print(f"🎛️  Quality settings: {quality_settings}")
    print(f"⚙️  Sample ratio: {sample_ratio} ({sample_ratio*100:.1f}%)")
    print(f"🎬 FPS: {fps}, Preset: {preset_sw}, Software: {force_software_encoding}")
    if use_per_frame_percentile:
        print(f"📊 Per-frame normalization: [{low_percentile}th, {high_percentile}th] percentile")
    else:
        print(f"📊 Normalization: Global min/max")
    if use_attenuation:
        print(f"📐 ATTENUATION MODE: Compressing μ = -ln((I-dark)/(flat-dark)) [clipped]")
    elif use_transmission:
        print(f"📐 TRANSMISSION MODE: Compressing T = (I-dark)/(flat-dark) [unclipped]")
    if use_cdf_normalization:
        print(f"📈 PER-FRAME CDF NORMALIZATION: Enabled (bins={cdf_num_bins})")
    print(f"🔄 Overwrite mode: {overwrite}")
    print()
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # CSV file for incremental saving
    results_csv_path = output_path / "tomography_compression_results.csv"
    
    # Load existing results if CSV exists (for skipping already processed entries)
    existing_results_df = None
    if results_csv_path.exists() and not overwrite:
        try:
            existing_results_df = pd.read_csv(results_csv_path)
            print(f"📂 Loaded existing results: {len(existing_results_df)} entries from {results_csv_path.name}")
        except Exception as e:
            print(f"⚠️  Could not load existing CSV: {e}. Starting fresh.")
            existing_results_df = None
    elif overwrite and results_csv_path.exists():
        print(f"🗑️  Overwrite mode: Ignoring existing results in {results_csv_path.name}")
    
    def result_exists(folder_name: str, quality: int) -> bool:
        """Check if a result already exists in the CSV for this folder/quality combination."""
        if existing_results_df is None or overwrite:
            return False
        mask = (existing_results_df['folder_name'] == folder_name) & \
               (existing_results_df['quality'] == quality)
        # Also check correction mode if columns exist
        if 'use_attenuation' in existing_results_df.columns:
            mask &= (existing_results_df['use_attenuation'] == use_attenuation)
        if 'use_transmission' in existing_results_df.columns:
            mask &= (existing_results_df['use_transmission'] == use_transmission)
        if 'use_cdf_normalization' in existing_results_df.columns:
            mask &= (existing_results_df['use_cdf_normalization'] == use_cdf_normalization)
        return mask.any()
    
    def save_result_to_csv(result_entry: dict):
        """Append a single result to the CSV file."""
        result_df = pd.DataFrame([result_entry])
        if results_csv_path.exists():
            # Append to existing CSV
            result_df.to_csv(results_csv_path, mode='a', header=False, index=False)
        else:
            # Create new CSV with header
            result_df.to_csv(results_csv_path, mode='w', header=True, index=False)
    
    # Find all folders with TIFF files (filter for _extracted folders to avoid compressed ones)
    if folder_id is not None:
        # Process only the specific folder with the given folder_id
        target_folder_name = f"file_{folder_id}_extracted"
        target_folder_path = ct_files_base_path / target_folder_name
        if target_folder_path.exists() and target_folder_path.is_dir():
            ct_folders = [target_folder_path]
            print(f"🎯 Processing single folder: {target_folder_name} (folder_id={folder_id})")
        else:
            print(f"❌ Error: Folder '{target_folder_name}' not found in {ct_files_base_path}")
            ct_folders = []
    else:
        # Process all _extracted folders
        ct_folders = [f for f in ct_files_base_path.iterdir() if f.is_dir() and '_extracted' in f.name]
        ct_folders = sorted(ct_folders)
    
    print(f"📂 Found {len(ct_folders)} folders to process:")
    for i, folder in enumerate(ct_folders, 1):
        print(f"   {i:2d}. {folder.name}")
    
    total_tasks = len(ct_folders) * len(quality_settings)
    print(f"\n🎯 Total tasks: {len(ct_folders)} folders × {len(quality_settings)} qualities = {total_tasks}")
    print()
    
    # Results tracking
    batch_results = []
    overall_start_time = time.time()
    
    # Process each folder
    for folder_idx, data_path in enumerate(ct_folders, 1):
        folder_name = data_path.name
        print(f"\n📂 [{folder_idx}/{len(ct_folders)}] Processing: {folder_name}")
        print("-" * 80)
        
        # Get TIFF files
        tiff_files = sorted(list(data_path.glob('*.tif')) + list(data_path.glob('*.tiff')))
        
        if len(tiff_files) == 0:
            print(f"   ⚠️  No TIFF files found, skipping...")
            continue
        
        print(f"   Found {len(tiff_files)} TIFF files")
        
        # Load tomography parameters
        try:
            tomo_params = load_tomography_params(data_path)
            # tomo_params['num_projections'] = len(tiff_files) - tomo_params['num_darks'] - tomo_params['num_flats']
            
            if tomo_params['num_projections'] <= 0:
                print(f"   ⚠️  Invalid structure, skipping...")
                continue
            
            print(f"   📊 Structure: {tomo_params['num_darks']} darks + {tomo_params['num_flats']} flats + {tomo_params['num_projections']} projections")
        except Exception as e:
            print(f"   ❌ Error loading parameters: {e}")
            continue
        
        # Compute dark/flat means if using correction mode
        dark_mean, flat_mean = None, None
        if use_correction:
            try:
                dark_mean, flat_mean = compute_mean_dark_flat(
                    tiff_files, tomo_params['num_darks'], tomo_params['num_flats']
                )
            except Exception as e:
                print(f"   ❌ Error computing dark/flat means: {e}")
                continue
        
        # Estimate independent ranges
        try:
            print(f"   🔍 Estimating independent ranges...")
            start_time = time.time()
            
            ranges = estimate_independent_ranges(
                tiff_files=tiff_files,
                params=tomo_params,
                sample_ratio=sample_ratio,
                min_samples=min_samples,
                max_samples=max_samples,
                create_histogram=create_histogram,
                use_per_frame_percentile=use_per_frame_percentile,
                low_percentile=low_percentile,
                high_percentile=high_percentile,
                use_attenuation=use_attenuation,
                use_transmission=use_transmission,
                dark_mean=dark_mean,
                flat_mean=flat_mean
            )
            
            estimation_time = time.time() - start_time
            print(f"   ✅ Range estimation completed in {estimation_time:.1f}s")
            
        except Exception as e:
            print(f"   ❌ Error estimating ranges: {e}")
            continue
        
        # Process with each quality setting
        for quality_idx, quality in enumerate(quality_settings, 1):
            # Check if this folder/quality combination already exists in CSV
            if result_exists(folder_name, quality):
                print(f"\n   ⏭️  [{quality_idx}/{len(quality_settings)}] Quality {quality} - SKIPPED (already in CSV)")
                continue
            
            print(f"\n   🎬 [{quality_idx}/{len(quality_settings)}] Quality {quality}")
            
            # Create output directory for this folder/quality
            if use_attenuation:
                suffix = "_attenuation"
            elif use_transmission:
                suffix = "_transmission"
            else:
                suffix = ""
            folder_output_path = output_path / f"{folder_name}_q{quality}{suffix}"
            folder_output_path.mkdir(parents=True, exist_ok=True)
            
            try:
                start_time = time.time()
                
                # Compress all three types independently
                compression_results = stream_tomography_to_hevc(
                    tiff_files=tiff_files,
                    params=tomo_params,
                    ranges=ranges,
                    output_path=folder_output_path,
                    folder_name=folder_name,
                    quality=quality,
                    fps=fps,
                    force_software=force_software_encoding,
                    preset_sw=preset_sw,
                    use_attenuation=use_attenuation,
                    use_transmission=use_transmission,
                    dark_mean=dark_mean,
                    flat_mean=flat_mean,
                    use_cdf_normalization=use_cdf_normalization,
                    cdf_num_bins=cdf_num_bins
                )
                
                total_compression_time = time.time() - start_time
                
                # Flatten results for CSV storage
                result_entry = {
                    'folder_name': folder_name,
                    'quality': quality,
                    'use_attenuation': use_attenuation,
                    'use_transmission': use_transmission,
                    'use_cdf_normalization': use_cdf_normalization,
                    'num_tiff_files_total': len(tiff_files),
                    'num_darks': tomo_params['num_darks'],
                    'num_flats': tomo_params['num_flats'],
                    'num_projections': tomo_params['num_projections'],
                    'width': ranges.get('projections', {}).get('width', 0),
                    'height': ranges.get('projections', {}).get('height', 0),
                    'total_compression_time_s': total_compression_time,
                    'timestamp': datetime.now().isoformat(),
                    'preset_sw': preset_sw,
                    'force_software': force_software_encoding,
                    'fps': fps
                }
                
                # Add data for each type
                for data_type in ['darks', 'flats', 'projections']:
                    if data_type in compression_results and 'error' not in compression_results[data_type]:
                        data = compression_results[data_type]
                        result_entry.update({
                            f'{data_type}_output_file': data.get('output_file', ''),
                            f'{data_type}_frames': data.get('processed_frames', 0),
                            f'{data_type}_original_mb': data.get('original_size_mb', 0),
                            f'{data_type}_compressed_mb': data.get('compressed_size_mb', 0),
                            f'{data_type}_compression_ratio': data.get('compression_ratio', 0),
                            f'{data_type}_space_savings_pct': data.get('space_savings_pct', 0),
                            f'{data_type}_bits_per_pixel': data.get('bits_per_pixel', 0),
                            f'{data_type}_compression_time_s': data.get('compression_time_s', 0),
                            f'{data_type}_range_min': data.get('range_min', 0),
                            f'{data_type}_range_max': data.get('range_max', 0),
                            f'{data_type}_range_delta': data.get('range_delta', 0)
                        })
                    else:
                        # Fill with zeros/empty if type not processed
                        result_entry.update({
                            f'{data_type}_output_file': '',
                            f'{data_type}_frames': 0,
                            f'{data_type}_original_mb': 0,
                            f'{data_type}_compressed_mb': 0,
                            f'{data_type}_compression_ratio': 0,
                            f'{data_type}_space_savings_pct': 0,
                            f'{data_type}_bits_per_pixel': 0,
                            f'{data_type}_compression_time_s': 0,
                            f'{data_type}_range_min': 0,
                            f'{data_type}_range_max': 0,
                            f'{data_type}_range_delta': 0
                        })
                
                batch_results.append(result_entry)
                
                # Save result immediately to CSV (incremental save)
                save_result_to_csv(result_entry)
                
                # Print summary
                total_original = sum([result_entry.get(f'{t}_original_mb', 0) for t in ['darks', 'flats', 'projections']])
                total_compressed = sum([result_entry.get(f'{t}_compressed_mb', 0) for t in ['darks', 'flats', 'projections']])
                overall_ratio = total_original / total_compressed if total_compressed > 0 else 0
                
                print(f"   ✅ Completed: {total_original:.1f} MB → {total_compressed:.1f} MB ({overall_ratio:.1f}:1)")
                print(f"   💾 Result saved to CSV")
                
            except Exception as e:
                print(f"   ❌ Error: {e}")
                # Store error result
                error_entry = {
                    'folder_name': folder_name,
                    'quality': quality,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                batch_results.append(error_entry)
                # Save error result immediately to CSV
                save_result_to_csv(error_entry)
        if limit_num_folders is not None and folder_idx >= limit_num_folders:
            print(f"\n🔔 Reached folder limit of {limit_num_folders}, stopping early.")
            break
    
    # Calculate overall statistics
    total_time = time.time() - overall_start_time
    successful_tasks = len([r for r in batch_results if 'error' not in r])
    failed_tasks = len([r for r in batch_results if 'error' in r])
    
    print(f"\n🏁 Pipeline Complete!")
    print("=" * 80)
    print(f"📊 Summary:")
    print(f"   Folders processed: {len(ct_folders)}")
    print(f"   Successful tasks: {successful_tasks}")
    print(f"   Failed tasks: {failed_tasks}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    if len(batch_results) > 0:
        print(f"   Average time per task: {total_time/len(batch_results):.1f} seconds")
    
    # Load the full CSV (which includes previously existing entries + new entries)
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    if results_csv_path.exists():
        results_df = pd.read_csv(results_csv_path)
        print(f"\n💾 Results incrementally saved to: {results_csv_path}")
        print(f"   Total entries in CSV: {len(results_df)}")
    else:
        # No results were processed (all were skipped or no folders found)
        results_df = pd.DataFrame(batch_results) if batch_results else pd.DataFrame()
        print(f"\n⚠️  No new results to save (all entries skipped or no folders found)")
    
    # Create summary report
    if len(results_df) > 0:
        success_df = results_df[~results_df['error'].notna() if 'error' in results_df.columns else [True] * len(results_df)]
        
        if len(success_df) > 0:
            summary_file = output_path / f"tomography_compression_summary_{timestamp_str}.txt"
            with open(summary_file, 'w') as f:
                f.write("Tomographic Data Compression Summary\n")
                f.write("=" * 60 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("INDEPENDENT COMPRESSION (Darks, Flats, Projections)\n")
                f.write("-" * 60 + "\n")
                f.write(f"Folders processed: {success_df['folder_name'].nunique()}\n")
                f.write(f"Quality settings: {sorted(success_df['quality'].unique())}\n\n")
                
                for data_type in ['darks', 'flats', 'projections']:
                    col_name = f'{data_type}_original_mb'
                    if col_name not in success_df.columns:
                        continue
                    total_original = success_df[f'{data_type}_original_mb'].sum()
                    total_compressed = success_df[f'{data_type}_compressed_mb'].sum()
                    if total_original > 0:
                        ratio = total_original / total_compressed if total_compressed > 0 else 0
                        f.write(f"\n{data_type.upper()}:\n")
                        f.write(f"  Original: {total_original:.1f} MB\n")
                        f.write(f"  Compressed: {total_compressed:.1f} MB\n")
                        f.write(f"  Compression ratio: {ratio:.1f}:1\n")
                        f.write(f"  Space savings: {(1 - total_compressed/total_original)*100:.1f}%\n")
            
            print(f"📄 Summary saved to: {summary_file}")
    
    print("\n✅ Pipeline execution completed!")
    return results_df


if __name__ == "__main__":
    """
    Example usage of the batch compression pipeline.
    
    This can be run directly or imported and called from other scripts/notebooks.
    """

    # Configuration
    CT_FILES_BASE_PATH = Path('/das/home/barbaf_l/p22274/compression_paper')
    OUTPUT_PATH = Path('/das/home/barbaf_l/p22274/compression_paper/streaming_output')
    QUALITY_SETTINGS = [100, 95, 90]
    SAMPLE_RATIO = 1.0
    MIN_SAMPLES = 5
    MAX_SAMPLES = 10000
    FPS = 30
    CREATE_HISTOGRAM = False
    FORCE_SOFTWARE_ENCODING = True
    PRESET_SW = "fast"
    LIMIT_NUM_FOLDERS = 1
    
    # Run the pipeline
    results_df = batch_compress_tomography(
        ct_files_base_path=CT_FILES_BASE_PATH,
        output_path=OUTPUT_PATH,
        quality_settings=QUALITY_SETTINGS,
        sample_ratio=SAMPLE_RATIO,
        min_samples=MIN_SAMPLES,
        max_samples=MAX_SAMPLES,
        fps=FPS,
        create_histogram=CREATE_HISTOGRAM,
        force_software_encoding=FORCE_SOFTWARE_ENCODING,
        preset_sw=PRESET_SW,
        limit_num_folders=LIMIT_NUM_FOLDERS
    )
    
    print(f"\n📊 Results DataFrame shape: {results_df.shape}")
    print(f"📊 Columns: {list(results_df.columns)}")
