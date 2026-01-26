"""
Utilities for handling normalization metadata in compressed tomographic data.

This module provides functions to save and load normalization parameters used
during compression, enabling accurate reconstruction of the original data.
"""

from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np


def load_normalization_metadata(npz_path: Union[str, Path]) -> Dict:
    """
    Load normalization metadata from a .npz file saved during compression.
    
    Parameters:
    -----------
    npz_path : str or Path
        Path to the .npz file containing normalization metadata
        
    Returns:
    --------
    metadata : dict
        Dictionary containing normalization parameters:
        - use_per_frame: bool indicating if per-frame normalization was used
        - global_min: float minimum value used for normalization
        - If use_per_frame is True:
            - per_frame_max: ndarray of per-frame max values (99th percentile)
            - percentile: float percentile value used (e.g., 99.0)
        - If use_per_frame is False:
            - global_max: float maximum value used for normalization
    
    Examples:
    ---------
    >>> metadata = load_normalization_metadata("output_q90_projections.npz")
    >>> if metadata['use_per_frame']:
    ...     # Use per-frame normalization for reconstruction
    ...     for i, frame in enumerate(frames):
    ...         reconstructed = frame * (metadata['per_frame_max'][i] - metadata['global_min']) + metadata['global_min']
    ... else:
    ...     # Use global normalization
    ...     reconstructed = frame * (metadata['global_max'] - metadata['global_min']) + metadata['global_min']
    """
    npz_path = Path(npz_path)
    
    if not npz_path.exists():
        raise FileNotFoundError(f"Normalization metadata file not found: {npz_path}")
    
    data = np.load(npz_path)
    metadata = {key: data[key] for key in data.files}
    
    # Convert scalar arrays to Python types for convenience
    for key in ['use_per_frame', 'global_min', 'global_max', 'percentile']:
        if key in metadata and metadata[key].ndim == 0:
            metadata[key] = metadata[key].item()
    
    return metadata


def denormalize_frame(
    normalized_frame: np.ndarray,
    metadata: Dict,
    frame_idx: Optional[int] = None
) -> np.ndarray:
    """
    Denormalize a frame using the provided normalization metadata.
    
    Parameters:
    -----------
    normalized_frame : ndarray
        Normalized frame in range [0, 1]
    metadata : dict
        Normalization metadata from load_normalization_metadata()
    frame_idx : int, optional
        Frame index (required if use_per_frame is True)
        
    Returns:
    --------
    denormalized : ndarray
        Original frame values before normalization
        
    Examples:
    ---------
    >>> metadata = load_normalization_metadata("output_q90_projections.npz")
    >>> for i, normalized_frame in enumerate(video_frames):
    ...     original_frame = denormalize_frame(normalized_frame, metadata, frame_idx=i)
    """
    if metadata['use_per_frame']:
        if frame_idx is None:
            raise ValueError("frame_idx is required when use_per_frame is True")
        
        per_frame_max = metadata['per_frame_max'][frame_idx]
        global_min = metadata['global_min']
        
        # Denormalize: value = normalized * (max - min) + min
        denormalized = normalized_frame * (per_frame_max - global_min) + global_min
    else:
        global_min = metadata['global_min']
        global_max = metadata['global_max']
        
        denormalized = normalized_frame * (global_max - global_min) + global_min
    
    return denormalized


def get_normalization_info(npz_path: Union[str, Path]) -> str:
    """
    Get a human-readable summary of normalization metadata.
    
    Parameters:
    -----------
    npz_path : str or Path
        Path to the .npz file containing normalization metadata
        
    Returns:
    --------
    info : str
        Human-readable description of the normalization parameters
    """
    metadata = load_normalization_metadata(npz_path)
    
    info_lines = [
        f"Normalization Metadata: {Path(npz_path).name}",
        "=" * 60,
        f"Type: {'Per-frame percentile' if metadata['use_per_frame'] else 'Global min/max'}",
        f"Global Min: {metadata['global_min']:.2f}",
    ]
    
    if metadata['use_per_frame']:
        per_frame_max = metadata['per_frame_max']
        percentile = metadata.get('percentile', 99.0)
        info_lines.extend([
            f"Percentile: {percentile}th",
            f"Per-frame Max Values:",
            f"  Count: {len(per_frame_max)}",
            f"  Range: [{per_frame_max.min():.2f}, {per_frame_max.max():.2f}]",
            f"  Mean: {per_frame_max.mean():.2f}",
            f"  Median: {np.median(per_frame_max):.2f}",
            f"  Std: {per_frame_max.std():.2f}",
        ])
    else:
        info_lines.append(f"Global Max: {metadata['global_max']:.2f}")
    
    return "\n".join(info_lines)


def find_normalization_file(video_path: Union[str, Path]) -> Optional[Path]:
    """
    Find the normalization metadata file corresponding to a compressed video file.
    
    The normalization file is expected to have the same name as the video file
    but with a .npz extension.
    
    Parameters:
    -----------
    video_path : str or Path
        Path to the compressed video file (e.g., "output_q90_projections.mov")
        
    Returns:
    --------
    npz_path : Path or None
        Path to the normalization metadata file, or None if not found
    """
    video_path = Path(video_path)
    npz_path = video_path.with_suffix('.npz')
    
    return npz_path if npz_path.exists() else None


# ============================================================================
# CDF-Based Histogram Mapping Functions
# ============================================================================

def compute_cdf_mapping(image: np.ndarray, num_bins: int = 1000) -> Dict:
    """
    Compute the CDF (Cumulative Distribution Function) mapping for an image.
    
    This creates a histogram equalization mapping that transforms pixel values
    to a uniform distribution in [0, 1]. The mapping is invertible and can be
    stored compactly for later reconstruction.
    
    Parameters:
    -----------
    image : ndarray
        Input image (can be raw intensity or attenuation values)
    num_bins : int, default=1000
        Number of bins for the histogram
        
    Returns:
    --------
    mapping : dict
        Dictionary containing:
        - 'bin_edges': array of bin edges (length num_bins + 1)
        - 'bin_centers': array of bin centers (length num_bins)
        - 'hist': histogram counts (length num_bins)
        - 'cdf': cumulative distribution function values [0, 1] (length num_bins)
        - 'min_val': minimum value in the image
        - 'max_val': maximum value in the image
        - 'num_bins': number of bins used
        
    Examples:
    ---------
    >>> mapping = compute_cdf_mapping(image, num_bins=500)
    >>> mapped_image = apply_cdf_mapping(image, mapping)
    >>> reconstructed = invert_cdf_mapping(mapped_image, mapping)
    """
    # Filter out invalid values (inf/nan) for attenuation images
    valid_data = image[np.isfinite(image)].flatten()
    
    min_val = valid_data.min()
    max_val = valid_data.max()
    
    # Compute histogram
    hist, bin_edges = np.histogram(valid_data, bins=num_bins, range=(min_val, max_val))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Compute CDF (normalized to [0, 1])
    cdf = np.cumsum(hist).astype(np.float64)
    cdf = cdf / cdf[-1]  # Normalize to [0, 1]
    
    return {
        'bin_edges': bin_edges,
        'bin_centers': bin_centers,
        'hist': hist,
        'cdf': cdf,
        'min_val': min_val,
        'max_val': max_val,
        'num_bins': num_bins
    }


def apply_cdf_mapping(image: np.ndarray, mapping: Dict) -> np.ndarray:
    """
    Apply CDF mapping to transform pixel values to [0, 1] via histogram equalization.
    
    This function maps each pixel value to its corresponding CDF value, effectively
    creating a uniform distribution of values in [0, 1].
    
    Parameters:
    -----------
    image : ndarray
        Input image with original value range
    mapping : dict
        CDF mapping from compute_cdf_mapping()
        
    Returns:
    --------
    mapped_image : ndarray
        Image with values mapped to [0, 1] via CDF
        Invalid (inf/nan) pixels are preserved as NaN
        
    Examples:
    ---------
    >>> mapping = compute_cdf_mapping(image, num_bins=500)
    >>> mapped_image = apply_cdf_mapping(image, mapping)
    >>> # mapped_image now has approximately uniform distribution in [0, 1]
    """
    # Find which bin each pixel belongs to
    bin_indices = np.digitize(image, mapping['bin_edges'][:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, len(mapping['cdf']) - 1)
    
    # Handle invalid values
    valid_mask = np.isfinite(image)
    mapped_image = np.zeros_like(image, dtype=np.float32)
    mapped_image[valid_mask] = mapping['cdf'][bin_indices[valid_mask]]
    mapped_image[~valid_mask] = np.nan
    
    return mapped_image


def invert_cdf_mapping(mapped_image: np.ndarray, mapping: Dict) -> np.ndarray:
    """
    Invert CDF mapping to recover original values from [0, 1] mapped values.
    
    This uses linear interpolation of the inverse CDF (quantile function) to
    map uniformly distributed values back to the original value range.
    
    Parameters:
    -----------
    mapped_image : ndarray
        Image with CDF-mapped values in [0, 1]
    mapping : dict
        CDF mapping from compute_cdf_mapping()
        
    Returns:
    --------
    original_image : ndarray
        Reconstructed image in original value range
        Invalid (NaN) pixels are preserved as NaN
        
    Examples:
    ---------
    >>> mapping = compute_cdf_mapping(image, num_bins=500)
    >>> mapped_image = apply_cdf_mapping(image, mapping)
    >>> reconstructed = invert_cdf_mapping(mapped_image, mapping)
    >>> # reconstructed ≈ image (up to binning discretization)
    """
    # Use linear interpolation: for each mapped value, find corresponding bin center
    valid_mask = np.isfinite(mapped_image)
    original_image = np.zeros_like(mapped_image, dtype=np.float32)
    
    # Interpolate: CDF values -> bin centers
    original_image[valid_mask] = np.interp(
        mapped_image[valid_mask],
        mapping['cdf'],
        mapping['bin_centers']
    )
    original_image[~valid_mask] = np.nan
    
    return original_image


def save_cdf_mapping(mapping: Dict, filepath: Union[str, Path]) -> None:
    """
    Save a CDF mapping to disk as a compressed .npz file.
    
    Parameters:
    -----------
    mapping : dict
        CDF mapping from compute_cdf_mapping()
    filepath : str or Path
        Path to save the mapping (should end with .npz)
        
    Examples:
    ---------
    >>> mapping = compute_cdf_mapping(image, num_bins=500)
    >>> save_cdf_mapping(mapping, "frame_0_cdf.npz")
    """
    filepath = Path(filepath)
    np.savez_compressed(
        filepath,
        bin_edges=mapping['bin_edges'],
        bin_centers=mapping['bin_centers'],
        cdf=mapping['cdf'],
        min_val=mapping['min_val'],
        max_val=mapping['max_val'],
        num_bins=mapping['num_bins']
    )


def load_cdf_mapping(filepath: Union[str, Path]) -> Dict:
    """
    Load a CDF mapping from disk.
    
    Parameters:
    -----------
    filepath : str or Path
        Path to the .npz file containing the CDF mapping
        
    Returns:
    --------
    mapping : dict
        CDF mapping dictionary compatible with apply_cdf_mapping() and invert_cdf_mapping()
        
    Examples:
    ---------
    >>> mapping = load_cdf_mapping("frame_0_cdf.npz")
    >>> mapped_image = apply_cdf_mapping(new_image, mapping)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CDF mapping file not found: {filepath}")
    
    data = np.load(filepath)
    mapping = {
        'bin_edges': data['bin_edges'],
        'bin_centers': data['bin_centers'],
        'cdf': data['cdf'],
        'min_val': float(data['min_val']),
        'max_val': float(data['max_val']),
        'num_bins': int(data['num_bins'])
    }
    
    # Add back histogram if not present (for backwards compatibility)
    if 'hist' not in mapping:
        mapping['hist'] = None
    
    return mapping
