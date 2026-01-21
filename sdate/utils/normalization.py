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
