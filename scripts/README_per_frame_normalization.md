# Per-Frame Percentile Normalization for Video Compression

## Overview

This update introduces **per-frame percentile-based normalization** for tomographic data compression, replacing the previous global min/max normalization approach. This provides better preservation of dynamic range and reduces the impact of outliers.

## Key Changes

### 1. Per-Frame Normalization
- **Previous**: Used a single global max value across all frames of each data type (darks, flats, projections)
- **New**: Computes the 99th percentile (configurable) for each frame individually
- **Benefit**: Each frame is normalized to its own dynamic range, preventing extreme values in one frame from affecting others

### 2. Normalization Metadata Storage
- Normalization parameters are automatically saved alongside compressed video files
- Saved as `.npz` files with the same base name as the video file
- Contains all information needed for accurate reconstruction

### 3. Backward Compatibility
- Global normalization is still available by setting `use_per_frame_percentile=False`
- Default behavior now uses per-frame 99th percentile normalization

## Usage

### Basic Compression

```python
from pathlib import Path
from sdate.pipelines.batch_compress_tomography import batch_compress_tomography

# Run compression with per-frame normalization (default)
results = batch_compress_tomography(
    ct_files_base_path=Path('data/ct_files'),
    output_path=Path('data/compressed'),
    quality_settings=[90, 95],
    use_per_frame_percentile=True,  # Default: True
    percentile=99.0  # Default: 99.0 (99th percentile)
)
```

### Using Global Normalization (Legacy)

```python
# Use global min/max normalization
results = batch_compress_tomography(
    ct_files_base_path=Path('data/ct_files'),
    output_path=Path('data/compressed'),
    quality_settings=[90, 95],
    use_per_frame_percentile=False  # Use global normalization
)
```

### Custom Percentile

```python
# Use 95th percentile instead of 99th
results = batch_compress_tomography(
    ct_files_base_path=Path('data/ct_files'),
    output_path=Path('data/compressed'),
    quality_settings=[90],
    use_per_frame_percentile=True,
    percentile=95.0  # Use 95th percentile
)
```

## Reconstruction

### Loading Normalization Metadata

```python
from sdate.utils.normalization import (
    load_normalization_metadata,
    denormalize_frame,
    get_normalization_info
)

# Load metadata
metadata = load_normalization_metadata('output_q90_projections.npz')

# Display info
print(get_normalization_info('output_q90_projections.npz'))
```

### Denormalizing Frames

```python
import numpy as np

# Load metadata
metadata = load_normalization_metadata('output_q90_projections.npz')

# Assume we have decompressed frames in [0, 1] range
for i, normalized_frame in enumerate(decompressed_frames):
    # Denormalize to original values
    original_frame = denormalize_frame(
        normalized_frame,
        metadata,
        frame_idx=i  # Required for per-frame normalization
    )
    
    # Process or save original_frame
    ...
```

### Complete Reconstruction Example

```python
from pathlib import Path
from sdate.utils.normalization import load_normalization_metadata, denormalize_frame
import tifffile

# 1. Find normalization file
video_path = Path('output_q90_projections.mov')
norm_path = video_path.with_suffix('.npz')

# 2. Load normalization metadata
metadata = load_normalization_metadata(norm_path)

# 3. Decompress video (example - actual decompression may vary)
# decompressed_frames = decompress_hevc_video(video_path)

# 4. Denormalize each frame
reconstructed_frames = []
for i, normalized_frame in enumerate(decompressed_frames):
    original = denormalize_frame(normalized_frame, metadata, frame_idx=i)
    reconstructed_frames.append(original)

# 5. Save reconstructed frames
for i, frame in enumerate(reconstructed_frames):
    tifffile.imwrite(f'reconstructed_{i:04d}.tif', frame.astype(np.uint16))
```

## Normalization Metadata Format

### Per-Frame Normalization (.npz file)
```python
{
    'use_per_frame': True,
    'global_min': 100.0,  # Minimum pixel value across all frames
    'per_frame_max': array([5234.2, 5189.7, ...]),  # 99th percentile for each frame
    'percentile': 99.0  # Percentile used
}
```

### Global Normalization (.npz file)
```python
{
    'use_per_frame': False,
    'global_min': 100.0,  # Minimum pixel value
    'global_max': 5500.0  # Maximum pixel value
}
```

## Files Modified/Created

### Modified
1. `sdate/pipelines/batch_compress_tomography.py`
   - Updated `estimate_independent_ranges()` to compute per-frame percentiles
   - Modified `stream_tomography_to_hevc()` to use per-frame normalization
   - Added normalization metadata saving

2. `scripts/run_tomography_compression.py`
   - Added parameters for per-frame normalization
   - Updated documentation

### Created
1. `sdate/utils/normalization.py`
   - Utility functions for loading/using normalization metadata
   - Functions: `load_normalization_metadata()`, `denormalize_frame()`, `get_normalization_info()`

2. `scripts/example_normalization_usage.py`
   - Complete examples of using normalization utilities
   - Demonstrates reconstruction workflow

3. `scripts/README_per_frame_normalization.md`
   - This documentation file

## Benefits

1. **Better Dynamic Range Preservation**: Each frame normalized to its own range
2. **Reduced Outlier Impact**: Using 99th percentile instead of max reduces extreme values
3. **Improved Compression**: Better utilization of available bit depth
4. **Flexible**: Can still use global normalization if needed
5. **Reconstruction-Ready**: All metadata saved for accurate reconstruction

## Performance Considerations

- **Per-frame mode**: Processes all frames to compute percentiles (slower estimation, better quality)
- **Global mode**: Samples frames to estimate range (faster estimation, may be less precise)
- Recommendation: Use per-frame mode for final compression, global mode for quick tests

## See Also

- `notebooks/TiffDynamicRangeAnalysis.ipynb` - Analyze frame-by-frame statistics
- `scripts/run_tomography_compression.py` - Main compression script
- `scripts/example_normalization_usage.py` - Reconstruction examples
