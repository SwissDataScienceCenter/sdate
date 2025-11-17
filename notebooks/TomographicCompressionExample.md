# Tomographic Data Compression - Quick Start

This notebook demonstrates how to use the tomographic compression pipeline.

## Setup

```python
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path.cwd().parent))

from sdate.pipelines.batch_compress_tomography import (
    batch_compress_tomography,
    estimate_independent_ranges,
    stream_tomography_to_hevc
)
import pandas as pd
```

## Configuration

```python
# Configure paths
CT_FILES_BASE_PATH = Path('../data/ct_files/')
OUTPUT_PATH = Path('/das/home/barbaf_l/p22274/compression_paper/streaming_output')

# Compression settings
QUALITY_SETTINGS = [100]  # Test multiple quality levels
SAMPLE_RATIO = 1.0                # Use 100% of files for range estimation
FPS = 30                          # Output video framerate
FORCE_SOFTWARE = True             # Use software encoding
PRESET_SW = "fast"                # Encoding preset (slow = better compression)
```

## Run Full Batch Pipeline

Process all folders with all quality settings:

```python
results_df = batch_compress_tomography(
    ct_files_base_path=CT_FILES_BASE_PATH,
    output_path=OUTPUT_PATH,
    quality_settings=QUALITY_SETTINGS,
    sample_ratio=SAMPLE_RATIO,
    fps=FPS,
    force_software_encoding=FORCE_SOFTWARE,
    preset_sw=PRESET_SW
)

print(f"Processed {len(results_df)} tasks")
results_df.head()
```

## Analyze Results

```python
# Calculate total compression
total_original = sum([
    results_df[f'{t}_original_mb'].sum() 
    for t in ['darks', 'flats', 'projections']
])
total_compressed = sum([
    results_df[f'{t}_compressed_mb'].sum() 
    for t in ['darks', 'flats', 'projections']
])

print(f"Original: {total_original:.1f} MB")
print(f"Compressed: {total_compressed:.1f} MB")
print(f"Ratio: {total_original/total_compressed:.1f}:1")
print(f"Savings: {(1 - total_compressed/total_original)*100:.1f}%")
```

## View Range Information

```python
# Show ranges for each type
for idx, row in results_df.iterrows():
    print(f"\n{row['folder_name']} - Quality {row['quality']}")
    print(f"  Darks:       [{row['darks_range_min']:.0f}, {row['darks_range_max']:.0f}]")
    print(f"  Flats:       [{row['flats_range_min']:.0f}, {row['flats_range_max']:.0f}]")
    print(f"  Projections: [{row['projections_range_min']:.0f}, {row['projections_range_max']:.0f}]")
```

## Process Single Folder (Advanced)

For more control, process a single folder manually:

```python
from sdate.pipelines.batch_compress_tomography import load_tomography_params

# Select a folder
data_path = CT_FILES_BASE_PATH / "file_3_extracted"
tiff_files = sorted(list(data_path.glob('*.tif*')))

# Load structure
params = load_tomography_params(data_path)
params['num_projections'] = len(tiff_files) - params['num_darks'] - params['num_flats']

print(f"Structure: {params['num_darks']} darks + {params['num_flats']} flats + {params['num_projections']} projections")

# Estimate ranges
ranges = estimate_independent_ranges(
    tiff_files=tiff_files,
    params=params,
    sample_ratio=1.0
)

print("\nRanges:")
for dtype, info in ranges.items():
    print(f"  {dtype}: [{info['min']:.0f}, {info['max']:.0f}]")

# Compress
results = stream_tomography_to_hevc(
    tiff_files=tiff_files,
    params=params,
    ranges=ranges,
    output_path=OUTPUT_PATH / "single_test",
    folder_name="file_3_extracted",
    quality=100,
    fps=30,
    force_software=True,
    preset_sw="slow"
)

print("\nCompression results:")
for dtype in ['darks', 'flats', 'projections']:
    if dtype in results:
        r = results[dtype]
        print(f"  {dtype}: {r['original_size_mb']:.1f} MB → {r['compressed_size_mb']:.1f} MB ({r['compression_ratio']:.1f}:1)")
```

## Load and Use Metadata for Decoding

```python
# Load the results CSV
results_file = OUTPUT_PATH / "tomography_compression_results_20251117_120000.csv"  # Update filename
df = pd.read_csv(results_file)

# Get metadata for a specific file
folder = "file_3_extracted"
quality = 100

row = df[(df['folder_name'] == folder) & (df['quality'] == quality)].iloc[0]

# Extract ranges for decoding
darks_range = (row['darks_range_min'], row['darks_range_max'])
flats_range = (row['flats_range_min'], row['flats_range_max'])
projections_range = (row['projections_range_min'], row['projections_range_max'])

print(f"Ranges for {folder} at quality {quality}:")
print(f"  Darks: {darks_range}")
print(f"  Flats: {flats_range}")
print(f"  Projections: {projections_range}")

# When decoding, denormalize like this:
# original_value = normalized_value * (range_max - range_min) + range_min
```

## Notes

- Each type (darks, flats, projections) is compressed to a separate `.mov` file
- Range metadata is stored in the CSV for accurate decoding
- The pipeline automatically handles edge cases (extra frames, missing types)
- Use `sample_ratio < 1.0` for faster range estimation on large datasets
