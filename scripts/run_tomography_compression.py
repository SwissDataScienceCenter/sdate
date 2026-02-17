#!/usr/bin/env python3
"""
Run Tomographic Data Compression Pipeline

This script runs the batch compression pipeline on tomographic TIFF sequences.
It compresses darks, flats, and projections independently with separate range
estimations for each type.

Usage:
    python run_tomography_compression.py [--folder-id FOLDER_ID]

Arguments:
    --folder-id : Optional folder ID to process only file_{folder_id}_extracted

Or customize the configuration below and run directly.
"""

from pathlib import Path
import sys
import argparse
import os

# Resolve project root based on this file's location
CURRENT_FILE = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_FILE.parent.parent   # /das/home/barbaf_l/sdate

# Add project root so we can import the `sdate` package
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    print(f"Added {PROJECT_ROOT} to Python path")

from sdate.pipelines.batch_compress_tomography import batch_compress_tomography


def main():
    """Run the tomographic compression pipeline with specified configuration."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run tomographic data compression pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--folder-id',
        type=str,
        default=None,
        help='Process only folder named file_{folder_id}_extracted. If not provided, process all _extracted folders.'
    )
    parser.add_argument(
        '--overwrite',
        action='store_true',
        default=False,
        help='Overwrite existing results in CSV. If not set, skip already processed folder/quality combinations.'
    )
    parser.add_argument(
        '--quality',
        type=str,
        default=None,
        help='Comma-separated quality settings (e.g., "70,60,50"). Overrides QUALITY environment variable.'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        default=None,
        help='Output directory path. Overrides OUTPUT_PATH environment variable.'
    )
    parser.add_argument(
        '--base-path',
        type=str,
        default=None,
        help='Base path for CT files. Overrides CT_FILES_BASE_PATH environment variable.'
    )
    args = parser.parse_args()
    
    folder_id = args.folder_id
    overwrite = args.overwrite
    
    # ========================================================================
    # CONFIGURATION - Parameters can be set via:
    #   1. Command-line arguments (highest priority)
    #   2. Environment variables
    #   3. Default values below (lowest priority)
    # ========================================================================
    
    # Base path containing folders with TIFF sequences
    default_base_path = '/das/home/barbaf_l/p22274/compression_paper'
    CT_FILES_BASE_PATH = Path(
        args.base_path or 
        os.environ.get('CT_FILES_BASE_PATH', default_base_path)
    )
    
    # Output directory for compressed files and reports
    default_output_path = '/das/home/barbaf_l/p22274/compression_paper/streaming_output'
    OUTPUT_PATH = Path(
        args.output_path or 
        os.environ.get('OUTPUT_PATH', default_output_path)
    )
    
    # Quality settings to test (0-100, higher = better quality)
    # Priority: CLI arg > env var > default
    default_qualities = [110]
    quality_str = args.quality or os.environ.get('QUALITY', None)
    if quality_str:
        QUALITY_SETTINGS = [int(q.strip()) for q in quality_str.split(',')]
    else:
        QUALITY_SETTINGS = default_qualities
    
    # Sampling ratio for dynamic range estimation (1.0 = 100% of files)
    SAMPLE_RATIO = 1.0  # Use 10% for testing
    
    # Min/max number of files to sample for range estimation
    MIN_SAMPLES = 5
    MAX_SAMPLES = 10000
    
    # Output video frame rate
    FPS = 30
    
    # Whether to create histograms during range estimation (useful for debugging)
    CREATE_HISTOGRAM = False
    
    # Force software encoding (set to False to try hardware encoding first)
    FORCE_SOFTWARE_ENCODING = True
    
    # Software encoding preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
    # Slower presets = better compression but slower encoding
    PRESET_SW = "slow"
    
    # Per-frame normalization using percentiles (recommended for better quality)
    USE_PER_FRAME_PERCENTILE = True  # Set to False to use global min/max
    LOW_PERCENTILE = 0.0   # Lower percentile for per-frame normalization (default: 1st)
    HIGH_PERCENTILE = 100.0  # Upper percentile for per-frame normalization (default: 99th)

    # Correction mode settings
    # USE_ATTENUATION: Compresses μ = -ln((I-dark)/(flat-dark)) - clips transmission, may lose info
    # USE_TRANSMISSION: Compresses T = (I-dark)/(flat-dark) - unclipped, preserves all information
    # Only one can be True at a time. If both False, raw projections are compressed.
    USE_ATTENUATION = False   # Whether to use attenuation correction (clips transmission)
    USE_TRANSMISSION = False   # Whether to use unclipped transmission (preserves all info)
    
    # CDF normalization settings (per-frame histogram equalization for more uniform distribution)
    USE_CDF_NORMALIZATION = False  # Set to True to apply per-frame CDF-based histogram equalization
    CDF_NUM_BINS = 2000  # Number of bins for CDF histogram computation
    
    # Keyframe spacing (GOP size) settings
    # Controls the interval between I-frames (keyframes):
    #   - KEY_FRAME_SPACING = 1: All-intra compression (JPEG-like, every frame is a keyframe)
    #   - KEY_FRAME_SPACING = 30: Keyframe every 30 frames (good for seeking)
    #   - KEY_FRAME_SPACING = None: Use encoder default (~250 frames, best compression)
    KEY_FRAME_SPACING = None  # Set to 1 for JPEG-like compression, or any integer for custom GOP
    
    # Note: To use lossless encoding, set a quality value > 100 (e.g., 101)
    # Quality values 0-100 use CRF-based lossy encoding
    
    # ========================================================================
    # RUN PIPELINE
    # ========================================================================
    
    print("=" * 80)
    print("TOMOGRAPHIC DATA COMPRESSION PIPELINE")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  Base path: {CT_FILES_BASE_PATH}")
    print(f"  Output path: {OUTPUT_PATH}")
    print(f"  Folder ID: {folder_id if folder_id else 'All folders'}")
    print(f"  Quality settings: {QUALITY_SETTINGS}")
    print(f"  Sample ratio: {SAMPLE_RATIO * 100:.0f}%")
    print(f"  FPS: {FPS}")
    print(f"  Encoding: {'Software' if FORCE_SOFTWARE_ENCODING else 'Hardware (fallback to software)'}")
    print(f"  Preset: {PRESET_SW}")
    print(f"  Overwrite existing: {overwrite}")
    if USE_PER_FRAME_PERCENTILE:
        print(f"  Normalization: Per-frame [{LOW_PERCENTILE}th, {HIGH_PERCENTILE}th] percentile")
    else:
        print(f"  Normalization: Global min/max")
    if USE_ATTENUATION:
        print(f"  Correction mode: Attenuation (μ = -ln(T), clipped)")
    elif USE_TRANSMISSION:
        print(f"  Correction mode: Transmission (T = (I-dark)/(flat-dark), unclipped)")
    else:
        print(f"  Correction mode: None (raw projections)")
    print(f"  CDF normalization: {'Per-frame (bins=' + str(CDF_NUM_BINS) + ')' if USE_CDF_NORMALIZATION else 'Disabled'}")
    if KEY_FRAME_SPACING is not None:
        if KEY_FRAME_SPACING == 1:
            print(f"  Keyframe spacing: {KEY_FRAME_SPACING} (all-intra / JPEG-like)")
        else:
            print(f"  Keyframe spacing: {KEY_FRAME_SPACING} frames")
    else:
        print(f"  Keyframe spacing: auto (encoder default ~250)")
    # Check if any quality setting triggers lossless mode
    lossless_qualities = [q for q in QUALITY_SETTINGS if q > 100]
    if lossless_qualities:
        print(f"  Lossless encoding: Yes (quality > 100 for values: {lossless_qualities})")
    else:
        print(f"  Lossless encoding: No (all quality values <= 100)")
    print("\n" + "=" * 80 + "\n")
    
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
        folder_id=folder_id,
        use_per_frame_percentile=USE_PER_FRAME_PERCENTILE,
        low_percentile=LOW_PERCENTILE,
        high_percentile=HIGH_PERCENTILE,
        use_attenuation=USE_ATTENUATION,
        use_transmission=USE_TRANSMISSION,
        use_cdf_normalization=USE_CDF_NORMALIZATION,
        cdf_num_bins=CDF_NUM_BINS,
        overwrite=overwrite,
        keyint=KEY_FRAME_SPACING
    )
    
    # Print results summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nProcessed {len(results_df)} tasks")
    
    if len(results_df) > 0:
        # Show sample of results
        print("\nSample results (first 5 rows):")
        print(results_df.head().to_string())
        
        # Show column names
        print(f"\nTotal columns: {len(results_df.columns)}")
        print("Columns include ranges for darks, flats, and projections")
        
        # Calculate totals
        if 'error' not in results_df.columns or results_df['error'].isna().all():
            total_original = sum([
                results_df[f'{t}_original_mb'].sum() 
                for t in ['darks', 'flats', 'projections']
            ])
            total_compressed = sum([
                results_df[f'{t}_compressed_mb'].sum() 
                for t in ['darks', 'flats', 'projections']
            ])
            
            if total_compressed > 0:
                overall_ratio = total_original / total_compressed
                print(f"\nOverall compression:")
                print(f"  Original: {total_original:.1f} MB")
                print(f"  Compressed: {total_compressed:.1f} MB")
                print(f"  Ratio: {overall_ratio:.1f}:1")
                print(f"  Space savings: {(1 - total_compressed/total_original)*100:.1f}%")
    
    print("\n" + "=" * 80)
    print("✅ Pipeline completed successfully!")
    print("=" * 80)
    
    return results_df


if __name__ == "__main__":
    main()
