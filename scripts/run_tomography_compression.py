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
    args = parser.parse_args()
    
    folder_id = args.folder_id
    
    # ========================================================================
    # CONFIGURATION - Modify these parameters as needed
    # ========================================================================
    
    # Base path containing folders with TIFF sequences
    CT_FILES_BASE_PATH = Path('/das/home/barbaf_l/p22274/compression_paper')
    
    # Output directory for compressed files and reports
    OUTPUT_PATH = Path('/das/home/barbaf_l/p22274/compression_paper/streaming_output')
    
    # Quality settings to test (0-100, higher = better quality)
    QUALITY_SETTINGS = [89,88, 87, 86, 85]
    
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
    PRESET_SW = "veryslow"
    
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
        folder_id=folder_id
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
