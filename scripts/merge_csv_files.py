#!/usr/bin/env python3
"""
Merge CSV Files Script
======================
Merges all CSV files in a given directory that have the same structure.

Usage:
    python merge_csv_files.py <data_path> [options]

Arguments:
    data_path           Directory containing CSV files to merge

Options:
    --output, -o        Output path for merged CSV (default: merged_output.csv)
    --pattern, -p       Glob pattern for CSV files (default: *.csv)
    --recursive, -r     Search recursively in subdirectories
    --no-header         Assume CSV files have no header row
    --sort-by           Column name to sort final result by
    --verbose, -v       Print detailed progress information

Examples:
    # Merge all CSV files in a directory
    python merge_csv_files.py /path/to/csvs

    # Merge with custom output name
    python merge_csv_files.py /path/to/csvs -o combined_results.csv

    # Merge CSV files recursively with pattern
    python merge_csv_files.py /path/to/csvs -p "*_results.csv" -r

    # Merge and sort by a column
    python merge_csv_files.py /path/to/csvs --sort-by timestamp
"""

import argparse
import glob
import os
import sys
from pathlib import Path
import pandas as pd


def find_csv_files(data_path, pattern="*.csv", recursive=False):
    """
    Find all CSV files matching the pattern in the given directory.
    
    Args:
        data_path: Directory to search
        pattern: Glob pattern for CSV files
        recursive: Whether to search recursively
        
    Returns:
        List of CSV file paths
    """
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Directory not found: {data_path}")
    
    if not data_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {data_path}")
    
    if recursive:
        csv_files = list(data_path.rglob(pattern))
    else:
        csv_files = list(data_path.glob(pattern))
    
    # Convert to absolute paths and filter out directories
    csv_files = [f.resolve() for f in csv_files if f.is_file()]
    
    return sorted(csv_files)


def merge_csv_files(csv_files, output_path, has_header=True, sort_by=None, verbose=False):
    """
    Merge multiple CSV files into a single file.
    
    Args:
        csv_files: List of CSV file paths to merge
        output_path: Output path for merged CSV
        has_header: Whether CSV files have header rows
        sort_by: Column name to sort by (optional)
        verbose: Print progress information
        
    Returns:
        Number of rows in merged file
    """
    if not csv_files:
        raise ValueError("No CSV files found to merge")
    
    if verbose:
        print(f"Found {len(csv_files)} CSV files to merge")
    
    # Read all CSV files
    dataframes = []
    for i, csv_file in enumerate(csv_files, 1):
        try:
            if verbose:
                print(f"Reading ({i}/{len(csv_files)}): {csv_file.name}")
            
            if has_header:
                df = pd.read_csv(csv_file)
            else:
                df = pd.read_csv(csv_file, header=None)
            
            # Add source file column for traceability
            df['_source_file'] = str(csv_file)
            
            dataframes.append(df)
            
            if verbose:
                print(f"  Loaded {len(df)} rows")
                
        except Exception as e:
            print(f"ERROR reading {csv_file}: {e}", file=sys.stderr)
            continue
    
    if not dataframes:
        raise ValueError("No CSV files could be read successfully")
    
    # Merge all dataframes
    if verbose:
        print("\nMerging dataframes...")
    
    merged_df = pd.concat(dataframes, ignore_index=True)
    
    if verbose:
        print(f"Total rows after merge: {len(merged_df)}")
    
    # Sort if requested
    if sort_by:
        if sort_by in merged_df.columns:
            if verbose:
                print(f"Sorting by column: {sort_by}")
            merged_df = merged_df.sort_values(by=sort_by).reset_index(drop=True)
        else:
            print(f"WARNING: Column '{sort_by}' not found. Available columns: {', '.join(merged_df.columns)}", 
                  file=sys.stderr)
    
    # Save merged dataframe
    if verbose:
        print(f"\nSaving merged CSV to: {output_path}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    merged_df.to_csv(output_path, index=False)
    
    if verbose:
        print(f"Successfully saved {len(merged_df)} rows")
        print(f"Columns: {', '.join(merged_df.columns)}")
    
    return len(merged_df)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Merge CSV files with the same structure from a directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "data_path",
        help="Directory containing CSV files to merge"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="merged_output.csv",
        help="Output path for merged CSV (default: merged_output.csv)"
    )
    
    parser.add_argument(
        "-p", "--pattern",
        default="*.csv",
        help="Glob pattern for CSV files (default: *.csv)"
    )
    
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Search recursively in subdirectories"
    )
    
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Assume CSV files have no header row"
    )
    
    parser.add_argument(
        "--sort-by",
        help="Column name to sort final result by"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed progress information"
    )
    
    args = parser.parse_args()
    
    try:
        # Find CSV files
        csv_files = find_csv_files(
            args.data_path,
            pattern=args.pattern,
            recursive=args.recursive
        )
        
        if not csv_files:
            print(f"ERROR: No CSV files matching pattern '{args.pattern}' found in {args.data_path}",
                  file=sys.stderr)
            return 1
        
        # Merge CSV files
        num_rows = merge_csv_files(
            csv_files,
            args.output,
            has_header=not args.no_header,
            sort_by=args.sort_by,
            verbose=args.verbose
        )
        
        if not args.verbose:
            print(f"Successfully merged {len(csv_files)} CSV files into {args.output} ({num_rows} total rows)")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
