#!/usr/bin/env python3
"""
Script to gather HDF5 files with a specific prefix pattern from nested directory structure.

Usage:
    python gather_hdf5_files.py --source /path/to/source --prefix first_diffusion --target /path/to/target
    python gather_hdf5_files.py --source /path/to/source --prefix reco --target /path/to/target --dry-run
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import List, Tuple


def find_matching_files(source_dir: Path, prefix: str, min_depth: int = 2) -> List[Tuple[Path, str]]:
    """
    Find all files matching the pattern {prefix}_XXX.hdf5 in the source directory.
    
    Args:
        source_dir: Root directory to search
        prefix: Prefix pattern to match (e.g., 'first_diffusion', 'reco')
        min_depth: Minimum depth level in directory structure (default: 2)
    
    Returns:
        List of tuples containing (file_path, relative_path_from_source)
    """
    matching_files = []
    pattern = f"{prefix}_*.hdf5"
    
    print(f"Searching for files matching pattern: {pattern}")
    print(f"Source directory: {source_dir}")
    
    # Walk through directory tree
    for root, dirs, files in os.walk(source_dir):
        current_path = Path(root)
        # Calculate depth relative to source_dir
        try:
            relative_depth = len(current_path.relative_to(source_dir).parts)
        except ValueError:
            continue
        
        # Only process files at minimum depth or deeper
        if relative_depth >= min_depth:
            for file in files:
                if file.startswith(f"{prefix}_") and file.endswith(".hdf5"):
                    file_path = current_path / file
                    relative_path = file_path.relative_to(source_dir)
                    matching_files.append((file_path, str(relative_path)))
    
    return matching_files


def copy_files(files: List[Tuple[Path, str]], target_dir: Path, preserve_structure: bool = False, dry_run: bool = False):
    """
    Copy files to target directory.
    
    Args:
        files: List of (file_path, relative_path) tuples
        target_dir: Target directory to copy files to
        preserve_structure: If True, preserve the directory structure; if False, copy all files to target root
        dry_run: If True, only print what would be done without actually copying
    """
    if not files:
        print("No files found to copy.")
        return
    
    print(f"\nFound {len(files)} matching files.")
    
    if dry_run:
        print("\n=== DRY RUN MODE - No files will be copied ===")
    
    # Create target directory if it doesn't exist
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
    
    copied_count = 0
    for file_path, relative_path in files:
        if preserve_structure:
            # Preserve directory structure
            target_path = target_dir / relative_path
            if not dry_run:
                target_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Copy all files to target root, flatten structure
            target_path = target_dir / file_path.name
        
        if dry_run:
            print(f"  Would copy: {file_path} -> {target_path}")
        else:
            print(f"  Copying: {file_path} -> {target_path}")
            shutil.copy2(file_path, target_path)
            copied_count += 1
    
    if not dry_run:
        print(f"\nSuccessfully copied {copied_count} files to {target_dir}")
    else:
        print(f"\nDry run complete. Would have copied {len(files)} files.")


def main():
    parser = argparse.ArgumentParser(
        description="Gather HDF5 files with a specific prefix from nested directories",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Copy all files matching 'first_diffusion_*.hdf5' to a target directory
  python gather_hdf5_files.py --source /myhome/lodopab_results --prefix first_diffusion --target /myhome/collected_files
  
  # Dry run to see what would be copied
  python gather_hdf5_files.py --source /myhome/lodopab_results --prefix reco --target /myhome/collected_files --dry-run
  
  # Preserve directory structure when copying
  python gather_hdf5_files.py --source /myhome/lodopab_results --prefix average_finetuned --target /myhome/collected_files --preserve-structure
  
  # Search with different minimum depth
  python gather_hdf5_files.py --source /myhome/lodopab_results --prefix reco --target /myhome/collected_files --min-depth 1
        """
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        required=True,
        help='Source directory to search for files'
    )
    
    parser.add_argument(
        '--prefix', '-p',
        type=str,
        required=True,
        help='Prefix pattern to match (e.g., "first_diffusion", "reco")'
    )
    
    parser.add_argument(
        '--target', '-t',
        type=str,
        required=True,
        help='Target directory to copy files to'
    )
    
    parser.add_argument(
        '--min-depth',
        type=int,
        default=2,
        help='Minimum depth level in directory structure (default: 2)'
    )
    
    parser.add_argument(
        '--preserve-structure',
        action='store_true',
        help='Preserve directory structure when copying (default: flatten to target root)'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually copying files'
    )
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    source_dir = Path(args.source)
    target_dir = Path(args.target)
    
    # Validate source directory
    if not source_dir.exists():
        print(f"Error: Source directory does not exist: {source_dir}")
        return 1
    
    if not source_dir.is_dir():
        print(f"Error: Source path is not a directory: {source_dir}")
        return 1
    
    # Find matching files
    print(f"{'='*60}")
    print(f"Gathering HDF5 files with prefix: {args.prefix}")
    print(f"{'='*60}")
    
    matching_files = find_matching_files(source_dir, args.prefix, args.min_depth)
    
    # Copy files
    copy_files(matching_files, target_dir, args.preserve_structure, args.dry_run)
    
    print(f"{'='*60}")
    print("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
