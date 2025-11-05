#!/usr/bin/env python3
"""
Simple runner script for HEIC to TIFF translation training.

Usage:
    python run_heic_to_tiff_training.py --data_path /path/to/tiff/data --output_dir ./outputs
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import and run the main training function which supports command line arguments
from sdate.training.compression.train_heic_to_tiff import main

if __name__ == "__main__":
    main()