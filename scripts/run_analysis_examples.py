#!/usr/bin/env python3
"""
Example usage of the TIFF compression analysis script

This shows how to run the analysis with different parameter configurations.
"""

import subprocess
import sys
from pathlib import Path

# Configuration examples
examples = [
    {
        "name": "High Quality Analysis", 
        "args": [
            "--data_path", "../data/ct_files/3a02cbef-042d-4d7e-8f9f-ad964a3bfde3_0_2020-05-26-13-33-52.tar_7d7ebd54-05b8-41ad-a221-d6702f9a0ad9",
            "--cq_hw", "100",  # Very high quality
            "--skip_frames", "5",  # More detailed analysis
            "--max_frames", "200",
            "--experiment_name", "high_quality_test"
        ]
    },
    {
        "name": "Medium Quality Analysis",
        "args": [
            "--data_path", "../data/ct_files/3a02cbef-042d-4d7e-8f9f-ad964a3bfde3_0_2020-05-26-13-33-52.tar_7d7ebd54-05b8-41ad-a221-d6702f9a0ad9",
            "--cq_hw", "23",  # Medium quality
            "--skip_frames", "10",
            "--max_frames", "400", 
            "--experiment_name", "medium_quality_test"
        ]
    },
    {
        "name": "Low Quality Analysis",
        "args": [
            "--data_path", "../data/ct_files/3a02cbef-042d-4d7e-8f9f-ad964a3bfde3_0_2020-05-26-13-33-52.tar_7d7ebd54-05b8-41ad-a221-d6702f9a0ad9",
            "--cq_hw", "10",  # Lower quality for higher compression
            "--skip_frames", "15",
            "--max_frames", "400",
            "--experiment_name", "low_quality_test"
        ]
    }
]

def run_analysis(example):
    """Run a single analysis configuration"""
    print(f"\n🚀 Running: {example['name']}")
    print("=" * 50)
    
    cmd = ["python", "tiff_compression_analysis.py"] + example["args"]
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✅ {example['name']} completed successfully!")
        print("Output:", result.stdout[-500:])  # Last 500 chars
    except subprocess.CalledProcessError as e:
        print(f"❌ {example['name']} failed!")
        print("Error:", e.stderr)
        return False
    return True

def main():
    print("TIFF Compression Analysis - Example Runner")
    print("=" * 60)
    
    # Check if script exists
    script_path = Path("tiff_compression_analysis.py")
    if not script_path.exists():
        print("❌ tiff_compression_analysis.py not found!")
        print("Make sure you're running this from the scripts directory.")
        sys.exit(1)
    
    print("Available analysis configurations:")
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example['name']}")
    
    print("\nOptions:")
    print("  1. Run a specific configuration (enter number)")
    print("  2. Run all configurations sequentially (enter 'all')")
    print("  3. Exit (enter 'q')")
    
    choice = input("\nEnter your choice: ").strip().lower()
    
    if choice == 'q':
        print("Exiting...")
        return
    elif choice == 'all':
        print("\n🚀 Running all configurations...")
        for example in examples:
            success = run_analysis(example)
            if not success:
                print("❌ Stopping due to failure")
                break
        print("\n🎉 All analyses completed!")
    else:
        try:
            index = int(choice) - 1
            if 0 <= index < len(examples):
                run_analysis(examples[index])
            else:
                print("❌ Invalid choice!")
        except ValueError:
            print("❌ Invalid input!")

if __name__ == "__main__":
    main()
