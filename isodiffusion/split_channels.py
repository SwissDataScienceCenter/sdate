import numpy as np
import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Split a 4D npy file into multiple 3D files along the last dimension (channels).")
    parser.add_argument("input_path", type=str, help="Path to the input .npy file")
    args = parser.parse_args()

    input_path = args.input_path
    if not os.path.exists(input_path):
        print(f"Error: File '{input_path}' does not exist.")
        sys.exit(1)

    # Extract directory and base name
    dir_name = os.path.dirname(os.path.abspath(input_path))
    base_name = os.path.basename(input_path)
    name_no_ext, ext = os.path.splitext(base_name)

    if ext != '.npy':
        print(f"Warning: Input file does not have a .npy extension ({ext}).")

    print(f"Loading {input_path}...")
    try:
        data = np.load(input_path)
    except Exception as e:
        print(f"Error loading {input_path}: {e}")
        sys.exit(1)

    print(f"Loaded data with shape: {data.shape}")
    if len(data.shape) != 4:
        print(f"Error: Expected 4D tensor (D, H, W, num_channels), but got shape {data.shape}")
        sys.exit(1)

    num_channels = data.shape[-1]
    
    # Create subfolder
    subfolder_path = os.path.join(dir_name, f"{name_no_ext}_channels")
    os.makedirs(subfolder_path, exist_ok=True)
    print(f"Output directory: {subfolder_path}")

    # Save channels
    for i in range(num_channels):
        channel_data = data[..., i]
        output_name = f"{name_no_ext}_channel_{i}.npy"
        output_path = os.path.join(subfolder_path, output_name)
        np.save(output_path, channel_data)
        print(f"Saved channel {i} to {output_name} with shape {channel_data.shape}")

    print("Done!")

if __name__ == "__main__":
    main()
