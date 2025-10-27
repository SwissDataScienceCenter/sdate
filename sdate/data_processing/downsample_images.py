import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import argparse

# Disable decompression bomb warnings
Image.MAX_IMAGE_PIXELS = None

def process_image(src_path, dst_path, target_size):
    """
    Opens an image from src_path, resizes it to target_size,
    converts it to grayscale (if not already), and saves to dst_path.
    """
    try:
        with Image.open(src_path) as img:
            # Handle 16-bit grayscale images by converting them to 8-bit grayscale
            if img.mode in ('I;16', 'I'):
                img = img.point(lambda i: i * (1.0 / 256)).convert('L')  # Convert 16-bit to 8-bit grayscale
            else:
                img = img.convert('L')  # Ensure output is in grayscale

            # Resize using high-quality resampling
            img_resized = img.resize(target_size, Image.LANCZOS)

            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

            # Save the downsampled grayscale image
            img_resized.save(dst_path)
            print(f"Processed: {src_path} -> {dst_path}")
    except Exception as e:
        print(f"Error processing {src_path}: {e}")

def gather_image_tasks(input_dir, output_dir, target_size, valid_exts=('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.tif')):
    tasks = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(valid_exts):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, input_dir)
                dst_path = os.path.join(output_dir, rel_path)
                tasks.append((src_path, dst_path, target_size))
    return tasks

def main(input_dir, output_dir, target_size, workers):
    tasks = gather_image_tasks(input_dir, output_dir, target_size)
    print(f"Found {len(tasks)} images to process.")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_image, src, dst, size) for src, dst, size in tasks]
        for future in futures:
            future.result()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Downsample images to a target size while preserving folder structure.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing images.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory for downsampled images.")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes (default: all CPU cores).")
    parser.add_argument("--target-size", type=int, nargs=2, metavar=('WIDTH', 'HEIGHT'), default=(256, 256),
                        help="Target size for downsampling (default: 256 256). Example: --target-size 512 512")

    args = parser.parse_args()

    main(args.input_dir, args.output_dir, tuple(args.target_size), args.workers)