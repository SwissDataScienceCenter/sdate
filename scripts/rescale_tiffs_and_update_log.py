#!/usr/bin/env python3
"""
Script to rescale TIFF files by a given factor and update the log file.
Rescales both dimensions by the same factor using area interpolation.
"""

import os
import sys
import glob
import tifffile
import numpy as np
from pathlib import Path
from tqdm import tqdm
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# Configuration
INPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else "/myhome/data/sdate/shared/compression_paper/small/file_1_extracted"
SCALE_FACTOR = 0.5  # e.g. 0.5 = half resolution

def rescale_tiff(input_path, scale_factor=SCALE_FACTOR):
    """
    Rescale a TIFF file by scale_factor in both dimensions.
    Uses cv2.INTER_AREA for downscaling (best quality), INTER_LINEAR for upscaling.
    Falls back to numpy slicing when OpenCV is not available.
    """
    img = tifffile.imread(input_path)
    original_height, original_width = img.shape[:2]

    new_width = max(1, int(round(original_width * scale_factor)))
    new_height = max(1, int(round(original_height * scale_factor)))

    if _HAS_CV2:
        interp = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_LINEAR
        rescaled = cv2.resize(img, (new_width, new_height), interpolation=interp)
    else:
        # Fallback: use PIL if available, otherwise numpy slice
        try:
            from PIL import Image
            pil_img = Image.fromarray(img)
            resample = Image.LANCZOS if scale_factor < 1.0 else Image.BILINEAR
            pil_img = pil_img.resize((new_width, new_height), resample=resample)
            rescaled = np.array(pil_img)
        except ImportError:
            # Last resort: nearest-neighbour via numpy indexing
            row_idx = (np.arange(new_height) / scale_factor).astype(int).clip(0, original_height - 1)
            col_idx = (np.arange(new_width) / scale_factor).astype(int).clip(0, original_width - 1)
            rescaled = img[np.ix_(row_idx, col_idx)] if img.ndim == 2 else img[np.ix_(row_idx, col_idx), :]

    return rescaled, original_width, original_height, new_width, new_height


def update_log_file(log_path, original_width, original_height, new_width, new_height):
    """
    Update the log file to reflect the new image dimensions.
    Replaces resolution strings and ROI entries.
    """
    with open(log_path, 'r') as f:
        content = f.read()

    old_res = f"{original_width}x{original_height}"
    new_res = f"{new_width}x{new_height}"
    content = content.replace(old_res, new_res)

    # Update X-ROI and Y-ROI if present (format: "1 - <size>")
    import re
    content = re.sub(
        r'(X-ROI\s*:\s*1\s*-\s*)' + str(original_width),
        r'\g<1>' + str(new_width),
        content
    )
    content = re.sub(
        r'(Y-ROI\s*:\s*1\s*-\s*)' + str(original_height),
        r'\g<1>' + str(new_height),
        content
    )

    with open(log_path, 'w') as f:
        f.write(content)

    print(f"Updated log file: {log_path}")


def main():
    tiff_pattern = os.path.join(INPUT_DIR, "*.tif")
    tiff_files = sorted(glob.glob(tiff_pattern))

    if not tiff_files:
        print(f"No TIFF files found in {INPUT_DIR}")
        return

    print(f"Found {len(tiff_files)} TIFF files to process (scale factor: {SCALE_FACTOR})")

    orig_w = orig_h = new_w = new_h = None

    for tiff_path in tqdm(tiff_files, desc="Rescaling TIFFs"):
        rescaled, orig_w, orig_h, new_w, new_h = rescale_tiff(tiff_path, SCALE_FACTOR)
        tifffile.imwrite(tiff_path, rescaled)

    print(f"\nSuccessfully rescaled {len(tiff_files)} TIFF files: "
          f"{orig_w}x{orig_h} -> {new_w}x{new_h} (factor {SCALE_FACTOR})")

    log_files = glob.glob(os.path.join(INPUT_DIR, "*.log"))
    if log_files:
        update_log_file(log_files[0], orig_w, orig_h, new_w, new_h)
    else:
        print(f"Warning: No .log file found in {INPUT_DIR}")


if __name__ == "__main__":
    main()
