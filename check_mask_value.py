#!/usr/bin/env python3
"""
Simple utility script to check the pixel values and details of the first mask
in the 'duts_data/train/masks' directory.
"""

from pathlib import Path
import cv2
import numpy as np

def main():
    mask_dir = Path("duts_data/train/masks")
    if not mask_dir.exists():
        print(f"Error: Directory '{mask_dir.resolve()}' does not exist.")
        return

    # Find the first png file, ignoring .gitkeep
    mask_files = sorted(p for p in mask_dir.glob("*.png") if p.name != ".gitkeep")

    if not mask_files:
        print(f"No mask files (.png) found in '{mask_dir}'.")
        return

    first_mask_path = mask_files[0]
    print(f"=== Mask Inspection ===")
    print(f"File Path    : {first_mask_path.resolve()}")
    print(f"Filename     : {first_mask_path.name}")

    # Read mask as grayscale
    mask = cv2.imread(str(first_mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print("Error: Failed to read image using OpenCV.")
        return

    # Basic Info
    print(f"Dimensions   : {mask.shape} (Height, Width)")
    print(f"Data Type    : {mask.dtype}")

    # Unique pixel values & distribution
    unique_vals = np.unique(mask)
    print(f"Unique Values: {unique_vals.tolist()}")
    
    total_pixels = mask.size
    print("\nPixel Value Distribution:")
    for val in unique_vals:
        count = np.sum(mask == val)
        percentage = (count / total_pixels) * 100
        print(f" - Value {val:3d}: {count:8d} pixels ({percentage:.2f}%)")

    # Double check if it matches project dataset expectations
    print("\nDataset Compatibility Check:")
    if set(unique_vals) <= {0, 1}:
        print(" ✓ COMPATIBLE: Mask only contains 0 and 1.")
    elif set(unique_vals) <= {0, 255}:
        print(" ✓ COMPATIBLE: Mask only contains 0 and 255. (Dataset loader will map 255 -> 1 on-the-fly)")
    else:
        print(" ✗ WARNING: Mask contains unexpected values. Our dataset loader requires values to be only {0, 1} or {0, 255}.")

if __name__ == "__main__":
    main()
