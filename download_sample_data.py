#!/usr/bin/env python3
"""
Download and structure the Penn-Fudan Pedestrian Dataset for Fast-SCNN.
Transforms multi-instance pedestrian masks into binary masks (foreground/background)
and splits them into data/train, data/val, and data/test.
"""

import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
import cv2
import numpy as np

# Configuration
DATASET_URL = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TEMP_DIR = PROJECT_ROOT / "temp_download"
ZIP_PATH = TEMP_DIR / "PennFudanPed.zip"

def download_progress(block_num, block_size, total_size):
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = min(100, read_so_far * 100 / total_size)
        print(f"\rDownloading dataset: {percent:.1f}% ({read_so_far / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB)", end="")
    else:
        print(f"\rDownloaded {read_so_far / 1024 / 1024:.1f}MB", end="")

def main():
    # 1. Setup directories
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    if DATA_DIR.exists():
        has_real_files = any(p.is_file() and p.name != ".gitkeep" for p in DATA_DIR.rglob("*"))
        if has_real_files:
            print(f"Data directory '{DATA_DIR}' already contains files. Please delete it first if you want to re-download.")
            return

    # 2. Download ZIP
    print(f"Fetching from: {DATASET_URL}")
    try:
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH, download_progress)
        print("\nDownload complete!")
    except Exception as e:
        print(f"\nFailed to download: {e}")
        print("Please check your internet connection and try again.")
        return

    # 3. Extract ZIP
    print("Extracting files...")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(TEMP_DIR)
    
    src_dir = TEMP_DIR / "PennFudanPed"
    
    # 4. Discover files
    image_paths = sorted(list((src_dir / "PNGImages").glob("*.png")))
    mask_paths = sorted(list((src_dir / "PedMasks").glob("*.png")))
    
    num_samples = len(image_paths)
    print(f"Found {num_samples} samples.")

    # Shuffle and split (130 train, 20 val, 20 test)
    np.random.seed(42)
    indices = np.random.permutation(num_samples)
    
    splits = {
        "train": indices[:130],
        "val": indices[130:150],
        "test": indices[150:]
    }

    # 5. Process and structure
    for split_name, idxs in splits.items():
        print(f"Structuring {split_name} split ({len(idxs)} samples)...")
        img_out = DATA_DIR / split_name / "images"
        msk_out = DATA_DIR / split_name / "masks"
        img_out.mkdir(parents=True, exist_ok=True)
        msk_out.mkdir(parents=True, exist_ok=True)

        for i, idx in enumerate(idxs):
            img_path = image_paths[idx]
            msk_path = mask_paths[idx]

            # Output filename (consistent stem)
            new_name = f"ped_{i:04d}"

            # Copy image (PennFudan is already PNG)
            shutil.copy(img_path, img_out / f"{new_name}.png")

            # Read instance mask and convert to binary mask (0/255)
            # Original mask contains values 0 for background, and 1, 2, ... for instance IDs
            inst_mask = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
            binary_mask = np.where(inst_mask > 0, 255, 0).astype(np.uint8)

            # Save binary mask
            cv2.imwrite(str(msk_out / f"{new_name}.png"), binary_mask)

    # 6. Cleanup
    print("Cleaning up temporary download files...")
    shutil.rmtree(TEMP_DIR)
    print("\nDataset ready!")
    print(f"Location: {DATA_DIR}")
    print("\nYou can now run training using:")
    print("python train.py --profile project --epochs 50 --batch-size 4")

if __name__ == "__main__":
    main()
