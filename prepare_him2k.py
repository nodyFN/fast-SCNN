#!/usr/bin/env python3
"""
HIM2K Dataset Preparation Script
Merges multi-instance person masks (alphas/a/*.png) into a single union mask (masks/a.png)
and splits the dataset into PyTorch project format (train/val/test splits).
"""

import argparse
import random
import shutil
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}

def combine_masks(mask_dir: Path) -> np.ndarray:
    """Find all png masks in mask_dir, read and merge them using pixel-wise maximum."""
    mask_files = sorted(list(mask_dir.glob("*.png")))
    if not mask_files:
        return None

    combined_mask = None
    for mf in mask_files:
        mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Failed to read mask file {mf}")
            continue
        if combined_mask is None:
            combined_mask = mask
        else:
            if combined_mask.shape != mask.shape:
                # Resize just in case of any shape anomaly
                mask = cv2.resize(mask, (combined_mask.shape[1], combined_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            combined_mask = np.maximum(combined_mask, mask)
    return combined_mask

def main():
    parser = argparse.ArgumentParser(description="Process and split HIM2K dataset into PyTorch project format")
    parser.add_argument("--src", type=str, required=True, help="Path to your HIM2K folder (containing image/ and alphas/)")
    parser.add_argument("--dest", type=str, default="data/him2k", help="Output directory path (default: data/him2k)")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio for training set (default: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Ratio for validation set (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Ratio for test set (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting (default: 42)")
    args = parser.parse_args()

    src_root = Path(args.src)
    dest_root = Path(args.dest)

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not (0.99 <= total_ratio <= 1.01):
        raise ValueError(f"Ratios must sum to 1.0. Currently sum to {total_ratio}")

    image_src_dir = src_root / "image"
    alphas_src_dir = src_root / "alphas"

    if not image_src_dir.exists():
        # Try checking for plural 'images' folder
        image_src_dir = src_root / "images"

    if not alphas_src_dir.exists():
        raise FileNotFoundError(f"Alphas directory not found under {src_root}. Expected 'alphas/'.")

    if not image_src_dir.exists():
        raise FileNotFoundError(f"Image directory not found under {src_root}. Expected 'image/' or 'images/'.")

    print(f"Reading images from: {image_src_dir}")
    print(f"Reading alphas from: {alphas_src_dir}")

    # Collect valid samples
    samples = []
    for img_path in image_src_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
            stem = img_path.stem
            mask_subdir = alphas_src_dir / stem
            if mask_subdir.exists() and mask_subdir.is_dir():
                samples.append({
                    "img_src": img_path,
                    "mask_subdir": mask_subdir,
                    "stem": stem
                })
            else:
                print(f"Warning: No corresponding alphas folder found for image '{img_path.name}' at {mask_subdir}")

    num_samples = len(samples)
    print(f"Found {num_samples} valid image-to-alphas pairings.")
    if num_samples == 0:
        print("No valid pairings found. Exiting.")
        return

    # Shuffle and split
    random.seed(args.seed)
    random.shuffle(samples)

    train_end = int(num_samples * args.train_ratio)
    val_end = train_end + int(num_samples * args.val_ratio)

    splits = {
        "train": samples[:train_end],
        "val": samples[train_end:val_end],
        "test": samples[val_end:]
    }

    # Clean up destination directory if it exists
    if dest_root.exists():
        print(f"Overwriting existing destination directory: {dest_root}")
        for p in list(dest_root.rglob("*")):
            if p.is_file():
                try:
                    p.unlink()
                except Exception:
                    pass

    # Process and write splits
    for split_name, split_samples in splits.items():
        if not split_samples:
            continue
        print(f"\nProcessing '{split_name}' split ({len(split_samples)} samples)...")
        img_dest_dir = dest_root / split_name / "images"
        mask_dest_dir = dest_root / split_name / "masks"

        img_dest_dir.mkdir(parents=True, exist_ok=True)
        mask_dest_dir.mkdir(parents=True, exist_ok=True)

        for sample in tqdm(split_samples):
            img_src = sample["img_src"]
            mask_subdir = sample["mask_subdir"]
            stem = sample["stem"]

            # 1. Combine masks
            combined = combine_masks(mask_subdir)
            if combined is None:
                print(f"Warning: No valid png masks found inside {mask_subdir}. Skipping sample.")
                continue

            # 2. Save combined mask
            target_mask_path = mask_dest_dir / f"{stem}.png"
            cv2.imwrite(str(target_mask_path), combined)

            # 3. Copy image
            target_img_path = img_dest_dir / img_src.name
            shutil.copy2(img_src, target_img_path)

    print("\nHIM2K dataset preparation and split completed successfully!")
    print(f"Target location: {dest_root.resolve()}")
    for split_name in splits:
        print(f" - {split_name}: {len(splits[split_name])} samples")

if __name__ == "__main__":
    main()
