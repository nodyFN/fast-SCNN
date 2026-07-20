#!/usr/bin/env python3
"""
Merge and split custom dataset (FG & NO_FG) into PyTorch project format.
Copies images and masks while ignoring JSON files and renaming to avoid collisions.
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".jpg", ".jpeg", ".png", ".PNG", ".JPG", ".JPEG"}

def collect_pairs(img_dir: Path, mask_dir: Path, prefix: str) -> List[Dict[str, Path]]:
    """Find image and mask pairs, filtering out json files."""
    pairs = []
    if not img_dir.exists() or not mask_dir.exists():
        print(f"Warning: Directory {img_dir} or {mask_dir} does not exist. Skipping.")
        return pairs

    # Build a lookup for masks by stem (file name without extension)
    mask_lookup = {p.stem: p for p in mask_dir.glob("*.png")}

    # Iterate images
    for img_path in img_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
            stem = img_path.stem
            if stem in mask_lookup:
                pairs.append({
                    "img_src": img_path,
                    "mask_src": mask_lookup[stem],
                    "prefix": prefix
                })
            else:
                print(f"Warning: Image '{img_path.name}' has no matching mask in {mask_dir.name}")
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Merge and split custom dataset into data/train, data/val, data/test")
    parser.add_argument("--src", type=str, required=True, help="Path to your custom 'my_dataset' folder")
    parser.add_argument("--dest", type=str, default="data", help="Output directory path (default: data)")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio for training set (default: 0.8)")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Ratio for validation set (default: 0.1)")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Ratio for test set (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting (default: 42)")
    args = parser.parse_args()

    src_root = Path(args.src)
    dest_root = Path(args.dest)

    # 1. Verification of inputs
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not (0.99 <= total_ratio <= 1.01):
        raise ValueError(f"Ratios must sum to 1.0. Currently sum to {total_ratio}")

    if not src_root.exists():
        raise FileNotFoundError(f"Source directory '{src_root}' not found.")

    # 2. Collect files from FG and NO_FG
    print("Collecting files...")
    fg_pairs = collect_pairs(src_root / "FG", src_root / "FG_masks", prefix="fg_")
    nofg_pairs = collect_pairs(src_root / "NO_FG", src_root / "NO_FG_masks", prefix="nofg_")
    
    all_pairs = fg_pairs + nofg_pairs
    num_samples = len(all_pairs)
    
    print(f"Found {len(fg_pairs)} FG samples and {len(nofg_pairs)} NO_FG samples.")
    print(f"Total valid image/mask pairs: {num_samples}")
    
    if num_samples == 0:
        print("No valid image/mask pairs found. Please verify directory names and file formats.")
        return

    # 3. Shuffle and split
    random.seed(args.seed)
    random.shuffle(all_pairs)

    train_end = int(num_samples * args.train_ratio)
    val_end = train_end + int(num_samples * args.val_ratio)

    splits = {
        "train": all_pairs[:train_end],
        "val": all_pairs[train_end:val_end],
        "test": all_pairs[val_end:]
    }

    # 4. Copy to destination
    for split_name, split_pairs in splits.items():
        print(f"Processing '{split_name}' split with {len(split_pairs)} samples...")
        img_dest_dir = dest_root / split_name / "images"
        mask_dest_dir = dest_root / split_name / "masks"
        
        # Create output directories (will retain structure / create automatically)
        img_dest_dir.mkdir(parents=True, exist_ok=True)
        mask_dest_dir.mkdir(parents=True, exist_ok=True)

        for pair in split_pairs:
            img_src = pair["img_src"]
            mask_src = pair["mask_src"]
            prefix = pair["prefix"]

            # Renamed target filename to avoid collision
            target_img_name = f"{prefix}{img_src.name}"
            target_mask_name = f"{prefix}{mask_src.name}"

            # Copy files
            shutil.copy2(img_src, img_dest_dir / target_img_name)
            shutil.copy2(mask_src, mask_dest_dir / target_mask_name)

    print("\nDataset split and fusion completed successfully!")
    print(f"Target location: {dest_root.resolve()}")
    print("Folder structures:")
    for split_name in splits:
         print(f" - {split_name}: {len(splits[split_name])} samples")

if __name__ == "__main__":
    main()
