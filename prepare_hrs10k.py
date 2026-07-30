#!/usr/bin/env python3
"""
Process and split the HRS10K dataset into the project format.
Supports optional offline resizing to standard resolution (e.g. 1024) to save VRAM and disk space.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List

# Try to import PIL for offline resizing
try:
    from PIL import Image
    PIL_AVAILABLE = True
    try:
        # Pillow >= 9.1.0
        RESAMPLE_BILINEAR = Image.Resampling.BILINEAR
        RESAMPLE_NEAREST = Image.Resampling.NEAREST
    except AttributeError:
        # Older Pillow versions
        RESAMPLE_BILINEAR = Image.BILINEAR
        RESAMPLE_NEAREST = Image.NEAREST
except ImportError:
    PIL_AVAILABLE = False


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"}


def find_directories(src_root: Path) -> tuple[Path, Path]:
    """Search for the image and mask directories in the source root."""
    img_candidates = ["image", "images", "imgs", "IMGS", "img", "im"]
    mask_candidates = ["mask", "masks", "gts", "GTS", "gt"]

    img_dir = None
    mask_dir = None

    for candidate in img_candidates:
        p = src_root / candidate
        if p.exists() and p.is_dir():
            img_dir = p
            break
    
    for candidate in mask_candidates:
        p = src_root / candidate
        if p.exists() and p.is_dir():
            mask_dir = p
            break

    # If not found directly, check subdirectories (1 level deep)
    if img_dir is None or mask_dir is None:
        for p in src_root.iterdir():
            if p.is_dir() and p.name != "." and p.name != "..":
                for img_cand in img_candidates:
                    ip = p / img_cand
                    if ip.exists() and ip.is_dir():
                        img_dir = ip
                for mask_cand in mask_candidates:
                    mp = p / mask_cand
                    if mp.exists() and mp.is_dir():
                        mask_dir = mp

    if img_dir is None:
        raise FileNotFoundError(f"Could not find images directory under '{src_root}'. Checked: {img_candidates}")
    if mask_dir is None:
        raise FileNotFoundError(f"Could not find masks directory under '{src_root}'. Checked: {mask_candidates}")

    return img_dir, mask_dir


def collect_pairs(img_dir: Path, mask_dir: Path) -> List[Dict[str, Path]]:
    """Find image and mask pairs by matching filenames stem."""
    pairs = []
    
    # Gather all mask files (.png is standard for masks)
    mask_lookup = {p.stem: p for p in mask_dir.glob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS}
    
    # Iterate images
    for img_path in img_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in IMAGE_EXTENSIONS:
            stem = img_path.stem
            if stem in mask_lookup:
                pairs.append({
                    "img_src": img_path,
                    "mask_src": mask_lookup[stem]
                })
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Prepare and split HRS10K dataset")
    parser.add_argument("--src", type=str, required=True, help="Path to downloaded HRS10K directory")
    parser.add_argument("--dest", type=str, default="hrs10k_data", help="Output directory path (default: hrs10k_data)")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio for training set")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Ratio for validation set")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Ratio for test set")
    parser.add_argument("--resize", action="store_true", help="Enable offline resizing of images and masks")
    parser.add_argument("--resize-longest-max-size", type=int, default=1024, help="Target size for longest side")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    args = parser.parse_args()

    src_root = Path(args.src)
    dest_root = Path(args.dest)

    # 1. Validation
    if args.resize and not PIL_AVAILABLE:
        print("Error: PIL (Pillow) is required for --resize. Please run 'pipenv install Pillow' or disable resizing.")
        sys.exit(1)

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not (0.99 <= total_ratio <= 1.01):
        raise ValueError(f"Ratios must sum to 1.0. Currently sum to {total_ratio}")

    if not src_root.exists():
        raise FileNotFoundError(f"Source directory '{src_root}' not found.")

    # 2. Find and match images/masks
    print("Locating directories...")
    tr_dir = src_root / "TR-HRS10K"
    te_dir = src_root / "TE-HRS10K"

    # Check lowercase candidates too
    if not tr_dir.exists():
        tr_dir = src_root / "tr-hrs10k"
    if not te_dir.exists():
        te_dir = src_root / "te-hrs10k"

    if tr_dir.exists() and te_dir.exists():
        print(f"Detected official split directories: {tr_dir.name} and {te_dir.name}")
        
        # Training/validation source
        try:
            tr_im_dir, tr_gt_dir = find_directories(tr_dir)
            tr_pairs = collect_pairs(tr_im_dir, tr_gt_dir)
        except FileNotFoundError as e:
            print(f"Error under {tr_dir.name}: {e}")
            sys.exit(1)

        # Test source
        try:
            te_im_dir, te_gt_dir = find_directories(te_dir)
            te_pairs = collect_pairs(te_im_dir, te_gt_dir)
        except FileNotFoundError as e:
            print(f"Error under {te_dir.name}: {e}")
            sys.exit(1)

        print(f"Collected {len(tr_pairs)} training pairs and {len(te_pairs)} test pairs.")
        if len(tr_pairs) == 0:
            print("Error: No training pairs found. Verify files.")
            sys.exit(1)

        # Shuffle training pairs and split into train and val
        random.seed(args.seed)
        random.shuffle(tr_pairs)

        total_train_ratio = args.train_ratio + args.val_ratio
        if total_train_ratio <= 0:
            raise ValueError("train-ratio and val-ratio must sum to a positive number")
        norm_train_ratio = args.train_ratio / total_train_ratio
        train_end = int(len(tr_pairs) * norm_train_ratio)

        splits = {
            "train": tr_pairs[:train_end],
            "val": tr_pairs[train_end:],
            "test": te_pairs
        }
    else:
        # Fallback to general single directory behavior
        try:
            img_dir, mask_dir = find_directories(src_root)
        except FileNotFoundError as e:
            print(f"Error: {e}")
            sys.exit(1)

        print(f"Found Image directory: {img_dir}")
        print(f"Found Mask directory: {mask_dir}")

        print("Collecting image-mask pairs...")
        all_pairs = collect_pairs(img_dir, mask_dir)
        num_samples = len(all_pairs)

        print(f"Total valid image-mask pairs found: {num_samples}")
        if num_samples == 0:
            print("No matched image-mask pairs found. Please verify file names.")
            sys.exit(1)

        # Shuffle and split
        random.seed(args.seed)
        random.shuffle(all_pairs)

        train_end = int(num_samples * args.train_ratio)
        val_end = train_end + int(num_samples * args.val_ratio)

        splits = {
            "train": all_pairs[:train_end],
            "val": all_pairs[train_end:val_end],
            "test": all_pairs[val_end:]
        }

    # Clean up destination directory if it exists
    if dest_root.exists():
        print(f"Destination '{dest_root}' already exists. Overwriting...")

    # 4. Copy / Resize and save
    for split_name, split_pairs in splits.items():
        print(f"\nProcessing '{split_name}' split ({len(split_pairs)} samples)...")
        img_dest_dir = dest_root / split_name / "images"
        mask_dest_dir = dest_root / split_name / "masks"

        img_dest_dir.mkdir(parents=True, exist_ok=True)
        mask_dest_dir.mkdir(parents=True, exist_ok=True)

        for i, pair in enumerate(split_pairs):
            img_src = pair["img_src"]
            mask_src = pair["mask_src"]

            # Maintain file extensions
            target_img_path = img_dest_dir / img_src.name
            target_mask_path = mask_dest_dir / mask_src.name

            if args.resize:
                try:
                    # Open
                    img = Image.open(img_src)
                    mask = Image.open(mask_src)

                    # Compute new dimensions
                    w, h = img.size
                    max_dim = max(w, h)
                    if max_dim > args.resize_longest_max_size:
                        scale = args.resize_longest_max_size / max_dim
                        new_w = int(w * scale)
                        new_h = int(h * scale)

                        # Resize
                        img_resized = img.resize((new_w, new_h), RESAMPLE_BILINEAR)
                        mask_resized = mask.resize((new_w, new_h), RESAMPLE_NEAREST)

                        # Save
                        img_resized.save(target_img_path)
                        mask_resized.save(target_mask_path)
                    else:
                        # No need to downscale if it's already smaller than target
                        shutil.copy2(img_src, target_img_path)
                        shutil.copy2(mask_src, target_mask_path)
                except Exception as e:
                    print(f"Warning: Failed to resize {img_src.name}: {e}. Copying original instead.")
                    shutil.copy2(img_src, target_img_path)
                    shutil.copy2(mask_src, target_mask_path)
            else:
                # Direct copying
                shutil.copy2(img_src, target_img_path)
                shutil.copy2(mask_src, target_mask_path)

            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(split_pairs)} samples...")

    print(f"\nHRS10K dataset preparation completed successfully!")
    print(f"Destination: {dest_root.resolve()}")
    for split_name in splits:
        print(f"  - {split_name}: {len(splits[split_name])} samples")


if __name__ == "__main__":
    main()
