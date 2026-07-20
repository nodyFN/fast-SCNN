#!/usr/bin/env python3
"""
Process and split the DUTS dataset into data/train, data/val, and data/test.
DUTS-TR (10,553 samples) -> 95% train, 5% val
DUTS-TE (5,019 samples) -> 100% test
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List

def collect_duts_pairs(img_dir: Path, mask_dir: Path) -> List[Dict[str, Path]]:
    """Match images in img_dir with masks in mask_dir by file stem."""
    pairs = []
    if not img_dir.exists() or not mask_dir.exists():
        print(f"Error: Subdirectory {img_dir} or {mask_dir} does not exist.")
        return pairs

    # Build mask map (stem -> Path)
    mask_map = {p.stem: p for p in mask_dir.glob("*.png")}

    # Scan for images (.jpg or .png)
    for img_path in img_dir.iterdir():
        if img_path.is_file() and img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
            stem = img_path.stem
            if stem in mask_map:
                pairs.append({
                    "img_src": img_path,
                    "mask_src": mask_map[stem]
                })
            else:
                print(f"Warning: Image '{img_path.name}' has no matching mask in {mask_dir.name}")
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Split and copy DUTS dataset into project data directory")
    parser.add_argument("--src", type=str, required=True, help="Path to the 'DUTS' folder")
    parser.add_argument("--dest", type=str, default="data", help="Output directory path (default: data)")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Ratio of TR dataset to split for val (default: 0.05)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    src_root = Path(args.src)
    dest_root = Path(args.dest)

    if not src_root.exists():
        raise FileNotFoundError(f"Source DUTS directory '{src_root}' not found.")

    # 1. Directories setup
    tr_img_dir = src_root / "DUTS-TR" / "DUTS-TR-Image"
    tr_msk_dir = src_root / "DUTS-TR" / "DUTS-TR-Mask"
    te_img_dir = src_root / "DUTS-TE" / "DUTS-TE-Image"
    te_msk_dir = src_root / "DUTS-TE" / "DUTS-TE-Mask"

    # Clean up destination files (keep structures / .gitkeep)
    if dest_root.exists():
        print(f"Destination directory '{dest_root}' already exists. Overwriting (cleaning old files)...")
        for p in list(dest_root.rglob("*")):
            if p.is_file() and p.name != ".gitkeep":
                try:
                    p.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete {p}: {e}")

    # 2. Collect Training/Val pairs
    print("Scanning DUTS-TR...")
    tr_pairs = collect_duts_pairs(tr_img_dir, tr_msk_dir)
    print(f"Found {len(tr_pairs)} valid training pairs.")

    # 3. Collect Test pairs
    print("Scanning DUTS-TE...")
    te_pairs = collect_duts_pairs(te_img_dir, te_msk_dir)
    print(f"Found {len(te_pairs)} valid testing pairs.")

    if len(tr_pairs) == 0 or len(te_pairs) == 0:
        print("Error: Could not find valid image/mask pairs. Check your directory structure.")
        return

    # 4. Shuffle and split DUTS-TR into Train (95%) and Val (5%)
    random.seed(args.seed)
    random.shuffle(tr_pairs)

    val_count = int(len(tr_pairs) * args.val_ratio)
    val_pairs = tr_pairs[:val_count]
    train_pairs = tr_pairs[val_count:]

    # Dictionary of target splits
    splits = {
        "train": train_pairs,
        "val": val_pairs,
        "test": te_pairs
    }

    # 5. Copy to destination
    for split_name, pairs in splits.items():
        print(f"Copying {len(pairs)} files to {split_name} split...")
        img_dest_dir = dest_root / split_name / "images"
        mask_dest_dir = dest_root / split_name / "masks"

        img_dest_dir.mkdir(parents=True, exist_ok=True)
        mask_dest_dir.mkdir(parents=True, exist_ok=True)

        for pair in pairs:
            img_src = pair["img_src"]
            mask_src = pair["mask_src"]

            # Keep original filename (e.g. ILSVRC2012_test_00000003.jpg/.png)
            shutil.copy2(img_src, img_dest_dir / img_src.name)
            shutil.copy2(mask_src, mask_dest_dir / mask_src.name)

    print("\nDUTS dataset processed successfully!")
    print(f"Destination: {dest_root.resolve()}")
    print(f" - train: {len(train_pairs)} samples")
    print(f" - val:   {len(val_pairs)} samples")
    print(f" - test:  {len(te_pairs)} samples")

if __name__ == "__main__":
    main()
