#!/usr/bin/env python3
"""
Utility script to generate Trimaps (0, 128, 255) from binary masks.

Trimap generation logic:
1. Erode binary mask -> definitely foreground (255)
2. Dilate binary mask -> definitely background (0)
3. Region in between -> unknown edge transition zone (128)
"""

import argparse
import logging
from pathlib import Path
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_trimap(
    mask_path: Path,
    output_path: Path,
    erode_radius: int,
    dilate_radius: int,
    label_mode: bool = False,
) -> None:
    """Generate trimap from a single binary mask file."""
    # Read mask as grayscale
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        logger.error(f"Failed to read mask: {mask_path}")
        return

    # Check value range and binarize
    if mask.max() == 1:
        mask = mask * 255
    _, binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 1. Erosion (Foreground core)
    if erode_radius > 0:
        kernel_size = erode_radius * 2 + 1
        kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        eroded = cv2.erode(binary, kernel_erode, iterations=1)
    else:
        eroded = binary.copy()

    # 2. Dilation (Background boundary)
    if dilate_radius > 0:
        kernel_size = dilate_radius * 2 + 1
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(binary, kernel_dilate, iterations=1)
    else:
        dilated = binary.copy()

    # 3. Create Trimap
    # Initialize all as unknown (128)
    trimap = np.full(binary.shape, 128, dtype=np.uint8)
    
    # Definitely background (0) where dilated mask is 0
    trimap[dilated == 0] = 0
    
    # Definitely foreground (255) where eroded mask is 255
    trimap[eroded == 255] = 255

    # Optional Matting label mode: 0 (BG), 1 (Unknown), 2 (FG) instead of 0, 128, 255
    if label_mode:
        label_trimap = np.zeros(trimap.shape, dtype=np.uint8)
        label_trimap[trimap == 128] = 1
        label_trimap[trimap == 255] = 2
        trimap = label_trimap

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), trimap)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert binary masks to trimaps via erosion and dilation.")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="Path to input mask file or directory of masks")
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="Path to output directory")
    parser.add_argument("-r", "--radius", type=int, default=10,
                        help="Radius for both erosion and dilation in pixels (default: 10)")
    parser.add_argument("--erode-radius", type=int, default=None,
                        help="Radius for erosion (overrides --radius)")
    parser.add_argument("--dilate-radius", type=int, default=None,
                        help="Radius for dilation (overrides --radius)")
    parser.add_argument("--labels", action="store_true",
                        help="Output as class labels (0=BG, 1=Unknown, 2=FG) instead of grayscale (0, 128, 255)")

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    erode_r = args.erode_radius if args.erode_radius is not None else args.radius
    dilate_r = args.dilate_radius if args.dilate_radius is not None else args.radius

    logger.info(f"Configuration: Erode Radius={erode_r}px, Dilate Radius={dilate_r}px, Label Mode={args.labels}")

    if input_path.is_file():
        # Single file execution
        out_name = input_path.stem + "_trimap.png"
        out_path = output_dir / out_name
        generate_trimap(input_path, out_path, erode_r, dilate_r, args.labels)
        logger.info(f"Successfully generated trimap: {out_path}")
    elif input_path.is_dir():
        # Directory batch execution
        extensions = {".png", ".jpg", ".jpeg", ".bmp"}
        mask_files = [f for f in input_path.iterdir() if f.suffix.lower() in extensions]

        if not mask_files:
            logger.warning(f"No image files found in {input_path}")
            return

        logger.info(f"Found {len(mask_files)} mask files. Processing...")

        try:
            from tqdm import tqdm
            pbar = tqdm(mask_files, desc="Generating Trimaps")
        except ImportError:
            pbar = mask_files

        success_count = 0
        for f in pbar:
            out_path = output_dir / f.name
            try:
                generate_trimap(f, out_path, erode_r, dilate_r, args.labels)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to process {f.name}: {e}")

        logger.info(f"Completed! Successfully processed {success_count}/{len(mask_files)} masks.")
    else:
        logger.error(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    main()
