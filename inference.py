#!/usr/bin/env python3
"""
Inference script for Fast-SCNN.

Supports single-image and folder inference with detailed timing breakdown.

Usage
-----
# Single image
python inference.py --checkpoint checkpoints/best_miou.pt --input image.jpg --output-dir results/

# Folder
python inference.py --checkpoint checkpoints/best_miou.pt --input data/test/images/ --output-dir results/
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from config import Config
from dataset import IMAGENET_MEAN, IMAGENET_STD
from models import FastSCNN, FastSCNNSalient
from utils.checkpoint import load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def preprocess(
    image_bgr: np.ndarray,
    height: int,
    width: int,
    device: torch.device,
) -> torch.Tensor:
    """Preprocess a single BGR image to model input tensor."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height), interpolation=cv2.INTER_LINEAR)
    image_float = image_resized.astype(np.float32) / 255.0
    # Normalize with ImageNet stats
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    image_norm = (image_float - mean) / std
    # HWC → CHW → BCHW
    tensor = torch.from_numpy(image_norm.transpose(2, 0, 1)).unsqueeze(0)
    return tensor.to(device, non_blocking=True)


def postprocess(
    output: torch.Tensor | Dict[str, torch.Tensor],
    original_h: int,
    original_w: int,
    model_name: str = "fast_scnn",
) -> Dict[str, np.ndarray]:
    """Convert model output to prediction maps at original resolution."""
    if model_name == "fast_scnn_salient" or isinstance(output, dict):
        fine_logits = output["fine_logits"]
        fine_prob = output["fine_prob"]
        
        # Upsample both to original size
        logits_up = F.interpolate(
            fine_logits, size=(original_h, original_w),
            mode="bilinear", align_corners=False,
        )
        prob_up = F.interpolate(
            fine_prob, size=(original_h, original_w),
            mode="bilinear", align_corners=False,
        )
        
        pred_class = (logits_up > 0.0).squeeze(0).squeeze(0).cpu().numpy().astype(np.uint8)
        prob_fg = prob_up.squeeze(0).squeeze(0).cpu().numpy()
    else:
        # Upsample to original size
        logits_up = F.interpolate(
            output, size=(original_h, original_w),
            mode="bilinear", align_corners=False,
        )
        probs = torch.softmax(logits_up, dim=1)
        pred_class = logits_up.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        prob_fg = probs[0, 1].cpu().numpy()
        
    mask_255 = (pred_class * 255).astype(np.uint8)

    return {
        "class_mask": pred_class,       # 0/1
        "binary_mask": mask_255,        # 0/255
        "fg_probability": prob_fg,      # float [0, 1]
    }


def create_overlay(
    image_bgr: np.ndarray,
    class_mask: np.ndarray,
    alpha: float = 0.4,
    color: tuple = (0, 255, 0),
) -> np.ndarray:
    """Create a coloured overlay of the predicted foreground."""
    overlay = image_bgr.copy()
    mask_rgb = np.zeros_like(image_bgr)
    mask_rgb[class_mask == 1] = color
    overlay = cv2.addWeighted(overlay, 1 - alpha, mask_rgb, alpha, 0)
    return overlay


def find_gt_mask_path(image_path: Path, gt_arg: Optional[str] = None) -> Optional[Path]:
    """Find the ground truth mask path for a given image.

    Checks:
      1. If gt_arg is a direct file path, returns it.
      2. If gt_arg is a directory, searches for img_name.png/jpg/jpeg inside it.
      3. Auto-detect: if image_path contains a folder named "images", tries
         to locate a sibling folder named "masks" with the same base name.
    """
    if gt_arg:
        gt_p = Path(gt_arg)
        if gt_p.is_file():
            return gt_p
        elif gt_p.is_dir():
            for ext in [".png", ".jpg", ".jpeg"]:
                candidate = gt_p / f"{image_path.stem}{ext}"
                if candidate.is_file():
                    return candidate

    # Auto-detect: replace "/images/" folder with "/masks/"
    parts = list(image_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "masks"
        parent_dir = Path(*parts[:-1])
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = image_path.parents[0].parent / "masks" / f"{image_path.stem}{ext}"
            if candidate.is_file():
                return candidate
    return None


def load_gt_mask(gt_path: Path) -> np.ndarray:
    """Load and normalize GT mask to binary {0, 1}."""
    mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise IOError(f"Failed to read GT mask: {gt_path}")
    
    unique = np.unique(mask)
    if set(unique.tolist()) <= {0, 1}:
        return mask.astype(np.uint8)
    if set(unique.tolist()) <= {0, 255}:
        return (mask // 255).astype(np.uint8)
    # Fallback to thresholding
    return (mask > 127).astype(np.uint8)


def create_comparison_overlay(
    image_bgr: np.ndarray,
    class_mask: np.ndarray,
    gt_mask: np.ndarray,
    alpha: float = 0.4,
) -> np.ndarray:
    """Create a colored overlay comparing model predictions vs ground truth.

    Color coding (BGR):
        - Green (0, 255, 0)   : True Positive (TP) - hit (pred=1, gt=1)
        - Blue (255, 0, 0)    : False Positive (FP) - false alarm (pred=1, gt=0)
        - Red (0, 0, 255)     : False Negative (FN) - miss (pred=0, gt=1)
    """
    overlay = image_bgr.copy()
    mask_rgb = np.zeros_like(image_bgr)

    tp = (class_mask == 1) & (gt_mask == 1)
    fp = (class_mask == 1) & (gt_mask == 0)
    fn = (class_mask == 0) & (gt_mask == 1)

    mask_rgb[tp] = (0, 255, 0)   # Green (True Positive)
    mask_rgb[fp] = (255, 0, 0)   # Blue (False Positive)
    mask_rgb[fn] = (0, 0, 255)   # Red (False Negative)

    # Combine with original image using alpha blending
    overlay = cv2.addWeighted(overlay, 1 - alpha, mask_rgb, alpha, 0)
    return overlay


def infer_single(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    height: int,
    width: int,
    output_dir: Path,
    model_name: str = "fast_scnn",
    gt_arg: Optional[str] = None,
    use_cuda_sync: bool = False,
) -> Dict[str, float]:
    """Run inference on a single image with timing."""
    timings: Dict[str, float] = {}

    # --- Load ---
    if use_cuda_sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise IOError(f"Failed to read image: {image_path}")
    original_h, original_w = image_bgr.shape[:2]
    if use_cuda_sync:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    timings["load_ms"] = (t1 - t0) * 1000

    # --- Preprocess ---
    if use_cuda_sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    tensor = preprocess(image_bgr, height, width, device)
    if use_cuda_sync:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    timings["preprocess_ms"] = (t1 - t0) * 1000

    # --- Model inference ---
    if use_cuda_sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        output = model(tensor)
    if use_cuda_sync:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    timings["model_ms"] = (t1 - t0) * 1000

    # --- Postprocess ---
    if use_cuda_sync:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    results = postprocess(output, original_h, original_w, model_name=model_name)
    overlay = create_overlay(image_bgr, results["class_mask"])

    # Load GT mask if available for comparison
    gt_mask = None
    gt_path = find_gt_mask_path(image_path, gt_arg)
    if gt_path:
        try:
            gt_mask_loaded = load_gt_mask(gt_path)
            if gt_mask_loaded.shape[:2] != (original_h, original_w):
                gt_mask = cv2.resize(
                    gt_mask_loaded, (original_w, original_h),
                    interpolation=cv2.INTER_NEAREST,
                )
            else:
                gt_mask = gt_mask_loaded
            logger.info(f"Loaded GT mask for comparison: {gt_path}")
        except Exception as e:
            logger.warning(f"Failed to load GT mask for comparison: {e}")

    if use_cuda_sync:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    timings["postprocess_ms"] = (t1 - t0) * 1000

    timings["e2e_ms"] = (
        timings["load_ms"] + timings["preprocess_ms"]
        + timings["model_ms"] + timings["postprocess_ms"]
    )

    # --- Save outputs ---
    stem = image_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_dir / f"{stem}_class.png"), results["class_mask"])
    cv2.imwrite(str(output_dir / f"{stem}_binary.png"), results["binary_mask"])
    cv2.imwrite(str(output_dir / f"{stem}_overlay.jpg"), overlay)

    # If GT mask exists, save the TP/FP/FN comparison overlay
    if gt_mask is not None:
        comp_overlay = create_comparison_overlay(image_bgr, results["class_mask"], gt_mask)
        cv2.imwrite(str(output_dir / f"{stem}_comparison.jpg"), comp_overlay)
        logger.info(f"Saved comparison overlay to {output_dir / f'{stem}_comparison.jpg'}")

    # Probability map as heatmap & grayscale
    prob_vis = (results["fg_probability"] * 255).astype(np.uint8)
    prob_color = cv2.applyColorMap(prob_vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(output_dir / f"{stem}_prob.jpg"), prob_color)
    cv2.imwrite(str(output_dir / f"{stem}_prob_gray.png"), prob_vis)

    return timings


def infer_folder(
    model: torch.nn.Module,
    input_dir: Path,
    device: torch.device,
    height: int,
    width: int,
    output_dir: Path,
    model_name: str = "fast_scnn",
    gt_arg: Optional[str] = None,
) -> None:
    """Run inference on all images in a folder."""
    images = sorted(
        p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        logger.error(f"No images found in {input_dir}")
        return

    use_cuda_sync = device.type == "cuda"
    all_model_ms = []
    all_e2e_ms = []

    for img_path in images:
        try:
            timings = infer_single(
                model, img_path, device, height, width, output_dir,
                model_name=model_name, gt_arg=gt_arg, use_cuda_sync=use_cuda_sync,
            )
            all_model_ms.append(timings["model_ms"])
            all_e2e_ms.append(timings["e2e_ms"])
            logger.info(
                f"{img_path.name}: model={timings['model_ms']:.1f}ms "
                f"e2e={timings['e2e_ms']:.1f}ms"
            )
        except Exception as e:
            logger.error(f"Failed on {img_path.name}: {e}")

    # Summary
    n = len(all_model_ms)
    if n > 0:
        avg_model = sum(all_model_ms) / n
        avg_e2e = sum(all_e2e_ms) / n
        model_fps = 1000.0 / avg_model if avg_model > 0 else 0
        e2e_fps = 1000.0 / avg_e2e if avg_e2e > 0 else 0
        print(f"\n{'='*50}")
        print(f"Folder Inference Summary")
        print(f"{'='*50}")
        print(f"  Total images          : {n}")
        print(f"  Avg model latency     : {avg_model:.1f} ms")
        print(f"  Avg end-to-end latency: {avg_e2e:.1f} ms")
        print(f"  Model FPS             : {model_fps:.1f}")
        print(f"  End-to-end FPS        : {e2e_fps:.1f}")
        print(f"{'='*50}")


def main() -> None:
    p = argparse.ArgumentParser(description="Fast-SCNN Inference")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model", choices=["fast_scnn", "fast_scnn_salient"], default="fast_scnn",
                   help="Model architecture of checkpoint (default: fast_scnn)")
    p.add_argument("--input", type=str, required=True,
                   help="Single image path or folder path")
    p.add_argument("--gt", type=str, default=None,
                   help="Optional single GT mask path or GT masks directory for comparison overlay")
    p.add_argument("--output-dir", type=str, default="inference_results")
    p.add_argument("--height", type=int, default=512, help="Model input height")
    p.add_argument("--width", type=int, default=1024, help="Model input width")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--num-classes", type=int, default=2)
    args = p.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir)

    # Config to access defaults if needed
    cfg = Config()

    # Model
    if args.model == "fast_scnn_salient":
        model = FastSCNNSalient(
            ppm_pool_sizes=cfg.ppm_pool_sizes,
            coarse_channels=cfg.coarse_channels,
            refinement_channels=cfg.refinement_channels,
            dropout_p=cfg.dropout_p,
        ).to(device)
    else:
        model = FastSCNN(num_classes=args.num_classes, aux=False).to(device)
        
    load_checkpoint(args.checkpoint, model, map_location=device, weights_only=True)
    model.eval()
    logger.info(f"Model ({args.model}) loaded on {device}")

    input_path = Path(args.input)
    if input_path.is_dir():
        infer_folder(
            model, input_path, device, args.height, args.width, output_dir,
            model_name=args.model, gt_arg=args.gt,
        )
    elif input_path.is_file():
        use_cuda_sync = device.type == "cuda"
        timings = infer_single(
            model, input_path, device, args.height, args.width,
            output_dir, model_name=args.model, gt_arg=args.gt, use_cuda_sync=use_cuda_sync,
        )
        print(f"\nTiming breakdown:")
        for k, v in timings.items():
            print(f"  {k:20s}: {v:.1f} ms")
    else:
        logger.error(f"Input path not found: {input_path}")


if __name__ == "__main__":
    main()
