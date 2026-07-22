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


def infer_single(
    model: torch.nn.Module,
    image_path: Path,
    device: torch.device,
    height: int,
    width: int,
    output_dir: Path,
    model_name: str = "fast_scnn",
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
                model, img_path, device, height, width, output_dir, model_name, use_cuda_sync,
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
        infer_folder(model, input_path, device, args.height, args.width, output_dir, model_name=args.model)
    elif input_path.is_file():
        use_cuda_sync = device.type == "cuda"
        timings = infer_single(
            model, input_path, device, args.height, args.width,
            output_dir, model_name=args.model, use_cuda_sync=use_cuda_sync,
        )
        print(f"\nTiming breakdown:")
        for k, v in timings.items():
            print(f"  {k:20s}: {v:.1f} ms")
    else:
        logger.error(f"Input path not found: {input_path}")


if __name__ == "__main__":
    main()
