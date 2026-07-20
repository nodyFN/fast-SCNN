#!/usr/bin/env python3
"""
Evaluate a trained Fast-SCNN checkpoint on val or test split.

Usage
-----
python evaluate.py --checkpoint checkpoints/best_miou.pt --split val
python evaluate.py --checkpoint checkpoints/best_miou.pt --split test --save-vis
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm import tqdm

from config import Config
from dataset import SegmentationDataset, build_dataloader, build_val_transform
from models.fast_scnn import FastSCNN
from utils.checkpoint import load_checkpoint
from utils.losses import CombinedSegmentationLoss, compute_total_loss
from utils.metrics import SegmentationMetrics
from utils.visualization import visualize_segmentation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate(
    checkpoint_path: str,
    split: str = "val",
    data_root: str | None = None,
    batch_size: int = 4,
    num_workers: int = 4,
    device_str: str = "auto",
    save_vis: bool = False,
    output_dir: str = "evaluation_results",
    val_height: int = 512,
    val_width: int = 1024,
    num_vis_samples: int = 8,
    allow_threshold: bool = False,
) -> None:
    cfg = Config()
    if data_root:
        cfg.data_root = Path(data_root)
        cfg.train_dir = cfg.data_root / "train"
        cfg.val_dir = cfg.data_root / "val"
        cfg.test_dir = cfg.data_root / "test"

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Dataset
    split_dir = cfg.val_dir if split == "val" else cfg.test_dir
    transform = build_val_transform(val_height, val_width)
    dataset = SegmentationDataset(split_dir, transform=transform, allow_threshold=allow_threshold)
    logger.info(f"Evaluating on {split} split: {len(dataset)} samples")

    dataloader = build_dataloader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    # Model
    model = FastSCNN(num_classes=cfg.num_classes, aux=True).to(device)
    ckpt = load_checkpoint(checkpoint_path, model, map_location=device, weights_only=True)
    logger.info(f"Loaded checkpoint: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

    model.eval()

    # Loss & Metrics
    criterion = CombinedSegmentationLoss(
        ce_weight=cfg.ce_weight, dice_weight=cfg.dice_weight,
        ignore_index=cfg.ignore_index,
    )
    metrics_obj = SegmentationMetrics(cfg.num_classes, cfg.ignore_index)

    total_loss = 0.0
    num_batches = 0
    vis_saved = False

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            logits = model(images)
            loss_dict = compute_total_loss(criterion, logits, masks, 0.0, 0.0)
            total_loss += loss_dict["total"].item()
            num_batches += 1
            metrics_obj.update(logits, masks)

            # Save visualization for first batch
            if save_vis and not vis_saved:
                preds = logits.argmax(dim=1)
                probs = torch.softmax(logits, dim=1)[:, 1]
                visualize_segmentation(
                    images, masks, preds, probs,
                    save_path=output_path / f"{split}_visualization.png",
                    num_samples=min(num_vis_samples, images.shape[0]),
                )
                vis_saved = True

    # Results
    results = metrics_obj.compute()
    results["avg_loss"] = total_loss / max(num_batches, 1)

    # Print
    print(f"\n{'='*50}")
    print(f"Evaluation Results ({split} split)")
    print(f"{'='*50}")
    print(f"  Avg Loss        : {results['avg_loss']:.4f}")
    print(f"  Pixel Accuracy  : {results['pixel_accuracy']:.4f}")
    print(f"  mIoU            : {results['miou']:.4f}")
    print(f"  Foreground IoU  : {results['foreground_iou']:.4f}")
    print(f"  Foreground Dice : {results['foreground_dice']:.4f}")
    print(f"  Mean Dice       : {results['mean_dice']:.4f}")
    for i, name in enumerate(cfg.class_names):
        print(f"  {name:15s} : IoU={results['per_class_iou'][i]:.4f}  "
              f"Dice={results['per_class_dice'][i]:.4f}")
    print(f"\nConfusion Matrix:")
    for row in results["confusion_matrix"]:
        print(f"  {row}")
    print(f"{'='*50}")

    # Save to JSON (convert non-serializable types)
    json_results = {
        k: v if not isinstance(v, float) or v == v else None  # NaN → null
        for k, v in results.items()
    }
    # Convert lists that may contain NaN
    for k in ["per_class_iou", "per_class_dice"]:
        if k in json_results:
            json_results[k] = [
                v if v == v else None for v in json_results[k]
            ]
    json_path = output_path / f"{split}_metrics.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"Metrics saved to {json_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate Fast-SCNN")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--save-vis", action="store_true", help="Save visualization")
    p.add_argument("--output-dir", type=str, default="evaluation_results")
    p.add_argument("--val-height", type=int, default=512)
    p.add_argument("--val-width", type=int, default=1024)
    p.add_argument("--num-vis-samples", type=int, default=8)
    p.add_argument("--allow-threshold", action="store_true",
                   help="Allow thresholding grayscale masks to binary (0/1)")
    args = p.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        split=args.split,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device_str=args.device,
        save_vis=args.save_vis,
        output_dir=args.output_dir,
        val_height=args.val_height,
        val_width=args.val_width,
        num_vis_samples=args.num_vis_samples,
        allow_threshold=args.allow_threshold,
    )


if __name__ == "__main__":
    main()
