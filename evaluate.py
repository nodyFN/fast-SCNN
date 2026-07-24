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
from dataset import (
    SegmentationDataset,
    MattingDataset,
    build_dataloader,
    build_val_transform,
    build_matting_val_transform,
)
from models import FastSCNN, FastSCNNSalient
from utils.checkpoint import load_checkpoint
from utils.ddc_loss import KnownRegionL1Loss
from utils.losses import CombinedSegmentationLoss, compute_total_loss, SalientSegmentationLoss
from utils.matting_metrics import MattingMetrics
from utils.metrics import SegmentationMetrics
from utils.visualization import visualize_segmentation, visualize_matting

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def evaluate(
    checkpoint_path: str,
    model_name: str = "fast_scnn",
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
    task_mode: str | None = None,
    threshold_sweep: bool = False,
) -> None:
    cfg = Config()
    cfg.model = model_name
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

    # 1. Quick Metadata Check from Checkpoint
    ckpt_meta = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = ckpt_meta.get("config", {})
    if task_mode is None:
        task_mode = checkpoint_config.get("task_mode", "segmentation")
    logger.info(f"Evaluation task mode: {task_mode}")

    is_matting = (task_mode == "ddc_matting")

    # 2. Dataset Setup
    split_dir = cfg.val_dir if split == "val" else cfg.test_dir
    if is_matting:
        transform = build_matting_val_transform(val_height, val_width)
        dataset = MattingDataset(
            split_dir, transform=transform,
            trimap_source=checkpoint_config.get("trimap_source", "binary_mask"),
            trimap_kernel_min=checkpoint_config.get("trimap_kernel_min", 1),
            trimap_kernel_max=checkpoint_config.get("trimap_kernel_max", 30),
            allow_threshold=allow_threshold,
            collapse_nonzero_to_foreground=checkpoint_config.get("collapse_nonzero_to_foreground", False),
        )
    else:
        transform = build_val_transform(val_height, val_width)
        dataset = SegmentationDataset(split_dir, transform=transform, allow_threshold=allow_threshold)
    logger.info(f"Evaluating on {split} split: {len(dataset)} samples")

    dataloader = build_dataloader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    # 3. Model Setup
    if cfg.model == "fast_scnn_salient":
        model = FastSCNNSalient(
            ppm_pool_sizes=checkpoint_config.get("ppm_pool_sizes", (1, 2, 3, 6)),
            coarse_channels=checkpoint_config.get("coarse_channels", 64),
            refinement_channels=checkpoint_config.get("refinement_channels", 64),
            dropout_p=checkpoint_config.get("dropout_p", 0.1),
            refinement_head=checkpoint_config.get("refinement_head", "multiscale"),
            prompt_gate_mode=checkpoint_config.get("prompt_gate_mode", "bidirectional"),
            prompt_gate_strength=checkpoint_config.get("prompt_gate_strength", 0.5),
            refine_h8_channels=checkpoint_config.get("refine_h8_channels", 96),
            h4_skip_channels=checkpoint_config.get("h4_skip_channels", 32),
            refine_h4_channels=checkpoint_config.get("refine_h4_channels", 64),
            h2_skip_channels=checkpoint_config.get("h2_skip_channels", 16),
            refine_h2_channels=checkpoint_config.get("refine_h2_channels", 32),
            fine_output_channels=checkpoint_config.get("fine_output_channels", 24),
            fine_dropout=checkpoint_config.get("fine_dropout", 0.1),
        ).to(device)
    else:
        model = FastSCNN(num_classes=cfg.num_classes, aux=True).to(device)

    # Load weights
    ckpt = load_checkpoint(checkpoint_path, model, map_location=device, weights_only=True)
    logger.info(f"Loaded checkpoint: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

    model.eval()

    # 4. Loss & Metrics Setup
    if is_matting:
        criterion = KnownRegionL1Loss().to(device)
        metrics_obj = MattingMetrics(
            foreground_threshold=checkpoint_config.get("foreground_threshold", 0.5),
            has_alpha_gt=False,  # Binary mask reference
        )
    elif cfg.model == "fast_scnn_salient":
        loss_profile = checkpoint_config.get("loss_profile", "legacy_salient")
        if loss_profile == "precision_salient":
            from utils.losses import PrecisionSalientLoss
            criterion = PrecisionSalientLoss(cfg).to(device)
        else:
            criterion = SalientSegmentationLoss(
                lambda_coarse=cfg.salient_lambda_coarse,
                lambda_fine=cfg.salient_lambda_fine,
                lambda_boundary=cfg.salient_lambda_boundary,
                coarse_bce_weight=cfg.salient_coarse_bce_weight,
                coarse_dice_weight=cfg.salient_coarse_dice_weight,
                fine_focal_weight=cfg.salient_fine_focal_weight,
                fine_dice_weight=cfg.salient_fine_dice_weight,
                focal_alpha=cfg.salient_focal_alpha,
                focal_gamma=cfg.salient_focal_gamma,
                pos_weight=cfg.salient_pos_weight,
            ).to(device)
        metrics_obj = SegmentationMetrics(2, cfg.ignore_index)
    else:
        criterion = CombinedSegmentationLoss(
            ce_weight=cfg.ce_weight, dice_weight=cfg.dice_weight,
            ignore_index=cfg.ignore_index,
        ).to(device)
        metrics_obj = SegmentationMetrics(cfg.num_classes, cfg.ignore_index)

    total_loss = 0.0
    num_batches = 0
    vis_saved = False

    is_salient = (cfg.model == "fast_scnn_salient")
    do_sweep = is_salient and threshold_sweep

    if do_sweep:
        thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        sweep_metrics = {t: SegmentationMetrics(2, cfg.ignore_index) for t in thresholds}
    else:
        metrics_obj.reset()

    # 5. Evaluation Loop
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            logits = model(images)

            if is_matting:
                trimaps = batch["trimap"].to(device, non_blocking=True)
                fine_alpha = logits["fine_prob"]
                loss_val = criterion(fine_alpha, trimaps)
                total_loss += loss_val.item()
                num_batches += 1
                gt = masks.unsqueeze(1).float()
                metrics_obj.update(fine_alpha, gt, trimap=trimaps)
            else:
                if is_salient:
                    targets = masks.unsqueeze(1).float()
                    loss_dict = criterion(
                        coarse_logits=logits["coarse_logits"],
                        fine_logits=logits["fine_logits"],
                        targets=targets,
                    )
                    if do_sweep:
                        probs = logits["fine_prob"].squeeze(1)
                    else:
                        best_t = checkpoint_config.get("best_validation_threshold", 0.5)
                        preds = (logits["fine_prob"] >= best_t).squeeze(1).long()
                else:
                    loss_dict = compute_total_loss(criterion, logits, masks, 0.0, 0.0)
                    preds = logits

                total_loss += loss_dict["total"].item()
                num_batches += 1
                
                if do_sweep:
                    for t, m_obj in sweep_metrics.items():
                        preds_t = (probs >= t).long()
                        m_obj.update(preds_t, masks)
                else:
                    metrics_obj.update(preds, masks)

            # Save visualization for first batch
            if save_vis and not vis_saved:
                if is_matting:
                    visualize_matting(
                        images=images,
                        trimaps=batch["trimap"].to(device),
                        coarse_alpha=logits["coarse_prob"],
                        fine_alpha=logits["fine_prob"],
                        ddc_images=batch["ddc_image"].to(device) if "ddc_image" in batch else None,
                        save_path=output_path / f"{split}_visualization.png",
                        num_samples=min(num_vis_samples, images.shape[0]),
                        threshold=checkpoint_config.get("foreground_threshold", 0.5),
                    )
                else:
                    if is_salient:
                        best_t = checkpoint_config.get("best_validation_threshold", 0.5)
                        preds_vis = (logits["fine_prob"] >= best_t).squeeze(1).long()
                        probs_vis = logits["fine_prob"].squeeze(1)
                    else:
                        preds_vis = logits.argmax(dim=1)
                        probs_vis = torch.softmax(logits, dim=1)[:, 1]
                    visualize_segmentation(
                        images, masks, preds_vis, probs_vis,
                        save_path=output_path / f"{split}_visualization.png",
                        num_samples=min(num_vis_samples, images.shape[0]),
                    )
                vis_saved = True

    # 6. Report and Save Results
    if do_sweep:
        logger.info("=== Threshold Sweep Results ===")
        for t in thresholds:
            m_obj = sweep_metrics[t]
            res = m_obj.compute()
            logger.info(
                f"Threshold: {t:.2f} | Dice: {res['foreground_dice']:.4f} | IoU: {res['foreground_iou']:.4f} | "
                f"Precision: {res['precision']:.4f} | Recall: {res['recall']:.4f} | FP Rate: {res['fp_rate']:.4f}"
            )
        best_t = checkpoint_config.get("best_validation_threshold", 0.5)
        best_t_idx = thresholds.index(best_t) if best_t in thresholds else thresholds.index(0.5)
        results = sweep_metrics[thresholds[best_t_idx]].compute()
        logger.info(f"★ Reporting metrics for loaded best threshold: {thresholds[best_t_idx]:.2f}")
    else:
        results = metrics_obj.compute()
    results["avg_loss"] = total_loss / max(num_batches, 1)

    # Print
    if is_matting:
        print(f"\n{'='*50}")
        print(f"Evaluation Results ({split} split) [DDC Matting]")
        print(f"{'='*50}")
        print(f"  Avg Loss (L1)   : {results['avg_loss']:.4f}")
        print(f"  Pixel Accuracy  : {results['pixel_accuracy']:.4f}")
        print(f"  mIoU            : {results['miou']:.4f}")
        print(f"  Foreground IoU  : {results['foreground_iou']:.4f}")
        print(f"  Foreground Dice : {results['foreground_dice']:.4f}")
        print(f"  SAD             : {results['sad']:.2f}")
        print(f"  MAD             : {results['mad']:.4f}")
        print(f"  MSE             : {results['mse']:.6f}")
        print(f"  Gradient Error  : {results['gradient_error']:.4f}")
        print(f"  SAD-T           : {results['sad_t']:.2f}")
        print(f"  MSE-T           : {results['mse_t']:.6f}")
        print(f"{'='*50}")
    else:
        print(f"\n{'='*50}")
        print(f"Evaluation Results ({split} split)")
        print(f"{'='*50}")
        print(f"  Avg Loss        : {results['avg_loss']:.4f}")
        print(f"  Pixel Accuracy  : {results['pixel_accuracy']:.4f}")
        print(f"  mIoU            : {results['miou']:.4f}")
        print(f"  Foreground IoU  : {results['foreground_iou']:.4f}")
        print(f"  Foreground Dice : {results['foreground_dice']:.4f}")
        print(f"  Mean Dice       : {results['mean_dice']:.4f}")
        if "precision" in results:
            print(f"  Precision       : {results['precision']:.4f}")
            print(f"  Recall          : {results['recall']:.4f}")
            print(f"  F1 (FG Dice)    : {results['f1']:.4f}")
            print(f"  False Pos Rate  : {results['fp_rate']:.4f}")
            print(f"  FP Pixel Ratio  : {results['fp_pixel_ratio']:.4f}")
        class_names = ["background", "foreground"] if cfg.model == "fast_scnn_salient" else cfg.class_names
        for i, name in enumerate(class_names):
            print(f"  {name:15s} : IoU={results['per_class_iou'][i]:.4f}  "
                  f"Dice={results['per_class_dice'][i]:.4f}")
        print(f"\nConfusion Matrix:")
        for row in results["confusion_matrix"]:
            print(f"  {row}")
        print(f"{'='*50}")

    # Save to JSON (convert non-serializable types like NaN)
    json_results = {
        k: v if not isinstance(v, float) or v == v else None  # NaN → null
        for k, v in results.items()
    }
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
    p.add_argument("--model", choices=["fast_scnn", "fast_scnn_salient"], default="fast_scnn",
                   help="Model architecture of checkpoint (default: fast_scnn)")
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
    p.add_argument("--task-mode", choices=["segmentation", "ddc_matting"], default=None,
                   help="Task mode for evaluation (default: auto-detected from checkpoint)")
    p.add_argument("--threshold-sweep", action="store_true",
                   help="Evaluate validation metrics across multiple threshold candidates")
    args = p.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
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
        task_mode=args.task_mode,
        threshold_sweep=args.threshold_sweep,
    )


if __name__ == "__main__":
    main()
