#!/usr/bin/env python3
"""
Fast-SCNN Salient Model Training Script.

Usage examples
--------------
# Train the salient model on your custom dataset (data/)
python train_salient.py --profile project --data-root data --train-height 540 --train-width 960

# Train on the DUTS dataset (requires --allow-threshold)
python train_salient.py --profile project --data-root duts_data --allow-threshold

# Fine-tune using pre-trained weights
python train_salient.py --profile project --data-root data --weights checkpoints/TIMESTAMP/best_miou.pt

# Freeze backbone and train only heads
python train_salient.py --profile project --data-root data --weights checkpoints/TIMESTAMP/best_miou.pt --freeze-backbone

# Smoke test with synthetic data
python train_salient.py --smoke-test
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config, get_paper_config, get_project_config
from dataset import (
    SegmentationDataset,
    build_dataloader,
    build_train_transform,
    build_val_transform,
)
from models.fast_scnn_salient import FastSCNNSalient, count_parameters
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.losses import SalientSegmentationLoss
from utils.metrics import SegmentationMetrics
from utils.seed import seed_everything
from utils.visualization import save_all_curves, visualize_segmentation

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Optimizer & Scheduler builders
# ===========================================================================


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Build optimizer with trainable parameters only."""
    # Filter out parameters that do not require gradients (e.g. if frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    if cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            trainable_params,
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(
            trainable_params,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")


def build_scheduler(
    scheduler_type: str,
    optimizer: torch.optim.Optimizer,
    total_iters: int,
    epochs: int,
    poly_power: float = 0.9,
    cosine_eta_min: float = 0.0,
) -> torch.optim.lr_scheduler._LRScheduler | None:
    """Build learning rate scheduler."""
    if scheduler_type.lower() == "poly":
        # PolyLR = initial_lr * (1 - iter / total_iters) ^ power
        lr_lambda = lambda step: (1.0 - step / total_iters) ** poly_power
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    elif scheduler_type.lower() == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=cosine_eta_min
        )
    return None


# ===========================================================================
# Epoch training and Validation loops
# ===========================================================================


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: SalientSegmentationLoss,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    cfg: Config,
    global_step: int,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    use_amp: bool = False,
) -> tuple:
    """Run one training epoch. Returns average losses."""
    model.train()
    running = {
        "total": 0.0, "coarse": 0.0, "fine": 0.0, "boundary": 0.0,
        "coarse_bce": 0.0, "coarse_dice": 0.0, "fine_focal": 0.0, "fine_dice": 0.0
    }
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        
        # Salient model expects float binary masks with shape [B, 1, H, W]
        targets = masks.unsqueeze(1).float()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(images)  # output dict
            losses = criterion(
                coarse_logits=output["coarse_logits"],
                fine_logits=output["fine_logits"],
                targets=targets,
            )

        total_loss = losses["total"]

        # NaN / Inf check
        if not torch.isfinite(total_loss):
            logger.error(f"Non-finite loss detected at step {global_step}: {total_loss.item()}")
            raise RuntimeError(f"Training diverged (loss={total_loss.item()}) at step {global_step}")

        if scaler is not None:
            scaler.scale(total_loss).backward()
            if cfg.gradient_clip_enabled:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            if cfg.gradient_clip_enabled:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.gradient_clip_max_norm)
            optimizer.step()

        # Step iteration-based scheduler (PolyLR)
        if cfg.scheduler.lower() == "poly":
            scheduler.step()

        global_step += 1
        num_batches += 1

        for k in running:
            running[k] += losses[k].item()

        pbar.set_postfix(loss=f"{total_loss.item():.4f}")

        # TensorBoard per-step
        if writer and global_step % 50 == 0:
            writer.add_scalar("Loss/train_step", total_loss.item(), global_step)

    # Average
    avg = {k: v / max(num_batches, 1) for k, v in running.items()}
    return avg, global_step


@torch.inference_mode()
def validate(
    model: nn.Module,
    dataloader,
    criterion: SalientSegmentationLoss,
    device: torch.device,
    metrics: SegmentationMetrics,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Run validation. Returns loss + metric dict."""
    model.eval()
    metrics.reset()
    running_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        targets = masks.unsqueeze(1).float()

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(images)
            losses = criterion(
                coarse_logits=output["coarse_logits"],
                fine_logits=output["fine_logits"],
                targets=targets,
            )

        running_loss += losses["total"].item()
        num_batches += 1

        # Calculate prediction classes (binary prediction map [B, H, W])
        preds_binary = (output["fine_logits"] > 0.0).squeeze(1).long()
        metrics.update(preds_binary, masks)

    avg_loss = running_loss / max(num_batches, 1)
    metric_results = metrics.compute()
    metric_results["val_loss"] = avg_loss
    return metric_results


# ===========================================================================
# Smoke test with synthetic data
# ===========================================================================


def run_smoke_test(cfg: Config) -> None:
    """Minimal training smoke test using synthetic data."""
    logger.info("=== SMOKE TEST (synthetic data) ===")
    device = cfg.resolve_device()
    cfg.ensure_dirs()

    model = FastSCNNSalient(
        ppm_pool_sizes=cfg.ppm_pool_sizes,
        coarse_channels=cfg.coarse_channels,
        refinement_channels=cfg.refinement_channels,
        dropout_p=cfg.dropout_p,
    ).to(device)
    optimizer = build_optimizer(model, cfg)

    total_iters = 2 * 2  # 2 epochs × 2 batches
    scheduler = build_scheduler(
        cfg.scheduler, optimizer, total_iters, cfg.epochs, cfg.poly_power,
    )

    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type) if use_amp else None

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
    )

    metrics_obj = SegmentationMetrics(
        num_classes=2, ignore_index=cfg.ignore_index,
    )

    h, w = 64, 128  # Small resolution for speed
    bs = 2

    # Fake data
    for epoch in range(2):
        model.train()
        for _ in range(2):
            images = torch.randn(bs, 3, h, w, device=device)
            masks = torch.randint(0, 2, (bs, h, w), device=device)
            targets = masks.unsqueeze(1).float()

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                output = model(images)
                losses = criterion(
                    coarse_logits=output["coarse_logits"],
                    fine_logits=output["fine_logits"],
                    targets=targets,
                )
            loss = losses["total"]
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            if cfg.scheduler.lower() == "poly":
                scheduler.step()
            logger.info(f"  Smoke epoch {epoch}, loss={loss.item():.4f}")

    logger.info("  ✓ Smoke test passed successfully.")


# ===========================================================================
# Main training pipeline
# ===========================================================================


def train(cfg: Config) -> None:
    """Full training pipeline."""
    device = cfg.resolve_device()
    cfg.ensure_dirs()
    seed_everything(cfg.seed, cfg.deterministic)

    logger.info(f"Device: {device}")
    logger.info(f"Profile: {cfg.profile}")
    logger.info(f"AMP: {cfg.amp and device.type == 'cuda'}")

    # Dataset & DataLoader
    train_transform = build_train_transform(
        cfg.train_height, cfg.train_width, cfg.aug_scale_min, cfg.aug_scale_max,
    )
    val_transform = build_val_transform(cfg.val_height, cfg.val_width)

    train_ds = SegmentationDataset(cfg.train_dir, transform=train_transform, allow_threshold=cfg.allow_threshold)
    val_ds = SegmentationDataset(cfg.val_dir, transform=val_transform, allow_threshold=cfg.allow_threshold)
    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    train_loader = build_dataloader(
        train_ds, cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers, drop_last=True,
        generator_seed=cfg.seed,
    )
    val_loader = build_dataloader(
        val_ds, cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers,
    )

    # Model
    model = FastSCNNSalient(
        ppm_pool_sizes=cfg.ppm_pool_sizes,
        coarse_channels=cfg.coarse_channels,
        refinement_channels=cfg.refinement_channels,
        dropout_p=cfg.dropout_p,
    ).to(device)
    total_p, trainable_p = count_parameters(model)
    logger.info(f"Parameters: total={total_p:,}  trainable={trainable_p:,}")

    # Load pre-trained weights (Transfer Learning / weights-only)
    if getattr(cfg, "weights", None):
        load_checkpoint(cfg.weights, model, map_location=device, weights_only=True)
        logger.info(f"Loaded pre-trained weights from {cfg.weights} for Transfer Learning.")

    # Freeze backbone if requested (Feature Extraction Mode)
    if getattr(cfg, "freeze_backbone", False):
        for param in model.backbone.learning_to_downsample.parameters():
            param.requires_grad = False
        for param in model.backbone.global_feature_extractor.parameters():
            param.requires_grad = False
        logger.info("Backbone modules (LearningToDownsample & GlobalFeatureExtractor) have been FROZEN.")
        # Recalculate parameters
        total_p, trainable_p = count_parameters(model)
        logger.info(f"Parameters after freezing: total={total_p:,}  trainable={trainable_p:,}")

    # Optimizer, scheduler, scaler
    optimizer = build_optimizer(model, cfg)
    total_iters = len(train_loader) * cfg.epochs
    scheduler = build_scheduler(
        cfg.scheduler, optimizer, total_iters, cfg.epochs, cfg.poly_power,
        cfg.cosine_eta_min,
    )
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type) if use_amp else None

    # Loss
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
    )

    # Metrics
    metrics_obj = SegmentationMetrics(num_classes=2, ignore_index=cfg.ignore_index)

    # TensorBoard
    writer = SummaryWriter(log_dir=str(cfg.tensorboard_dir))

    # History
    history: Dict[str, List[float]] = {
        "train_loss": [], "val_loss": [],
        "pixel_accuracy": [], "miou": [],
        "foreground_iou": [], "foreground_dice": [],
        "learning_rate": [],
    }

    # Resume
    start_epoch = 0
    global_step = 0
    best_miou = 0.0
    if cfg.resume:
        ckpt = load_checkpoint(
            cfg.resume, model, optimizer, scheduler, scaler,
            map_location=device,
        )
        start_epoch = ckpt.get("epoch", 0) + 1
        global_step = ckpt.get("global_step", 0)
        best_miou = ckpt.get("best_miou", 0.0)
        history = ckpt.get("history", history)
        logger.info(f"Resumed from epoch {start_epoch}, step {global_step}, best mIoU={best_miou:.4f}")

    # Early stopping
    patience_counter = 0

    try:
        for epoch in range(start_epoch, cfg.epochs):
            t0 = time.time()

            # Train
            train_losses, global_step = train_one_epoch(
                model, train_loader, criterion, optimizer, scheduler,
                scaler, device, cfg, global_step, epoch, writer, use_amp,
            )

            # Epoch-based scheduler step (CosineAnnealing)
            if cfg.scheduler.lower() == "cosine":
                scheduler.step()

            # Validate
            val_results = validate(model, val_loader, criterion, device, metrics_obj, use_amp)

            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]

            # Log
            logger.info(
                f"Epoch {epoch}/{cfg.epochs - 1} ({elapsed:.1f}s) | "
                f"Train loss={train_losses['total']:.4f} (Coarse={train_losses['coarse']:.4f} "
                f"Fine={train_losses['fine']:.4f} "
                f"Boundary={train_losses['boundary']:.4f}) | "
                f"Val loss={val_results['val_loss']:.4f} | "
                f"PA={val_results['pixel_accuracy']:.4f} "
                f"mIoU={val_results['miou']:.4f} "
                f"FG_IoU={val_results['foreground_iou']:.4f} "
                f"FG_Dice={val_results['foreground_dice']:.4f} | "
                f"LR={current_lr:.6f} | Best mIoU={best_miou:.4f}"
            )
            for i, name in enumerate(["background", "foreground"]):
                logger.info(f"  {name}: IoU={val_results['per_class_iou'][i]:.4f}  "
                            f"Dice={val_results['per_class_dice'][i]:.4f}")

            # History
            history["train_loss"].append(train_losses["total"])
            history["val_loss"].append(val_results["val_loss"])
            history["pixel_accuracy"].append(val_results["pixel_accuracy"])
            history["miou"].append(val_results["miou"])
            history["foreground_iou"].append(val_results["foreground_iou"])
            history["foreground_dice"].append(val_results["foreground_dice"])
            history["learning_rate"].append(current_lr)

            # TensorBoard
            writer.add_scalar("Loss/train", train_losses["total"], epoch)
            writer.add_scalar("Loss/validation", val_results["val_loss"], epoch)
            writer.add_scalar("Loss/coarse", train_losses["coarse"], epoch)
            writer.add_scalar("Loss/fine", train_losses["fine"], epoch)
            writer.add_scalar("Loss/boundary", train_losses["boundary"], epoch)
            writer.add_scalar("Metrics/pixel_accuracy", val_results["pixel_accuracy"], epoch)
            writer.add_scalar("Metrics/miou", val_results["miou"], epoch)
            writer.add_scalar("Metrics/foreground_iou", val_results["foreground_iou"], epoch)
            writer.add_scalar("Metrics/foreground_dice", val_results["foreground_dice"], epoch)
            writer.add_scalar("LearningRate", current_lr, epoch)

            # Visualization (save a few samples periodically)
            if epoch % max(cfg.epochs // 20, 1) == 0 or epoch == cfg.epochs - 1:
                try:
                    model.eval()
                    sample_batch = next(iter(val_loader))
                    imgs = sample_batch["image"].to(device)
                    msks = sample_batch["mask"].to(device)
                    with torch.inference_mode():
                        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                            preds_outputs = model(imgs)
                    # Squeeze channel dim for visualization
                    preds = (preds_outputs["fine_logits"] > 0.0).squeeze(1).long()
                    probs = preds_outputs["fine_prob"].squeeze(1)
                    visualize_segmentation(
                        imgs, msks, preds, probs,
                        save_path=cfg.training_image_dir / f"epoch_{epoch:04d}.png",
                        num_samples=cfg.num_vis_samples,
                    )
                except Exception as e:
                    logger.warning(f"Visualization failed: {e}")

            # Save curves
            save_all_curves(history, cfg.training_image_dir)

            # Checkpoints
            save_checkpoint(
                cfg.checkpoint_dir / "latest.pt",
                epoch, global_step, model, optimizer, scheduler, scaler,
                best_miou, history, asdict(cfg), ["background", "foreground"],
                2, cfg.seed,
            )
            if val_results["miou"] > best_miou:
                best_miou = val_results["miou"]
                save_checkpoint(
                    cfg.checkpoint_dir / "best_miou.pt",
                    epoch, global_step, model, optimizer, scheduler, scaler,
                    best_miou, history, asdict(cfg), ["background", "foreground"],
                    2, cfg.seed,
                )
                logger.info(f"  ★ New best mIoU: {best_miou:.4f}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if cfg.early_stopping_enabled and patience_counter >= cfg.early_stopping_patience:
                logger.info(f"Early stopping triggered (patience={cfg.early_stopping_patience})")
                break

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt — saving latest checkpoint before exit")
        save_checkpoint(
            cfg.checkpoint_dir / "latest.pt",
            epoch, global_step, model, optimizer, scheduler, scaler,
            best_miou, history, asdict(cfg), ["background", "foreground"],
            2, cfg.seed,
        )

    # Save history as JSON
    history_path = cfg.checkpoint_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    writer.close()
    logger.info(f"Training complete.  Best mIoU: {best_miou:.4f}")


# ===========================================================================
# CLI arguments
# ===========================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fast-SCNN Salient segmentation training")
    p.add_argument("--profile", type=str, default="project", choices=["paper", "project"])
    p.add_argument("--data-root", type=str, default=None, help="Root folder for dataset splits")
    p.add_argument("--train-height", type=int, default=None)
    p.add_argument("--train-width", type=int, default=None)
    p.add_argument("--val-height", type=int, default=None)
    p.add_argument("--val-width", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-amp", action="store_true", help="Disable AMP")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p.add_argument("--weights", type=str, default=None,
                   help="Path to pre-trained weights for transfer learning (weights-only)")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="Freeze the backbone for transfer learning")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--early-stopping", type=int, default=None,
                   help="Enable early stopping with given patience (0=disable)")
    p.add_argument("--allow-threshold", action="store_true", default=None,
                   help="Allow thresholding grayscale masks to binary (0/1)")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run a minimal smoke test with synthetic data")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Base config from profile
    if args.profile == "paper":
        cfg = get_paper_config()
    elif args.profile == "project":
        cfg = get_project_config()
    else:
        cfg = Config()

    # Manual CLI Overrides
    if args.data_root:
        root = Path(args.data_root)
        cfg.data_root = root
        cfg.train_dir = root / "train"
        cfg.val_dir = root / "val"
        cfg.test_dir = root / "test"
    if args.train_height is not None:
        cfg.train_height = args.train_height
    if args.train_width is not None:
        cfg.train_width = args.train_width
    if args.val_height is not None:
        cfg.val_height = args.val_height
    if args.val_width is not None:
        cfg.val_width = args.val_width
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.no_amp:
        cfg.amp = False
    if args.resume:
        cfg.resume = args.resume
    if args.weights:
        cfg.weights = args.weights
    if args.freeze_backbone:
        cfg.freeze_backbone = True
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device:
        cfg.device = args.device
    if args.early_stopping is not None:
        if args.early_stopping > 0:
            cfg.early_stopping_enabled = True
            cfg.early_stopping_patience = args.early_stopping
        else:
            cfg.early_stopping_enabled = False
    if args.allow_threshold is not None:
        cfg.allow_threshold = args.allow_threshold

    # Run mode
    if args.smoke_test:
        run_smoke_test(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
