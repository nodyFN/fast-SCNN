#!/usr/bin/env python3
"""
Fast-SCNN training script.

Usage examples
--------------
# Project profile (AdamW + CE+Dice + AMP)
python train.py --profile project --epochs 200

# Paper profile (SGD + CE only, 1000 epochs)
python train.py --profile paper --epochs 1000 --batch-size 12

# Resume training
python train.py --resume checkpoints/latest.pt

# Smoke test with synthetic data (no real dataset required)
python train.py --smoke-test

Windows note: run inside ``if __name__ == '__main__':`` or set num_workers=0.
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
from models import FastSCNN, FastSCNNSalient, count_parameters
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.losses import CombinedSegmentationLoss, compute_total_loss, SalientSegmentationLoss
from utils.metrics import SegmentationMetrics
from utils.scheduler import build_scheduler
from utils.seed import seed_everything
from utils.visualization import save_all_curves, visualize_segmentation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Optimizer factory
# ===========================================================================


def build_optimizer(
    model: nn.Module,
    cfg: Config,
) -> torch.optim.Optimizer:
    """Build optimizer with optional per-group weight decay.

    Paper profile: depthwise convolution weights → weight_decay=0;
    BatchNorm params → weight_decay=0; bias → weight_decay=0.
    """
    if cfg.profile == "paper":
        # Separate parameter groups
        depthwise_params = []
        bn_params = []
        bias_params = []
        other_params = []

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "depthwise" in name and "weight" in name and param.ndim == 4:
                depthwise_params.append(param)
            elif "bn" in name or "batch" in name.lower():
                bn_params.append(param)
            elif "bias" in name:
                bias_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {"params": other_params, "weight_decay": cfg.weight_decay},
            {"params": depthwise_params, "weight_decay": cfg.depthwise_weight_decay},
            {"params": bn_params, "weight_decay": 0.0},
            {"params": bias_params, "weight_decay": 0.0},
        ]
        logger.info(
            f"Paper-profile parameter groups: "
            f"conv={len(other_params)}, dw={len(depthwise_params)}, "
            f"bn={len(bn_params)}, bias={len(bias_params)}"
        )
    else:
        param_groups = [
            {"params": [p for p in model.parameters() if p.requires_grad]}
        ]

    if cfg.optimizer.lower() == "sgd":
        return torch.optim.SGD(
            param_groups,
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer.lower() == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


# ===========================================================================
# Training and validation loops
# ===========================================================================


def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion: nn.Module,
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
    """Run one training epoch. Returns (loss_dict, updated_global_step)."""
    model.train()
    running = {}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(images)
            if cfg.model == "fast_scnn_salient":
                targets = masks.unsqueeze(1).float()
                losses = criterion(
                    coarse_logits=output["coarse_logits"],
                    fine_logits=output["fine_logits"],
                    targets=targets,
                )
            else:
                losses = compute_total_loss(
                    criterion, output, masks,
                    aux_downsample_weight=cfg.aux_downsample_weight,
                    aux_global_weight=cfg.aux_global_weight,
                )

        total_loss = losses["total"]

        # Dynamic running loss dictionary creation
        if not running:
            running = {k: 0.0 for k in losses.keys()}

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
    criterion: nn.Module,
    device: torch.device,
    metrics: SegmentationMetrics,
    cfg: Config,
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

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(images)
            if cfg.model == "fast_scnn_salient":
                targets = masks.unsqueeze(1).float()
                losses = criterion(
                    coarse_logits=output["coarse_logits"],
                    fine_logits=output["fine_logits"],
                    targets=targets,
                )
                preds = (output["fine_logits"] > 0.0).squeeze(1).long()
            else:
                losses = compute_total_loss(
                    criterion, output, masks,
                    aux_downsample_weight=0.0,
                    aux_global_weight=0.0,
                )
                preds = output

        running_loss += losses["total"].item()
        num_batches += 1
        metrics.update(preds, masks)

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

    if cfg.model == "fast_scnn_salient":
        model = FastSCNNSalient(
            ppm_pool_sizes=cfg.ppm_pool_sizes,
            coarse_channels=cfg.coarse_channels,
            refinement_channels=cfg.refinement_channels,
            dropout_p=cfg.dropout_p,
        ).to(device)
    else:
        model = FastSCNN(num_classes=cfg.num_classes, aux=cfg.aux, dropout_p=cfg.dropout_p).to(device)
    optimizer = build_optimizer(model, cfg)

    total_iters = 2 * 2  # 2 epochs × 2 batches
    scheduler = build_scheduler(cfg.scheduler, optimizer, total_iters, cfg.epochs, cfg.poly_power)

    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type) if use_amp else None

    if cfg.model == "fast_scnn_salient":
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
    else:
        criterion = CombinedSegmentationLoss(
            ce_weight=cfg.ce_weight, dice_weight=cfg.dice_weight,
            focal_weight=cfg.focal_weight, ignore_index=cfg.ignore_index,
        )

    metrics_obj = SegmentationMetrics(
        num_classes=2 if cfg.model == "fast_scnn_salient" else cfg.num_classes,
        ignore_index=cfg.ignore_index
    )

    h, w = 64, 128  # Small resolution for speed
    bs = 2

    # Fake data
    for epoch in range(2):
        model.train()
        for _ in range(2):
            images = torch.randn(bs, 3, h, w, device=device)
            masks = torch.randint(
                0, 2 if cfg.model == "fast_scnn_salient" else cfg.num_classes,
                (bs, h, w), device=device
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                output = model(images)
                if cfg.model == "fast_scnn_salient":
                    targets = masks.unsqueeze(1).float()
                    losses = criterion(
                        coarse_logits=output["coarse_logits"],
                        fine_logits=output["fine_logits"],
                        targets=targets,
                    )
                else:
                    losses = compute_total_loss(criterion, output, masks,
                                                cfg.aux_downsample_weight, cfg.aux_global_weight)
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

        # Validation
        model.eval()
        metrics_obj.reset()
        with torch.inference_mode():
            images = torch.randn(bs, 3, h, w, device=device)
            masks = torch.randint(
                0, 2 if cfg.model == "fast_scnn_salient" else cfg.num_classes,
                (bs, h, w), device=device
            )
            output = model(images)
            if cfg.model == "fast_scnn_salient":
                preds = (output["fine_logits"] > 0.0).squeeze(1).long()
            else:
                preds = output
            metrics_obj.update(preds, masks)
        res = metrics_obj.compute()
        logger.info(f"  Smoke val mIoU={res['miou']:.4f}")

        if cfg.scheduler.lower() == "cosine":
            scheduler.step()

    # Checkpoint save/load
    ckpt_path = cfg.checkpoint_dir / "smoke_test.pt"
    save_checkpoint(
        ckpt_path, epoch=1, global_step=4, model=model, optimizer=optimizer,
        scheduler=scheduler, scaler=scaler, best_miou=m["miou"],
        num_classes=cfg.num_classes, seed=cfg.seed,
    )
    loaded = load_checkpoint(ckpt_path, model, optimizer, scheduler, scaler,
                             map_location=device)
    assert loaded["epoch"] == 1
    assert loaded["global_step"] == 4
    logger.info(f"  Checkpoint save/load OK: {ckpt_path}")

    logger.info("=== SMOKE TEST PASSED ===")


# ===========================================================================
# Main training loop
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
    if cfg.model == "fast_scnn_salient":
        model = FastSCNNSalient(
            ppm_pool_sizes=cfg.ppm_pool_sizes,
            coarse_channels=cfg.coarse_channels,
            refinement_channels=cfg.refinement_channels,
            dropout_p=cfg.dropout_p,
        ).to(device)
    else:
        model = FastSCNN(
            num_classes=cfg.num_classes, aux=cfg.aux,
            ppm_pool_sizes=cfg.ppm_pool_sizes, dropout_p=cfg.dropout_p,
        ).to(device)
    total_p, trainable_p = count_parameters(model)
    logger.info(f"Parameters: total={total_p:,}  trainable={trainable_p:,}")

    # Load pre-trained weights (Transfer Learning / weights-only)
    if getattr(cfg, "weights", None):
        load_checkpoint(cfg.weights, model, map_location=device, weights_only=True)
        logger.info(f"Loaded pre-trained weights from {cfg.weights} for Transfer Learning.")

    # Freeze backbone if requested (Feature Extraction Mode)
    if getattr(cfg, "freeze_backbone", False):
        if cfg.model == "fast_scnn_salient":
            for param in model.backbone.learning_to_downsample.parameters():
                param.requires_grad = False
            for param in model.backbone.global_feature_extractor.parameters():
                param.requires_grad = False
        else:
            for param in model.learning_to_downsample.parameters():
                param.requires_grad = False
            for param in model.global_feature_extractor.parameters():
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
    if cfg.model == "fast_scnn_salient":
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
    else:
        criterion = CombinedSegmentationLoss(
            ce_weight=cfg.ce_weight, dice_weight=cfg.dice_weight,
            focal_weight=cfg.focal_weight, focal_alpha=cfg.focal_alpha,
            focal_gamma=cfg.focal_gamma, class_weights=cfg.class_weights,
            ignore_index=cfg.ignore_index,
        ).to(device)

    # Metrics
    metrics_obj = SegmentationMetrics(
        2 if cfg.model == "fast_scnn_salient" else cfg.num_classes,
        cfg.ignore_index
    )

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
            val_results = validate(model, val_loader, criterion, device, metrics_obj, cfg, use_amp)

            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]

            # Log
            if cfg.model == "fast_scnn_salient":
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
            else:
                logger.info(
                    f"Epoch {epoch}/{cfg.epochs - 1} ({elapsed:.1f}s) | "
                    f"Train loss={train_losses['total']:.4f} (CE={train_losses['ce']:.4f} "
                    f"Dice={train_losses['dice']:.4f} "
                    f"aux_ds={train_losses['aux_downsample']:.4f} "
                    f"aux_gl={train_losses['aux_global']:.4f}) | "
                    f"Val loss={val_results['val_loss']:.4f} | "
                    f"PA={val_results['pixel_accuracy']:.4f} "
                    f"mIoU={val_results['miou']:.4f} "
                    f"FG_IoU={val_results['foreground_iou']:.4f} "
                    f"FG_Dice={val_results['foreground_dice']:.4f} | "
                    f"LR={current_lr:.6f} | Best mIoU={best_miou:.4f}"
                )
                for i, name in enumerate(cfg.class_names):
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
            if cfg.model == "fast_scnn_salient":
                writer.add_scalar("Loss/coarse", train_losses["coarse"], epoch)
                writer.add_scalar("Loss/fine", train_losses["fine"], epoch)
                writer.add_scalar("Loss/boundary", train_losses["boundary"], epoch)
            else:
                writer.add_scalar("Loss/aux_downsample", train_losses["aux_downsample"], epoch)
                writer.add_scalar("Loss/aux_global", train_losses["aux_global"], epoch)
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
                    if cfg.model == "fast_scnn_salient":
                        preds = (preds_outputs["fine_logits"] > 0.0).squeeze(1).long()
                        probs = preds_outputs["fine_prob"].squeeze(1)
                    else:
                        preds = preds_outputs.argmax(dim=1)
                        probs = torch.softmax(preds_outputs, dim=1)[:, 1]
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
                best_miou, history, asdict(cfg), cfg.class_names,
                cfg.num_classes, cfg.seed,
            )
            if val_results["miou"] > best_miou:
                best_miou = val_results["miou"]
                save_checkpoint(
                    cfg.checkpoint_dir / "best_miou.pt",
                    epoch, global_step, model, optimizer, scheduler, scaler,
                    best_miou, history, asdict(cfg), cfg.class_names,
                    cfg.num_classes, cfg.seed,
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
            best_miou, history, asdict(cfg), cfg.class_names,
            cfg.num_classes, cfg.seed,
        )

    # Save history as JSON
    history_path = cfg.checkpoint_dir / "history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    writer.close()
    logger.info(f"Training complete.  Best mIoU: {best_miou:.4f}")


# ===========================================================================
# CLI
# ===========================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Fast-SCNN")
    p.add_argument("--profile", choices=["paper", "project"], default=None,
                   help="Training profile (default: use config.py defaults)")
    p.add_argument("--model", choices=["fast_scnn", "fast_scnn_salient"], default=None,
                   help="Model architecture to train (default: fast_scnn)")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, dest="learning_rate")
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default=None)
    p.add_argument("--scheduler", choices=["poly", "cosine"], default=None)
    p.add_argument("--train-height", type=int, default=None)
    p.add_argument("--train-width", type=int, default=None)
    p.add_argument("--val-height", type=int, default=None)
    p.add_argument("--val-width", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-aux", action="store_true", help="Disable auxiliary heads")
    p.add_argument("--no-amp", action="store_true", help="Disable AMP")
    p.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    p.add_argument("--weights", type=str, default=None,
                   help="Path to pre-trained weights for transfer learning / fine-tuning (weights-only)")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="Freeze the backbone (LearningToDownsample & GlobalFeatureExtractor) for transfer learning")
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

    # CLI overrides
    if args.model:
        cfg.model = args.model
    if args.data_root:
        cfg.data_root = Path(args.data_root)
        cfg.train_dir = cfg.data_root / "train"
        cfg.val_dir = cfg.data_root / "val"
        cfg.test_dir = cfg.data_root / "test"
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.learning_rate = args.learning_rate
    if args.optimizer:
        cfg.optimizer = args.optimizer
    if args.scheduler:
        cfg.scheduler = args.scheduler
    if args.train_height is not None:
        cfg.train_height = args.train_height
    if args.train_width is not None:
        cfg.train_width = args.train_width
    if args.val_height is not None:
        cfg.val_height = args.val_height
    if args.val_width is not None:
        cfg.val_width = args.val_width
    if args.num_workers is not None:
        cfg.num_workers = args.num_workers
    if args.no_aux:
        cfg.aux = False
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

    # Generate timestamp and redirect config directories
    from datetime import datetime
    if cfg.resume:
        resume_path = Path(cfg.resume)
        timestamp = resume_path.parent.name
        # Fallback to new timestamp if parent dir is not a timestamp
        if timestamp == "checkpoints" or not timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg.checkpoint_dir = cfg.checkpoint_dir / timestamp
    cfg.training_image_dir = cfg.training_image_dir / timestamp
    cfg.tensorboard_dir = cfg.tensorboard_dir / timestamp

    if args.smoke_test:
        run_smoke_test(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
