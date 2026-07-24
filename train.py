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
import os
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

from config import (
    Config, get_paper_config, get_project_config,
    get_ddc_am2k_config, get_ddc_p3m_config, get_ddc_tv_config,
)
from dataset import (
    SegmentationDataset,
    MattingDataset,
    build_dataloader,
    build_train_transform,
    build_val_transform,
    build_matting_train_transform,
    build_matting_val_transform,
)
from models import FastSCNN, FastSCNNSalient, UNet, UNetSalientAdapter, count_parameters
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.ddc_loss import KnownRegionL1Loss, DirectionalDistanceConsistencyLoss
from utils.losses import CombinedSegmentationLoss, compute_total_loss, SalientSegmentationLoss, PrecisionSalientLoss, compute_kd_loss
from utils.matting_metrics import MattingMetrics
from utils.metrics import SegmentationMetrics
from utils.scheduler import build_scheduler
from utils.seed import seed_everything
from utils.visualization import save_all_curves, visualize_segmentation, visualize_matting

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
    teacher: Optional[nn.Module] = None,
) -> tuple:
    """Run one training epoch. Returns (loss_dict, updated_global_step)."""
    model.train()
    running = {}
    num_batches = 0

    # Disable progress bar on non-master ranks to avoid output clutter, or if no_tqdm is requested
    disable_tqdm = (os.environ.get("RANK", "0") != "0") or getattr(cfg, 'no_tqdm', False)
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False, disable=disable_tqdm)
    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(images)
            if cfg.model == "fast_scnn_salient" or (cfg.model == "unet" and "salient" in getattr(cfg, "loss_profile", "")):
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

            if teacher is not None:
                with torch.no_grad():
                    teacher_out = teacher(images)
                
                is_sal = (cfg.model == "fast_scnn_salient")
                kd_loss = compute_kd_loss(
                    student_out=output,
                    teacher_out=teacher_out,
                    loss_type=getattr(cfg, "kd_loss_type", "mse"),
                    temp=getattr(cfg, "kd_temperature", 1.0),
                    is_salient=is_sal,
                )
                
                alpha = getattr(cfg, "kd_alpha", 0.5)
                losses["student_total"] = losses["total"]
                losses["kd"] = kd_loss
                losses["total"] = (1.0 - alpha) * losses["student_total"] + alpha * kd_loss

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
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message="Detected call of.*")
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

    # Gather training losses across processes if DDP is active
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        for k in avg:
            loss_tensor = torch.tensor([avg[k]], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg[k] = (loss_tensor / dist.get_world_size()).item()

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
    running_loss = 0.0
    num_batches = 0

    is_salient = (cfg.model == "fast_scnn_salient" or (cfg.model == "unet" and "salient" in getattr(cfg, "loss_profile", "")))
    do_sweep = is_salient and getattr(cfg, "threshold_sweep", False)

    if do_sweep:
        thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
        sweep_metrics = {t: SegmentationMetrics(num_classes=2) for t in thresholds}
    else:
        metrics.reset()

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(images)
            if is_salient:
                targets = masks.unsqueeze(1).float()
                losses = criterion(
                    coarse_logits=output["coarse_logits"],
                    fine_logits=output["fine_logits"],
                    targets=targets,
                )
                if do_sweep:
                    probs = torch.sigmoid(output["fine_logits"]).squeeze(1)
                else:
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
        
        if do_sweep:
            for t, m_obj in sweep_metrics.items():
                preds_t = (probs >= t).long()
                m_obj.update(preds_t, masks)
        else:
            metrics.update(preds, masks)

    # Gather metrics & loss across DDP processes if active
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        # Reduce average loss
        loss_tensor = torch.tensor([running_loss / max(num_batches, 1)], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (loss_tensor / dist.get_world_size()).item()
    else:
        avg_loss = running_loss / max(num_batches, 1)

    if do_sweep:
        sweep_results = {}
        best_t = 0.50
        best_dice = -1.0
        for t in thresholds:
            m_obj = sweep_metrics[t]
            m_obj.all_reduce()
            res = m_obj.compute()
            dice = res["foreground_dice"]
            iou = res["foreground_iou"]
            prec = res["precision"]
            rec = res["recall"]
            fpr = res["fp_rate"]
            if os.environ.get("RANK", "0") == "0":
                logger.info(
                    f"Threshold: {t:.2f} | Dice: {dice:.4f} | IoU: {iou:.4f} | "
                    f"Precision: {prec:.4f} | Recall: {rec:.4f} | FP Rate: {fpr:.4f}"
                )
            sweep_results[t] = res
            if dice > best_dice:
                best_dice = dice
                best_t = t

        if os.environ.get("RANK", "0") == "0":
            logger.info(f"★ Best validation threshold selected: {best_t:.2f} with Dice {best_dice:.4f}")
            
        cfg.best_validation_threshold = best_t
        metric_results = sweep_results[best_t]
    else:
        metrics.all_reduce()
        metric_results = metrics.compute()

    metric_results["val_loss"] = avg_loss
    return metric_results


# ===========================================================================
# DDC Matting Training and Validation Loops
# ===========================================================================


def train_one_epoch_matting(
    model: nn.Module,
    dataloader,
    known_loss_fn: KnownRegionL1Loss,
    ddc_loss_fn: DirectionalDistanceConsistencyLoss,
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
    """Run one DDC matting training epoch.

    Returns (loss_dict, updated_global_step).
    """
    model.train()
    running = {}
    num_batches = 0

    # DDC Lambda Warmup: linearly scale ddc_lambda over the first cfg.ddc_warmup_epochs
    if getattr(cfg, "ddc_warmup_epochs", 0) > 0:
        current_ddc_lambda = cfg.ddc_lambda * min(1.0, epoch / cfg.ddc_warmup_epochs)
    else:
        current_ddc_lambda = cfg.ddc_lambda

    disable_tqdm = (os.environ.get("RANK", "0") != "0") or getattr(cfg, 'no_tqdm', False)
    if os.environ.get("RANK", "0") == "0":
        logger.info(f"Epoch {epoch} [matting] - Active DDC Lambda: {current_ddc_lambda:.4f}")
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [matting]", leave=False, disable=disable_tqdm)

    for batch in pbar:
        images = batch["image"].to(device, non_blocking=True)
        ddc_images = batch["ddc_image"].to(device, non_blocking=True)
        trimaps = batch["trimap"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Model forward (inside AMP)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(images)

            # Extract alpha predictions (sigmoid already applied in model)
            coarse_alpha = output["coarse_prob"]  # [B, 1, H, W]
            fine_alpha = output["fine_prob"]       # [B, 1, H, W]

            # Known L1 losses (AMP-safe, operates on float alpha)
            coarse_known_loss = known_loss_fn(coarse_alpha, trimaps)
            fine_known_loss = known_loss_fn(fine_alpha, trimaps)

        # DDC Loss — computed OUTSIDE AMP for numerical stability (float32)
        with torch.autocast(device_type=device.type, enabled=False):
            ddc_loss = ddc_loss_fn(
                fine_alpha.float(),
                ddc_images.float(),
            )

        # Total loss
        total_loss = (
            cfg.lambda_coarse_known * coarse_known_loss
            + cfg.lambda_fine_known * fine_known_loss
            + current_ddc_lambda * ddc_loss
        )

        # Build loss dict
        losses = {
            "total": total_loss,
            "coarse_known": coarse_known_loss,
            "fine_known": fine_known_loss,
            "ddc": ddc_loss,
        }

        if not running:
            running = {k: 0.0 for k in losses.keys()}

        # NaN / Inf check
        if not torch.isfinite(total_loss):
            logger.error(f"Non-finite loss at step {global_step}: {total_loss.item()}")
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
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, message="Detected call of.*")
                scheduler.step()

        global_step += 1
        num_batches += 1

        for k in running:
            running[k] += losses[k].item()

        pbar.set_postfix(loss=f"{total_loss.item():.4f}")

        # TensorBoard per-step logging
        if writer and global_step % 50 == 0:
            writer.add_scalar("Loss/train_step", total_loss.item(), global_step)
            writer.add_scalar("Loss/ddc_step", ddc_loss.item(), global_step)

            # Statistics
            with torch.no_grad():
                known_mask = ((trimaps < 0.25) | (trimaps > 0.75))
                unknown_mask = ((trimaps > 0.25) & (trimaps < 0.75))
                n_total = trimaps.numel()
                writer.add_scalar("Statistics/known_pixel_ratio",
                                  known_mask.float().sum().item() / max(n_total, 1), global_step)
                writer.add_scalar("Statistics/unknown_pixel_ratio",
                                  unknown_mask.float().sum().item() / max(n_total, 1), global_step)
                writer.add_scalar("Statistics/alpha_mean", fine_alpha.mean().item(), global_step)
                writer.add_scalar("Statistics/alpha_min", fine_alpha.min().item(), global_step)
                writer.add_scalar("Statistics/alpha_max", fine_alpha.max().item(), global_step)

    # Average losses
    avg = {k: v / max(num_batches, 1) for k, v in running.items()}

    # DDP all-reduce
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        for k in avg:
            loss_tensor = torch.tensor([avg[k]], device=device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
            avg[k] = (loss_tensor / dist.get_world_size()).item()

    return avg, global_step


@torch.inference_mode()
def validate_matting(
    model: nn.Module,
    dataloader,
    known_loss_fn: KnownRegionL1Loss,
    device: torch.device,
    metrics: MattingMetrics,
    cfg: Config,
    use_amp: bool = False,
) -> Dict[str, float]:
    """Run matting validation. Returns loss + metric dict."""
    model.eval()
    metrics.reset()
    running_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        trimaps = batch["trimap"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            output = model(images)
            fine_alpha = output["fine_prob"]  # [B, 1, H, W]
            coarse_alpha = output["coarse_prob"]

            # Loss on known regions only
            fine_known_loss = known_loss_fn(fine_alpha, trimaps)

        running_loss += fine_known_loss.item()
        num_batches += 1

        # Update matting metrics
        # Use binary mask as reference (since we don't have GT alpha)
        gt = masks.unsqueeze(1).float()  # [B, 1, H, W]
        metrics.update(fine_alpha, gt, trimap=trimaps)

    # DDP reduce
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        loss_tensor = torch.tensor([running_loss / max(num_batches, 1)], device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = (loss_tensor / dist.get_world_size()).item()
    else:
        avg_loss = running_loss / max(num_batches, 1)

    metrics.all_reduce()
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
            refinement_head=cfg.refinement_head,
            prompt_gate_mode=cfg.prompt_gate_mode,
            prompt_gate_strength=cfg.prompt_gate_strength,
            refine_h8_channels=cfg.refine_h8_channels,
            h4_skip_channels=cfg.h4_skip_channels,
            refine_h4_channels=cfg.refine_h4_channels,
            h2_skip_channels=cfg.h2_skip_channels,
            refine_h2_channels=cfg.refine_h2_channels,
            fine_output_channels=cfg.fine_output_channels,
            fine_dropout=cfg.fine_dropout,
        ).to(device)
    else:
        model = FastSCNN(num_classes=cfg.num_classes, aux=cfg.aux, dropout_p=cfg.dropout_p).to(device)
    optimizer = build_optimizer(model, cfg)

    total_iters = 2 * 2  # 2 epochs × 2 batches
    scheduler = build_scheduler(cfg.scheduler, optimizer, total_iters, cfg.epochs, cfg.poly_power)

    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type) if use_amp else None

    is_salient_mode = (cfg.model == "fast_scnn_salient" or (cfg.model == "unet" and "salient" in getattr(cfg, "loss_profile", "")))
    if is_salient_mode:
        if cfg.loss_profile == "precision_salient":
            criterion = PrecisionSalientLoss(cfg)
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
            )
    else:
        criterion = CombinedSegmentationLoss(
            ce_weight=cfg.ce_weight, dice_weight=cfg.dice_weight,
            focal_weight=cfg.focal_weight, ignore_index=cfg.ignore_index,
        )

    metrics_obj = SegmentationMetrics(
        num_classes=2 if is_salient_mode else cfg.num_classes,
        ignore_index=cfg.ignore_index
    )

    # Load Teacher Model for KD in smoke test if requested
    teacher = None
    if getattr(cfg, "teacher_weights", None):
        logger.info(f"  Smoke test: loading teacher from {cfg.teacher_weights}...")
        teacher_out_ch = 1 if cfg.model == "fast_scnn_salient" else cfg.num_classes
        teacher = UNet(in_channels=3, out_channels=teacher_out_ch).to(device)
        load_checkpoint(cfg.teacher_weights, teacher, map_location=device, weights_only=True)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False

    h, w = 64, 128  # Small resolution for speed
    bs = 2

    # Fake data
    for epoch in range(2):
        model.train()
        for _ in range(2):
            images = torch.randn(bs, 3, h, w, device=device)
            masks = torch.randint(
                0, 2 if is_salient_mode else cfg.num_classes,
                (bs, h, w), device=device
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                output = model(images)
                if is_salient_mode:
                    targets = masks.unsqueeze(1).float()
                    losses = criterion(
                        coarse_logits=output["coarse_logits"],
                        fine_logits=output["fine_logits"],
                        targets=targets,
                    )
                else:
                    losses = compute_total_loss(criterion, output, masks,
                                                cfg.aux_downsample_weight, cfg.aux_global_weight)

                if teacher is not None:
                    with torch.no_grad():
                        teacher_out = teacher(images)
                    is_sal = is_salient_mode
                    kd_loss = compute_kd_loss(
                        student_out=output,
                        teacher_out=teacher_out,
                        loss_type=getattr(cfg, "kd_loss_type", "mse"),
                        temp=getattr(cfg, "kd_temperature", 1.0),
                        is_salient=is_sal,
                    )
                    alpha = getattr(cfg, "kd_alpha", 0.5)
                    losses["student_total"] = losses["total"]
                    losses["kd"] = kd_loss
                    losses["total"] = (1.0 - alpha) * losses["student_total"] + alpha * kd_loss

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
                0, 2 if is_salient_mode else cfg.num_classes,
                (bs, h, w), device=device
            )
            output = model(images)
            if is_salient_mode:
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
        scheduler=scheduler, scaler=scaler, best_miou=res["miou"],
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
    import os
    import torch.distributed as dist
    from torch.utils.data.distributed import DistributedSampler
    from torch.nn.parallel import DistributedDataParallel as DDP

    # Detect and initialize DDP environment
    is_ddp = "RANK" in os.environ
    if is_ddp:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        # Suppress logging on non-master ranks
        if rank != 0:
            logger.setLevel(logging.WARNING)
    else:
        local_rank = 0
        rank = 0
        world_size = 1
        device = cfg.resolve_device()

    seed_everything(cfg.seed, cfg.deterministic)

    # Attach FileHandler to save logs to train.log in the active checkpoint folder (Rank 0 only)
    if rank == 0:
        cfg.ensure_dirs()
        log_file = cfg.checkpoint_dir / "train.log"
        file_handler = logging.FileHandler(str(log_file), encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)

    logger.info(f"Device: {device}")
    logger.info(f"Profile: {cfg.profile}")
    logger.info(f"Task mode: {cfg.task_mode}")
    logger.info(f"AMP: {cfg.amp and device.type == 'cuda'}")

    is_matting = (cfg.task_mode == "ddc_matting")

    # Dataset & DataLoader
    if is_matting:
        crop_h = cfg.matting_crop_height
        crop_w = cfg.matting_crop_width
        train_transform = build_matting_train_transform(
            crop_h, crop_w, cfg.aug_scale_min, cfg.aug_scale_max,
            longest_max_size=getattr(cfg, "longest_max_size", None),
        )
        val_transform = build_matting_val_transform(crop_h, crop_w)
        train_ds = MattingDataset(
            cfg.train_dir, transform=train_transform,
            trimap_source=cfg.trimap_source,
            trimap_kernel_min=cfg.trimap_kernel_min,
            trimap_kernel_max=cfg.trimap_kernel_max,
            allow_threshold=cfg.allow_threshold,
            collapse_nonzero_to_foreground=cfg.collapse_nonzero_to_foreground,
        )
        val_ds = MattingDataset(
            cfg.val_dir, transform=val_transform,
            trimap_source=cfg.trimap_source,
            trimap_kernel_min=cfg.trimap_kernel_min,
            trimap_kernel_max=cfg.trimap_kernel_max,
            allow_threshold=cfg.allow_threshold,
            collapse_nonzero_to_foreground=cfg.collapse_nonzero_to_foreground,
        )
    else:
        train_transform = build_train_transform(
            cfg.train_height, cfg.train_width, cfg.aug_scale_min, cfg.aug_scale_max,
            longest_max_size=getattr(cfg, "longest_max_size", None),
        )
        val_transform = build_val_transform(cfg.val_height, cfg.val_width)
        train_ds = SegmentationDataset(cfg.train_dir, transform=train_transform, allow_threshold=cfg.allow_threshold)
        val_ds = SegmentationDataset(cfg.val_dir, transform=val_transform, allow_threshold=cfg.allow_threshold)
    logger.info(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")

    if is_ddp:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
        train_shuffle = False
    else:
        train_sampler = None
        val_sampler = None
        train_shuffle = True

    train_loader = build_dataloader(
        train_ds, cfg.batch_size, shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        persistent_workers=cfg.persistent_workers, drop_last=True,
        generator_seed=cfg.seed,
    )
    val_loader = build_dataloader(
        val_ds, cfg.batch_size, shuffle=False,
        sampler=val_sampler,
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
            refinement_head=cfg.refinement_head,
            prompt_gate_mode=cfg.prompt_gate_mode,
            prompt_gate_strength=cfg.prompt_gate_strength,
            refine_h8_channels=cfg.refine_h8_channels,
            h4_skip_channels=cfg.h4_skip_channels,
            refine_h4_channels=cfg.refine_h4_channels,
            h2_skip_channels=cfg.h2_skip_channels,
            refine_h2_channels=cfg.refine_h2_channels,
            fine_output_channels=cfg.fine_output_channels,
            fine_dropout=cfg.fine_dropout,
        ).to(device)
    elif cfg.model == "unet":
        is_salient_task = "salient" in getattr(cfg, "loss_profile", "")
        unet_out_ch = 1 if is_salient_task else cfg.num_classes
        unet_model = UNet(in_channels=3, out_channels=unet_out_ch).to(device)
        if is_salient_task:
            model = UNetSalientAdapter(unet_model)
        else:
            model = unet_model
    else:
        # For matting, standard FastSCNN outputs 1 channel and disables auxiliary heads
        num_classes = 1 if is_matting else cfg.num_classes
        aux = False if is_matting else cfg.aux
        model = FastSCNN(
            num_classes=num_classes, aux=aux,
            ppm_pool_sizes=cfg.ppm_pool_sizes, dropout_p=cfg.dropout_p,
        ).to(device)

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
        elif cfg.model == "unet":
            unet_ref = model.unet if isinstance(model, UNetSalientAdapter) else model
            for layer in [unet_ref.encoder1, unet_ref.encoder2, unet_ref.encoder3, unet_ref.encoder4]:
                for param in layer.parameters():
                    param.requires_grad = False
        else:
            for param in model.learning_to_downsample.parameters():
                param.requires_grad = False
            for param in model.global_feature_extractor.parameters():
                param.requires_grad = False
        logger.info("Backbone modules/Encoder layers have been FROZEN.")

    # Wrap FastSCNN with matting adapter if needed
    if is_matting and cfg.model == "fast_scnn":
        from models.fast_scnn import FastSCNNMattingAdapter
        model = FastSCNNMattingAdapter(model).to(device)

    # Load Teacher Model for Knowledge Distillation if requested
    teacher = None
    if getattr(cfg, "teacher_weights", None):
        logger.info(f"Setting up UNet Teacher Model from {cfg.teacher_weights}...")
        teacher_out_ch = 1 if cfg.model == "fast_scnn_salient" else cfg.num_classes
        teacher = UNet(in_channels=3, out_channels=teacher_out_ch).to(device)
        load_checkpoint(cfg.teacher_weights, teacher, map_location=device, weights_only=True)
        teacher.eval()
        for param in teacher.parameters():
            param.requires_grad = False


    total_p, trainable_p = count_parameters(model)
    logger.info(f"Parameters: total={total_p:,}  trainable={trainable_p:,}")

    # Wrap model with DDP if distributed training is enabled
    if is_ddp:
        # find_unused_parameters is only needed if there are parameters in the model 
        # that are not used in forward (e.g. standard FastSCNN with aux=False)
        find_unused = (cfg.model == "fast_scnn" and not cfg.aux)
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,
        )

    # Optimizer, scheduler, scaler
    optimizer = build_optimizer(model, cfg)
    total_iters = len(train_loader) * cfg.epochs
    scheduler = build_scheduler(
        cfg.scheduler, optimizer, total_iters, cfg.epochs, cfg.poly_power,
        cfg.cosine_eta_min,
        milestones=cfg.scheduler_milestones,
        gamma=cfg.scheduler_gamma,
    )
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler(device=device.type) if use_amp else None

    # Loss
    if is_matting:
        known_loss_fn = KnownRegionL1Loss().to(device)
        ddc_loss_fn = DirectionalDistanceConsistencyLoss(
            window_size=cfg.ddc_window_size,
            num_neighbors=cfg.ddc_num_neighbors,
            padding_mode=cfg.ddc_padding_mode,
            exclude_center=cfg.ddc_exclude_center,
            reduction=cfg.ddc_reduction,
            chunk_size=cfg.ddc_chunk_size,
            downsample_factor=cfg.ddc_downsample_factor,
        ).to(device)
        criterion = None  # Not used for matting
        logger.info(
            f"DDC Matting Loss: λ_coarse_known={cfg.lambda_coarse_known}, "
            f"λ_fine_known={cfg.lambda_fine_known}, λ_ddc={cfg.ddc_lambda}, "
            f"window={cfg.ddc_window_size}, neighbors={cfg.ddc_num_neighbors}, "
            f"chunk_size={cfg.ddc_chunk_size}, downsample={cfg.ddc_downsample_factor}"
        )
    elif cfg.model == "fast_scnn_salient" or (cfg.model == "unet" and "salient" in getattr(cfg, "loss_profile", "")):
        if cfg.loss_profile == "precision_salient":
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
        known_loss_fn = None
        ddc_loss_fn = None
    else:
        criterion = CombinedSegmentationLoss(
            ce_weight=cfg.ce_weight, dice_weight=cfg.dice_weight,
            focal_weight=cfg.focal_weight, focal_alpha=cfg.focal_alpha,
            focal_gamma=cfg.focal_gamma, class_weights=cfg.class_weights,
            ignore_index=cfg.ignore_index,
        ).to(device)
        known_loss_fn = None
        ddc_loss_fn = None

    # Metrics
    if is_matting:
        metrics_obj = MattingMetrics(
            foreground_threshold=cfg.foreground_threshold,
            has_alpha_gt=False,  # Binary mask validation by default
        )
    else:
        metrics_obj = SegmentationMetrics(
            2 if cfg.model == "fast_scnn_salient" else cfg.num_classes,
            cfg.ignore_index
        )

    # TensorBoard
    writer = SummaryWriter(log_dir=str(cfg.tensorboard_dir))

    # History
    if is_matting:
        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "coarse_known_loss": [], "fine_known_loss": [], "ddc_loss": [],
            "mad": [], "mse": [],
            "foreground_iou": [], "foreground_dice": [],
            "learning_rate": [],
        }
    else:
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
            map_location=device, task_mode=cfg.task_mode,
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
            if is_ddp:
                train_sampler.set_epoch(epoch)
            t0 = time.time()

            # Train
            if is_matting:
                train_losses, global_step = train_one_epoch_matting(
                    model, train_loader, known_loss_fn, ddc_loss_fn, optimizer, scheduler,
                    scaler, device, cfg, global_step, epoch, writer, use_amp,
                )
            else:
                train_losses, global_step = train_one_epoch(
                    model, train_loader, criterion, optimizer, scheduler,
                    scaler, device, cfg, global_step, epoch, writer, use_amp,
                    teacher=teacher,
                )

            # Epoch-based scheduler step (CosineAnnealing / MultiStep)
            if cfg.scheduler.lower() in ("cosine", "multistep"):
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=UserWarning, message="Detected call of.*")
                    scheduler.step()

            # Validate
            if is_matting:
                val_results = validate_matting(model, val_loader, known_loss_fn, device, metrics_obj, cfg, use_amp)
            else:
                val_results = validate(model, val_loader, criterion, device, metrics_obj, cfg, use_amp)

            elapsed = time.time() - t0
            current_lr = optimizer.param_groups[0]["lr"]

            # Log
            if is_matting:
                logger.info(
                    f"Epoch {epoch}/{cfg.epochs - 1} ({elapsed:.1f}s) | "
                    f"Train loss={train_losses['total']:.4f} (CoarseL1={train_losses['coarse_known']:.4f} "
                    f"FineL1={train_losses['fine_known']:.4f} DDC={train_losses['ddc']:.4f}) | "
                    f"Val loss={val_results['val_loss']:.4f} | "
                    f"mIoU={val_results['miou']:.4f} "
                    f"FG_IoU={val_results['foreground_iou']:.4f} "
                    f"FG_Dice={val_results['foreground_dice']:.4f} | "
                    f"LR={current_lr:.6f} | Best mIoU={best_miou:.4f}"
                )
                logger.info(
                    f"  Matting validation: SAD={val_results['sad']:.1f} MAD={val_results['mad']:.4f} "
                    f"MSE={val_results['mse']:.6f} GradErr={val_results['gradient_error']:.4f} "
                    f"SAD-T={val_results['sad_t']:.1f} MSE-T={val_results['mse_t']:.6f}"
                )
            elif cfg.model == "fast_scnn_salient":
                loss_str = " ".join([f"{k}={v:.4f}" for k, v in train_losses.items() if k != "total"])
                logger.info(
                    f"Epoch {epoch}/{cfg.epochs - 1} ({elapsed:.1f}s) | "
                    f"Train loss={train_losses['total']:.4f} ({loss_str}) | "
                    f"Val loss={val_results['val_loss']:.4f} | "
                    f"PA={val_results['pixel_accuracy']:.4f} "
                    f"mIoU={val_results['miou']:.4f} "
                    f"FG_IoU={val_results['foreground_iou']:.4f} "
                    f"FG_Dice={val_results['foreground_dice']:.4f} | "
                    f"LR={current_lr:.6f} | Best mIoU={best_miou:.4f}"
                )
                if "precision" in val_results:
                    logger.info(
                        f"  Metrics: Prec={val_results['precision']:.4f}  "
                        f"Recall={val_results['recall']:.4f}  F1={val_results['f1']:.4f}  "
                        f"FPR={val_results['fp_rate']:.4f}"
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

            # History (Rank 0 only)
            if rank == 0:
                history["train_loss"].append(train_losses["total"])
                history["val_loss"].append(val_results["val_loss"])
                for k, v in train_losses.items():
                    if k != "total":
                        hist_key = f"train_{k}"
                        if hist_key not in history:
                            history[hist_key] = [0.0] * epoch
                        history[hist_key].append(v)
                if not is_matting:
                    history["pixel_accuracy"].append(val_results["pixel_accuracy"])
                    history["miou"].append(val_results["miou"])
                    for k in ["precision", "recall", "f1", "fp_rate", "fp_pixel_ratio"]:
                        if k in val_results:
                            if k not in history:
                                history[k] = [0.0] * epoch
                            history[k].append(val_results[k])
                else:
                    history["mad"].append(val_results["mad"])
                    history["mse"].append(val_results["mse"])
                history["foreground_iou"].append(val_results["foreground_iou"])
                history["foreground_dice"].append(val_results["foreground_dice"])
                history["learning_rate"].append(current_lr)

            # TensorBoard (Rank 0 only)
            if rank == 0 and writer:
                writer.add_scalar("Loss/train", train_losses["total"], epoch)
                writer.add_scalar("Loss/validation", val_results["val_loss"], epoch)
                for k, v in train_losses.items():
                    if k != "total":
                        writer.add_scalar(f"Loss/{k}", v, epoch)
                if not is_matting:
                    writer.add_scalar("Metrics/pixel_accuracy", val_results["pixel_accuracy"], epoch)
                    writer.add_scalar("Metrics/miou", val_results["miou"], epoch)
                    for k in ["precision", "recall", "f1", "fp_rate", "fp_pixel_ratio"]:
                        if k in val_results:
                            writer.add_scalar(f"Metrics/{k}", val_results[k], epoch)
                else:
                    writer.add_scalar("Metrics/mad", val_results["mad"], epoch)
                    writer.add_scalar("Metrics/mse", val_results["mse"], epoch)
                    writer.add_scalar("Metrics/miou", val_results["miou"], epoch)
                writer.add_scalar("Metrics/foreground_iou", val_results["foreground_iou"], epoch)
                writer.add_scalar("Metrics/foreground_dice", val_results["foreground_dice"], epoch)
                writer.add_scalar("LearningRate", current_lr, epoch)

            # Visualization & Curves (Rank 0 only)
            if rank == 0:
                if epoch % cfg.vis_interval == 0 or epoch == cfg.epochs - 1:
                    try:
                        model.eval()
                        sample_batch = next(iter(val_loader))
                        imgs = sample_batch["image"].to(device)
                        msks = sample_batch["mask"].to(device)
                        with torch.inference_mode():
                            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                                preds_outputs = model(imgs)
                        if is_matting:
                            visualize_matting(
                                images=imgs,
                                trimaps=sample_batch["trimap"].to(device),
                                coarse_alpha=preds_outputs["coarse_prob"],
                                fine_alpha=preds_outputs["fine_prob"],
                                ddc_images=sample_batch["ddc_image"].to(device) if "ddc_image" in sample_batch else None,
                                save_path=cfg.training_image_dir / f"epoch_{epoch:04d}.png",
                                num_samples=cfg.num_vis_samples,
                                threshold=cfg.foreground_threshold,
                            )
                        else:
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

            # Checkpoints (Save on Rank 0, but update metrics & early stopping on all ranks to stay synced)
            if rank == 0:
                save_checkpoint(
                    cfg.checkpoint_dir / "latest.pt",
                    epoch, global_step, model, optimizer, scheduler, scaler,
                    best_miou, history, asdict(cfg), cfg.class_names,
                    cfg.num_classes, cfg.seed,
                )
            if val_results["miou"] > best_miou:
                best_miou = val_results["miou"]
                if rank == 0:
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
        if rank == 0:
            logger.info("KeyboardInterrupt — saving latest checkpoint before exit")
            save_checkpoint(
                cfg.checkpoint_dir / "latest.pt",
                epoch, global_step, model, optimizer, scheduler, scaler,
                best_miou, history, asdict(cfg), cfg.class_names,
                cfg.num_classes, cfg.seed,
            )

    # Save history as JSON (Rank 0 only)
    if rank == 0:
        history_path = cfg.checkpoint_dir / "history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(f"Training history saved to {history_path}")

        if writer:
            writer.close()
        logger.info(f"Training complete.  Best mIoU: {best_miou:.4f}")

        # Flush, close and remove FileHandler
        file_handler.close()
        logging.getLogger().removeHandler(file_handler)

    # Cleanup DDP process group
    if is_ddp:
        dist.destroy_process_group()


# ===========================================================================
# CLI
# ===========================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Fast-SCNN")
    p.add_argument("--profile", choices=["paper", "project", "paper_am2k", "paper_p3m", "tv_ddc"], default=None,
                   help="Training profile (default: use config.py defaults)")
    p.add_argument("--no-tqdm", action="store_true", help="Disable tqdm progress bars")
    p.add_argument("--model", choices=["fast_scnn", "fast_scnn_salient", "unet"], default=None,
                   help="Model architecture to train (default: fast_scnn)")
    p.add_argument("--data-root", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, dest="learning_rate")
    p.add_argument("--longest-max-size", type=int, default=None,
                   help="Resize longest side of images to this size before augmentation")
    p.add_argument("--optimizer", choices=["sgd", "adamw"], default=None)
    p.add_argument("--scheduler", choices=["poly", "cosine", "multistep"], default=None)
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

    # DDC Matting specific arguments
    p.add_argument("--task-mode", choices=["segmentation", "ddc_matting"], default=None)
    p.add_argument("--loss-profile", choices=["legacy", "legacy_salient", "ddc_matting", "precision_salient"], default=None)
    
    # Multiscale Fine Head & Precision Loss arguments
    p.add_argument("--refinement-head", choices=["legacy_h8", "multiscale"], default=None,
                   help="Refinement head architecture")
    p.add_argument("--prompt-gate-mode", choices=["legacy_additive", "bidirectional"], default=None,
                   help="Spatial prompt gating mode")
    p.add_argument("--prompt-gate-strength", type=float, default=None,
                   help="Strength of bidirectional spatial prompt gating")
    p.add_argument("--tversky-fp-weight", type=float, default=None,
                   help="False Positive penalty weight in Tversky Loss")
    p.add_argument("--tversky-fn-weight", type=float, default=None,
                   help="False Negative penalty weight in Tversky Loss")
    p.add_argument("--boundary-kernel-size", type=int, default=None,
                   help="Kernel size for Boundary Weighted BCE Loss")
    p.add_argument("--boundary-extra-weight", type=float, default=None,
                   help="Extra weight for boundary region in Boundary Weighted BCE Loss")
    p.add_argument("--hard-negative-ratio", type=float, default=None,
                   help="Top ratio of background pixels to mine in Hard Negative BCE Loss")
    p.add_argument("--threshold-sweep", action="store_true", default=None,
                   help="Run evaluation/validation across multiple threshold candidates")
    p.add_argument("--trimap-source", choices=["binary_mask", "file"], default=None)
    p.add_argument("--trimap-kernel-min", type=int, default=None)
    p.add_argument("--trimap-kernel-max", type=int, default=None)
    p.add_argument("--collapse-nonzero-to-foreground", action="store_true", default=None)
    p.add_argument("--ddc-window-size", type=int, default=None)
    p.add_argument("--ddc-num-neighbors", type=int, default=None)
    p.add_argument("--ddc-lambda", type=float, default=None)
    p.add_argument("--ddc-warmup-epochs", type=int, default=None,
                   help="Number of epochs to linearly scale DDC lambda from 0 to target")
    p.add_argument("--ddc-reduction", choices=["paper", "mean_neighbors"], default=None,
                   help="Reduction mode for DDC loss: paper (sum over neighbors) or mean_neighbors")
    p.add_argument("--ddc-chunk-size", type=int, default=None)
    p.add_argument("--ddc-downsample-factor", type=int, default=None)
    p.add_argument("--lambda-coarse-known", type=float, default=None)
    p.add_argument("--lambda-fine-known", type=float, default=None)
    p.add_argument("--matting-crop-height", type=int, default=None)
    p.add_argument("--matting-crop-width", type=int, default=None)
    p.add_argument("--foreground-threshold", type=float, default=None)
    p.add_argument("--scheduler-milestones", type=int, nargs="+", default=None)
    p.add_argument("--scheduler-gamma", type=float, default=None)
    p.add_argument("--vis-interval", type=int, default=None,
                   help="Save validation visualization images every N epochs")
    # Knowledge Distillation (KD) arguments
    p.add_argument("--teacher-weights", type=str, default=None,
                   help="Path to pre-trained UNet teacher weights checkpoint")
    p.add_argument("--kd-alpha", type=float, default=None,
                   help="Weight for KD loss component (between 0.0 and 1.0)")
    p.add_argument("--kd-temperature", type=float, default=None,
                   help="Scaling temperature for distillation logits/probabilities")
    p.add_argument("--kd-loss-type", choices=["mse", "l1", "kl"], default=None,
                   help="Loss type for distillation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Base config from profile
    if args.profile == "paper":
        cfg = get_paper_config()
    elif args.profile == "project":
        cfg = get_project_config()
    elif args.profile == "paper_am2k":
        cfg = get_ddc_am2k_config()
    elif args.profile == "paper_p3m":
        cfg = get_ddc_p3m_config()
    elif args.profile == "tv_ddc":
        cfg = get_ddc_tv_config()
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
    if args.longest_max_size is not None:
        cfg.longest_max_size = args.longest_max_size
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
    if args.no_tqdm:
        cfg.no_tqdm = True

    # DDC Matting specific overrides
    if args.task_mode:
        cfg.task_mode = args.task_mode
    if args.loss_profile:
        cfg.loss_profile = args.loss_profile
    if args.trimap_source:
        cfg.trimap_source = args.trimap_source
    if args.trimap_kernel_min is not None:
        cfg.trimap_kernel_min = args.trimap_kernel_min
    if args.trimap_kernel_max is not None:
        cfg.trimap_kernel_max = args.trimap_kernel_max
    if args.collapse_nonzero_to_foreground is not None:
        cfg.collapse_nonzero_to_foreground = args.collapse_nonzero_to_foreground
    if args.ddc_window_size is not None:
        cfg.ddc_window_size = args.ddc_window_size
    if args.ddc_num_neighbors is not None:
        cfg.ddc_num_neighbors = args.ddc_num_neighbors
    if args.ddc_lambda is not None:
        cfg.ddc_lambda = args.ddc_lambda
    if args.ddc_warmup_epochs is not None:
        cfg.ddc_warmup_epochs = args.ddc_warmup_epochs
    if args.ddc_chunk_size is not None:
        cfg.ddc_chunk_size = args.ddc_chunk_size
    if args.ddc_reduction is not None:
        cfg.ddc_reduction = args.ddc_reduction
    if args.vis_interval is not None:
        cfg.vis_interval = args.vis_interval
    if args.ddc_downsample_factor is not None:
        cfg.ddc_downsample_factor = args.ddc_downsample_factor
    if args.lambda_coarse_known is not None:
        cfg.lambda_coarse_known = args.lambda_coarse_known
    if args.lambda_fine_known is not None:
        cfg.lambda_fine_known = args.lambda_fine_known
    if args.matting_crop_height is not None:
        cfg.matting_crop_height = args.matting_crop_height
    if args.matting_crop_width is not None:
        cfg.matting_crop_width = args.matting_crop_width
    if args.foreground_threshold is not None:
        cfg.foreground_threshold = args.foreground_threshold
    if args.scheduler_milestones is not None:
        cfg.scheduler_milestones = args.scheduler_milestones
    if args.scheduler_gamma is not None:
        cfg.scheduler_gamma = args.scheduler_gamma
    if args.refinement_head is not None:
        cfg.refinement_head = args.refinement_head
    if args.prompt_gate_mode is not None:
        cfg.prompt_gate_mode = args.prompt_gate_mode
    if args.prompt_gate_strength is not None:
        cfg.prompt_gate_strength = args.prompt_gate_strength
    if args.tversky_fp_weight is not None:
        cfg.tversky_fp_weight = args.tversky_fp_weight
    if args.tversky_fn_weight is not None:
        cfg.tversky_fn_weight = args.tversky_fn_weight
    if args.boundary_kernel_size is not None:
        cfg.boundary_kernel_size = args.boundary_kernel_size
    if args.boundary_extra_weight is not None:
        cfg.boundary_extra_weight = args.boundary_extra_weight
    if args.hard_negative_ratio is not None:
        cfg.hard_negative_ratio = args.hard_negative_ratio
    if args.threshold_sweep is not None:
        cfg.threshold_sweep = args.threshold_sweep
    if args.teacher_weights is not None:
        cfg.teacher_weights = args.teacher_weights
    if args.kd_alpha is not None:
        cfg.kd_alpha = args.kd_alpha
    if args.kd_temperature is not None:
        cfg.kd_temperature = args.kd_temperature
    if args.kd_loss_type is not None:
        cfg.kd_loss_type = args.kd_loss_type

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
