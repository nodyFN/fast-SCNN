"""
Centralized configuration for Fast-SCNN semantic segmentation and matting project.

This module defines all tunable parameters in one place. CLI arguments in
train.py, evaluate.py, inference.py and export.py can override these defaults.

Terminology
-----------
- **Paper setting**: values explicitly stated in the Fast-SCNN paper.
- **Project decision**: values chosen for this project that are NOT specified
  by the paper (marked with [PROJECT DECISION] in comments).
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project root (the directory that contains this file)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent


@dataclass
class Config:
    """All-in-one configuration container."""

    # ── Training profile ──────────────────────────────────────────────
    # "paper" → reproduce paper settings; "project" → project-tuned defaults
    profile: str = "project"
    model: str = "fast_scnn"  # "fast_scnn" | "fast_scnn_salient"

    # ── Data paths ────────────────────────────────────────────────────
    data_root: Path = PROJECT_ROOT / "data"
    train_dir: Path = PROJECT_ROOT / "data" / "train"
    val_dir: Path = PROJECT_ROOT / "data" / "val"
    test_dir: Path = PROJECT_ROOT / "data" / "test"

    # ── Class settings ────────────────────────────────────────────────
    num_classes: int = 2
    class_names: List[str] = field(default_factory=lambda: ["background", "foreground"])
    ignore_index: int = 255  # standard ignore label for CE loss

    # ── Input / crop sizes ────────────────────────────────────────────
    # Original image resolution (for reference / inference output)
    original_height: int = 1080
    original_width: int = 1920

    # Training crop size — [PROJECT DECISION] smaller than original to save VRAM
    train_height: int = 512
    train_width: int = 1024

    # Validation resize — [PROJECT DECISION]
    val_height: int = 512
    val_width: int = 1024

    # ── DataLoader ────────────────────────────────────────────────────
    batch_size: int = 4
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    # ── Optimizer (paper: SGD; project: AdamW) ────────────────────────
    optimizer: str = "adamw"  # "sgd" | "adamw"
    learning_rate: float = 1e-3
    momentum: float = 0.9  # SGD only (paper: 0.9)
    weight_decay: float = 1e-4
    # Paper setting: depthwise conv layers use weight_decay = 0
    depthwise_weight_decay: float = 0.0

    # ── Scheduler ─────────────────────────────────────────────────────
    scheduler: str = "poly"  # "poly" | "cosine"
    poly_power: float = 0.9  # Paper: 0.9
    cosine_eta_min: float = 1e-6  # [PROJECT DECISION]

    # ── Training ──────────────────────────────────────────────────────
    epochs: int = 200
    amp: bool = True  # Only effective when CUDA is available
    no_tqdm: bool = False  # Disable tqdm progress bars
    gradient_clip_max_norm: float = 1.0  # [PROJECT DECISION]
    gradient_clip_enabled: bool = True  # [PROJECT DECISION]

    # ── Auxiliary outputs (paper: both weights = 0.4) ─────────────────
    aux: bool = True
    aux_downsample_weight: float = 0.4  # Paper: 0.4
    aux_global_weight: float = 0.4  # Paper: 0.4

    # ── Loss ──────────────────────────────────────────────────────────
    ce_weight: float = 1.0
    dice_weight: float = 1.0  # [PROJECT DECISION] Combined CE + Dice
    focal_weight: float = 0.0  # [PROJECT DECISION] disabled by default
    focal_alpha: float = 0.25  # [PROJECT DECISION]
    focal_gamma: float = 2.0  # [PROJECT DECISION]
    class_weights: Optional[List[float]] = None  # e.g. [0.3, 0.7]

    # ── Model ─────────────────────────────────────────────────────────
    # PPM pool sizes — [PROJECT DECISION] paper does not specify for Fast-SCNN
    ppm_pool_sizes: Tuple[int, ...] = (1, 2, 3, 6)
    # Dropout in classifier — [PROJECT DECISION] paper does not specify exact p
    dropout_p: float = 0.1

    # ── Directories ───────────────────────────────────────────────────
    checkpoint_dir: Path = PROJECT_ROOT / "checkpoints"
    training_image_dir: Path = PROJECT_ROOT / "training_results"
    tensorboard_dir: Path = PROJECT_ROOT / "runs"
    export_dir: Path = PROJECT_ROOT / "exports"

    # ── Reproducibility ───────────────────────────────────────────────
    seed: int = 42
    deterministic: bool = False  # True may slow down training

    # ── Device ────────────────────────────────────────────────────────
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    allow_threshold: bool = False  # Set to True to allow thresholding grayscale masks

    # ── Checkpoint resume ─────────────────────────────────────────────
    resume: Optional[str] = None  # path to checkpoint to resume from
    weights: Optional[str] = None  # path to pre-trained weights for transfer learning (weights-only)
    freeze_backbone: bool = False  # Set to True to freeze backbone during transfer learning

    # ── Early stopping ────────────────────────────────────────────────
    early_stopping_patience: int = 50  # [PROJECT DECISION] 0 = disabled
    early_stopping_enabled: bool = False  # [PROJECT DECISION]

    # ── ONNX export ───────────────────────────────────────────────────
    onnx_opset: int = 17  # [PROJECT DECISION] good PyTorch/TensorRT compat
    onnx_dynamic_axes: bool = True

    # ── Visualization ─────────────────────────────────────────────────
    num_vis_samples: int = 4
    vis_interval: int = 1  # Save validation visualization images every N epochs

    # ── Augmentation (paper-compatible range) ─────────────────────────
    aug_scale_min: float = 0.5  # Paper: 0.5
    aug_scale_max: float = 2.0  # Paper: 2.0
    longest_max_size: Optional[int] = None  # Pre-resize high-res images before scaling/cropping

    # ── Salient Dual-Head Model ──────────────────────────────────────
    coarse_channels: int = 64
    refinement_channels: int = 64
    salient_coarse_bce_weight: float = 1.0
    salient_coarse_dice_weight: float = 1.0
    salient_fine_focal_weight: float = 1.0
    salient_fine_dice_weight: float = 1.0
    salient_boundary_weight: float = 1.0
    salient_lambda_coarse: float = 1.0
    salient_lambda_fine: float = 1.0
    salient_lambda_boundary: float = 0.5
    salient_focal_alpha: float = 0.25
    salient_focal_gamma: float = 2.0
    salient_pos_weight: Optional[float] = None

    # ── DDC Alpha-Free Matting ────────────────────────────────────────
    # Based on: "Training Matting Models without Alpha Labels"
    task_mode: str = "segmentation"   # "segmentation" | "ddc_matting"
    loss_profile: str = "legacy"      # "legacy" | "legacy_salient" | "ddc_matting"

    # Trimap
    trimap_source: str = "binary_mask"  # "binary_mask" | "file"
    trimap_kernel_min: int = 1
    trimap_kernel_max: int = 30
    collapse_nonzero_to_foreground: bool = False

    # DDC Loss
    ddc_window_size: int = 11
    ddc_num_neighbors: int = 11
    ddc_lambda: float = 10.0
    ddc_warmup_epochs: int = 5  # Warmup epochs for DDC loss to prevent initial weight collapse
    ddc_chunk_size: int = 4096
    ddc_padding_mode: str = "replicate"
    ddc_exclude_center: bool = True
    ddc_reduction: str = "mean_neighbors"
    ddc_downsample_factor: int = 1

    # DDC Matting Loss weights
    lambda_coarse_known: float = 1.0
    lambda_fine_known: float = 1.0

    # Matting crop (paper: 512×512)
    matting_crop_height: int = 512
    matting_crop_width: int = 512

    # Matting evaluation
    foreground_threshold: float = 0.5

    # Scheduler milestones (for MultiStepLR, used by DDC paper profiles)
    scheduler_milestones: Optional[List[int]] = None
    scheduler_gamma: float = 0.1

    def resolve_device(self) -> torch.device:
        """Return the torch.device to use."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def ensure_dirs(self) -> None:
        """Create output directories if they do not exist."""
        for d in [
            self.checkpoint_dir,
            self.training_image_dir,
            self.tensorboard_dir,
            self.export_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


def get_paper_config() -> Config:
    """Return a Config that reproduces the paper's training settings."""
    return Config(
        profile="paper",
        optimizer="sgd",
        learning_rate=0.045,
        momentum=0.9,
        weight_decay=0.00004,
        depthwise_weight_decay=0.0,
        scheduler="poly",
        poly_power=0.9,
        epochs=1000,
        batch_size=12,
        ce_weight=1.0,
        dice_weight=0.0,  # Paper uses CE only
        focal_weight=0.0,
        aux=True,
        aux_downsample_weight=0.4,
        aux_global_weight=0.4,
        amp=False,  # Paper does not mention AMP
        gradient_clip_enabled=False,
        early_stopping_enabled=False,
    )


def get_project_config() -> Config:
    """Return a Config tuned for this project's dataset / workflow."""
    return Config(
        profile="project",
        optimizer="adamw",
        learning_rate=1e-3,
        weight_decay=1e-4,
        depthwise_weight_decay=0.0,
        scheduler="poly",
        poly_power=0.9,
        epochs=200,
        batch_size=4,
        ce_weight=1.0,
        dice_weight=1.0,
        focal_weight=0.0,
        aux=True,
        aux_downsample_weight=0.4,
        aux_global_weight=0.4,
        amp=True,
        gradient_clip_enabled=True,
        gradient_clip_max_norm=1.0,
        early_stopping_enabled=True,
        early_stopping_patience=50,
    )


def get_ddc_am2k_config() -> Config:
    """DDC matting config reproducing paper AM-2K animal matting settings."""
    return Config(
        profile="paper_am2k",
        task_mode="ddc_matting",
        loss_profile="ddc_matting",
        model="fast_scnn_salient",
        optimizer="adamw",
        learning_rate=5e-4,
        weight_decay=0.1,
        scheduler="multistep",
        scheduler_milestones=[60, 90],
        scheduler_gamma=0.1,
        epochs=100,
        batch_size=16,
        train_height=512,
        train_width=512,
        matting_crop_height=512,
        matting_crop_width=512,
        ddc_window_size=11,
        ddc_num_neighbors=11,
        ddc_lambda=10.0,
        lambda_coarse_known=1.0,
        lambda_fine_known=1.0,
        amp=True,
        gradient_clip_enabled=True,
        gradient_clip_max_norm=1.0,
        aux=False,  # Disable aux heads for matting
        early_stopping_enabled=False,
    )


def get_ddc_p3m_config() -> Config:
    """DDC matting config reproducing paper P3M-10K portrait matting settings."""
    return Config(
        profile="paper_p3m",
        task_mode="ddc_matting",
        loss_profile="ddc_matting",
        model="fast_scnn_salient",
        optimizer="adamw",
        learning_rate=5e-4,
        weight_decay=0.1,
        scheduler="multistep",
        scheduler_milestones=[300, 450],
        scheduler_gamma=0.1,
        epochs=500,
        batch_size=16,
        train_height=512,
        train_width=512,
        matting_crop_height=512,
        matting_crop_width=512,
        ddc_window_size=11,
        ddc_num_neighbors=11,
        ddc_lambda=10.0,
        lambda_coarse_known=1.0,
        lambda_fine_known=1.0,
        amp=True,
        gradient_clip_enabled=True,
        gradient_clip_max_norm=1.0,
        aux=False,
        early_stopping_enabled=False,
    )


def get_ddc_tv_config() -> Config:
    """DDC matting config tuned for this project's TV broadcast dataset.

    [PROJECT DECISION] Crop, batch, epoch, and DDC params may differ from paper.
    """
    return Config(
        profile="tv_ddc",
        task_mode="ddc_matting",
        loss_profile="ddc_matting",
        model="fast_scnn_salient",
        optimizer="adamw",
        learning_rate=5e-4,
        weight_decay=0.01,
        scheduler="poly",
        poly_power=0.9,
        epochs=200,
        batch_size=8,
        train_height=512,
        train_width=512,
        matting_crop_height=512,
        matting_crop_width=512,
        ddc_window_size=11,
        ddc_num_neighbors=11,
        ddc_lambda=10.0,
        ddc_chunk_size=4096,
        lambda_coarse_known=1.0,
        lambda_fine_known=1.0,
        amp=True,
        gradient_clip_enabled=True,
        gradient_clip_max_norm=1.0,
        aux=False,
        early_stopping_enabled=True,
        early_stopping_patience=30,
    )
