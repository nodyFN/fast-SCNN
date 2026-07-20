"""
Centralized configuration for Fast-SCNN semantic segmentation project.

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

    # ── Early stopping ────────────────────────────────────────────────
    early_stopping_patience: int = 50  # [PROJECT DECISION] 0 = disabled
    early_stopping_enabled: bool = False  # [PROJECT DECISION]

    # ── ONNX export ───────────────────────────────────────────────────
    onnx_opset: int = 17  # [PROJECT DECISION] good PyTorch/TensorRT compat
    onnx_dynamic_axes: bool = True

    # ── Visualization ─────────────────────────────────────────────────
    num_vis_samples: int = 4

    # ── Augmentation (paper-compatible range) ─────────────────────────
    aug_scale_min: float = 0.5  # Paper: 0.5
    aug_scale_max: float = 2.0  # Paper: 2.0

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
