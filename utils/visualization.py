"""
Visualization utilities.

- Training curve plots (loss, metrics, LR) saved to ``training_images/``.
- Segmentation result visualization (image, GT, prediction, overlay).
- Uses non-interactive Matplotlib backend (Agg) for headless servers.
- Closes all figures after saving to prevent memory leaks.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be before pyplot import

import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Sequence


# ImageNet de-normalisation
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denormalize(image: np.ndarray) -> np.ndarray:
    """Reverse ImageNet normalisation.  Input: [C, H, W] or [H, W, C]."""
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.transpose(1, 2, 0)
    img = image * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


# ===========================================================================
# Training curves
# ===========================================================================


def plot_loss_curves(
    history: Dict[str, List[float]],
    save_dir: Path | str,
) -> None:
    """Plot train/val loss curves."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    if "train_loss" in history:
        ax.plot(history["train_loss"], label="Train Loss")
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "loss_curve.png", dpi=150)
    plt.close(fig)


def plot_metrics_curves(
    history: Dict[str, List[float]],
    save_dir: Path | str,
) -> None:
    """Plot metrics curves: PA, mIoU, FG IoU, FG Dice."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    keys_labels = [
        ("pixel_accuracy", "Pixel Accuracy"),
        ("miou", "mIoU"),
        ("foreground_iou", "Foreground IoU"),
        ("foreground_dice", "Foreground Dice"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    for key, label in keys_labels:
        if key in history:
            ax.plot(history[key], label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Validation Metrics")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(save_dir / "metrics_curve.png", dpi=150)
    plt.close(fig)


def plot_lr_curve(
    history: Dict[str, List[float]],
    save_dir: Path | str,
) -> None:
    """Plot learning rate schedule."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    if "learning_rate" not in history:
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(history["learning_rate"])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "learning_rate_curve.png", dpi=150)
    plt.close(fig)


def save_all_curves(
    history: Dict[str, List[float]],
    save_dir: Path | str,
) -> None:
    """Save all training curve plots."""
    plot_loss_curves(history, save_dir)
    plot_metrics_curves(history, save_dir)
    plot_lr_curve(history, save_dir)


# ===========================================================================
# Segmentation visualization
# ===========================================================================


def visualize_segmentation(
    images: torch.Tensor,
    masks_gt: torch.Tensor,
    masks_pred: torch.Tensor,
    probs_fg: Optional[torch.Tensor] = None,
    save_path: Optional[Path | str] = None,
    num_samples: int = 4,
    class_colors: Optional[Dict[int, tuple]] = None,
) -> None:
    """Visualize segmentation results.

    Parameters
    ----------
    images : [B, C, H, W] normalised float
    masks_gt : [B, H, W] long
    masks_pred : [B, H, W] long
    probs_fg : [B, H, W] float, optional foreground probability map
    save_path : where to save the figure
    num_samples : max number of samples to plot
    """
    if class_colors is None:
        class_colors = {0: (0, 0, 0), 1: (0, 255, 0)}  # black BG, green FG

    n = min(num_samples, images.shape[0])
    ncols = 5 if probs_fg is not None else 4
    fig, axes = plt.subplots(n, ncols, figsize=(ncols * 4, n * 3.5))
    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Image", "Ground Truth", "Prediction", "Overlay"]
    if probs_fg is not None:
        col_titles.insert(3, "FG Probability")

    for row in range(n):
        # Denormalize image
        img = denormalize(images[row].cpu().numpy())
        gt = masks_gt[row].cpu().numpy()
        pred = masks_pred[row].cpu().numpy()

        # Color masks
        gt_color = _colorize_mask(gt, class_colors)
        pred_color = _colorize_mask(pred, class_colors)

        # Overlay
        overlay = (img * 0.6 + pred_color / 255.0 * 0.4).clip(0, 1)

        col = 0
        axes[row, col].imshow(img)
        axes[row, col].set_title(col_titles[col] if row == 0 else "")
        axes[row, col].axis("off")
        col += 1

        axes[row, col].imshow(gt_color)
        axes[row, col].set_title(col_titles[col] if row == 0 else "")
        axes[row, col].axis("off")
        col += 1

        axes[row, col].imshow(pred_color)
        axes[row, col].set_title(col_titles[col] if row == 0 else "")
        axes[row, col].axis("off")
        col += 1

        if probs_fg is not None:
            prob = probs_fg[row].cpu().numpy()
            axes[row, col].imshow(prob, cmap="hot", vmin=0, vmax=1)
            axes[row, col].set_title(col_titles[col] if row == 0 else "")
            axes[row, col].axis("off")
            col += 1

        axes[row, col].imshow(overlay)
        axes[row, col].set_title(col_titles[col] if row == 0 else "")
        axes[row, col].axis("off")

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _colorize_mask(mask: np.ndarray, colors: Dict[int, tuple]) -> np.ndarray:
    """Convert single-channel class mask to RGB."""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in colors.items():
        rgb[mask == cls_id] = color
    return rgb
