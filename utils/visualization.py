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


# ===========================================================================
# Matting visualization
# ===========================================================================


def visualize_matting(
    images: torch.Tensor,
    trimaps: torch.Tensor,
    coarse_alpha: torch.Tensor,
    fine_alpha: torch.Tensor,
    ddc_images: Optional[torch.Tensor] = None,
    gt_alpha: Optional[torch.Tensor] = None,
    save_path: Optional[Path | str] = None,
    num_samples: int = 4,
    threshold: float = 0.5,
) -> None:
    """Visualize matting training results.

    Parameters
    ----------
    images : [B, C, H, W]  ImageNet-normalized
    trimaps : [B, 1, H, W]  {0.0, 0.5, 1.0}
    coarse_alpha : [B, 1, H, W]  [0, 1]
    fine_alpha : [B, 1, H, W]  [0, 1]
    ddc_images : [B, 3, H, W]  raw RGB [0, 1], optional
    gt_alpha : [B, 1, H, W]  ground truth alpha, optional
    """
    n = min(num_samples, images.shape[0])
    has_gt = gt_alpha is not None
    has_ddc = ddc_images is not None
    ncols = 6 + (1 if has_ddc else 0) + (2 if has_gt else 0)

    fig, axes = plt.subplots(n, ncols, figsize=(ncols * 3, n * 3))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row in range(n):
        col = 0
        img = denormalize(images[row].cpu().numpy())

        # 1. Original RGB
        axes[row, col].imshow(img)
        if row == 0:
            axes[row, col].set_title("RGB", fontsize=8)
        axes[row, col].axis("off")
        col += 1

        # 2. Trimap (color-coded)
        tri = trimaps[row, 0].cpu().numpy()
        tri_rgb = np.zeros((*tri.shape, 3), dtype=np.float32)
        tri_rgb[tri > 0.75] = [1, 1, 1]  # FG = white
        tri_rgb[(tri > 0.25) & (tri < 0.75)] = [0.5, 0.5, 0.5]  # Unknown = gray
        # BG stays black
        axes[row, col].imshow(tri_rgb)
        if row == 0:
            axes[row, col].set_title("Trimap", fontsize=8)
        axes[row, col].axis("off")
        col += 1

        # 3. Coarse Alpha
        ca = coarse_alpha[row, 0].cpu().numpy()
        axes[row, col].imshow(ca, cmap="gray", vmin=0, vmax=1)
        if row == 0:
            axes[row, col].set_title("Coarse α", fontsize=8)
        axes[row, col].axis("off")
        col += 1

        # 4. Fine Alpha
        fa = fine_alpha[row, 0].cpu().numpy()
        axes[row, col].imshow(fa, cmap="gray", vmin=0, vmax=1)
        if row == 0:
            axes[row, col].set_title("Fine α", fontsize=8)
        axes[row, col].axis("off")
        col += 1

        # 5. Fine Binary Preview
        fb = (fa >= threshold).astype(np.float32)
        axes[row, col].imshow(fb, cmap="gray", vmin=0, vmax=1)
        if row == 0:
            axes[row, col].set_title(f"Binary (≥{threshold})", fontsize=8)
        axes[row, col].axis("off")
        col += 1

        # 6. Fine Alpha Overlay
        overlay = img * fa[..., np.newaxis]  # alpha-blended
        axes[row, col].imshow(overlay.clip(0, 1))
        if row == 0:
            axes[row, col].set_title("α Overlay", fontsize=8)
        axes[row, col].axis("off")
        col += 1

        # 7. DDC RGB (if available)
        if has_ddc:
            ddc_rgb = ddc_images[row].cpu().numpy().transpose(1, 2, 0)
            axes[row, col].imshow(ddc_rgb.clip(0, 1))
            if row == 0:
                axes[row, col].set_title("DDC RGB", fontsize=8)
            axes[row, col].axis("off")
            col += 1

        # 8-9. GT Alpha + Error (if available)
        if has_gt:
            gta = gt_alpha[row, 0].cpu().numpy()
            axes[row, col].imshow(gta, cmap="gray", vmin=0, vmax=1)
            if row == 0:
                axes[row, col].set_title("GT α", fontsize=8)
            axes[row, col].axis("off")
            col += 1

            err = np.abs(fa - gta)
            axes[row, col].imshow(err, cmap="hot", vmin=0, vmax=1)
            if row == 0:
                axes[row, col].set_title("|Error|", fontsize=8)
            axes[row, col].axis("off")

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)

