"""
Segmentation metrics with confusion-matrix accumulation.

Metrics are accumulated across batches via an integer confusion matrix stored
on CPU to avoid unnecessary GPU memory growth.

Absent-class strategy
---------------------
- A class absent from *both* predictions and ground truth yields IoU = NaN.
- NaN classes are **excluded** when computing mIoU / mDice.
- This is clearly documented and tested.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


class SegmentationMetrics:
    """Accumulate a confusion matrix and compute standard metrics.

    Parameters
    ----------
    num_classes : int
        Number of classes (including background).
    ignore_index : int
        Label value to exclude from metric computation.
    """

    def __init__(self, num_classes: int = 2, ignore_index: int = 255) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # Confusion matrix on CPU (int64)
        self.confusion_matrix = torch.zeros(
            num_classes, num_classes, dtype=torch.long
        )

    def reset(self) -> None:
        """Reset the accumulated confusion matrix."""
        self.confusion_matrix.zero_()

    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Update confusion matrix with a batch.

        Parameters
        ----------
        predictions : [B, C, H, W] logits or [B, H, W] class indices
        targets : [B, H, W] long class indices
        """
        # If logits, convert to class indices
        if predictions.ndim == 4:
            predictions = predictions.argmax(dim=1)

        predictions = predictions.detach().cpu().long()
        targets = targets.detach().cpu().long()

        # Mask out ignore pixels
        valid = targets != self.ignore_index
        preds = predictions[valid]
        tgts = targets[valid]

        if preds.numel() == 0:
            return  # nothing to accumulate

        # Flat index into confusion matrix
        indices = tgts * self.num_classes + preds
        cm_flat = torch.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += cm_flat.reshape(self.num_classes, self.num_classes)

    def all_reduce(self) -> None:
        """Sum the confusion matrix across all DDP processes in place."""
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            # Get active device
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
            cm_gpu = self.confusion_matrix.to(device)
            dist.all_reduce(cm_gpu, op=dist.ReduceOp.SUM)
            self.confusion_matrix = cm_gpu.cpu()

    def compute(self) -> Dict[str, object]:
        """Compute all metrics from the accumulated confusion matrix.

        Returns
        -------
        dict with keys:
            pixel_accuracy : float
            per_class_iou : list[float]  (NaN for absent classes)
            miou : float  (mean of valid classes)
            foreground_iou : float
            per_class_dice : list[float]
            mean_dice : float
            foreground_dice : float
            confusion_matrix : list[list[int]]
        """
        cm = self.confusion_matrix.float()
        total_correct = cm.trace()
        total_pixels = cm.sum()

        # Pixel accuracy
        if total_pixels == 0:
            pixel_accuracy = 0.0
        else:
            pixel_accuracy = (total_correct / total_pixels).item()

        # Per-class IoU
        per_class_iou = []
        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            denom = tp + fp + fn
            if denom == 0:
                per_class_iou.append(float("nan"))
            else:
                per_class_iou.append((tp / denom).item())

        # mIoU (only over valid / present classes)
        valid_iou = [v for v in per_class_iou if v == v]  # filter NaN
        miou = sum(valid_iou) / len(valid_iou) if valid_iou else 0.0

        # Foreground IoU (class 1)
        fg_iou = per_class_iou[1] if self.num_classes > 1 else float("nan")

        # Per-class Dice
        per_class_dice = []
        for c in range(self.num_classes):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            denom = 2 * tp + fp + fn
            if denom == 0:
                per_class_dice.append(float("nan"))
            else:
                per_class_dice.append((2 * tp / denom).item())

        valid_dice = [v for v in per_class_dice if v == v]
        mean_dice = sum(valid_dice) / len(valid_dice) if valid_dice else 0.0
        fg_dice = per_class_dice[1] if self.num_classes > 1 else float("nan")

        return {
            "pixel_accuracy": pixel_accuracy,
            "per_class_iou": per_class_iou,
            "miou": miou,
            "foreground_iou": fg_iou,
            "per_class_dice": per_class_dice,
            "mean_dice": mean_dice,
            "foreground_dice": fg_dice,
            "confusion_matrix": self.confusion_matrix.tolist(),
        }
