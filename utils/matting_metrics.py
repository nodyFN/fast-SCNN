"""
Alpha matting evaluation metrics.

Metrics
-------
- SAD (Sum of Absolute Differences)
- MAD (Mean Absolute Difference)
- MSE (Mean Squared Error)
- Gradient Error (Sobel-based gradient comparison)
- SAD-T / MSE-T (Transition-region SAD/MSE, computed only where trimap == 0.5)

For binary-mask-only validation:
- Thresholded alpha → IoU, Dice (labeled as "binary-reference")
- Binary-reference MAE

Interface matches SegmentationMetrics: reset(), update(), all_reduce(), compute().
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MattingMetrics:
    """Accumulate and compute alpha matting evaluation metrics.

    Parameters
    ----------
    foreground_threshold : float
        Threshold for converting continuous alpha to binary mask
        when computing segmentation-style metrics (IoU, Dice).
    has_alpha_gt : bool
        If True, expect ground-truth alpha matte for SAD/MAD/MSE.
        If False, expect binary mask and compute binary-reference metrics.
    """

    def __init__(
        self,
        foreground_threshold: float = 0.5,
        has_alpha_gt: bool = False,
    ) -> None:
        self.foreground_threshold = foreground_threshold
        self.has_alpha_gt = has_alpha_gt
        self.reset()

    def reset(self) -> None:
        """Reset all accumulated values."""
        self._sad = 0.0
        self._mse_sum = 0.0
        self._pixel_count = 0
        self._grad_error = 0.0
        self._grad_count = 0

        # Transition metrics (only where trimap == 0.5)
        self._sad_t = 0.0
        self._mse_t_sum = 0.0
        self._pixel_count_t = 0

        # Binary segmentation metrics (for binary-mask validation)
        self._tp = 0
        self._fp = 0
        self._fn = 0
        self._tn = 0

        self._batch_count = 0

    def update(
        self,
        pred_alpha: torch.Tensor,
        gt: torch.Tensor,
        trimap: Optional[torch.Tensor] = None,
    ) -> None:
        """Update metrics with a batch.

        Parameters
        ----------
        pred_alpha : [B, 1, H, W]
            Predicted alpha matte (values in [0, 1]).
        gt : [B, 1, H, W]
            Ground truth alpha (continuous [0,1]) or binary mask ({0,1}).
        trimap : [B, 1, H, W], optional
            Trimap for transition-region metrics. If None, transition
            metrics are skipped.
        """
        pred = pred_alpha.detach().float().clamp(0, 1)
        target = gt.detach().float()

        # Flatten spatial dims
        B = pred.shape[0]

        # SAD and MSE (full image)
        abs_diff = torch.abs(pred - target)
        self._sad += abs_diff.sum().item()
        self._mse_sum += (abs_diff ** 2).sum().item()
        self._pixel_count += pred.numel()

        # Gradient error (Sobel-based)
        grad_err = self._compute_gradient_error(pred, target)
        self._grad_error += grad_err.sum().item()
        self._grad_count += grad_err.numel()

        # Transition-region metrics
        if trimap is not None:
            trimap_f = trimap.detach().float()
            trans_mask = ((trimap_f > 0.4) & (trimap_f < 0.6)).float()
            n_trans = trans_mask.sum().item()
            if n_trans > 0:
                self._sad_t += (abs_diff * trans_mask).sum().item()
                self._mse_t_sum += ((abs_diff ** 2) * trans_mask).sum().item()
                self._pixel_count_t += int(n_trans)

        # Binary segmentation metrics (thresholded)
        pred_binary = (pred >= self.foreground_threshold).long()
        target_binary = (target >= self.foreground_threshold).long()

        self._tp += ((pred_binary == 1) & (target_binary == 1)).sum().item()
        self._fp += ((pred_binary == 1) & (target_binary == 0)).sum().item()
        self._fn += ((pred_binary == 0) & (target_binary == 1)).sum().item()
        self._tn += ((pred_binary == 0) & (target_binary == 0)).sum().item()

        self._batch_count += B

    def _compute_gradient_error(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Sobel-based gradient error between pred and target alpha."""
        # Sobel kernels
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=pred.dtype, device=pred.device,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=pred.dtype, device=pred.device,
        ).view(1, 1, 3, 3)

        # Pred gradients
        pred_gx = F.conv2d(pred, sobel_x, padding=1)
        pred_gy = F.conv2d(pred, sobel_y, padding=1)

        # Target gradients
        target_gx = F.conv2d(target, sobel_x, padding=1)
        target_gy = F.conv2d(target, sobel_y, padding=1)

        # Gradient magnitude difference
        grad_diff_x = pred_gx - target_gx
        grad_diff_y = pred_gy - target_gy
        grad_error = torch.sqrt(grad_diff_x ** 2 + grad_diff_y ** 2 + 1e-8)

        return grad_error

    def all_reduce(self) -> None:
        """Sum metrics across all DDP processes."""
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            return

        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        values = torch.tensor(
            [
                self._sad, self._mse_sum, self._pixel_count,
                self._grad_error, self._grad_count,
                self._sad_t, self._mse_t_sum, self._pixel_count_t,
                self._tp, self._fp, self._fn, self._tn,
                self._batch_count,
            ],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
        values = values.cpu()

        (
            self._sad, self._mse_sum, self._pixel_count,
            self._grad_error, self._grad_count,
            self._sad_t, self._mse_t_sum, self._pixel_count_t,
            self._tp, self._fp, self._fn, self._tn,
            self._batch_count,
        ) = [v.item() for v in values]

    def compute(self) -> Dict[str, float]:
        """Compute all metrics from accumulated values.

        Returns
        -------
        dict with keys:
            sad, mad, mse : float (full-image alpha metrics)
            gradient_error : float
            sad_t, mse_t : float (transition-region metrics, NaN if no transition pixels)
            foreground_iou, foreground_dice : float (binary-thresholded metrics)
            pixel_accuracy : float
            miou : float (mean of BG and FG IoU)
        """
        n = max(self._pixel_count, 1)
        n_t = max(self._pixel_count_t, 1)
        n_grad = max(self._grad_count, 1)

        sad = self._sad
        mad = self._sad / n
        mse = self._mse_sum / n
        gradient_error = self._grad_error / n_grad

        # Transition metrics
        if self._pixel_count_t > 0:
            sad_t = self._sad_t
            mse_t = self._mse_t_sum / self._pixel_count_t
        else:
            sad_t = float("nan")
            mse_t = float("nan")

        # Binary segmentation metrics
        tp, fp, fn, tn = self._tp, self._fp, self._fn, self._tn

        pixel_accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)

        fg_iou = tp / max(tp + fp + fn, 1)
        bg_iou = tn / max(tn + fn + fp, 1)
        miou = (fg_iou + bg_iou) / 2.0

        fg_dice = (2 * tp) / max(2 * tp + fp + fn, 1)
        bg_dice = (2 * tn) / max(2 * tn + fn + fp, 1)

        return {
            "sad": sad,
            "mad": mad,
            "mse": mse,
            "gradient_error": gradient_error,
            "sad_t": sad_t,
            "mse_t": mse_t,
            "pixel_accuracy": pixel_accuracy,
            "miou": miou,
            "foreground_iou": fg_iou,
            "foreground_dice": fg_dice,
            "per_class_iou": [bg_iou, fg_iou],
            "per_class_dice": [bg_dice, fg_dice],
        }
