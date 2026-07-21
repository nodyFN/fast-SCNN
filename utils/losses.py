"""
Loss functions for binary / multiclass semantic segmentation.

This project uses **2-channel logits + CrossEntropyLoss** (NOT 1-channel BCE).

Components
----------
- CrossEntropyLoss wrapper (with class weights, ignore_index)
- Multiclass Dice Loss (softmax-based, not argmax-based)
- Focal Loss (optional)
- CombinedSegmentationLoss (CE + Dice + optional Focal)
- compute_total_loss (main + two auxiliary losses)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropySegLoss(nn.Module):
    """Cross-entropy loss wrapper.

    Parameters
    ----------
    weight : list of float, optional
        Per-class weights (moved to logits device automatically).
    ignore_index : int
        Label value to ignore (default 255).
    """

    def __init__(
        self,
        weight: Optional[List[float]] = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.register_buffer(
            "weight",
            torch.tensor(weight, dtype=torch.float32) if weight else None,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : [B, C, H, W] raw logits
        targets : [B, H, W] long class indices
        """
        weight = self.weight
        if weight is not None:
            weight = weight.to(logits.device)
        return F.cross_entropy(
            logits, targets, weight=weight, ignore_index=self.ignore_index
        )


class DiceLoss(nn.Module):
    """Multiclass soft Dice loss computed on softmax probabilities.

    Dice is computed per class then averaged (optionally weighted).
    Pixels with ``ignore_index`` label are excluded.

    Parameters
    ----------
    smooth : float
        Smoothing constant to avoid division by zero.
    ignore_index : int
        Label value to ignore.
    weight : list of float, optional
        Per-class weights for the averaging step.
    """

    def __init__(
        self,
        smooth: float = 1.0,
        ignore_index: int = 255,
        weight: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.register_buffer(
            "weight",
            torch.tensor(weight, dtype=torch.float32) if weight else None,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : [B, C, H, W]
        targets : [B, H, W] long
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)  # [B, C, H, W]

        # Build valid mask
        valid = targets != self.ignore_index  # [B, H, W]

        # Safe one-hot: replace ignore pixels with 0 for scatter
        safe_targets = targets.clone()
        safe_targets[~valid] = 0
        one_hot = torch.zeros_like(probs)  # [B, C, H, W]
        one_hot.scatter_(1, safe_targets.unsqueeze(1), 1)

        # Zero out ignore positions in both one_hot and probs
        valid_mask = valid.unsqueeze(1).float()  # [B, 1, H, W]
        one_hot = one_hot * valid_mask
        probs = probs * valid_mask

        # Per-class Dice
        dims = (0, 2, 3)  # sum over batch, H, W
        intersection = (probs * one_hot).sum(dim=dims)
        cardinality = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Weighted average
        weight = self.weight
        if weight is not None:
            weight = weight.to(dice.device)
            dice_loss = 1.0 - (dice * weight).sum() / weight.sum()
        else:
            dice_loss = 1.0 - dice.mean()

        return dice_loss


class FocalLoss(nn.Module):
    """Focal Loss for multiclass segmentation.

    [PROJECT DECISION] Optional, disabled by default (focal_weight=0).

    Parameters
    ----------
    alpha : float
        Balancing factor.
    gamma : float
        Focusing parameter.
    ignore_index : int
        Label value to ignore.
    weight : list of float, optional
        Per-class weights.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = 255,
        weight: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.register_buffer(
            "weight",
            torch.tensor(weight, dtype=torch.float32) if weight else None,
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits,
            targets,
            weight=self.weight.to(logits.device) if self.weight is not None else None,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        probs = F.softmax(logits, dim=1)
        # Gather class probability for each pixel's true class
        safe_targets = targets.clone()
        safe_targets[targets == self.ignore_index] = 0
        pt = probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)

        focal_weight = self.alpha * (1.0 - pt) ** self.gamma
        loss = focal_weight * ce

        # Mask ignore pixels
        valid = targets != self.ignore_index
        if valid.sum() == 0:
            return loss.sum() * 0.0  # no valid pixels
        return loss[valid].mean()


class CombinedSegmentationLoss(nn.Module):
    """Weighted combination of CE + Dice + optional Focal.

    Parameters
    ----------
    ce_weight, dice_weight, focal_weight : float
        Scalar weights for each component.  Set to 0 to disable a component.
    """

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_weight: float = 0.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        class_weights: Optional[List[float]] = None,
        ignore_index: int = 255,
    ) -> None:
        super().__init__()
        self.ce_w = ce_weight
        self.dice_w = dice_weight
        self.focal_w = focal_weight

        self.ce = CrossEntropySegLoss(weight=class_weights, ignore_index=ignore_index) if ce_weight > 0 else None
        self.dice = DiceLoss(ignore_index=ignore_index, weight=class_weights) if dice_weight > 0 else None
        self.focal = (
            FocalLoss(alpha=focal_alpha, gamma=focal_gamma, ignore_index=ignore_index, weight=class_weights)
            if focal_weight > 0
            else None
        )

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Return dict with 'total', 'ce', 'dice', 'focal' keys."""
        zero = torch.tensor(0.0, device=logits.device)
        ce_loss = self.ce(logits, targets) if self.ce is not None else zero
        dice_loss = self.dice(logits, targets) if self.dice is not None else zero
        focal_loss = self.focal(logits, targets) if self.focal is not None else zero

        total = (
            self.ce_w * ce_loss
            + self.dice_w * dice_loss
            + self.focal_w * focal_loss
        )
        return {
            "total": total,
            "ce": ce_loss,
            "dice": dice_loss,
            "focal": focal_loss,
        }


def compute_total_loss(
    criterion: CombinedSegmentationLoss,
    model_output: dict | torch.Tensor,
    targets: torch.Tensor,
    aux_downsample_weight: float = 0.4,
    aux_global_weight: float = 0.4,
) -> Dict[str, torch.Tensor]:
    """Compute main + auxiliary losses.

    Parameters
    ----------
    model_output : dict or Tensor
        If dict, expected keys: "out", "aux_downsample", "aux_global".
        If Tensor, only the main loss is computed.

    Returns
    -------
    dict with keys: total, main, ce, dice, focal, aux_downsample, aux_global
    """
    zero = torch.tensor(0.0, device=targets.device)

    if isinstance(model_output, dict):
        main_logits = model_output["out"]
        aux_ds_logits = model_output["aux_downsample"]
        aux_gl_logits = model_output["aux_global"]
    else:
        main_logits = model_output
        aux_ds_logits = None
        aux_gl_logits = None

    main_losses = criterion(main_logits, targets)

    # Auxiliary losses
    if aux_ds_logits is not None:
        aux_ds_losses = criterion(aux_ds_logits, targets)
        aux_ds_loss = aux_ds_losses["total"]
    else:
        aux_ds_loss = zero

    if aux_gl_logits is not None:
        aux_gl_losses = criterion(aux_gl_logits, targets)
        aux_gl_loss = aux_gl_losses["total"]
    else:
        aux_gl_loss = zero

    total = (
        main_losses["total"]
        + aux_downsample_weight * aux_ds_loss
        + aux_global_weight * aux_gl_loss
    )

    return {
        "total": total,
        "main": main_losses["total"],
        "ce": main_losses["ce"],
        "dice": main_losses["dice"],
        "focal": main_losses["focal"],
        "aux_downsample": aux_ds_loss,
        "aux_global": aux_gl_loss,
    }


# ===========================================================================
# Binary losses for FastSCNNSalient (1-channel logits)
# ===========================================================================


class BinaryDiceLoss(nn.Module):
    """Binary Dice Loss using sigmoid probabilities.

    Computes batch-wise Dice coefficient on 1-channel predictions.

    Parameters
    ----------
    smooth : float
        Smoothing constant to avoid division by zero (default 1.0).
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : [B, 1, H, W]  raw logits
        targets : [B, 1, H, W]  float (0.0 or 1.0)
        """
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions: [B, N]
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1).float()

        intersection = (probs_flat * targets_flat).sum(dim=1)
        pred_sum = probs_flat.sum(dim=1)
        target_sum = targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (
            pred_sum + target_sum + self.smooth
        )

        return 1.0 - dice.mean()


class BinaryFocalLoss(nn.Module):
    """Binary Focal Loss using numerically stable logits formulation.

    Uses ``F.binary_cross_entropy_with_logits`` internally.

    Parameters
    ----------
    alpha : float
        Balancing factor (default 0.25).
    gamma : float
        Focusing parameter (default 2.0).
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : [B, 1, H, W]  raw logits
        targets : [B, 1, H, W]  float (0.0 or 1.0)
        """
        bce = F.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none",
        )

        prob = torch.sigmoid(logits)

        # p_t = prob if target=1, (1-prob) if target=0
        p_t = prob * targets + (1.0 - prob) * (1.0 - targets)

        # alpha_t = alpha if target=1, (1-alpha) if target=0
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)

        focal = alpha_t * (1.0 - p_t).pow(self.gamma) * bce

        return focal.mean()


class SobelBoundaryLoss(nn.Module):
    """Boundary loss using fixed Sobel edge detection kernels.

    Computes edge magnitude for both prediction and ground truth using
    Sobel operators, then minimizes L1 distance between them.

    Sobel kernels are registered as buffers (non-trainable, device-following).

    Parameters
    ----------
    eps : float
        Small epsilon for numerical stability in sqrt (default 1e-6).
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

        # Sobel X kernel [1, 1, 3, 3]
        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0],
             [-2.0, 0.0, 2.0],
             [-1.0, 0.0, 1.0]],
        ).unsqueeze(0).unsqueeze(0)

        # Sobel Y kernel [1, 1, 3, 3]
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0],
             [ 0.0,  0.0,  0.0],
             [ 1.0,  2.0,  1.0]],
        ).unsqueeze(0).unsqueeze(0)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _edge_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Compute edge magnitude from [B, 1, H, W] input."""
        edge_x = F.conv2d(x, self.sobel_x, padding=1)
        edge_y = F.conv2d(x, self.sobel_y, padding=1)
        return torch.sqrt(edge_x.square() + edge_y.square() + self.eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : [B, 1, H, W]  fine logits (gradients flow through sigmoid)
        targets : [B, 1, H, W]  float (0.0 or 1.0)
        """
        fine_prob = torch.sigmoid(logits)

        pred_edge = self._edge_magnitude(fine_prob)
        gt_edge = self._edge_magnitude(targets.float())

        return F.l1_loss(pred_edge, gt_edge)


class SalientSegmentationLoss(nn.Module):
    """Combined loss for FastSCNNSalient.

    Total loss::

        L_total = lambda_coarse  * L_coarse
                + lambda_fine    * L_fine
                + lambda_boundary * L_boundary

    Where::

        L_coarse   = bce_weight * BCEWithLogits + dice_weight * BinaryDice
        L_fine     = focal_weight * BinaryFocal + dice_weight * BinaryDice
        L_boundary = SobelBoundaryLoss (on fine prediction only)

    Parameters
    ----------
    lambda_coarse : float
        Weight for coarse loss (default 1.0).
    lambda_fine : float
        Weight for fine loss (default 1.0).
    lambda_boundary : float
        Weight for boundary loss (default 0.5).
    coarse_bce_weight : float
        Weight for BCE in coarse loss (default 1.0).
    coarse_dice_weight : float
        Weight for Dice in coarse loss (default 1.0).
    fine_focal_weight : float
        Weight for Focal in fine loss (default 1.0).
    fine_dice_weight : float
        Weight for Dice in fine loss (default 1.0).
    focal_alpha : float
        Alpha for BinaryFocalLoss (default 0.25).
    focal_gamma : float
        Gamma for BinaryFocalLoss (default 2.0).
    pos_weight : float or None
        Positive class weight for BCEWithLogitsLoss (default None).
    """

    def __init__(
        self,
        lambda_coarse: float = 1.0,
        lambda_fine: float = 1.0,
        lambda_boundary: float = 0.5,
        coarse_bce_weight: float = 1.0,
        coarse_dice_weight: float = 1.0,
        fine_focal_weight: float = 1.0,
        fine_dice_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.lambda_coarse = lambda_coarse
        self.lambda_fine = lambda_fine
        self.lambda_boundary = lambda_boundary

        self.coarse_bce_w = coarse_bce_weight
        self.coarse_dice_w = coarse_dice_weight
        self.fine_focal_w = fine_focal_weight
        self.fine_dice_w = fine_dice_weight

        # pos_weight for BCEWithLogitsLoss
        self.register_buffer(
            "pos_weight",
            torch.tensor([pos_weight], dtype=torch.float32)
            if pos_weight is not None
            else None,
        )

        # Loss components
        self.binary_dice = BinaryDiceLoss()
        self.binary_focal = BinaryFocalLoss(
            alpha=focal_alpha, gamma=focal_gamma,
        )
        self.boundary = SobelBoundaryLoss()

    def forward(
        self,
        coarse_logits: torch.Tensor,
        fine_logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        coarse_logits : [B, 1, H, W]
        fine_logits : [B, 1, H, W]
        targets : [B, 1, H, W]  float (0.0 or 1.0)

        Returns
        -------
        dict with keys:
            total, coarse, fine, boundary,
            coarse_bce, coarse_dice, fine_focal, fine_dice
        """
        targets_float = targets.float()
        zero = torch.tensor(0.0, device=coarse_logits.device)

        # ── Coarse Loss: BCEWithLogits + BinaryDice ───────────────────
        pw = self.pos_weight
        if pw is not None:
            pw = pw.to(coarse_logits.device)

        coarse_bce = (
            F.binary_cross_entropy_with_logits(
                coarse_logits, targets_float, pos_weight=pw,
            )
            if self.coarse_bce_w > 0
            else zero
        )
        coarse_dice = (
            self.binary_dice(coarse_logits, targets_float)
            if self.coarse_dice_w > 0
            else zero
        )
        coarse_loss = (
            self.coarse_bce_w * coarse_bce
            + self.coarse_dice_w * coarse_dice
        )

        # ── Fine Loss: BinaryFocal + BinaryDice ──────────────────────
        fine_focal = (
            self.binary_focal(fine_logits, targets_float)
            if self.fine_focal_w > 0
            else zero
        )
        fine_dice = (
            self.binary_dice(fine_logits, targets_float)
            if self.fine_dice_w > 0
            else zero
        )
        fine_loss = (
            self.fine_focal_w * fine_focal
            + self.fine_dice_w * fine_dice
        )

        # ── Boundary Loss: Sobel on fine prediction only ──────────────
        boundary_loss = (
            self.boundary(fine_logits, targets_float)
            if self.lambda_boundary > 0
            else zero
        )

        # ── Total ─────────────────────────────────────────────────────
        total = (
            self.lambda_coarse * coarse_loss
            + self.lambda_fine * fine_loss
            + self.lambda_boundary * boundary_loss
        )

        return {
            "total": total,
            "coarse": coarse_loss,
            "fine": fine_loss,
            "boundary": boundary_loss,
            "coarse_bce": coarse_bce,
            "coarse_dice": coarse_dice,
            "fine_focal": fine_focal,
            "fine_dice": fine_dice,
        }
