import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryTverskyLoss(nn.Module):
    """Binary Tversky Loss using sigmoid probabilities.

    Tversky Loss generalizes Dice Loss by allowing different weights on
    False Positives (FP) and False Negatives (FN).

    Parameters
    ----------
    fp_weight : float
        Penalty weight for False Positives (default 0.7).
    fn_weight : float
        Penalty weight for False Negatives (default 0.3).
    smooth : float
        Smoothing constant to avoid division by zero (default 1.0).
    eps : float
        Small constant for numerical stability (default 1e-7).
    """
    def __init__(
        self,
        fp_weight: float = 0.7,
        fn_weight: float = 0.3,
        smooth: float = 1.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        self.fp_weight = fp_weight
        self.fn_weight = fn_weight
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits : [B, 1, H, W]
        targets : [B, 1, H, W]
        """
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions: [B, N]
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1).float()

        tp = (probs_flat * targets_flat).sum(dim=1)
        fp = (probs_flat * (1.0 - targets_flat)).sum(dim=1)
        fn = ((1.0 - probs_flat) * targets_flat).sum(dim=1)

        tversky = (tp + self.smooth) / (
            tp + self.fp_weight * fp + self.fn_weight * fn + self.smooth + self.eps
        )
        return 1.0 - tversky.mean()


class BoundaryWeightedBCELoss(nn.Module):
    """BCE loss with extra weight applied to the boundary band of the target mask.

    Parameters
    ----------
    boundary_kernel_size : int
        Size of dilation/erosion structuring element (must be odd, default 7).
    boundary_extra_weight : float
        Extra weight added to the boundary region (default 4.0).
    eps : float
        Stability constant (default 1e-7).
    """
    def __init__(
        self,
        boundary_kernel_size: int = 7,
        boundary_extra_weight: float = 4.0,
        eps: float = 1e-7,
    ) -> None:
        super().__init__()
        assert boundary_kernel_size % 2 == 1, "boundary_kernel_size must be odd"
        self.boundary_kernel_size = boundary_kernel_size
        self.boundary_extra_weight = boundary_extra_weight
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # target dilation & erosion to find boundary band
        dilated = F.max_pool2d(
            targets,
            kernel_size=self.boundary_kernel_size,
            stride=1,
            padding=self.boundary_kernel_size // 2,
        )
        eroded = -F.max_pool2d(
            -targets,
            kernel_size=self.boundary_kernel_size,
            stride=1,
            padding=self.boundary_kernel_size // 2,
        )
        boundary_band = (dilated - eroded).clamp(0.0, 1.0)

        pixel_weight = 1.0 + self.boundary_extra_weight * boundary_band

        per_pixel_bce = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            reduction="none",
        )

        loss = (per_pixel_bce * pixel_weight).sum() / pixel_weight.sum().clamp_min(self.eps)
        return loss


class HardNegativeBCELoss(nn.Module):
    """BCE loss targeting only the top-k highest loss background pixels.

    Parameters
    ----------
    hard_negative_ratio : float
        Fraction of background pixels to select (default 0.10).
    hard_negative_min_pixels : int
        Minimum number of hard negative pixels to select (default 256).
    """
    def __init__(
        self,
        hard_negative_ratio: float = 0.10,
        hard_negative_min_pixels: int = 256,
    ) -> None:
        super().__init__()
        self.hard_negative_ratio = hard_negative_ratio
        self.hard_negative_min_pixels = hard_negative_min_pixels

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Shape: [B, 1, H, W]
        per_pixel_bce = F.binary_cross_entropy_with_logits(
            logits,
            targets.float(),
            reduction="none",
        )

        B = logits.size(0)
        image_losses = []

        for i in range(B):
            img_target = targets[i]
            img_bce = per_pixel_bce[i]

            bg_mask = img_target < 0.5
            bg_losses = img_bce[bg_mask]

            num_bg_pixels = bg_losses.numel()
            if num_bg_pixels == 0:
                # If no background pixels, return 0.0 with gradient tracking
                image_losses.append(logits[i].sum() * 0.0)
                continue

            num_hard = max(
                int(num_bg_pixels * self.hard_negative_ratio),
                self.hard_negative_min_pixels,
            )
            num_hard = min(num_hard, num_bg_pixels)

            topk_losses = torch.topk(
                bg_losses,
                k=num_hard,
                largest=True,
            ).values

            image_losses.append(topk_losses.mean())

        return torch.stack(image_losses).mean()
