"""
DDC (Directional Distance Consistency) Loss for alpha-free matting training.

Based on: "Training Matting Models without Alpha Labels"

Components
----------
- KnownRegionL1Loss : L1 supervision on trimap-known (FG/BG) pixels only.
- DirectionalDistanceConsistencyLoss : DDC loss encouraging alpha consistency
  with local RGB similarity, applied to the ENTIRE alpha matte.

Key design decisions
--------------------
- DDC uses F.unfold for efficient sliding-window extraction (no for-loops).
- Chunked spatial computation bounds peak GPU memory.
- Forces float32 internally for numerical stability (sqrt, topk).
- Alpha gradient flows through; ddc_image has no gradient.
- DDC is applied to the entire alpha matte (not just unknown region).
"""

from __future__ import annotations

import logging
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ===========================================================================
# Known Region L1 Loss
# ===========================================================================


class KnownRegionL1Loss(nn.Module):
    """L1 loss computed only on trimap-known foreground/background pixels.

    Paper formula::

        L_known = (1/N_known) * Σ_i  m_i * |α_i - t_i|

    Where m_i = 1 for known pixels (trimap == 0.0 or trimap == 1.0),
    t_i is the trimap value (0 or 1), and N_known is the count of known pixels.

    Equivalently implemented as::

        L_known = mean(|α - trimap| * known_mask) * (N_total / N_known)

    Parameters
    ----------
    eps : float
        Small value to avoid division by zero when N_known is 0.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred_alpha: torch.Tensor,
        trimap: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pred_alpha : [B, 1, H, W]
            Predicted alpha (after sigmoid, values in [0, 1]).
        trimap : [B, 1, H, W]
            Trimap with values {0.0, 0.5, 1.0}.

        Returns
        -------
        Scalar loss tensor with gradient.
        """
        # Known mask: trimap is exactly 0.0 or 1.0 (NOT 0.5)
        known_mask = ((trimap < 0.25) | (trimap > 0.75)).float()

        # Count known pixels
        n_known = known_mask.sum()
        n_total = known_mask.numel()

        if n_known < 1.0:
            warnings.warn(
                "KnownRegionL1Loss: no known pixels in batch. "
                "Returning zero loss.",
                RuntimeWarning,
                stacklevel=2,
            )
            return pred_alpha.sum() * 0.0

        # L1 error, masked to known pixels only
        abs_error = torch.abs(pred_alpha - trimap)
        masked_error = abs_error * known_mask

        # Paper scaling: mean over all pixels × (N_total / N_known)
        # This equals mean over known pixels only
        loss = masked_error.sum() / n_known.clamp_min(self.eps)

        return loss


# ===========================================================================
# Directional Distance Consistency Loss
# ===========================================================================


class DirectionalDistanceConsistencyLoss(nn.Module):
    """DDC Loss: encourages alpha consistency with local RGB similarity.

    Paper formula::

        L_DDC = (1/N) Σ_i Σ_{j ∈ S_i} |α_i - α_j - ‖I_i - I_j‖₂|

    Where S_i = TopK(-‖I_i - I_j‖₂) selects the K most RGB-similar
    neighbors from a local K×K window around pixel i.

    This is DIRECTIONAL — preserving the sign of (α_i - α_j) — which
    distinguishes DDC from standard DC loss.

    Parameters
    ----------
    window_size : int
        Local window size (must be odd, >= 3). Paper default: 11.
    num_neighbors : int
        Number of most-similar RGB neighbors to select. Paper default: 11.
    padding_mode : str
        Padding mode for F.pad. Default: "replicate".
    exclude_center : bool
        If True, exclude the center pixel from neighbor candidates.
    reduction : str
        "paper" — sum over neighbors, then mean over pixels.
        "mean_neighbors" — mean over both (changes scale, requires λ re-tuning).
    chunk_size : int
        Number of spatial positions per chunk. Controls peak memory.
        0 = no chunking (compute all at once).
    downsample_factor : int
        If > 1, bilinear-downsample image and alpha before DDC computation.
        Marked as PROJECT optimization, not paper-exact.
    """

    def __init__(
        self,
        window_size: int = 11,
        num_neighbors: int = 11,
        padding_mode: str = "replicate",
        exclude_center: bool = True,
        reduction: str = "paper",
        chunk_size: int = 4096,
        downsample_factor: int = 1,
    ) -> None:
        super().__init__()
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be odd and >= 3, got {window_size}"
            )
        if reduction not in ("paper", "mean_neighbors"):
            raise ValueError(
                f"reduction must be 'paper' or 'mean_neighbors', got '{reduction}'"
            )

        self.window_size = window_size
        self.num_neighbors = num_neighbors
        self.padding_mode = padding_mode
        self.exclude_center = exclude_center
        self.reduction = reduction
        self.chunk_size = chunk_size
        self.downsample_factor = downsample_factor

        # Total positions in window
        total_positions = window_size * window_size
        if exclude_center:
            total_positions -= 1
        if num_neighbors > total_positions:
            raise ValueError(
                f"num_neighbors ({num_neighbors}) exceeds available positions "
                f"({total_positions}) in {window_size}×{window_size} window "
                f"(exclude_center={exclude_center})"
            )

    def forward(
        self,
        alpha: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        alpha : [B, 1, H, W]
            Predicted alpha matte (after sigmoid, values in [0, 1]).
            Gradients MUST flow through this tensor.
        image : [B, 3, H, W]
            Raw RGB image in [0, 1] range. NOT ImageNet-normalized.
            No gradient computation on this tensor.

        Returns
        -------
        Scalar loss tensor with gradient flowing to alpha.
        """
        # Ensure no gradient flows to image
        image = image.detach()

        # Force float32 for numerical stability
        alpha_f32 = alpha.float()
        image_f32 = image.float()

        # Optional downsampling (PROJECT optimization)
        if self.downsample_factor > 1:
            ds = self.downsample_factor
            h, w = alpha_f32.shape[-2:]
            new_h, new_w = h // ds, w // ds
            alpha_f32 = F.interpolate(
                alpha_f32, size=(new_h, new_w),
                mode="bilinear", align_corners=False,
            )
            image_f32 = F.interpolate(
                image_f32, size=(new_h, new_w),
                mode="bilinear", align_corners=False,
            )

        B, _, H, W = alpha_f32.shape
        K = self.window_size
        pad = K // 2

        # Pad image and alpha with replicate/reflect padding
        image_padded = F.pad(
            image_f32, [pad, pad, pad, pad], mode=self.padding_mode,
        )  # [B, 3, H+2p, W+2p]
        alpha_padded = F.pad(
            alpha_f32, [pad, pad, pad, pad], mode=self.padding_mode,
        )  # [B, 1, H+2p, W+2p]

        N = H * W  # total spatial positions

        if self.chunk_size > 0 and N > self.chunk_size:
            return self._forward_chunked(
                image_padded, alpha_padded, alpha_f32,
                B, H, W, K, N,
            )
        else:
            return self._forward_full(
                image_padded, alpha_padded, alpha_f32,
                B, H, W, K, N,
            )

    def _forward_full(
        self,
        image_padded: torch.Tensor,
        alpha_padded: torch.Tensor,
        alpha_orig: torch.Tensor,
        B: int, H: int, W: int, K: int, N: int,
    ) -> torch.Tensor:
        """Compute DDC without chunking."""
        KK = K * K

        # Unfold image patches: [B, 3*K*K, N]
        img_patches = F.unfold(
            image_padded, kernel_size=K,
        )  # [B, 3*KK, N]

        # Unfold alpha patches: [B, K*K, N]
        alpha_patches = F.unfold(
            alpha_padded, kernel_size=K,
        )  # [B, KK, N]

        # Reshape image patches: [B, 3, KK, N]
        img_patches = img_patches.view(B, 3, KK, N)

        # Center pixel RGB: [B, 3, 1, N]
        center_idx = KK // 2
        center_rgb = img_patches[:, :, center_idx:center_idx+1, :]

        # Center pixel alpha: [B, 1, N] — use original (non-padded) for gradient
        center_alpha = alpha_orig.view(B, 1, N)

        # Neighbor RGB distances: [B, KK, N]
        rgb_diff = img_patches - center_rgb  # [B, 3, KK, N]
        rgb_dist = torch.sqrt(
            (rgb_diff * rgb_diff).sum(dim=1) + 1e-8
        )  # [B, KK, N]

        # Neighbor alpha: [B, KK, N]
        neighbor_alpha = alpha_patches  # [B, KK, N]

        if self.exclude_center:
            # Remove center pixel from candidates
            # Create mask for all positions except center
            keep = list(range(KK))
            keep.pop(center_idx)
            keep = torch.tensor(keep, device=rgb_dist.device)

            rgb_dist = rgb_dist[:, keep, :]           # [B, KK-1, N]
            neighbor_alpha = neighbor_alpha[:, keep, :]  # [B, KK-1, N]

        # Select TopK most similar neighbors (smallest RGB distance)
        # topk with largest=False gives smallest distances
        _, topk_indices = torch.topk(
            rgb_dist, k=self.num_neighbors, dim=1, largest=False,
        )  # [B, num_neighbors, N]

        # Gather selected neighbor alpha values
        selected_alpha = torch.gather(
            neighbor_alpha, dim=1, index=topk_indices,
        )  # [B, num_neighbors, N]

        # Gather selected RGB distances
        selected_rgb_dist = torch.gather(
            rgb_dist, dim=1, index=topk_indices,
        )  # [B, num_neighbors, N]

        # DDC: |α_i - α_j - ‖I_i - I_j‖₂|
        # NOTE: DIRECTIONAL — NOT |α_i - α_j| - ...
        alpha_diff = center_alpha.unsqueeze(1) - selected_alpha  # [B, K_sel, N]
        pair_loss = torch.abs(alpha_diff - selected_rgb_dist)    # [B, K_sel, N]

        # Reduction
        if self.reduction == "paper":
            # Sum over neighbors, then mean over pixels
            loss = pair_loss.sum(dim=1).mean()  # sum(K_sel) → [B, N] → mean
        else:  # mean_neighbors
            loss = pair_loss.mean()

        return loss

    def _forward_chunked(
        self,
        image_padded: torch.Tensor,
        alpha_padded: torch.Tensor,
        alpha_orig: torch.Tensor,
        B: int, H: int, W: int, K: int, N: int,
    ) -> torch.Tensor:
        """Compute DDC with spatial chunking to bound memory."""
        chunk_size = self.chunk_size
        KK = K * K
        center_idx = KK // 2

        # Pre-compute unfold for the full spatial extent
        # We chunk over spatial dimension N
        img_patches_full = F.unfold(
            image_padded, kernel_size=K,
        )  # [B, 3*KK, N]
        alpha_patches_full = F.unfold(
            alpha_padded, kernel_size=K,
        )  # [B, KK, N]

        img_patches_full = img_patches_full.view(B, 3, KK, N)
        center_alpha_full = alpha_orig.view(B, 1, N)

        total_loss = torch.tensor(0.0, device=alpha_orig.device)
        total_count = 0

        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_n = end - start

            # Slice spatial chunk
            img_chunk = img_patches_full[:, :, :, start:end]    # [B, 3, KK, chunk_n]
            alpha_chunk = alpha_patches_full[:, :, start:end]    # [B, KK, chunk_n]
            center_alpha_chunk = center_alpha_full[:, :, start:end]  # [B, 1, chunk_n]

            # Center pixel RGB
            center_rgb = img_chunk[:, :, center_idx:center_idx+1, :]  # [B, 3, 1, chunk_n]

            # RGB distances
            rgb_diff = img_chunk - center_rgb  # [B, 3, KK, chunk_n]
            rgb_dist = torch.sqrt(
                (rgb_diff * rgb_diff).sum(dim=1) + 1e-8
            )  # [B, KK, chunk_n]

            neighbor_alpha = alpha_chunk  # [B, KK, chunk_n]

            if self.exclude_center:
                keep = list(range(KK))
                keep.pop(center_idx)
                keep = torch.tensor(keep, device=rgb_dist.device)
                rgb_dist = rgb_dist[:, keep, :]
                neighbor_alpha = neighbor_alpha[:, keep, :]

            # TopK
            _, topk_indices = torch.topk(
                rgb_dist, k=self.num_neighbors, dim=1, largest=False,
            )

            selected_alpha = torch.gather(
                neighbor_alpha, dim=1, index=topk_indices,
            )
            selected_rgb_dist = torch.gather(
                rgb_dist, dim=1, index=topk_indices,
            )

            # DDC pair loss
            alpha_diff = center_alpha_chunk.unsqueeze(1) - selected_alpha
            pair_loss = torch.abs(alpha_diff - selected_rgb_dist)

            if self.reduction == "paper":
                # Sum over neighbors for this chunk
                chunk_loss = pair_loss.sum(dim=1).sum()  # scalar
                total_loss = total_loss + chunk_loss
                total_count += B * chunk_n
            else:
                chunk_loss = pair_loss.sum()
                total_loss = total_loss + chunk_loss
                total_count += pair_loss.numel()

        # Final mean
        loss = total_loss / max(total_count, 1)
        return loss
