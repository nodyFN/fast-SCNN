"""
Fast-SCNN Salient: Single-Backbone, Dual-Head, Coarse-to-Fine
Salient Foreground Segmentation Model
======================================================================

Architecture overview::

    Input [B, 3, H, W]
      │
      ▼
    SharedFastSCNNBackbone  (executed ONCE)
      │  Learning to Downsample
      │  Global Feature Extractor
      │  Pyramid Pooling Module
      │  Feature Fusion Module
      │
      └─► F_shared [B, 128, H/8, W/8]
            │
            ├──► CoarseHead ──► coarse_logits_lowres [B, 1, H/8, W/8]
            │                       │
            │                 sigmoid + detach
            │                       │
            │                 coarse_prompt [B, 1, H/8, W/8]  (stop-gradient)
            │                       │
            └──► RefinementHead ◄───┘
                      │
                      └──► fine_logits_lowres [B, 1, H/8, W/8]

Output format (1-channel logits):
    - coarse_logits:        [B, 1, H, W]
    - coarse_prob:          [B, 1, H, W]
    - fine_logits:          [B, 1, H, W]
    - fine_prob:            [B, 1, H, W]
    - coarse_logits_lowres: [B, 1, H/8, W/8]
    - coarse_prompt:        [B, 1, H/8, W/8]  (requires_grad=False)

Uses BCEWithLogitsLoss / BinaryFocalLoss / BinaryDiceLoss (NOT CrossEntropyLoss).

Design constraints for edge deployment:
    - Single backbone execution (no duplicate feature extraction)
    - No external spatial prompts (GT mask, bbox, SAM, etc.)
    - Broadcasting for spatial attention (no .repeat())
    - Sobel kernels as fixed buffers (not trainable)
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse existing Fast-SCNN building blocks — NO code duplication
from models.fast_scnn import (
    ConvBNReLU,
    DepthwiseSeparableConv,
    FeatureFusionModule,
    GlobalFeatureExtractor,
    LearningToDownsample,
    PyramidPooling,
    count_parameters,
)


# ===========================================================================
# Shared Backbone
# ===========================================================================


class SharedFastSCNNBackbone(nn.Module):
    """Shared backbone for the dual-head salient segmentation model.

    Reuses the same building blocks as FastSCNN:
        Learning to Downsample → Global Feature Extractor
        → Pyramid Pooling → Feature Fusion

    Does NOT include the original Classifier or Auxiliary Heads.

    Parameters
    ----------
    ppm_pool_sizes : tuple of int
        Pool sizes for the Pyramid Pooling Module.
    """

    def __init__(
        self,
        ppm_pool_sizes: Tuple[int, ...] = (1, 2, 3, 6),
    ) -> None:
        super().__init__()
        self.learning_to_downsample = LearningToDownsample()
        self.global_feature_extractor = GlobalFeatureExtractor()
        self.ppm = PyramidPooling(
            in_channels=128,
            out_channels=128,
            pool_sizes=ppm_pool_sizes,
        )
        self.ffm = FeatureFusionModule(
            high_channels=64, low_channels=128, out_channels=128,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Single-pass feature extraction.

        Parameters
        ----------
        x : [B, 3, H, W]

        Returns
        -------
        fused : [B, 128, H/8, W/8]
        """
        ltd_out = self.learning_to_downsample(x)
        gfe_out = self.global_feature_extractor(ltd_out)
        ppm_out = self.ppm(gfe_out)
        fused = self.ffm(high_res=ltd_out, low_res=ppm_out)
        return fused


# ===========================================================================
# Coarse Head
# ===========================================================================


class CoarseHead(nn.Module):
    """First decoder — fast, high-recall foreground predictor.

    Structure::

        DSConv 128 → coarse_channels (default 64), stride=1
        Dropout (optional, default p=0.1)
        1×1 Conv coarse_channels → 1

    Output: low-resolution logits [B, 1, H/8, W/8].

    Parameters
    ----------
    in_channels : int
        Input channels from shared backbone (default 128).
    coarse_channels : int
        Intermediate channel width (default 64).
    dropout_p : float
        Dropout probability (default 0.1).
    """

    def __init__(
        self,
        in_channels: int = 128,
        coarse_channels: int = 64,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.dsconv = DepthwiseSeparableConv(
            in_channels, coarse_channels, stride=1,
        )
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        self.conv = nn.Conv2d(coarse_channels, 1, kernel_size=1)

    def forward(self, shared_feature: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        shared_feature : [B, 128, H_s, W_s]

        Returns
        -------
        coarse_logits_lowres : [B, 1, H_s, W_s]
        """
        x = self.dsconv(shared_feature)
        x = self.dropout(x)
        return self.conv(x)


# ===========================================================================
# Refinement Head
# ===========================================================================


class RefinementHead(nn.Module):
    """Second decoder — precision refinement with spatial attention.

    Pipeline::

        1. Spatial Attention:
           f_attended = F_shared + F_shared * coarse_prompt   (broadcasting)

        2. Channel Concatenation:
           refinement_input = cat([f_attended, coarse_prompt], dim=1)
           → C + 1 channels (default 129)

        3. Convolution Stack:
           1×1 ConvBNReLU  (C+1) → refinement_channels
           3×3 DSConv      refinement_channels → refinement_channels
           Dropout (optional)
           1×1 Conv         refinement_channels → 1

    Parameters
    ----------
    in_channels : int
        Channel count of shared feature (default 128).
    refinement_channels : int
        Intermediate channel width (default 64).
    dropout_p : float
        Dropout probability (default 0.1).
    """

    def __init__(
        self,
        in_channels: int = 128,
        refinement_channels: int = 64,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        # 1×1 projection: (C + 1) → refinement_channels
        self.proj = ConvBNReLU(
            in_channels + 1, refinement_channels,
            kernel_size=1, stride=1, padding=0,
        )
        # 3×3 DSConv: refinement_channels → refinement_channels
        self.dsconv = DepthwiseSeparableConv(
            refinement_channels, refinement_channels, stride=1,
        )
        self.dropout = nn.Dropout2d(p=dropout_p) if dropout_p > 0 else nn.Identity()
        # Final 1×1 prediction
        self.conv = nn.Conv2d(refinement_channels, 1, kernel_size=1)

    def forward(
        self,
        shared_feature: torch.Tensor,
        coarse_prompt: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        shared_feature : [B, C, H_s, W_s]
            Feature fusion output (gradients flow through here).
        coarse_prompt : [B, 1, H_s, W_s]
            Detached coarse probability (requires_grad=False).

        Returns
        -------
        fine_logits_lowres : [B, 1, H_s, W_s]
        """
        # Spatial Attention: F_attended = F_shared + F_shared * coarse_prompt
        # Uses broadcasting: [B, C, H, W] * [B, 1, H, W] → [B, C, H, W]
        f_attended = shared_feature + shared_feature * coarse_prompt

        # Channel concatenation: [B, C, H, W] + [B, 1, H, W] → [B, C+1, H, W]
        refinement_input = torch.cat([f_attended, coarse_prompt], dim=1)

        # Convolution stack
        x = self.proj(refinement_input)
        x = self.dsconv(x)
        x = self.dropout(x)
        return self.conv(x)


# ===========================================================================
# Full Salient Model
# ===========================================================================


class FastSCNNSalient(nn.Module):
    """Single-Backbone, Dual-Head, Coarse-to-Fine Salient Segmentation.

    Class-agnostic binary foreground segmentation model optimized for
    edge devices with limited DRAM bandwidth.

    Key design constraints:
        - Single backbone execution per input image.
        - Internal spatial prompt (no external GT/bbox/SAM).
        - Stop-gradient on coarse prompt to Refinement Head.
        - 1-channel logits with BCEWithLogitsLoss (not 2-channel CE).

    Parameters
    ----------
    ppm_pool_sizes : tuple of int
        Pool sizes for the Pyramid Pooling Module (default (1,2,3,6)).
    coarse_channels : int
        Intermediate channels in CoarseHead (default 64).
    refinement_channels : int
        Intermediate channels in RefinementHead (default 64).
    dropout_p : float
        Dropout probability for both heads (default 0.1).

    Returns (forward)
    -----------------
    dict with keys:
        coarse_logits        : [B, 1, H, W]
        coarse_prob          : [B, 1, H, W]
        fine_logits          : [B, 1, H, W]
        fine_prob            : [B, 1, H, W]
        coarse_logits_lowres : [B, 1, H_s, W_s]
        coarse_prompt        : [B, 1, H_s, W_s]  (requires_grad=False)
    """

    def __init__(
        self,
        ppm_pool_sizes: Tuple[int, ...] = (1, 2, 3, 6),
        coarse_channels: int = 64,
        refinement_channels: int = 64,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = SharedFastSCNNBackbone(
            ppm_pool_sizes=ppm_pool_sizes,
        )
        self.coarse_head = CoarseHead(
            in_channels=128,
            coarse_channels=coarse_channels,
            dropout_p=dropout_p,
        )
        self.refinement_head = RefinementHead(
            in_channels=128,
            refinement_channels=refinement_channels,
            dropout_p=dropout_p,
        )
        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming initialization for conv weights; BN weight=1, bias=0."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[-2:]  # (H, W)

        # ── Shared Backbone (executed ONCE) ───────────────────────────
        shared_feature = self.backbone(x)  # [B, 128, H/8, W/8]

        # ── Coarse Head ───────────────────────────────────────────────
        coarse_logits_lowres = self.coarse_head(shared_feature)
        # [B, 1, H/8, W/8]

        # Coarse probability at low resolution
        coarse_prob_lowres = torch.sigmoid(coarse_logits_lowres)

        # Stop-gradient: only detach the probability prompt
        # Fine Loss updates Refinement Head + Backbone, but NOT Coarse Head
        coarse_prompt = coarse_prob_lowres.detach()  # requires_grad=False

        # ── Refinement Head ───────────────────────────────────────────
        fine_logits_lowres = self.refinement_head(
            shared_feature, coarse_prompt,
        )  # [B, 1, H/8, W/8]

        # ── Upsample to full resolution ───────────────────────────────
        coarse_logits = F.interpolate(
            coarse_logits_lowres,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        coarse_prob = torch.sigmoid(coarse_logits)

        fine_logits = F.interpolate(
            fine_logits_lowres,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        fine_prob = torch.sigmoid(fine_logits)

        return {
            "coarse_logits": coarse_logits,
            "coarse_prob": coarse_prob,
            "fine_logits": fine_logits,
            "fine_prob": fine_prob,
            "coarse_logits_lowres": coarse_logits_lowres,
            "coarse_prompt": coarse_prompt,
        }


# ===========================================================================
# CLI test & benchmark
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FastSCNNSalient model test & benchmark",
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--device", type=str, default="auto",
        help="auto | cuda | cpu",
    )
    parser.add_argument(
        "--full-res", action="store_true",
        help="Use 1080×1920 instead of default",
    )
    args = parser.parse_args()

    if args.full_res:
        args.height, args.width = 1080, 1920

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Building FastSCNNSalient")
    model = FastSCNNSalient().to(device)

    total, trainable = count_parameters(model)
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(f"Model size (approx)  : {total * 4 / 1024 / 1024:.2f} MB (FP32)")

    # --- Shape Test (eval mode) ---
    print(f"\n--- Shape Test (eval mode, {args.height}×{args.width}) ---")
    model.eval()
    x = torch.randn(
        args.batch_size, 3, args.height, args.width, device=device,
    )
    with torch.inference_mode():
        out = model(x)

    assert isinstance(out, dict), "Expected dict output"
    expected_keys = [
        "coarse_logits", "coarse_prob",
        "fine_logits", "fine_prob",
        "coarse_logits_lowres", "coarse_prompt",
    ]
    for key in expected_keys:
        assert key in out, f"Missing key '{key}'"
        print(f"  {key}: {list(out[key].shape)}")

    # Full-res shapes
    for key in ("coarse_logits", "coarse_prob", "fine_logits", "fine_prob"):
        assert out[key].shape == (
            args.batch_size, 1, args.height, args.width,
        ), f"{key}: unexpected shape {list(out[key].shape)}"

    # Low-res shapes (approx H/8, W/8)
    h_s = out["coarse_logits_lowres"].shape[2]
    w_s = out["coarse_logits_lowres"].shape[3]
    print(f"  Low-res spatial: {h_s}×{w_s}")
    for key in ("coarse_logits_lowres", "coarse_prompt"):
        assert out[key].shape == (args.batch_size, 1, h_s, w_s)

    print("  ✓ shape test passed")

    # --- Gradient Flow Test ---
    print(f"\n--- Gradient Flow Test ---")
    model.train()
    x_grad = torch.randn(
        2, 3, args.height, args.width, device=device, requires_grad=False,
    )
    out_grad = model(x_grad)

    # coarse_prompt must NOT require grad
    assert not out_grad["coarse_prompt"].requires_grad, (
        "coarse_prompt should have requires_grad=False"
    )
    print("  ✓ coarse_prompt.requires_grad = False")

    # fine_logits must require grad
    assert out_grad["fine_logits"].requires_grad, (
        "fine_logits should have requires_grad=True"
    )
    print("  ✓ fine_logits.requires_grad = True")

    # coarse_logits must require grad
    assert out_grad["coarse_logits"].requires_grad, (
        "coarse_logits should have requires_grad=True"
    )
    print("  ✓ coarse_logits.requires_grad = True")

    # Backward through fine_logits should NOT update CoarseHead
    fine_loss = out_grad["fine_logits"].mean()
    fine_loss.backward(retain_graph=True)

    coarse_head_grads = [
        p.grad for p in model.coarse_head.parameters()
        if p.grad is not None
    ]
    assert len(coarse_head_grads) == 0, (
        "Fine loss should NOT produce gradients in CoarseHead"
    )
    print("  ✓ fine_loss does NOT update CoarseHead")

    # Backward through fine_logits SHOULD update Backbone
    backbone_grads = [
        p.grad for p in model.backbone.parameters()
        if p.grad is not None
    ]
    assert len(backbone_grads) > 0, (
        "Fine loss should produce gradients in Backbone"
    )
    print("  ✓ fine_loss DOES update Backbone")

    # Backward through fine_logits SHOULD update RefinementHead
    refinement_grads = [
        p.grad for p in model.refinement_head.parameters()
        if p.grad is not None
    ]
    assert len(refinement_grads) > 0, (
        "Fine loss should produce gradients in RefinementHead"
    )
    print("  ✓ fine_loss DOES update RefinementHead")

    # Reset gradients, test coarse_logits backward
    model.zero_grad()
    out_grad2 = model(x_grad)
    coarse_loss = out_grad2["coarse_logits"].mean()
    coarse_loss.backward()

    coarse_head_grads2 = [
        p.grad for p in model.coarse_head.parameters()
        if p.grad is not None
    ]
    assert len(coarse_head_grads2) > 0, (
        "Coarse loss should produce gradients in CoarseHead"
    )
    print("  ✓ coarse_loss DOES update CoarseHead")

    backbone_grads2 = [
        p.grad for p in model.backbone.parameters()
        if p.grad is not None
    ]
    assert len(backbone_grads2) > 0, (
        "Coarse loss should produce gradients in Backbone"
    )
    print("  ✓ coarse_loss DOES update Backbone")

    print("  ✓ gradient flow test passed")

    # --- Odd-size Test ---
    print(f"\n--- Odd-size Test ---")
    model.eval()
    for h, w in [(513, 1025), (65, 129), (127, 255)]:
        x_odd = torch.randn(1, 3, h, w, device=device)
        with torch.inference_mode():
            out_odd = model(x_odd)
        assert out_odd["fine_logits"].shape == (1, 1, h, w), (
            f"Odd-size {h}×{w}: unexpected shape "
            f"{list(out_odd['fine_logits'].shape)}"
        )
        print(f"  {h}×{w} → fine_logits {list(out_odd['fine_logits'].shape)}  ✓")

    # --- Benchmark ---
    print(f"\n--- Benchmark ---")
    model.eval()
    dummy = torch.randn(
        args.batch_size, 3, args.height, args.width, device=device,
    )
    use_cuda = device.type == "cuda"

    # Warm-up
    with torch.inference_mode():
        for _ in range(args.warmup):
            _ = model(dummy)
    if use_cuda:
        torch.cuda.synchronize()

    # Timed
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(args.iterations):
            _ = model(dummy)
    if use_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_time = t1 - t0
    total_images = args.batch_size * args.iterations
    avg_latency_ms = (total_time / args.iterations) * 1000
    fps = total_images / total_time

    print(f"\n{'='*60}")
    print(f"FastSCNNSalient Benchmark Results")
    print(f"{'='*60}")
    print(f"  Device           : {device}")
    print(f"  Input resolution : {args.height}×{args.width}")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  Warm-up iters    : {args.warmup}")
    print(f"  Timed iters      : {args.iterations}")
    print(f"  Total images     : {total_images}")
    print(f"  Total time       : {total_time:.3f} s")
    print(f"  Avg latency/batch: {avg_latency_ms:.2f} ms")
    print(f"  FPS (images/sec) : {fps:.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
