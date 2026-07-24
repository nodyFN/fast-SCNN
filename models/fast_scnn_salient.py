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

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Single-pass multiscale feature extraction.

        Parameters
        ----------
        x : [B, 3, H, W]

        Returns
        -------
        Dict containing:
            feature_h2 : [B, 32, H/2, W/2]
            feature_h4 : [B, 48, H/4, W/4]
            feature_h8 : [B, 128, H/8, W/8]
        """
        ltd_feats = self.learning_to_downsample.forward_features(x)
        feat_h2 = ltd_feats["feat_h2"]
        feat_h4 = ltd_feats["feat_h4"]
        ltd_out = ltd_feats["feat_h8_skip"]

        gfe_out = self.global_feature_extractor(ltd_out)
        ppm_out = self.ppm(gfe_out)
        feat_h8 = self.ffm(high_res=ltd_out, low_res=ppm_out)

        return {
            "feature_h2": feat_h2,
            "feature_h4": feat_h4,
            "feature_h8": feat_h8,
        }

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


class LegacyRefinementHead(nn.Module):
    """Legacy H/8 refinement head for backward compatibility."""
    def __init__(
        self,
        in_channels: int = 128,
        refinement_channels: int = 64,
        dropout_p: float = 0.1,
        prompt_gate_mode: str = "legacy_additive",
        prompt_gate_strength: float = 0.5,
    ) -> None:
        super().__init__()
        self.prompt_gate_mode = prompt_gate_mode
        self.prompt_gate_strength = prompt_gate_strength
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
        # Spatial Gating
        if self.prompt_gate_mode == "bidirectional":
            gate = 1.0 + self.prompt_gate_strength * (2.0 * coarse_prompt - 1.0)
            f_attended = shared_feature * gate
        else:
            f_attended = shared_feature + shared_feature * coarse_prompt

        # Channel concatenation: [B, C, H, W] + [B, 1, H, W] → [B, C+1, H, W]
        refinement_input = torch.cat([f_attended, coarse_prompt], dim=1)

        # Convolution stack
        x = self.proj(refinement_input)
        x = self.dsconv(x)
        x = self.dropout(x)
        return self.conv(x)


# Keep RefinementHead as alias for legacy code
RefinementHead = LegacyRefinementHead


class MultiscaleRefinementHead(nn.Module):
    """Upgrade Decoder — multi-scale feature refinement leveraging H/2, H/4, and H/8 features."""
    def __init__(
        self,
        in_channels: int = 128,
        feature_h4_channels: int = 48,
        feature_h2_channels: int = 32,
        refine_h8_channels: int = 96,
        h4_skip_channels: int = 32,
        refine_h4_channels: int = 64,
        h2_skip_channels: int = 16,
        refine_h2_channels: int = 32,
        fine_output_channels: int = 24,
        fine_dropout: float = 0.1,
        prompt_gate_mode: str = "bidirectional",
        prompt_gate_strength: float = 0.5,
    ) -> None:
        super().__init__()
        self.prompt_gate_mode = prompt_gate_mode
        self.prompt_gate_strength = prompt_gate_strength

        # H/8 Block: Concat(F_attended, coarse_prompt) -> (in_channels + 1) to refine_h8_channels
        self.h8_proj = ConvBNReLU(
            in_channels + 1, refine_h8_channels,
            kernel_size=1, stride=1, padding=0,
        )
        self.h8_dsconv = DepthwiseSeparableConv(
            refine_h8_channels, refine_h8_channels, stride=1,
        )

        # H/4 Skip Fusion
        self.h4_skip_proj = ConvBNReLU(
            feature_h4_channels, h4_skip_channels,
            kernel_size=1, stride=1, padding=0,
        )
        self.h4_fusion_proj = ConvBNReLU(
            refine_h8_channels + h4_skip_channels, refine_h4_channels,
            kernel_size=1, stride=1, padding=0,
        )
        self.h4_dsconv = DepthwiseSeparableConv(
            refine_h4_channels, refine_h4_channels, stride=1,
        )

        # H/2 Skip Fusion
        self.h2_skip_proj = ConvBNReLU(
            feature_h2_channels, h2_skip_channels,
            kernel_size=1, stride=1, padding=0,
        )
        self.h2_fusion_proj = ConvBNReLU(
            refine_h4_channels + h2_skip_channels, refine_h2_channels,
            kernel_size=1, stride=1, padding=0,
        )
        self.h2_dsconv = DepthwiseSeparableConv(
            refine_h2_channels, refine_h2_channels, stride=1,
        )

        # Full-resolution Output Block
        self.out_conv = ConvBNReLU(
            refine_h2_channels, fine_output_channels,
            kernel_size=3, stride=1, padding=1,
        )
        self.dropout = nn.Dropout2d(p=fine_dropout) if fine_dropout > 0 else nn.Identity()
        self.pred_conv = nn.Conv2d(fine_output_channels, 1, kernel_size=1)

    def forward(
        self,
        feature_h8: torch.Tensor,
        feature_h4: torch.Tensor,
        feature_h2: torch.Tensor,
        coarse_prompt: torch.Tensor,
        input_size: Tuple[int, int],
    ) -> torch.Tensor:
        # Spatial Attention Gating on feature_h8
        if self.prompt_gate_mode == "bidirectional":
            gate = 1.0 + self.prompt_gate_strength * (2.0 * coarse_prompt - 1.0)
            f_attended = feature_h8 * gate
        else:
            f_attended = feature_h8 + feature_h8 * coarse_prompt

        # H/8 Block
        x = torch.cat([f_attended, coarse_prompt], dim=1)
        x = self.h8_proj(x)
        x = self.h8_dsconv(x)

        # H/4 Skip Fusion
        x = F.interpolate(
            x,
            size=feature_h4.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        h4_skip = self.h4_skip_proj(feature_h4)
        x = torch.cat([x, h4_skip], dim=1)
        x = self.h4_fusion_proj(x)
        x = self.h4_dsconv(x)

        # H/2 Skip Fusion
        x = F.interpolate(
            x,
            size=feature_h2.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        h2_skip = self.h2_skip_proj(feature_h2)
        x = torch.cat([x, h2_skip], dim=1)
        x = self.h2_fusion_proj(x)
        x = self.h2_dsconv(x)

        # Full-resolution Output
        x = F.interpolate(
            x,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        x = self.out_conv(x)
        x = self.dropout(x)
        fine_logits = self.pred_conv(x)

        return fine_logits


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
    refinement_head : str
        "legacy_h8" | "multiscale" (default "multiscale").
    prompt_gate_mode : str
        "legacy_additive" | "bidirectional" (default "bidirectional").
    prompt_gate_strength : float
        Strength of bidirectional spatial gating (default 0.5).
    """

    def __init__(
        self,
        ppm_pool_sizes: Tuple[int, ...] = (1, 2, 3, 6),
        coarse_channels: int = 64,
        refinement_channels: int = 64,
        dropout_p: float = 0.1,
        refinement_head: str = "multiscale",
        prompt_gate_mode: str = "bidirectional",
        prompt_gate_strength: float = 0.5,
        refine_h8_channels: int = 96,
        h4_skip_channels: int = 32,
        refine_h4_channels: int = 64,
        h2_skip_channels: int = 16,
        refine_h2_channels: int = 32,
        fine_output_channels: int = 24,
        fine_dropout: float = 0.1,
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
        self.refinement_head_type = refinement_head
        if refinement_head == "legacy_h8":
            self.refinement_head = LegacyRefinementHead(
                in_channels=128,
                refinement_channels=refinement_channels,
                dropout_p=dropout_p,
                prompt_gate_mode=prompt_gate_mode,
                prompt_gate_strength=prompt_gate_strength,
            )
        elif refinement_head == "multiscale":
            self.refinement_head = MultiscaleRefinementHead(
                in_channels=128,
                feature_h4_channels=48,
                feature_h2_channels=32,
                refine_h8_channels=refine_h8_channels,
                h4_skip_channels=h4_skip_channels,
                refine_h4_channels=refine_h4_channels,
                h2_skip_channels=h2_skip_channels,
                refine_h2_channels=refine_h2_channels,
                fine_output_channels=fine_output_channels,
                fine_dropout=fine_dropout,
                prompt_gate_mode=prompt_gate_mode,
                prompt_gate_strength=prompt_gate_strength,
            )
        else:
            raise ValueError(f"Unknown refinement_head: {refinement_head}")
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

        if self.refinement_head_type == "multiscale":
            # ── Shared Backbone (executed ONCE) ───────────────────────────
            feats = self.backbone.forward_features(x)
            shared_feature = feats["feature_h8"]
            feature_h4 = feats["feature_h4"]
            feature_h2 = feats["feature_h2"]
        else:
            shared_feature = self.backbone(x)
            feature_h4 = None
            feature_h2 = None

        # ── Coarse Head ───────────────────────────────────────────────
        coarse_logits_lowres = self.coarse_head(shared_feature)
        # [B, 1, H/8, W/8]

        # Coarse probability at low resolution
        coarse_prob_lowres = torch.sigmoid(coarse_logits_lowres)

        # Stop-gradient: only detach the probability prompt
        # Fine Loss updates Refinement Head + Backbone, but NOT Coarse Head
        coarse_prompt = coarse_prob_lowres.detach()  # requires_grad=False

        # ── Refinement Head ───────────────────────────────────────────
        if self.refinement_head_type == "multiscale":
            fine_logits = self.refinement_head(
                shared_feature, feature_h4, feature_h2, coarse_prompt, input_size
            )
        else:
            fine_logits_lowres = self.refinement_head(
                shared_feature, coarse_prompt,
            )  # [B, 1, H/8, W/8]
            # ── Upsample legacy fine logits to full resolution ──────────
            fine_logits = F.interpolate(
                fine_logits_lowres,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            )

        # ── Upsample coarse logits to full resolution ──────────────────
        coarse_logits = F.interpolate(
            coarse_logits_lowres,
            size=input_size,
            mode="bilinear",
            align_corners=False,
        )
        coarse_prob = torch.sigmoid(coarse_logits)
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
