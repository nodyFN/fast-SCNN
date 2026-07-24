"""
Fast-SCNN: Fast Semantic Segmentation Network
==============================================

Implementation follows the paper architecture from Table 1 / Table 2 / Table 3:

    Input
      → Learning to Downsample   (3 layers, output 1/8 res, 64-ch)
      → Global Feature Extractor  (MobileNetV2 bottlenecks, output 1/32 res, 128-ch)
      → Pyramid Pooling Module    (multi-scale context, 128-ch)
      → Feature Fusion Module     (skip from LtD + PPM output, 128-ch at 1/8 res)
      → Classifier               (2×DSConv → Dropout → 1×1 → upsample)

Paper settings
--------------
- Activation: ReLU (NOT ReLU6)
- BatchNorm after every conv (except where noted)
- Linear bottleneck: no activation after final 1×1 projection
- DSConv: no ReLU between depthwise and pointwise
- Single skip connection from LtD to FFM

Project implementation decisions (NOT from paper)
-------------------------------------------------
- PPM pool_sizes = (1, 2, 3, 6), branch_channels = 32
- Dropout probability = 0.1
- AuxiliaryHead: 3×3 ConvBNReLU → Dropout → 1×1 Conv
- align_corners = False for all bilinear interpolation
- Kaiming initialization for conv weights, BN weight=1 / bias=0
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================================================
# Building blocks
# ===========================================================================


class ConvBNReLU(nn.Module):
    """Conv2d → BatchNorm2d → ReLU (optional).

    When conv is followed by BN, bias is disabled by default.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        relu: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if relu:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise Separable Convolution.

    Structure (per paper):
        3×3 Depthwise Conv → BN → 1×1 Pointwise Conv → BN → ReLU

    No ReLU between depthwise and pointwise as specified by the paper.
    The final ReLU can be disabled via ``relu=False`` for modules that
    require a linear output before addition (e.g. FFM branches).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        relu: bool = True,
    ) -> None:
        super().__init__()
        # Depthwise
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.bn_dw = nn.BatchNorm2d(in_channels)
        # Pointwise
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn_pw = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn_dw(self.depthwise(x))
        x = self.bn_pw(self.pointwise(x))
        if self.relu is not None:
            x = self.relu(x)
        return x


class LinearBottleneck(nn.Module):
    """MobileNetV2-style Inverted Residual Linear Bottleneck.

    Structure:
        1×1 expand (c_in → t*c_in) → BN → ReLU
        3×3 depthwise (t*c_in, stride=s) → BN → ReLU
        1×1 project (t*c_in → c_out) → BN → NO activation

    Residual connection only when stride == 1 AND c_in == c_out.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int = 6,
        stride: int = 1,
    ) -> None:
        super().__init__()
        mid_channels = in_channels * expansion
        self.use_residual = (stride == 1) and (in_channels == out_channels)

        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=mid_channels,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        # Linear projection – no activation
        self.project = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)
        if self.use_residual:
            out = out + x
        return out


# ===========================================================================
# Major modules
# ===========================================================================


class LearningToDownsample(nn.Module):
    """Learning to Downsample — three layers with total stride 8.

    Layer 1: standard 3×3 Conv (stride 2) — standard conv because 3 input
              channels make DSConv inefficient.
    Layer 2: 3×3 DSConv (stride 2)  32 → 48
    Layer 3: 3×3 DSConv (stride 2)  48 → 64
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = ConvBNReLU(3, 32, kernel_size=3, stride=2, padding=1)
        self.dsconv1 = DepthwiseSeparableConv(32, 48, stride=2)
        self.dsconv2 = DepthwiseSeparableConv(48, 64, stride=2)

    def forward_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward features and return intermediate outputs at H/2, H/4, and H/8."""
        feat_h2 = self.conv(x)
        feat_h4 = self.dsconv1(feat_h2)
        feat_h8_skip = self.dsconv2(feat_h4)
        return {
            "feat_h2": feat_h2,
            "feat_h4": feat_h4,
            "feat_h8_skip": feat_h8_skip,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return x


class GlobalFeatureExtractor(nn.Module):
    """Global Feature Extractor using MobileNetV2 bottlenecks.

    Three stages (paper Table 1):
        Stage 1:  64 → 64,  t=6, n=3, s=2   (1/8 → 1/16)
        Stage 2:  64 → 96,  t=6, n=3, s=2   (1/16 → 1/32)
        Stage 3:  96 → 128, t=6, n=3, s=1   (stays 1/32)
    """

    def __init__(self) -> None:
        super().__init__()
        self.bottlenecks = nn.Sequential(
            # Stage 1: 64 → 64
            *self._make_stage(64, 64, expansion=6, num_blocks=3, stride=2),
            # Stage 2: 64 → 96
            *self._make_stage(64, 96, expansion=6, num_blocks=3, stride=2),
            # Stage 3: 96 → 128
            *self._make_stage(96, 128, expansion=6, num_blocks=3, stride=1),
        )

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        expansion: int,
        num_blocks: int,
        stride: int,
    ) -> List[LinearBottleneck]:
        layers: List[LinearBottleneck] = []
        # First block uses the given stride and changes channels
        layers.append(
            LinearBottleneck(in_channels, out_channels, expansion, stride)
        )
        # Remaining blocks: stride=1, channels stay at out_channels
        for _ in range(1, num_blocks):
            layers.append(
                LinearBottleneck(out_channels, out_channels, expansion, 1)
            )
        return layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bottlenecks(x)


class PyramidPooling(nn.Module):
    """Pyramid Pooling Module (PPM).

    [PROJECT DECISION] The paper references PSPNet-style PPM but does not
    specify pool_sizes or branch channels for Fast-SCNN.

    Default: pool_sizes=(1, 2, 3, 6), branch_channels=32, out=128.

    WARNING: When batch_size=1, the 1×1 pooling branch produces a single
    spatial value per channel, which can cause BatchNorm instability during
    training.  Use batch_size > 1 for training, or freeze BN / switch to
    GroupNorm if needed.
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 128,
        pool_sizes: Tuple[int, ...] = (1, 2, 3, 6),
        branch_channels: int = 32,
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList()
        for ps in pool_sizes:
            self.branches.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(ps),
                    nn.Conv2d(in_channels, branch_channels, 1, bias=False),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=True),
                )
            )
        # Fusion: concat(original + branches) → 1×1 conv
        concat_channels = in_channels + branch_channels * len(pool_sizes)
        self.fusion = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_size = x.shape[-2:]
        branch_outs = [x]
        for branch in self.branches:
            b = branch(x)
            b = F.interpolate(
                b, size=input_size, mode="bilinear", align_corners=False
            )
            branch_outs.append(b)
        out = torch.cat(branch_outs, dim=1)
        return self.fusion(out)


class FeatureFusionModule(nn.Module):
    """Feature Fusion Module (FFM).

    Fuses high-resolution skip features from LtD (64-ch, ~1/8) with
    low-resolution global features from GFE+PPM (128-ch, ~1/32).

    Low-res branch (paper Table 3):
        Bilinear upsample to high-res spatial size
        → 3×3 DW conv (dilation=4, padding=4)
        → BN → ReLU
        → 1×1 PW conv → BN (no activation before add)

    High-res branch:
        1×1 Conv → BN (no activation before add)

    Fusion: element-wise add → ReLU

    ``align_corners=False`` is used for bilinear interpolation for stability
    across arbitrary input sizes and consistency between PyTorch / ONNX Runtime.
    """

    def __init__(
        self,
        high_channels: int = 64,
        low_channels: int = 128,
        out_channels: int = 128,
    ) -> None:
        super().__init__()
        # Low-resolution branch
        self.low_dw = nn.Conv2d(
            low_channels,
            low_channels,
            kernel_size=3,
            stride=1,
            padding=4,
            dilation=4,
            groups=low_channels,
            bias=False,
        )
        self.low_bn_dw = nn.BatchNorm2d(low_channels)
        self.low_relu = nn.ReLU(inplace=True)
        self.low_pw = nn.Conv2d(low_channels, out_channels, 1, bias=False)
        self.low_bn_pw = nn.BatchNorm2d(out_channels)

        # High-resolution branch
        self.high_proj = nn.Conv2d(high_channels, out_channels, 1, bias=False)
        self.high_bn = nn.BatchNorm2d(out_channels)

        # Post-fusion activation
        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        high_res: torch.Tensor,
        low_res: torch.Tensor,
    ) -> torch.Tensor:
        # Use actual spatial size of high-res feature for alignment
        target_size = high_res.shape[-2:]

        # Low-resolution branch: upsample first, then DW→BN→ReLU→PW→BN
        low = F.interpolate(
            low_res, size=target_size, mode="bilinear", align_corners=False
        )
        low = self.low_relu(self.low_bn_dw(self.low_dw(low)))
        low = self.low_bn_pw(self.low_pw(low))  # no activation

        # High-resolution branch: 1×1→BN (no activation)
        high = self.high_bn(self.high_proj(high_res))

        # Element-wise addition → ReLU
        return self.relu(high + low)


class Classifier(nn.Module):
    """Classifier head.

    Structure (paper):
        DSConv 128→128 (stride 1)
        DSConv 128→128 (stride 1)
        Dropout  [PROJECT DECISION: p=0.1]
        1×1 Conv 128→num_classes
    """

    def __init__(
        self,
        in_channels: int = 128,
        num_classes: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.dsconv1 = DepthwiseSeparableConv(in_channels, in_channels, stride=1)
        self.dsconv2 = DepthwiseSeparableConv(in_channels, in_channels, stride=1)
        self.dropout = nn.Dropout2d(p=dropout_p)
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        x = self.dropout(x)
        return self.conv(x)


class AuxiliaryHead(nn.Module):
    """Lightweight auxiliary classification head.

    [PROJECT DECISION] The paper does not specify the exact auxiliary head
    architecture.  We use a simple:
        3×3 ConvBNReLU → Dropout → 1×1 Conv → bilinear upsample

    Used at two locations during training:
        1. After Learning to Downsample (in_channels=64)
        2. After Global Feature Extractor + PPM (in_channels=128)
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int = 2,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            ConvBNReLU(in_channels, in_channels, kernel_size=3, padding=1),
            nn.Dropout2d(p=dropout_p),
            nn.Conv2d(in_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        x = self.head(x)
        return F.interpolate(
            x, size=target_size, mode="bilinear", align_corners=False
        )


# ===========================================================================
# Full model
# ===========================================================================


class FastSCNN(nn.Module):
    """Fast-SCNN: Fast Semantic Segmentation Network.

    Parameters
    ----------
    num_classes : int
        Number of segmentation classes (default 2 for binary segmentation).
    aux : bool
        If True, build auxiliary heads for training.  When ``aux=True`` and
        ``self.training is True``, forward returns a dict with keys
        ``"out"``, ``"aux_downsample"``, ``"aux_global"``.
        Otherwise forward returns a single ``[B, num_classes, H, W]`` tensor.
    ppm_pool_sizes : tuple of int
        Pool sizes for the Pyramid Pooling Module.
        [PROJECT DECISION] default = (1, 2, 3, 6).
    dropout_p : float
        Dropout probability in the Classifier and AuxiliaryHeads.
        [PROJECT DECISION] default = 0.1.

    Outputs (raw logits — no softmax / sigmoid / argmax):
        Training with aux : dict  {"out", "aux_downsample", "aux_global"}
        Eval or aux=False : Tensor [B, num_classes, H, W]
    """

    def __init__(
        self,
        num_classes: int = 2,
        aux: bool = True,
        ppm_pool_sizes: Tuple[int, ...] = (1, 2, 3, 6),
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.aux = aux

        # Main backbone
        self.learning_to_downsample = LearningToDownsample()
        self.global_feature_extractor = GlobalFeatureExtractor()
        self.ppm = PyramidPooling(
            in_channels=128,
            out_channels=128,
            pool_sizes=ppm_pool_sizes,
        )
        self.ffm = FeatureFusionModule(
            high_channels=64, low_channels=128, out_channels=128
        )
        self.classifier = Classifier(
            in_channels=128, num_classes=num_classes, dropout_p=dropout_p
        )

        # Auxiliary heads (only built when aux=True)
        if self.aux:
            self.aux_head_downsample = AuxiliaryHead(
                in_channels=64, num_classes=num_classes, dropout_p=dropout_p
            )
            self.aux_head_global = AuxiliaryHead(
                in_channels=128, num_classes=num_classes, dropout_p=dropout_p
            )

        self._init_weights()

    def _init_weights(self) -> None:
        """Kaiming initialization for conv weights; BN weight=1, bias=0."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the fused feature map without applying the classifier.

        This is used by downstream models (e.g. FastSCNNSalient) that share
        the backbone but attach their own heads.

        Returns
        -------
        fused : Tensor  [B, 128, H/8, W/8]
            Feature-fusion output at ~1/8 input resolution.
        """
        ltd_out = self.learning_to_downsample(x)
        gfe_out = self.global_feature_extractor(ltd_out)
        ppm_out = self.ppm(gfe_out)
        fused = self.ffm(high_res=ltd_out, low_res=ppm_out)
        return fused

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        input_size = x.shape[-2:]  # (H, W)

        # Learning to Downsample — produces skip feature at ~1/8
        ltd_out = self.learning_to_downsample(x)

        # Global Feature Extractor — ~1/32
        gfe_out = self.global_feature_extractor(ltd_out)

        # Pyramid Pooling — still ~1/32, 128-ch
        ppm_out = self.ppm(gfe_out)

        # Feature Fusion — combines skip (1/8) + global (1/32) → 1/8
        fused = self.ffm(high_res=ltd_out, low_res=ppm_out)

        # Classifier → logits at ~1/8, then upsample to input size
        logits = self.classifier(fused)
        logits = F.interpolate(
            logits, size=input_size, mode="bilinear", align_corners=False
        )

        # Auxiliary outputs for training
        if self.aux and self.training:
            aux_ds = self.aux_head_downsample(ltd_out, target_size=input_size)
            aux_gl = self.aux_head_global(ppm_out, target_size=input_size)
            return {
                "out": logits,
                "aux_downsample": aux_ds,
                "aux_global": aux_gl,
            }

        return logits


# ===========================================================================
# Fast-SCNN Matting Adapter (for DDC Alpha-Free Matting task mode)
# ===========================================================================


class FastSCNNMattingAdapter(nn.Module):
    """Adapter wrapper converting standard single-channel Fast-SCNN output
    to match the dual-head interface expected by DDC matting training loops.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.model(x)  # [B, 1, H, W]
        prob = torch.sigmoid(logits)
        return {
            "coarse_logits": logits,
            "coarse_prob": prob,
            "fine_logits": logits,
            "fine_prob": prob,
            "coarse_logits_lowres": logits,
            "coarse_prompt": prob,
        }


# ===========================================================================
# Utilities for benchmarking
# ===========================================================================


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (total_params, trainable_params)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _benchmark(
    model: nn.Module,
    device: torch.device,
    height: int,
    width: int,
    batch_size: int = 1,
    warmup: int = 10,
    iterations: int = 100,
) -> None:
    """Run FPS / latency benchmark."""
    model.eval()
    # Pre-allocate input on device
    dummy = torch.randn(batch_size, 3, height, width, device=device)

    use_cuda = device.type == "cuda"

    # Warm-up
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(dummy)
    if use_cuda:
        torch.cuda.synchronize()

    # Timed iterations
    if use_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(iterations):
            _ = model(dummy)
    if use_cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    total_time = t1 - t0
    total_images = batch_size * iterations
    avg_latency_ms = (total_time / iterations) * 1000
    fps = total_images / total_time

    print(f"\n{'='*60}")
    print(f"Benchmark Results")
    print(f"{'='*60}")
    print(f"  Device           : {device}")
    print(f"  Input resolution : {height}×{width}")
    print(f"  Batch size       : {batch_size}")
    print(f"  Warm-up iters    : {warmup}")
    print(f"  Timed iters      : {iterations}")
    print(f"  Total images     : {total_images}")
    print(f"  Total time       : {total_time:.3f} s")
    print(f"  Avg latency/batch: {avg_latency_ms:.2f} ms")
    print(f"  FPS (images/sec) : {fps:.1f}")
    print(f"{'='*60}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast-SCNN model test & benchmark")
    parser.add_argument("--height", type=int, default=512, help="Input height (default: 512)")
    parser.add_argument("--width", type=int, default=1024, help="Input width (default: 1024)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up iterations")
    parser.add_argument("--iterations", type=int, default=100, help="Timed iterations")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto|cuda|cpu")
    parser.add_argument("--aux", action="store_true", help="Enable auxiliary heads")
    parser.add_argument(
        "--full-res",
        action="store_true",
        help="Use 1080×1920 instead of default 512×1024",
    )
    args = parser.parse_args()

    if args.full_res:
        args.height, args.width = 1080, 1920

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Building FastSCNN (num_classes={args.num_classes}, aux={args.aux})")
    model = FastSCNN(num_classes=args.num_classes, aux=args.aux).to(device)

    total, trainable = count_parameters(model)
    print(f"Total parameters     : {total:,}")
    print(f"Trainable parameters : {trainable:,}")
    print(f"Model size (approx)  : {total * 4 / 1024 / 1024:.2f} MB (FP32)")

    # --- Shape test ---
    print(f"\n--- Shape Test (eval mode) ---")
    model.eval()
    x = torch.randn(args.batch_size, 3, args.height, args.width, device=device)
    with torch.inference_mode():
        out = model(x)
    if isinstance(out, dict):
        for k, v in out.items():
            print(f"  {k}: {list(v.shape)}")
    else:
        print(f"  output: {list(out.shape)}")
        assert out.shape == (args.batch_size, args.num_classes, args.height, args.width), (
            f"Expected [{args.batch_size}, {args.num_classes}, {args.height}, {args.width}], "
            f"got {list(out.shape)}"
        )
    print("  ✓ shape test passed")

    # --- Aux dict test ---
    if args.aux:
        print(f"\n--- Auxiliary Output Test (train mode) ---")
        model.train()
        with torch.no_grad():
            out_dict = model(x)
        assert isinstance(out_dict, dict), "Expected dict output in train mode with aux=True"
        for key in ("out", "aux_downsample", "aux_global"):
            assert key in out_dict, f"Missing key '{key}'"
            shape = out_dict[key].shape
            assert shape == (args.batch_size, args.num_classes, args.height, args.width), (
                f"{key}: expected [{args.batch_size}, {args.num_classes}, {args.height}, {args.width}], "
                f"got {list(shape)}"
            )
            print(f"  {key}: {list(shape)}  ✓")
        model.eval()

    # --- Non-32-multiple test ---
    print(f"\n--- Odd-size Test ---")
    for h, w in [(513, 1025), (65, 129), (127, 255)]:
        x_odd = torch.randn(1, 3, h, w, device=device)
        with torch.inference_mode():
            out_odd = model(x_odd)
        assert out_odd.shape == (1, args.num_classes, h, w), (
            f"Odd-size {h}×{w}: expected [1, {args.num_classes}, {h}, {w}], got {list(out_odd.shape)}"
        )
        print(f"  {h}×{w} → {list(out_odd.shape)}  ✓")

    # --- Benchmark ---
    _benchmark(
        model,
        device,
        args.height,
        args.width,
        batch_size=args.batch_size,
        warmup=args.warmup,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
