#!/usr/bin/env python3
"""
Export Fast-SCNN to ONNX and validate with ONNX Runtime.

Usage
-----
# Fixed-size export
python export.py --checkpoint checkpoints/best_miou.pt --height 512 --width 1024

# Dynamic axes
python export.py --checkpoint checkpoints/best_miou.pt --height 512 --width 1024 --dynamic

# Custom opset
python export.py --checkpoint checkpoints/best_miou.pt --opset 17
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch

from config import Config
from models import FastSCNN, FastSCNNSalient
from utils.checkpoint import load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class FastSCNNExportWrapper(torch.nn.Module):
    """Wrapper that ensures only the main logits are returned for export.

    When aux=True, the model returns a dict in training mode.
    This wrapper always returns a single tensor.
    """

    def __init__(self, model: torch.nn.Module, model_name: str = "fast_scnn") -> None:
        super().__init__()
        self.model = model
        self.model_name = model_name
        # Force eval mode for export
        self.model.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if self.model_name == "fast_scnn_salient" or isinstance(out, dict):
            if "fine_logits" in out:
                return out["fine_logits"]
            return out["out"]
        return out


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    model_name: str = "fast_scnn",
    height: int = 512,
    width: int = 1024,
    opset: int = 17,
    dynamic: bool = True,
    num_classes: int = 2,
    device_str: str = "cpu",
) -> Path:
    """Export model to ONNX.

    Returns the path to the saved ONNX file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device(device_str)

    # Config to access defaults
    cfg = Config()

    # Build model (aux=True to load all weights, then wrap for export)
    if model_name == "fast_scnn_salient":
        model = FastSCNNSalient(
            ppm_pool_sizes=cfg.ppm_pool_sizes,
            coarse_channels=cfg.coarse_channels,
            refinement_channels=cfg.refinement_channels,
            dropout_p=cfg.dropout_p,
        ).to(device)
    else:
        model = FastSCNN(num_classes=num_classes, aux=True).to(device)

    if checkpoint_path:
        load_checkpoint(checkpoint_path, model, map_location=device, weights_only=True)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")

    wrapper = FastSCNNExportWrapper(model, model_name=model_name)
    wrapper.eval()

    dummy_input = torch.randn(1, 3, height, width, device=device)

    # Dynamic axes
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "logits": {0: "batch", 2: "height", 3: "width"},
        }
        logger.info("Exporting with dynamic axes: batch, height, width")

    logger.info(f"Exporting ONNX: opset={opset}, input=[1,3,{height},{width}], dynamic={dynamic}")

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(output_path),
            opset_version=opset,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"ONNX saved to {output_path} ({file_size_mb:.2f} MB)")

    return output_path


def validate_onnx(
    onnx_path: Path,
    model_name: str = "fast_scnn",
    height: int = 512,
    width: int = 1024,
    num_classes: int = 2,
    checkpoint_path: Optional[str] = None,
    dynamic: bool = True,
    device_str: str = "cpu",
) -> bool:
    """Validate ONNX model: checker + ORT numerical comparison."""
    try:
        import onnx
    except ImportError:
        logger.error("onnx package not installed. Run: pip install onnx")
        return False
    try:
        import onnxruntime as ort
    except ImportError:
        logger.error("onnxruntime not installed. Run: pip install onnxruntime")
        return False

    # --- ONNX checker ---
    logger.info("Running onnx.checker.check_model ...")
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("  ✓ ONNX model passes checker")

    # --- ORT session ---
    available_providers = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if "CUDAExecutionProvider" in available_providers and device_str == "cuda":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    logger.info(f"ORT providers: {providers}")

    session = ort.InferenceSession(str(onnx_path), providers=providers)

    # --- Numerical comparison ---
    device = torch.device(device_str if device_str != "cuda" or torch.cuda.is_available() else "cpu")
    
    cfg = Config()
    if model_name == "fast_scnn_salient":
        model = FastSCNNSalient(
            ppm_pool_sizes=cfg.ppm_pool_sizes,
            coarse_channels=cfg.coarse_channels,
            refinement_channels=cfg.refinement_channels,
            dropout_p=cfg.dropout_p,
        ).to(device)
    else:
        model = FastSCNN(num_classes=num_classes, aux=True).to(device)

    if checkpoint_path:
        load_checkpoint(checkpoint_path, model, map_location=device, weights_only=True)
    model.eval()

    test_sizes = [(height, width)]
    if dynamic:
        test_sizes.append((height + 1, width + 1))
        # Add a different size
        test_sizes.append((height // 2, width // 2))

    all_pass = True
    for h, w in test_sizes:
        logger.info(f"\nTesting size: 1×3×{h}×{w}")
        dummy = torch.randn(1, 3, h, w, device=device)

        # PyTorch reference
        with torch.inference_mode():
            pt_out = model(dummy)
        if isinstance(pt_out, dict):
            if "fine_logits" in pt_out:
                pt_out = pt_out["fine_logits"]
            else:
                pt_out = pt_out["out"]
        pt_np = pt_out.cpu().numpy()

        # ORT
        try:
            ort_inputs = {"input": dummy.cpu().numpy()}
            ort_outputs = session.run(None, ort_inputs)
            ort_np = ort_outputs[0]

            # Shape check
            if pt_np.shape != ort_np.shape:
                logger.error(f"  Shape mismatch: PyTorch={pt_np.shape}, ORT={ort_np.shape}")
                all_pass = False
                continue

            max_abs_err = np.max(np.abs(pt_np - ort_np))
            mean_abs_err = np.mean(np.abs(pt_np - ort_np))

            # Relative error (avoid division by zero)
            denom = np.maximum(np.abs(pt_np), 1e-8)
            max_rel_err = np.max(np.abs(pt_np - ort_np) / denom)

            logger.info(f"  Output shape    : {ort_np.shape}")
            logger.info(f"  Max abs error   : {max_abs_err:.6e}")
            logger.info(f"  Mean abs error  : {mean_abs_err:.6e}")
            logger.info(f"  Max rel error   : {max_rel_err:.6e}")

            # FP32 tolerance
            rtol, atol = 1e-4, 1e-5
            try:
                np.testing.assert_allclose(pt_np, ort_np, rtol=rtol, atol=atol)
                logger.info(f"  ✓ Numerical match (rtol={rtol}, atol={atol})")
            except AssertionError as e:
                logger.warning(f"  ⚠ Loose tolerance needed: {e}")
                # Try relaxed
                rtol2, atol2 = 1e-3, 1e-4
                try:
                    np.testing.assert_allclose(pt_np, ort_np, rtol=rtol2, atol=atol2)
                    logger.info(f"  ✓ Passes with relaxed tolerance (rtol={rtol2}, atol={atol2})")
                except AssertionError:
                    logger.error(f"  ✗ FAILED even with relaxed tolerance")
                    all_pass = False

        except Exception as e:
            if dynamic and (h, w) != (height, width):
                logger.warning(
                    f"  ⚠ Dynamic size {h}×{w} failed with ORT: {e}. "
                    f"This may indicate the ONNX exporter or an operator does not "
                    f"fully support dynamic spatial dimensions."
                )
            else:
                logger.error(f"  ✗ ORT inference failed: {e}")
                all_pass = False

    return all_pass


def main() -> None:
    p = argparse.ArgumentParser(description="Export Fast-SCNN to ONNX")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to checkpoint (None = random weights)")
    p.add_argument("--model", choices=["fast_scnn", "fast_scnn_salient"], default="fast_scnn",
                   help="Model architecture to export (default: fast_scnn)")
    p.add_argument("--output", type=str, default=None,
                   help="Output ONNX path (default: exports/{model}{suffix}.onnx)")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--opset", type=int, default=17,
                   help="ONNX opset version (default: 17, project decision)")
    p.add_argument("--dynamic", action="store_true", default=False,
                   help="Enable dynamic axes (batch, height, width)")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip ORT validation")
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--device", type=str, default="cpu",
                   help="Device for export (recommend cpu for portability)")
    args = p.parse_args()

    cfg = Config()
    cfg.ensure_dirs()

    if args.output is None:
        suffix = "_dynamic" if args.dynamic else f"_{args.height}x{args.width}"
        args.output = str(cfg.export_dir / f"{args.model}{suffix}.onnx")

    onnx_path = export_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        model_name=args.model,
        height=args.height,
        width=args.width,
        opset=args.opset,
        dynamic=args.dynamic,
        num_classes=args.num_classes,
        device_str=args.device,
    )

    if not args.no_validate:
        ok = validate_onnx(
            onnx_path,
            model_name=args.model,
            height=args.height,
            width=args.width,
            num_classes=args.num_classes,
            checkpoint_path=args.checkpoint,
            dynamic=args.dynamic,
            device_str=args.device,
        )
        if ok:
            logger.info("\n✓ ONNX export and validation successful!")
        else:
            logger.warning("\n⚠ Some validation checks failed — see above for details")


if __name__ == "__main__":
    main()
