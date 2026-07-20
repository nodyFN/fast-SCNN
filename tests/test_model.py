"""
Unit tests for Fast-SCNN model.

Tests cover:
- Basic output shape (eval, no aux)
- Auxiliary output dict (train, aux=True)
- Eval mode with aux=True → single tensor
- Various input sizes (odd, non-32-multiple)
- Backward pass / gradient check
- Residual connection conditions
- Parameter count sanity check
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from models.fast_scnn import (
    FastSCNN,
    LinearBottleneck,
    count_parameters,
)

NUM_CLASSES = 2
DEVICE = torch.device("cpu")


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def model_no_aux() -> FastSCNN:
    return FastSCNN(num_classes=NUM_CLASSES, aux=False).to(DEVICE).eval()


@pytest.fixture
def model_aux() -> FastSCNN:
    return FastSCNN(num_classes=NUM_CLASSES, aux=True).to(DEVICE)


# ===========================================================================
# Basic output
# ===========================================================================


class TestBasicOutput:
    def test_eval_returns_tensor(self, model_no_aux: FastSCNN) -> None:
        x = torch.randn(1, 3, 64, 128, device=DEVICE)
        with torch.inference_mode():
            out = model_no_aux(x)
        assert isinstance(out, torch.Tensor)

    def test_output_channels(self, model_no_aux: FastSCNN) -> None:
        x = torch.randn(1, 3, 64, 128, device=DEVICE)
        with torch.inference_mode():
            out = model_no_aux(x)
        assert out.shape[1] == NUM_CLASSES

    def test_output_matches_input_spatial(self, model_no_aux: FastSCNN) -> None:
        h, w = 128, 256
        x = torch.randn(1, 3, h, w, device=DEVICE)
        with torch.inference_mode():
            out = model_no_aux(x)
        assert out.shape == (1, NUM_CLASSES, h, w)


# ===========================================================================
# Auxiliary output
# ===========================================================================


class TestAuxiliaryOutput:
    def test_train_returns_dict(self, model_aux: FastSCNN) -> None:
        model_aux.train()
        x = torch.randn(2, 3, 64, 128, device=DEVICE)
        with torch.no_grad():
            out = model_aux(x)
        assert isinstance(out, dict)

    def test_dict_keys(self, model_aux: FastSCNN) -> None:
        model_aux.train()
        x = torch.randn(2, 3, 64, 128, device=DEVICE)
        with torch.no_grad():
            out = model_aux(x)
        assert "out" in out
        assert "aux_downsample" in out
        assert "aux_global" in out

    def test_all_shapes_match(self, model_aux: FastSCNN) -> None:
        h, w = 64, 128
        model_aux.train()
        x = torch.randn(2, 3, h, w, device=DEVICE)
        with torch.no_grad():
            out = model_aux(x)
        for key in ("out", "aux_downsample", "aux_global"):
            assert out[key].shape == (2, NUM_CLASSES, h, w), (
                f"{key}: expected [2, {NUM_CLASSES}, {h}, {w}], got {list(out[key].shape)}"
            )


# ===========================================================================
# Eval mode with aux=True
# ===========================================================================


class TestEvalWithAux:
    def test_eval_returns_tensor(self, model_aux: FastSCNN) -> None:
        model_aux.eval()
        x = torch.randn(1, 3, 64, 128, device=DEVICE)
        with torch.inference_mode():
            out = model_aux(x)
        assert isinstance(out, torch.Tensor)
        assert out.shape == (1, NUM_CLASSES, 64, 128)


# ===========================================================================
# Different input sizes
# ===========================================================================


class TestDifferentSizes:
    @pytest.mark.parametrize(
        "h,w",
        [(64, 128), (65, 129), (127, 255), (513, 1025), (96, 160)],
        ids=["64x128", "65x129", "127x255", "513x1025", "96x160"],
    )
    def test_various_sizes(self, model_no_aux: FastSCNN, h: int, w: int) -> None:
        x = torch.randn(1, 3, h, w, device=DEVICE)
        with torch.inference_mode():
            out = model_no_aux(x)
        assert out.shape == (1, NUM_CLASSES, h, w)

    def test_batch_size_2(self, model_no_aux: FastSCNN) -> None:
        x = torch.randn(2, 3, 64, 128, device=DEVICE)
        with torch.inference_mode():
            out = model_no_aux(x)
        assert out.shape[0] == 2


# ===========================================================================
# Backward pass
# ===========================================================================


class TestBackward:
    def test_backward_runs(self) -> None:
        """Test that forward→loss→backward produces valid gradients."""
        # Use batch_size=2 for BatchNorm compatibility
        model = FastSCNN(num_classes=NUM_CLASSES, aux=True).to(DEVICE)
        model.train()
        x = torch.randn(2, 3, 64, 128, device=DEVICE)
        target = torch.randint(0, NUM_CLASSES, (2, 64, 128), device=DEVICE)

        out = model(x)
        loss = nn.CrossEntropyLoss()(out["out"], target)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


# ===========================================================================
# Residual connection conditions
# ===========================================================================


class TestResidual:
    def test_residual_when_same_channels_stride_1(self) -> None:
        block = LinearBottleneck(64, 64, expansion=6, stride=1)
        assert block.use_residual is True

    def test_no_residual_when_stride_2(self) -> None:
        block = LinearBottleneck(64, 64, expansion=6, stride=2)
        assert block.use_residual is False

    def test_no_residual_when_channels_differ(self) -> None:
        block = LinearBottleneck(64, 96, expansion=6, stride=1)
        assert block.use_residual is False

    def test_no_residual_when_both_differ(self) -> None:
        block = LinearBottleneck(64, 96, expansion=6, stride=2)
        assert block.use_residual is False


# ===========================================================================
# Parameter count
# ===========================================================================


class TestParameterCount:
    def test_lightweight(self) -> None:
        """Fast-SCNN should be lightweight (~1.1M in TF).
        PyTorch version may differ slightly due to implementation details.
        We check it's below 3M total as a generous upper bound.
        """
        model = FastSCNN(num_classes=NUM_CLASSES, aux=True)
        total, trainable = count_parameters(model)
        # The model should be well under 3M params
        assert total < 3_000_000, (
            f"Model has {total:,} params — expected < 3M for Fast-SCNN"
        )
        # And at least some reasonable minimum
        assert total > 500_000, (
            f"Model has only {total:,} params — too few, likely missing components"
        )
