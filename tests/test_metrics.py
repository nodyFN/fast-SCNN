"""
Unit tests for SegmentationMetrics.

All tests use hand-crafted small tensors with manually verifiable values.
"""

from __future__ import annotations

import math

import pytest
import torch

from utils.metrics import SegmentationMetrics

NUM_CLASSES = 2


# ===========================================================================
# Helper
# ===========================================================================


def _m() -> SegmentationMetrics:
    return SegmentationMetrics(num_classes=NUM_CLASSES, ignore_index=255)


# ===========================================================================
# Perfect prediction
# ===========================================================================


class TestPerfect:
    def test_perfect_all_metrics(self) -> None:
        m = _m()
        # 2×4×4: all background
        preds = torch.zeros(2, 4, 4, dtype=torch.long)
        targets = torch.zeros(2, 4, 4, dtype=torch.long)
        m.update(preds, targets)
        r = m.compute()
        assert r["pixel_accuracy"] == pytest.approx(1.0)
        # Background IoU = 1.0, foreground absent → NaN
        assert r["per_class_iou"][0] == pytest.approx(1.0)
        assert math.isnan(r["per_class_iou"][1])

    def test_perfect_both_classes(self) -> None:
        m = _m()
        # Top half = 0, bottom half = 1
        preds = torch.zeros(1, 4, 4, dtype=torch.long)
        preds[0, 2:, :] = 1
        targets = preds.clone()
        m.update(preds, targets)
        r = m.compute()
        assert r["pixel_accuracy"] == pytest.approx(1.0)
        assert r["miou"] == pytest.approx(1.0)
        assert r["foreground_iou"] == pytest.approx(1.0)
        assert r["foreground_dice"] == pytest.approx(1.0)


# ===========================================================================
# Completely wrong
# ===========================================================================


class TestWrong:
    def test_completely_wrong(self) -> None:
        m = _m()
        preds = torch.zeros(1, 4, 4, dtype=torch.long)
        targets = torch.ones(1, 4, 4, dtype=torch.long)
        m.update(preds, targets)
        r = m.compute()
        assert r["pixel_accuracy"] == pytest.approx(0.0)
        # Both classes have IoU = 0
        assert r["per_class_iou"][0] == pytest.approx(0.0)
        assert r["per_class_iou"][1] == pytest.approx(0.0)
        assert r["miou"] == pytest.approx(0.0)


# ===========================================================================
# Absent classes
# ===========================================================================


class TestAbsent:
    def test_foreground_absent(self) -> None:
        """When foreground never appears → foreground IoU = NaN."""
        m = _m()
        preds = torch.zeros(1, 4, 4, dtype=torch.long)
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        m.update(preds, targets)
        r = m.compute()
        assert math.isnan(r["foreground_iou"])
        assert math.isnan(r["foreground_dice"])
        # mIoU should only average over background
        assert r["miou"] == pytest.approx(1.0)

    def test_background_absent(self) -> None:
        """When background never appears → background IoU = NaN."""
        m = _m()
        preds = torch.ones(1, 4, 4, dtype=torch.long)
        targets = torch.ones(1, 4, 4, dtype=torch.long)
        m.update(preds, targets)
        r = m.compute()
        assert math.isnan(r["per_class_iou"][0])
        assert r["foreground_iou"] == pytest.approx(1.0)
        assert r["miou"] == pytest.approx(1.0)

    def test_class_absent_both_pred_and_target(self) -> None:
        """Class absent from both → NaN, excluded from mIoU."""
        m = _m()
        preds = torch.zeros(1, 4, 4, dtype=torch.long)
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        m.update(preds, targets)
        r = m.compute()
        # Foreground absent in both → NaN
        assert math.isnan(r["per_class_iou"][1])


# ===========================================================================
# Ignore index
# ===========================================================================


class TestIgnoreIndex:
    def test_ignore_index_excluded(self) -> None:
        m = _m()
        preds = torch.zeros(1, 4, 4, dtype=torch.long)
        targets = torch.full((1, 4, 4), 255, dtype=torch.long)
        m.update(preds, targets)
        r = m.compute()
        # All pixels ignored → no valid pixels → PA=0
        assert r["pixel_accuracy"] == 0.0

    def test_partial_ignore(self) -> None:
        m = _m()
        # 2×2: top-left valid, rest ignored
        preds = torch.zeros(1, 2, 2, dtype=torch.long)
        targets = torch.full((1, 2, 2), 255, dtype=torch.long)
        targets[0, 0, 0] = 0  # one valid pixel, correctly predicted
        m.update(preds, targets)
        r = m.compute()
        assert r["pixel_accuracy"] == pytest.approx(1.0)


# ===========================================================================
# Cross-batch accumulation
# ===========================================================================


class TestAccumulation:
    def test_accumulation(self) -> None:
        m = _m()
        # Batch 1: all background correct
        m.update(
            torch.zeros(1, 4, 4, dtype=torch.long),
            torch.zeros(1, 4, 4, dtype=torch.long),
        )
        # Batch 2: all foreground correct
        m.update(
            torch.ones(1, 4, 4, dtype=torch.long),
            torch.ones(1, 4, 4, dtype=torch.long),
        )
        r = m.compute()
        assert r["pixel_accuracy"] == pytest.approx(1.0)
        assert r["miou"] == pytest.approx(1.0)

    def test_reset_clears(self) -> None:
        m = _m()
        m.update(
            torch.zeros(1, 4, 4, dtype=torch.long),
            torch.zeros(1, 4, 4, dtype=torch.long),
        )
        m.reset()
        r = m.compute()
        assert r["pixel_accuracy"] == 0.0


# ===========================================================================
# Logits input
# ===========================================================================


class TestLogitsInput:
    def test_logits_converted(self) -> None:
        """When 4D logits are passed, argmax(dim=1) should be applied."""
        m = _m()
        # logits: [1, 2, 4, 4] — class 0 has higher logit everywhere
        logits = torch.zeros(1, 2, 4, 4)
        logits[:, 0, :, :] = 10.0  # strongly predict class 0
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        m.update(logits, targets)
        r = m.compute()
        assert r["pixel_accuracy"] == pytest.approx(1.0)


# ===========================================================================
# Manual IoU / Dice verification
# ===========================================================================


class TestManualVerification:
    def test_specific_iou(self) -> None:
        """Manually compute IoU for a known configuration.

        preds:   0 0 1 1     targets: 0 0 0 1
                 0 0 1 1              0 0 0 1
                 0 0 1 1              0 0 1 1
                 0 0 1 1              0 0 1 1

        Class 0: TP=8, FP=2, FN=0 → IoU = 8/10 = 0.8
        Class 1: TP=6, FP=0, FN=2 → IoU = 6/8 = 0.75
        mIoU = (0.8 + 0.75) / 2 = 0.775
        """
        m = _m()
        preds = torch.zeros(1, 4, 4, dtype=torch.long)
        preds[0, :, 2:] = 1
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        targets[0, :, 3] = 1
        targets[0, 2:, 2] = 1
        m.update(preds, targets)
        r = m.compute()
        assert r["per_class_iou"][0] == pytest.approx(0.8, abs=1e-6)
        assert r["per_class_iou"][1] == pytest.approx(0.75, abs=1e-6)
        assert r["miou"] == pytest.approx(0.775, abs=1e-6)

    def test_specific_dice(self) -> None:
        """Same configuration as above.

        Class 0: TP=8, FP=2, FN=0 → Dice = 16/18 ≈ 0.8889
        Class 1: TP=6, FP=0, FN=2 → Dice = 12/14 ≈ 0.8571
        """
        m = _m()
        preds = torch.zeros(1, 4, 4, dtype=torch.long)
        preds[0, :, 2:] = 1
        targets = torch.zeros(1, 4, 4, dtype=torch.long)
        targets[0, :, 3] = 1
        targets[0, 2:, 2] = 1
        m.update(preds, targets)
        r = m.compute()
        assert r["per_class_dice"][0] == pytest.approx(16 / 18, abs=1e-6)
        assert r["per_class_dice"][1] == pytest.approx(12 / 14, abs=1e-6)


# ===========================================================================
# Edge case: empty valid pixels
# ===========================================================================


class TestEmptyPixels:
    def test_all_ignored(self) -> None:
        m = _m()
        preds = torch.zeros(1, 4, 4, dtype=torch.long)
        targets = torch.full((1, 4, 4), 255, dtype=torch.long)
        m.update(preds, targets)
        r = m.compute()
        # No valid pixels → PA=0, all IoU=NaN
        assert r["pixel_accuracy"] == 0.0
        for iou in r["per_class_iou"]:
            assert math.isnan(iou)
