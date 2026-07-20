"""
Unit tests for SegmentationDataset.

Uses pytest tmp_path to create temporary fake data.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
import torch
from pathlib import Path

from dataset import SegmentationDataset, build_train_transform, build_val_transform


# ===========================================================================
# Helpers
# ===========================================================================


def _create_fake_data(
    root: Path,
    num_images: int = 3,
    height: int = 64,
    width: int = 128,
    mask_max_value: int = 1,
) -> None:
    """Create synthetic image/mask pairs under root/images and root/masks."""
    img_dir = root / "images"
    mask_dir = root / "masks"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_images):
        img = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (height, width), dtype=np.uint8) * mask_max_value

        cv2.imwrite(str(img_dir / f"frame_{i:04d}.jpg"), img)
        cv2.imwrite(str(mask_dir / f"frame_{i:04d}.png"), mask)


# ===========================================================================
# Basic functionality
# ===========================================================================


class TestBasicPairing:
    def test_correct_pairing(self, tmp_path: Path) -> None:
        _create_fake_data(tmp_path, num_images=5)
        ds = SegmentationDataset(tmp_path)
        assert len(ds) == 5

    def test_01_mask_values(self, tmp_path: Path) -> None:
        """Mask with {0, 1} should be used directly."""
        _create_fake_data(tmp_path, num_images=2, mask_max_value=1)
        ds = SegmentationDataset(tmp_path, transform=build_val_transform(32, 64))
        sample = ds[0]
        assert sample["mask"].dtype == torch.long
        unique = torch.unique(sample["mask"])
        for v in unique:
            assert v.item() in (0, 1)

    def test_0255_mask_conversion(self, tmp_path: Path) -> None:
        """Mask with {0, 255} should convert to {0, 1}."""
        _create_fake_data(tmp_path, num_images=2, mask_max_value=255)
        ds = SegmentationDataset(tmp_path, transform=build_val_transform(32, 64))
        sample = ds[0]
        unique = torch.unique(sample["mask"])
        for v in unique:
            assert v.item() in (0, 1)

    def test_image_tensor_shape(self, tmp_path: Path) -> None:
        _create_fake_data(tmp_path, num_images=1, height=64, width=128)
        ds = SegmentationDataset(tmp_path, transform=build_val_transform(32, 64))
        sample = ds[0]
        assert sample["image"].ndim == 3
        assert sample["image"].shape[0] == 3  # C, H, W

    def test_image_dtype(self, tmp_path: Path) -> None:
        _create_fake_data(tmp_path, num_images=1)
        ds = SegmentationDataset(tmp_path, transform=build_val_transform(32, 64))
        assert ds[0]["image"].dtype == torch.float32

    def test_mask_dtype(self, tmp_path: Path) -> None:
        _create_fake_data(tmp_path, num_images=1)
        ds = SegmentationDataset(tmp_path, transform=build_val_transform(32, 64))
        assert ds[0]["mask"].dtype == torch.long

    def test_mask_shape(self, tmp_path: Path) -> None:
        _create_fake_data(tmp_path, num_images=1)
        ds = SegmentationDataset(tmp_path, transform=build_val_transform(32, 64))
        mask = ds[0]["mask"]
        assert mask.ndim == 2  # [H, W]


# ===========================================================================
# Error handling
# ===========================================================================


class TestErrors:
    def test_missing_mask(self, tmp_path: Path) -> None:
        """Should raise FileNotFoundError when mask is missing."""
        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        img = np.zeros((32, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test_001.jpg"), img)
        # No mask for test_001

        with pytest.raises(FileNotFoundError, match="Mask not found"):
            SegmentationDataset(tmp_path)

    def test_illegal_mask_values(self, tmp_path: Path) -> None:
        """Mask with values like {0, 128, 255} should raise ValueError."""
        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        img = np.zeros((32, 64, 3), dtype=np.uint8)
        mask = np.full((32, 64), 128, dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.jpg"), img)
        cv2.imwrite(str(mask_dir / "test.png"), mask)

        ds = SegmentationDataset(tmp_path)
        with pytest.raises(ValueError, match="unexpected values"):
            ds[0]

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Empty images/masks directories should raise RuntimeError."""
        (tmp_path / "images").mkdir(parents=True)
        (tmp_path / "masks").mkdir(parents=True)

        with pytest.raises(RuntimeError, match="No image/mask pairs"):
            SegmentationDataset(tmp_path)

    def test_nonexistent_root(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            SegmentationDataset(tmp_path / "nonexistent")

    def test_size_mismatch(self, tmp_path: Path) -> None:
        """Image and mask with different spatial sizes should raise ValueError."""
        img_dir = tmp_path / "images"
        mask_dir = tmp_path / "masks"
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        img = np.zeros((64, 128, 3), dtype=np.uint8)
        mask = np.zeros((32, 64), dtype=np.uint8)
        cv2.imwrite(str(img_dir / "test.jpg"), img)
        cv2.imwrite(str(mask_dir / "test.png"), mask)

        ds = SegmentationDataset(tmp_path)
        with pytest.raises(ValueError, match="mismatch"):
            ds[0]


# ===========================================================================
# Transforms
# ===========================================================================


class TestTransforms:
    def test_train_transform_sync(self, tmp_path: Path) -> None:
        """Train transform should produce same-size image and mask."""
        _create_fake_data(tmp_path, num_images=2, height=128, width=256)
        transform = build_train_transform(64, 128)
        ds = SegmentationDataset(tmp_path, transform=transform)
        sample = ds[0]
        img_h, img_w = sample["image"].shape[1], sample["image"].shape[2]
        mask_h, mask_w = sample["mask"].shape
        assert (img_h, img_w) == (mask_h, mask_w)

    def test_val_transform_deterministic(self, tmp_path: Path) -> None:
        """Val transform should produce identical outputs across calls."""
        _create_fake_data(tmp_path, num_images=1, height=128, width=256)
        transform = build_val_transform(64, 128)
        ds = SegmentationDataset(tmp_path, transform=transform)
        s1 = ds[0]
        s2 = ds[0]
        assert torch.allclose(s1["image"], s2["image"])
        assert torch.equal(s1["mask"], s2["mask"])
