"""
Unit tests for trimap generation, quantization, and file loading.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from utils.trimap import (
    generate_trimap_from_mask,
    load_trimap_from_file,
    quantize_trimap,
    validate_trimap_values,
)

DEVICE = torch.device("cpu")


def test_generate_trimap_from_mask_shapes_and_values() -> None:
    # Batch size 2, 1 channel, 128x128
    mask = torch.zeros(2, 1, 128, 128, dtype=torch.float32, device=DEVICE)
    mask[:, :, 32:96, 32:96] = 1.0  # square in center

    trimap = generate_trimap_from_mask(mask, kernel_min=5, kernel_max=5)
    
    assert trimap.shape == (2, 1, 128, 128)
    assert trimap.device == DEVICE
    assert trimap.dtype == torch.float32

    # Validate that trimap contains exactly {0.0, 0.5, 1.0}
    assert validate_trimap_values(trimap)

    # Known background check: far corner must be 0
    assert trimap[0, 0, 0, 0] == 0.0
    # Known foreground check: center of the square must be 1
    assert trimap[0, 0, 64, 64] == 1.0
    # Transition zone check: around boundary of the square must be 0.5
    # The original boundary was at 32, so with radius 5, the transition zone is [32-5, 32+5] -> 27 to 37
    assert trimap[0, 0, 32, 32] == 0.5


def test_quantize_trimap() -> None:
    # Create trimap with float noise
    noisy = torch.tensor([0.02, 0.24, 0.26, 0.49, 0.74, 0.76, 0.99], dtype=torch.float32)
    expected = torch.tensor([0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0], dtype=torch.float32)
    
    quantized = quantize_trimap(noisy)
    assert torch.allclose(quantized, expected)


def test_load_trimap_from_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "test_trimap.png"

        # 1. Test 0/128/255 format
        arr_255 = np.zeros((64, 64), dtype=np.uint8)
        arr_255[10:20, 10:20] = 128
        arr_255[20:30, 20:30] = 255
        cv2.imwrite(str(tmp_path), arr_255)

        loaded = load_trimap_from_file(tmp_path)
        assert loaded.shape == (64, 64)
        assert loaded.dtype == np.float32
        assert np.allclose(np.unique(loaded), [0.0, 0.5, 1.0])
        assert loaded[15, 15] == 0.5
        assert loaded[25, 25] == 1.0

        # 2. Test 0/1/2 format
        arr_labels = np.zeros((64, 64), dtype=np.uint8)
        arr_labels[10:20, 10:20] = 1
        arr_labels[20:30, 20:30] = 2
        cv2.imwrite(str(tmp_path), arr_labels)

        loaded_labels = load_trimap_from_file(tmp_path)
        assert loaded_labels.shape == (64, 64)
        assert loaded_labels.dtype == np.float32
        assert np.allclose(np.unique(loaded_labels), [0.0, 0.5, 1.0])
        assert loaded_labels[15, 15] == 0.5
        assert loaded_labels[25, 25] == 1.0

        # 3. Test invalid values raises ValueError
        arr_invalid = np.zeros((64, 64), dtype=np.uint8)
        arr_invalid[10:20, 10:20] = 42
        cv2.imwrite(str(tmp_path), arr_invalid)

        with pytest.raises(ValueError):
            load_trimap_from_file(tmp_path)
