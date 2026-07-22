"""
Trimap generation and processing utilities for DDC matting training.

Provides GPU-accelerated trimap generation from binary masks using
F.max_pool2d (no OpenCV dependency during training), file loading,
and floating-point quantization after geometric transforms.

Trimap value convention
-----------------------
- 0.0  = Known Background
- 0.5  = Unknown / Transition
- 1.0  = Known Foreground

File format: 0/128/255 grayscale PNG → normalized to 0.0/0.5/1.0.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def generate_trimap_from_mask(
    mask: torch.Tensor,
    kernel_min: int = 1,
    kernel_max: int = 30,
) -> torch.Tensor:
    """Generate trimap from binary mask using random erosion/dilation.

    Uses F.max_pool2d for morphological operations (GPU-accelerated,
    no OpenCV dependency during training).

    Parameters
    ----------
    mask : [B, 1, H, W]
        Binary foreground mask (0.0 or 1.0, float32).
    kernel_min : int
        Minimum kernel radius (pixels) for erosion/dilation.
    kernel_max : int
        Maximum kernel radius (pixels) for erosion/dilation.

    Returns
    -------
    trimap : [B, 1, H, W]
        Trimap with values exactly {0.0, 0.5, 1.0}.
    """
    B = mask.shape[0]
    device = mask.device
    dtype = mask.dtype

    trimaps = []
    for i in range(B):
        single_mask = mask[i:i+1]  # [1, 1, H, W]

        # Random kernel size for this sample
        radius = torch.randint(
            kernel_min, kernel_max + 1, (1,),
        ).item()
        kernel_size = radius * 2 + 1
        padding = radius

        # Dilation: max_pool2d on the mask
        # Pixels near the boundary become foreground → expands FG
        dilated = F.max_pool2d(
            single_mask,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )  # [1, 1, H, W]

        # Erosion: max_pool2d on the inverted mask, then invert back
        # This shrinks the foreground region
        eroded = 1.0 - F.max_pool2d(
            1.0 - single_mask,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
        )  # [1, 1, H, W]

        # Build trimap
        # Start with all unknown (0.5)
        trimap = torch.full_like(single_mask, 0.5)

        # Known background: where dilated mask is 0 (definitely outside)
        trimap[dilated < 0.5] = 0.0

        # Known foreground: where eroded mask is 1 (definitely inside)
        trimap[eroded > 0.5] = 1.0

        trimaps.append(trimap)

    return torch.cat(trimaps, dim=0)


def quantize_trimap(trimap: torch.Tensor) -> torch.Tensor:
    """Snap floating-point trimap values to nearest {0.0, 0.5, 1.0}.

    Use after geometric transforms (resize, affine) that may introduce
    interpolation artifacts in trimap values.

    Parameters
    ----------
    trimap : [B, 1, H, W] or [H, W]
        Trimap with potentially noisy float values.

    Returns
    -------
    Trimap with values exactly {0.0, 0.5, 1.0}.
    """
    # Threshold boundaries: < 0.25 → 0.0, 0.25–0.75 → 0.5, > 0.75 → 1.0
    result = torch.full_like(trimap, 0.5)
    result[trimap < 0.25] = 0.0
    result[trimap > 0.75] = 1.0
    return result


def load_trimap_from_file(
    path: Path | str,
) -> np.ndarray:
    """Load a trimap image file and normalize to {0.0, 0.5, 1.0}.

    Accepts file formats:
    - {0, 128, 255} grayscale → {0.0, 0.5, 1.0}
    - {0, 1, 2} labels → {0.0, 0.5, 1.0}

    Parameters
    ----------
    path : Path or str
        Path to trimap image file.

    Returns
    -------
    trimap : np.ndarray
        [H, W] float32 array with values {0.0, 0.5, 1.0}.

    Raises
    ------
    IOError
        If the file cannot be read.
    ValueError
        If unexpected pixel values are found.
    """
    path = Path(path)
    trimap = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if trimap is None:
        raise IOError(f"Failed to read trimap: {path}")

    unique = set(np.unique(trimap).tolist())

    # {0, 128, 255} format
    if unique <= {0, 128, 255}:
        result = np.zeros_like(trimap, dtype=np.float32)
        result[trimap == 128] = 0.5
        result[trimap == 255] = 1.0
        return result

    # {0, 1, 2} label format
    if unique <= {0, 1, 2}:
        result = np.zeros_like(trimap, dtype=np.float32)
        result[trimap == 1] = 0.5
        result[trimap == 2] = 1.0
        return result

    raise ValueError(
        f"Trimap '{path}' contains unexpected values: {sorted(unique)}. "
        f"Expected {{0, 128, 255}} or {{0, 1, 2}}."
    )


def validate_trimap_values(trimap: torch.Tensor) -> bool:
    """Check that a trimap only contains {0.0, 0.5, 1.0} values.

    Parameters
    ----------
    trimap : torch.Tensor
        Trimap tensor.

    Returns
    -------
    True if valid, False otherwise.
    """
    valid_bg = (trimap < 0.01).float()
    valid_unknown = ((trimap > 0.49) & (trimap < 0.51)).float()
    valid_fg = (trimap > 0.99).float()
    valid = valid_bg + valid_unknown + valid_fg
    return bool((valid > 0.5).all().item())
