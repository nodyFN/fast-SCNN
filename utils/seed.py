"""
Reproducibility utilities.

``seed_everything`` sets seeds for Python, NumPy, PyTorch CPU/CUDA, and
configures cuDNN determinism.

Notes
-----
- **Deterministic mode** (``deterministic=True``) enables
  ``torch.use_deterministic_algorithms`` and disables cuDNN benchmark.
  This **may significantly slow down training** because some CUDA kernels
  fall back to slower deterministic implementations.
- Even with all seeds fixed, some CUDA operations (e.g. atomicAdd in
  backward passes) may introduce tiny floating-point differences across runs.
- ``torch.backends.cudnn.benchmark = True`` (the default when
  ``deterministic=False``) lets cuDNN auto-tune kernel selection for fixed
  input sizes, improving throughput at the cost of reproducibility.
"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """Set all random seeds for reproducibility.

    Parameters
    ----------
    seed : int
        Global random seed.
    deterministic : bool
        If True, enable fully deterministic algorithms (slower).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
