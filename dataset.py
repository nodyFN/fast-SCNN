"""
Dataset and DataLoader utilities for binary semantic segmentation.

Data layout
-----------
::

    data/
    ├── train/
    │   ├── images/   (*.jpg, *.jpeg, *.png)
    │   └── masks/    (*.png — single-channel, values 0/1 or 0/255)
    ├── val/
    │   └── ...
    └── test/
        └── ...

Image and mask are paired by matching file stem (e.g. ``frame_0001``).

Mask conversion rules
---------------------
- {0, 1} → use directly
- {0, 255} → map 255 → 1
- Other values → raise ``ValueError`` (unless ``allow_threshold=True``)
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from typing import Any, Callable, Dict, List, Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ===========================================================================
# Dataset
# ===========================================================================


class SegmentationDataset(Dataset):
    """Binary segmentation dataset.

    Parameters
    ----------
    root : Path or str
        Directory containing ``images/`` and ``masks/`` sub-directories.
    transform : callable, optional
        An Albumentations Compose pipeline.
    allow_threshold : bool
        If True, values > 127 are mapped to 1 and ≤ 127 to 0 (silences the
        strict mask-value check).  **Default False** — unexpected values raise
        ``ValueError``.
    """

    def __init__(
        self,
        root: Path | str,
        transform: Optional[Callable] = None,
        allow_threshold: bool = False,
    ) -> None:
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.transform = transform
        self.allow_threshold = allow_threshold

        # Validate directories
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")

        # Discover and pair images + masks
        self.pairs = self._discover_pairs()
        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No image/mask pairs found in {self.root}. "
                f"Ensure images/ and masks/ contain files with matching stems."
            )

    def _discover_pairs(self) -> List[Tuple[Path, Path]]:
        """Find all (image_path, mask_path) pairs, sorted by stem."""
        # Build stem→path maps
        image_map: Dict[str, Path] = {}
        for p in sorted(self.image_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                image_map[p.stem] = p

        mask_map: Dict[str, Path] = {}
        for p in sorted(self.mask_dir.iterdir()):
            if p.suffix.lower() == ".png":
                mask_map[p.stem] = p

        pairs: List[Tuple[Path, Path]] = []
        for stem in sorted(image_map.keys()):
            if stem not in mask_map:
                raise FileNotFoundError(
                    f"Mask not found for image '{image_map[stem].name}'. "
                    f"Expected a corresponding mask with stem '{stem}' in {self.mask_dir}"
                )
            pairs.append((image_map[stem], mask_map[stem]))
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_path, mask_path = self.pairs[idx]

        # Read image (BGR → RGB)
        image = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read mask (single-channel)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise IOError(f"Failed to read mask: {mask_path}")

        # Validate image / mask spatial dimensions
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(
                f"Image and mask size mismatch for '{img_path.stem}': "
                f"image={image.shape[:2]}, mask={mask.shape[:2]}"
            )

        # Convert mask values
        mask = self._convert_mask(mask, mask_path)

        # Apply augmentations
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]  # float32 [C, H, W]
            mask = transformed["mask"]  # [H, W]

        # Ensure correct dtypes
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask)
        mask = mask.long()

        return {"image": image, "mask": mask}

    def _convert_mask(self, mask: np.ndarray, path: Path) -> np.ndarray:
        """Validate and convert mask to {0, 1} uint8."""
        unique = set(np.unique(mask).tolist())
        if unique <= {0, 1}:
            return mask.astype(np.uint8)
        if unique <= {0, 255}:
            return (mask // 255).astype(np.uint8)
        if self.allow_threshold:
            return (mask > 127).astype(np.uint8)
        raise ValueError(
            f"Mask '{path}' contains unexpected values: {sorted(unique)}. "
            f"Expected {{0, 1}} or {{0, 255}}.  Set allow_threshold=True to "
            f"threshold at 127 instead."
        )


# ===========================================================================
# Transforms
# ===========================================================================

# ImageNet statistics for normalisation
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(
    height: int = 512,
    width: int = 1024,
    scale_min: float = 0.5,
    scale_max: float = 2.0,
) -> A.Compose:
    """Training augmentation pipeline.

    Paper-compatible augmentations:
        - Random resizing (0.5–2.0)
        - Translation / crop
        - Horizontal flip
        - Brightness augmentation
        - Color channel noise

    Mask interpolation uses nearest-neighbor.
    """
    return A.Compose(
        [
            # Random scale (paper: 0.5–2.0)
            A.RandomScale(scale_limit=(scale_min - 1.0, scale_max - 1.0), p=1.0),
            # Pad if smaller than crop after scaling
            A.PadIfNeeded(
                min_height=height,
                min_width=width,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
            ),
            # Random crop
            A.RandomCrop(height=height, width=width),
            # Horizontal flip (paper)
            A.HorizontalFlip(p=0.5),
            # Brightness and contrast (paper: brightness augmentation)
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5
            ),
            # Color channel noise (paper)
            A.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3
            ),
            # Affine (translation component — paper)
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                rotate=(-5, 5),
                shear=(-3, 3),
                mode=cv2.BORDER_CONSTANT,
                p=0.3,
            ),
            # Normalize + to tensor
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def build_val_transform(
    height: int = 512,
    width: int = 1024,
) -> A.Compose:
    """Deterministic validation / test preprocessing.

    No random augmentation — only resize + normalize.
    """
    return A.Compose(
        [
            A.Resize(height=height, width=width, interpolation=cv2.INTER_LINEAR),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


# Alias — test uses the same deterministic pipeline
build_test_transform = build_val_transform


# ===========================================================================
# DataLoader factory
# ===========================================================================


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    drop_last: bool = False,
    generator_seed: Optional[int] = None,
) -> DataLoader:
    """Create a DataLoader with safe defaults.

    Notes
    -----
    - ``persistent_workers`` is forced to False when ``num_workers == 0``
      (in-process loading does not use workers).
    - On **Windows**, DataLoader with ``num_workers > 0`` must be called
      inside ``if __name__ == '__main__':`` to avoid multiprocessing errors.
    """
    if num_workers == 0:
        persistent_workers = False

    generator = None
    if generator_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(generator_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        generator=generator,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
    )


def _worker_init_fn(worker_id: int) -> None:
    """Seed each DataLoader worker for reproducibility."""
    import random

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
