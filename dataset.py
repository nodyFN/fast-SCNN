"""
Dataset and DataLoader utilities for binary semantic segmentation and alpha matting.

Data layout (segmentation)
--------------------------
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
    longest_max_size: Optional[int] = None,
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
    transforms = []
    if longest_max_size is not None:
        transforms.append(A.LongestMaxSize(max_size=longest_max_size, p=1.0))

    transforms.extend(
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
    return A.Compose(transforms)


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
    sampler: Optional[torch.utils.data.Sampler] = None,
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

    if sampler is not None:
        shuffle = False

    generator = None
    if generator_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(generator_seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
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


# ===========================================================================
# Matting Dataset (for DDC Alpha-Free Matting Training)
# ===========================================================================


class MattingDataset(Dataset):
    """Dataset for DDC matting training.

    Returns both ImageNet-normalized and raw [0,1] RGB images, plus
    trimap generated from binary mask or loaded from file.

    Data layout::

        data/
        ├── train/
        │   ├── images/   (*.jpg, *.jpeg, *.png)
        │   ├── masks/    (*.png — binary foreground mask)
        │   └── trimaps/  (*.png — optional, 0/128/255 trimap files)
        └── val/
            └── ...

    Parameters
    ----------
    root : Path or str
        Directory containing images/, masks/, and optionally trimaps/.
    transform : callable, optional
        Albumentations Compose for geometric augmentations.
        Must NOT include Normalize or ToTensorV2.
    trimap_source : str
        "binary_mask" — generate trimap from binary mask on-the-fly.
        "file" — load trimap from trimaps/ subdirectory.
    trimap_kernel_min : int
        Minimum kernel radius for trimap generation (default 1).
    trimap_kernel_max : int
        Maximum kernel radius for trimap generation (default 30).
    allow_threshold : bool
        Allow thresholding grayscale masks.
    collapse_nonzero_to_foreground : bool
        If True, any non-zero mask value → foreground.
    """

    def __init__(
        self,
        root: Path | str,
        transform: Optional[Callable] = None,
        trimap_source: str = "binary_mask",
        trimap_kernel_min: int = 1,
        trimap_kernel_max: int = 30,
        allow_threshold: bool = False,
        collapse_nonzero_to_foreground: bool = False,
    ) -> None:
        self.root = Path(root)
        self.image_dir = self.root / "images"
        self.mask_dir = self.root / "masks"
        self.trimap_dir = self.root / "trimaps"
        self.transform = transform
        self.trimap_source = trimap_source
        self.trimap_kernel_min = trimap_kernel_min
        self.trimap_kernel_max = trimap_kernel_max
        self.allow_threshold = allow_threshold
        self.collapse_nonzero_to_foreground = collapse_nonzero_to_foreground

        # Validate directories
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")
        if not self.mask_dir.exists():
            raise FileNotFoundError(f"Mask directory not found: {self.mask_dir}")
        if trimap_source == "file" and not self.trimap_dir.exists():
            raise FileNotFoundError(
                f"Trimap directory not found: {self.trimap_dir}. "
                f"Use trimap_source='binary_mask' to generate trimaps on-the-fly."
            )

        # Discover pairs
        self.pairs = self._discover_pairs()
        if len(self.pairs) == 0:
            raise RuntimeError(
                f"No image/mask pairs found in {self.root}."
            )

    def _discover_pairs(self) -> List[Tuple[Path, Path]]:
        """Find (image_path, mask_path) pairs sorted by stem."""
        image_map: Dict[str, Path] = {}
        for p in sorted(self.image_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXTENSIONS:
                image_map[p.stem] = p

        mask_map: Dict[str, Path] = {}
        for p in sorted(self.mask_dir.iterdir()):
            if p.suffix.lower() == ".png":
                mask_map[p.stem] = p

        pairs = []
        for stem in sorted(image_map.keys()):
            if stem not in mask_map:
                raise FileNotFoundError(
                    f"Mask not found for image '{image_map[stem].name}'."
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

        # Convert mask to binary {0, 1}
        mask = self._convert_mask(mask, mask_path)

        # Load trimap from file if requested
        trimap_from_file = None
        if self.trimap_source == "file":
            from utils.trimap import load_trimap_from_file
            trimap_path = self.trimap_dir / f"{img_path.stem}.png"
            trimap_from_file = load_trimap_from_file(trimap_path)

        # Apply geometric augmentations (same transform for image, mask, trimap)
        if self.transform is not None:
            if trimap_from_file is not None:
                transformed = self.transform(
                    image=image, mask=mask,
                    trimap=trimap_from_file,
                )
                trimap_np = transformed["trimap"]
            else:
                transformed = self.transform(image=image, mask=mask)
                trimap_np = None
            image = transformed["image"]  # uint8 HWC numpy
            mask = transformed["mask"]    # uint8 HW numpy
        else:
            trimap_np = trimap_from_file

        # Convert image to float [0, 1]
        if isinstance(image, np.ndarray):
            image_float = image.astype(np.float32) / 255.0
        else:
            image_float = image.float() / 255.0

        # ddc_image: raw RGB [0,1] — NOT ImageNet-normalized
        if isinstance(image_float, np.ndarray):
            ddc_image = torch.from_numpy(
                image_float.transpose(2, 0, 1)
            )  # [3, H, W]
        else:
            ddc_image = image_float

        # image: ImageNet-normalized
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        if isinstance(image_float, np.ndarray):
            image_norm = (image_float - mean) / std
            image_tensor = torch.from_numpy(
                image_norm.transpose(2, 0, 1)
            )  # [3, H, W]
        else:
            # Already tensor
            image_tensor = image_float

        # mask → float tensor [1, H, W]
        if isinstance(mask, np.ndarray):
            mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        else:
            mask_tensor = mask.float().unsqueeze(0)

        # Trimap: generate from mask or use loaded file
        if self.trimap_source == "binary_mask":
            # On-the-fly generation using random kernel
            from utils.trimap import generate_trimap_from_mask
            trimap_tensor = generate_trimap_from_mask(
                mask_tensor.unsqueeze(0),  # [1, 1, H, W]
                kernel_min=self.trimap_kernel_min,
                kernel_max=self.trimap_kernel_max,
            ).squeeze(0)  # [1, H, W]
        else:
            # From file — quantize after any geometric transforms
            from utils.trimap import quantize_trimap
            if isinstance(trimap_np, np.ndarray):
                trimap_tensor = torch.from_numpy(
                    trimap_np.astype(np.float32)
                ).unsqueeze(0)  # [1, H, W]
            else:
                trimap_tensor = trimap_np.float().unsqueeze(0)
            trimap_tensor = quantize_trimap(trimap_tensor)

        return {
            "image": image_tensor,
            "ddc_image": ddc_image,
            "mask": mask_tensor.squeeze(0).long().squeeze(0),  # [H, W] long for compat
            "trimap": trimap_tensor,  # [1, H, W] float {0.0, 0.5, 1.0}
        }

    def _convert_mask(self, mask: np.ndarray, path: Path) -> np.ndarray:
        """Validate and convert mask to {0, 1} uint8."""
        unique = set(np.unique(mask).tolist())
        if unique <= {0, 1}:
            return mask.astype(np.uint8)
        if unique <= {0, 255}:
            return (mask // 255).astype(np.uint8)
        if self.collapse_nonzero_to_foreground:
            return (mask > 0).astype(np.uint8)
        if self.allow_threshold:
            return (mask > 127).astype(np.uint8)
        raise ValueError(
            f"Mask '{path}' contains unexpected values: {sorted(unique)}. "
            f"Expected {{0, 1}} or {{0, 255}}. "
            f"Set allow_threshold=True or collapse_nonzero_to_foreground=True."
        )


# ===========================================================================
# Matting Transforms
# ===========================================================================


def build_matting_train_transform(
    height: int = 512,
    width: int = 512,
    scale_min: float = 0.5,
    scale_max: float = 2.0,
    longest_max_size: Optional[int] = None,
) -> A.Compose:
    """Training augmentation for matting (NO Normalize or ToTensorV2).

    Geometric augmentations only — Normalize and tensor conversion
    are done manually in MattingDataset to produce both normalized
    and raw [0,1] images.

    Trimap uses nearest-neighbor interpolation via additional_targets.
    """
    transforms = []
    if longest_max_size is not None:
        transforms.append(A.LongestMaxSize(max_size=longest_max_size, p=1.0))

    transforms.extend(
        [
            A.RandomScale(scale_limit=(scale_min - 1.0, scale_max - 1.0), p=1.0),
            A.PadIfNeeded(
                min_height=height, min_width=width,
                border_mode=cv2.BORDER_CONSTANT,
                value=0, mask_value=0,
            ),
            A.RandomCrop(height=height, width=width),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, contrast_limit=0.2, p=0.5,
            ),
            A.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3,
            ),
        ]
    )
    return A.Compose(transforms, additional_targets={"trimap": "mask"})


def build_matting_val_transform(
    height: int = 512,
    width: int = 512,
) -> A.Compose:
    """Deterministic validation transform for matting (NO Normalize/ToTensorV2).

    Trimap uses nearest-neighbor interpolation via additional_targets.
    """
    return A.Compose(
        [
            A.Resize(
                height=height, width=width,
                interpolation=cv2.INTER_LINEAR,
            ),
        ],
        additional_targets={"trimap": "mask"},
    )

