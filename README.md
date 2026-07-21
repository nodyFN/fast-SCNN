# Fast-SCNN: Fast Semantic Segmentation Network

A complete PyTorch implementation of [Fast-SCNN](https://arxiv.org/abs/1902.04502) for **binary semantic segmentation** (2 classes: background / foreground) on 1920×1080 TV screen images.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Network Architecture Table](#network-architecture-table)
- [Module Details](#module-details)
- [Project vs Paper Settings](#project-vs-paper-settings)
- [FastSCNNSalient (Dual-Head Model)](#fastscnnsalient-dual-head-model)
  - [Salient Model Architecture](#salient-model-architecture)
  - [Loss Functions](#salient-loss-functions)
  - [Gradient Flow](#gradient-flow)
  - [Salient Model Config](#salient-model-config)
  - [Salient Model Test & Benchmark](#salient-model-test--benchmark)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Formats](#dataset-formats)
  - [Custom Dataset (`data/`)](#1-custom-dataset-data)
  - [DUTS Dataset (`duts_data/`)](#2-duts-dataset-duts_data)
- [Training](#training)
  - [Standard Training](#standard-training)
  - [Focal Loss Tuning](#focal-loss-tuning)
  - [Transfer Learning (轉移學習 / 微調)](#transfer-learning-轉移學習--微調)
  - [Resuming Training](#resuming-training)
  - [Smoke Test](#smoke-test)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [ONNX Export](#onnx-export)
- [TensorBoard](#tensorboard)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)

---

## Architecture Overview

Fast-SCNN uses a two-branch architecture with a single skip connection:

```
Input [B, 3, H, W]
  │
  ▼
Learning to Downsample (3 layers → 1/8 res, 64-ch)
  ├── skip feature (64-ch, ~H/8 × W/8) ──────────────┐
  ▼                                                     │
Global Feature Extractor (MobileNetV2 bottlenecks)      │
  → 1/32 res, 128-ch                                   │
  ▼                                                     │
Pyramid Pooling Module (multi-scale context, 128-ch)    │
  ▼                                                     │
Feature Fusion Module ◄─────────────────────────────────┘
  → element-wise add + ReLU (128-ch, ~H/8 × W/8)
  ▼
Classifier (2×DSConv → Dropout → 1×1 Conv)
  ▼
Bilinear Upsample → [B, num_classes, H, W]
```

**Single skip connection** from Learning to Downsample to Feature Fusion Module — no U-Net style multi-layer skips.

---

## Network Architecture Table

Based on paper Table 1, using `num_classes = 2`:

| Module | Operator | t | c | n | s | Output |
|---|---|---|---|---|---|---|
| Learning to Downsample | 3×3 Conv2D | – | 32 | 1 | 2 | ~H/2 × W/2 |
| Learning to Downsample | 3×3 DSConv | – | 48 | 1 | 2 | ~H/4 × W/4 |
| Learning to Downsample | 3×3 DSConv | – | 64 | 1 | 2 | ~H/8 × W/8 |
| GFE Bottleneck Stage 1 | Bottleneck | 6 | 64 | 3 | 2 | ~H/16 × W/16 |
| GFE Bottleneck Stage 2 | Bottleneck | 6 | 96 | 3 | 2 | ~H/32 × W/32 |
| GFE Bottleneck Stage 3 | Bottleneck | 6 | 128 | 3 | 1 | ~H/32 × W/32 |
| Pyramid Pooling | PPM | – | 128 | – | – | ~H/32 × W/32 |
| Feature Fusion | FFM | – | 128 | – | – | ~H/8 × W/8 |
| Classifier | 2×DSConv + 1×1 | – | 2 | – | 1 | ~H/8 × W/8 |
| Output | Bilinear Upsample | – | 2 | – | – | H × W |

Where: `t` = expansion factor, `c` = output channels, `n` = repetitions, `s` = first stride.

---

## Module Details

### Learning to Downsample

Three layers total, each stride 2:
1. **Standard 3×3 Conv** (3→32) — standard conv because 3 input channels make DSConv inefficient
2. **DSConv** (32→48)
3. **DSConv** (48→64) — output serves as skip feature for FFM

### Depthwise Separable Convolution (DSConv)

```
3×3 Depthwise Conv → BN → 1×1 Pointwise Conv → BN → ReLU
```

**No ReLU between depthwise and pointwise** (paper specification).

### Bottleneck Stages (Global Feature Extractor)

MobileNetV2-style inverted residual with **linear** projection (no activation after final 1×1 conv).

Residual connection **only** when `stride == 1 AND in_channels == out_channels`.

**Stage 1**: 64→64, t=6, n=3, s=2 (Block 1: no residual; Blocks 2-3: residual)
**Stage 2**: 64→96, t=6, n=3, s=2 (Block 1: no residual; Blocks 2-3: residual)
**Stage 3**: 96→128, t=6, n=3, s=1 (Block 1: no residual; Blocks 2-3: residual)

### Pyramid Pooling Module (PPM)

> **[PROJECT DECISION]** The paper references PSPNet-style PPM but does not specify pool sizes or branch channels for Fast-SCNN.

Default settings:
- `pool_sizes = (1, 2, 3, 6)`
- `branch_channels = 32`
- Concat → 1×1 Conv fusion → 128-ch output
- Configurable via `ppm_pool_sizes` in config

**BatchNorm + batch_size=1 warning**: The 1×1 pooling branch produces a single value per channel, which can destabilize BatchNorm. **Use batch_size > 1 for training.**

### Feature Fusion Module (FFM)

**Low-res branch** (from PPM, 128-ch):
```
Bilinear upsample to high-res spatial size
→ 3×3 DW Conv (dilation=4, padding=4) → BN → ReLU
→ 1×1 PW Conv → BN (no activation)
```

**High-res branch** (from LtD, 64-ch):
```
1×1 Conv (64→128) → BN (no activation)
```

**Fusion**: `ReLU(high + low)` — element-wise addition, NOT concatenation.

Upsample uses actual tensor shape (`high_res.shape[-2:]`), not fixed scale_factor, to handle non-32-multiple heights like 1080.

### Auxiliary Heads

> **[PROJECT DECISION]** Paper specifies two auxiliary losses (weight 0.4 each) but not the exact head architecture.

Two lightweight heads used **only during training**:
1. After Learning to Downsample (in=64)
2. After PPM (in=128)

Structure: `3×3 ConvBNReLU → Dropout(0.1) → 1×1 Conv → Bilinear upsample`

### Classifier

```
DSConv (128→128) × 2 → Dropout(0.1) → 1×1 Conv (128→2) → Bilinear upsample
```

> **[PROJECT DECISION]** `dropout_p = 0.1` — paper mentions dropout in classifier but does not specify the probability.

### `align_corners=False`

All bilinear interpolation uses `align_corners=False` because:
- Avoids output dependency on input endpoint alignment
- More stable for arbitrary input sizes
- Better consistency between PyTorch and ONNX Runtime
- Standard for segmentation resize behaviour

### Model Output

The model outputs **raw logits** `[B, 2, H, W]` — **no softmax, sigmoid, or argmax** inside the model. This is required for `CrossEntropyLoss`.

For inference:
```python
prediction = logits.argmax(dim=1)        # class index
probability = torch.softmax(logits, 1)   # probability map
```

---

## Project vs Paper Settings

| Setting | Paper | Project Default | Notes |
|---|---|---|---|
| Optimizer | SGD | AdamW | |
| Learning rate | 0.045 | 0.001 | |
| Momentum | 0.9 | 0.9 | SGD only |
| Scheduler | PolyLR | PolyLR | |
| Poly power | 0.9 | 0.9 | |
| Weight decay | 0.00004 | 0.0001 | |
| DW weight decay | 0.0 | 0.0 | Paper: no L2 on DW conv |
| Epochs | 1000 | 200 | |
| Batch size | 12 | 4 | |
| Loss | CE | CE + Dice | |
| Aux weights | 0.4 each | 0.4 each | Paper setting |
| AMP | Not mentioned | Enabled (CUDA) | |
| Grad clipping | Not mentioned | 1.0 | |
| Activation | ReLU | ReLU | NOT ReLU6 |

Use `--profile paper` or `--profile project` to switch between profiles.

---

## Project Structure

```
fast-SCNN/
├── .gitignore
├── Pipfile               # Dependencies (CPU-default)
├── requirements.txt      # Versionless dependencies (Linux headless server friendly)
├── README.md
├── config.py             # Centralized configuration
├── dataset.py            # Dataset + transforms + DataLoader
├── train.py              # Training script (supports both models via --model)
├── evaluate.py           # Evaluation script
├── inference.py          # Single/folder inference
├── export.py             # ONNX export + validation
├── prepare_my_dataset.py # Custom dataset fusion/splitter
├── prepare_duts_dataset.py # DUTS dataset splitter
├── check_mask_value.py   # Utility to inspect mask pixel values
├── models/
│   ├── __init__.py
│   ├── fast_scnn.py          # Original Fast-SCNN (2-class CE)
│   └── fast_scnn_salient.py  # Dual-Head Salient Model (1-ch binary)
├── utils/
│   ├── __init__.py
│   ├── losses.py         # CE, Dice, Focal, Combined + Binary losses
│   ├── metrics.py        # Confusion matrix metrics
│   ├── scheduler.py      # PolyLR, CosineAnnealing
│   ├── checkpoint.py     # Save/load checkpoints
│   ├── visualization.py  # Training curves + segmentation vis
│   └── seed.py           # Reproducibility
├── tests/
│   ├── __init__.py
│   ├── test_model.py
│   ├── test_dataset.py
│   └── test_metrics.py
├── checkpoints/          # Saved model checkpoints (Subdivided by timestamp)
├── training_results/     # Training curve plots & visualisations (Subdivided by timestamp)
├── runs/                 # TensorBoard logs (Subdivided by timestamp)
├── exports/              # ONNX models
├── data/                 # Custom Dataset Directory (Git-ignored skeleton)
└── duts_data/            # DUTS Dataset Directory (Git-ignored skeleton)
```

*Note: Output files in `checkpoints/`, `training_results/`, and `runs/` are organized inside subdirectories named after the training run's timestamp (e.g. `20260720_145440`) to keep experiments separated.*

---

## Installation

### Prerequisites

- Python 3.10.2
- Pipenv

### Setup virtual environment

```bash
# Install pipenv if needed
pip install pipenv

# Create environment and install dependencies
pipenv install --dev

# Activate the environment
pipenv shell
```

### Headless Linux Server / Versionless Installation
If installing on a headless Linux server with Python 3.10.2, use the versionless `requirements.txt`. It defaults to `opencv-python-headless` to avoid missing X11/GUI system library dependencies:

```bash
pip install -r requirements.txt
```

### CUDA / GPU PyTorch
For NVIDIA GPU support, install the CUDA version of PyTorch first, then install the rest of the packages:

```bash
# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

Verify GPU availability:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Dataset Formats

All datasets must structure images and masks symmetrically by file stem (e.g., `images/img_001.jpg` ↔ `masks/img_001.png`). Masks must be single-channel `.png` files.

### 1. Custom Dataset (`data/`)
For segmenting custom foreground objects against background. Has folders:
```
data/
├── train/
│   ├── images/
│   └── masks/ (Values: 0 for background, 255/1 for foreground)
├── val/
│   └── ...
└── test/
    └── ...
```
If you have raw `FG` (foreground) and `NO_FG` (no foreground) images in a `my_dataset` folder, you can merge and split them into `data/` using:
```bash
python prepare_my_dataset.py --src /path/to/my_dataset
```
*This handles file name collisions by prefixing stems with `fg_` or `nofg_` and splits 8:1:1.*

### 2. DUTS Dataset (`duts_data/`)
For general salient object detection. Download the DUTS dataset folder containing `DUTS-TR` and `DUTS-TE` and execute the splitter script:
```bash
python prepare_duts_dataset.py --src /path/to/DUTS
```
*This splits `DUTS-TR` (95% to `duts_data/train/`, 5% to `duts_data/val/`) and sends `DUTS-TE` directly to `duts_data/test/`.*

---

## Training

### Standard Training
Run standard training by targeting your preferred dataset with the `--data-root` flag:

```bash
# Train on your custom dataset (data/)
python train.py --profile project --data-root data

# Train on the DUTS dataset (duts_data/).
# NOTE: DUTS masks contain grayscale soft edges, so you MUST add --allow-threshold
# to binarize them into strict 0/1 target formats.
python train.py --profile project --data-root duts_data --allow-threshold
```

### Focal Loss Tuning
You can enable and adjust the weight of Focal Loss in the combined loss function by modifying `focal_weight` in `config.py` (specifically under the active profile's configuration function, such as `get_project_config()`):
```python
# In config.py:
def get_project_config() -> Config:
    return Config(
        ...
        focal_weight=0.5,  # Enable Focal Loss and set its weight
        ...
    )
```

### Transfer Learning (轉移學習 / 微調)
If you have trained a backbone on DUTS and want to use it as pre-trained weights for your custom dataset, you have two options:

```bash
# Option A: Fine-tune the entire network (Recommended)
python train.py --profile project --data-root data --weights checkpoints/DUTS_TIMESTAMP/best_miou.pt

# Option B: Freeze the backbone and only train classifier/fusion layers (Feature Extraction Mode)
python train.py --profile project --data-root data --weights checkpoints/DUTS_TIMESTAMP/best_miou.pt --freeze-backbone
```
*The `--weights` flag performs a **weights-only initialization**. Unlike `--resume`, it starts the epoch counter from 0 and initializes a fresh optimizer and scheduler. Adding `--freeze-backbone` sets `requires_grad=False` on the `LearningToDownsample` and `GlobalFeatureExtractor` modules, only optimizing the fusion and classifier layers.*

### Resuming Training
To resume an interrupted training run (restores epoch, steps, optimizer states, and scheduler decay progress exactly):
```bash
python train.py --resume checkpoints/TIMESTAMP/latest.pt --epochs 100
```
*Note: Make sure to increase `--epochs` to a value higher than the epoch at which the checkpoint was saved, otherwise it will exit immediately.*

### Smoke Test
Run a quick, dataset-free, GPU-free sanity check to verify the model builds, trains, backward passes, and saves checkpoints:
```bash
python train.py --smoke-test
```

---

## Evaluation

Evaluate a saved checkpoint against the validation or test split of your dataset:

```bash
# Evaluate on custom validation split
python evaluate.py --checkpoint checkpoints/TIMESTAMP/best_miou.pt --split val --data-root data

# Evaluate on DUTS test split with visual output (requires --allow-threshold)
python evaluate.py --checkpoint checkpoints/TIMESTAMP/best_miou.pt --split test --data-root duts_data --allow-threshold --save-vis
```

---

## Inference

Run inference on a single image or a folder of images. Preprocessing automatically resizes the image to the model input dimensions, and postprocessing upsamples the prediction mask back to the original image dimensions.

```bash
# Single image
python inference.py --checkpoint checkpoints/TIMESTAMP/best_miou.pt --input image.jpg --output-dir results/

# Folder of images
python inference.py --checkpoint checkpoints/TIMESTAMP/best_miou.pt --input data/test/images/ --output-dir results/
```
Inference outputs `_class.png` (0/1), `_binary.png` (0/255), `_prob.jpg` (saliency heatmap), `_prob_gray.png` (grayscale probability mask [0, 255]), and `_overlay.jpg` (colour blend) for each image. It logs model-only latency and end-to-end latency separately.

---

## ONNX Export

### Fixed-size export
```bash
python export.py --checkpoint checkpoints/TIMESTAMP/best_miou.pt --height 512 --width 1024
```

### Dynamic axes export (Recommended for varying resolutions)
```bash
python export.py --checkpoint checkpoints/TIMESTAMP/best_miou.pt --height 512 --width 1024 --dynamic
```

After export, ONNX Runtime validation runs automatically: checking the model structure, output shape, and numerical deviation vs PyTorch. Dynamic exports are tested at multiple input resolutions.

---

## TensorBoard

Visualize loss curves, auxiliary losses, metric scores (PA, mIoU, Foreground IoU, Foreground Dice), and learning rate changes:

```bash
tensorboard --logdir runs/
```

---

## Testing

Run unit tests via `pytest` to verify components locally:

```bash
# Run all tests
pytest tests/ -v

# Run specific modules
pytest tests/test_model.py -v
pytest tests/test_dataset.py -v
pytest tests/test_metrics.py -v
```

### Model latency & FPS Benchmark
Test Fast-SCNN's pure forward pass latency and average FPS on a dummy tensor of choice:
```bash
# Default resolution (512x1024)
python models/fast_scnn.py --device cuda

# Full TV resolution (1080x1920)
python models/fast_scnn.py --device cuda --full-res
```

---

## FastSCNNSalient (Dual-Head Model)

A **Single-Backbone, Dual-Head, Coarse-to-Fine Salient Foreground Segmentation** model optimized for edge devices (TV SoC) with limited DRAM bandwidth.

This model is completely independent from the original `FastSCNN`. It uses **1-channel binary logits** with `BCEWithLogitsLoss`, not the original 2-channel `CrossEntropyLoss`.

### Salient Model Architecture

```
Input [B, 3, H, W]
  │
  ▼
SharedFastSCNNBackbone (executed ONCE)
  │  Learning to Downsample
  │  Global Feature Extractor
  │  Pyramid Pooling Module
  │  Feature Fusion Module
  │
  └─► F_shared [B, 128, H/8, W/8]
        │
        ├──► CoarseHead ──► coarse_logits [B, 1, H, W]
        │         │
        │    sigmoid + detach (stop-gradient)
        │         │
        │    coarse_prompt [B, 1, H/8, W/8]
        │         │
        └──► RefinementHead ◄──┘
                  │
                  └──► fine_logits [B, 1, H, W]  (final output)
```

**Key design constraints:**
- **Single backbone execution**: the shared backbone runs exactly once per input image.
- **Internal spatial prompt**: coarse probability is generated internally, no external GT/bbox/SAM prompts.
- **Stop-gradient**: `coarse_prompt = sigmoid(coarse_logits).detach()` — Fine Loss cannot update CoarseHead through the prompt path.
- **Broadcasting spatial attention**: `F_attended = F_shared + F_shared × coarse_prompt` uses broadcasting (no `.repeat()`).

**Module details:**

| Module | Structure | Output |
|---|---|---|
| `SharedFastSCNNBackbone` | LtD → GFE → PPM → FFM | `[B, 128, H/8, W/8]` |
| `CoarseHead` | DSConv 128→64 → Dropout → 1×1 Conv 64→1 | `[B, 1, H/8, W/8]` |
| `RefinementHead` | SpatialAttn → Cat(129ch) → 1×1 Conv 129→64 → DSConv 64→64 → Dropout → 1×1 Conv 64→1 | `[B, 1, H/8, W/8]` |

### Salient Loss Functions

Total loss:
```
L_total = λ_coarse × L_coarse + λ_fine × L_fine + λ_boundary × L_boundary
```

| Loss | Formula | Target |
|---|---|---|
| **L_coarse** | `BCEWithLogitsLoss(pos_weight) + BinaryDiceLoss` | High Recall (find all foreground) |
| **L_fine** | `BinaryFocalLoss(α=0.25, γ=2.0) + BinaryDiceLoss` | High Precision (refine edges) |
| **L_boundary** | `SobelBoundaryLoss` (L1 on Sobel edge magnitudes, fine only) | Sharp contours |

Default weights: `λ_coarse=1.0`, `λ_fine=1.0`, `λ_boundary=0.5`.

### Gradient Flow

| Loss | Backbone | CoarseHead | RefinementHead |
|---|---|---|---|
| Coarse Loss | ✓ updates | ✓ updates | — |
| Fine Loss | ✓ updates | ✗ blocked by detach | ✓ updates |
| Boundary Loss | ✓ updates | ✗ blocked by detach | ✓ updates |

### Salient Model Config

All salient model parameters are configurable in `config.py`:

| Field | Default | Description |
|---|---|---|
| `coarse_channels` | 64 | CoarseHead intermediate channels |
| `refinement_channels` | 64 | RefinementHead intermediate channels |
| `salient_lambda_coarse` | 1.0 | Weight for coarse loss |
| `salient_lambda_fine` | 1.0 | Weight for fine loss |
| `salient_lambda_boundary` | 0.5 | Weight for boundary loss |
| `salient_focal_alpha` | 0.25 | Focal Loss alpha |
| `salient_focal_gamma` | 2.0 | Focal Loss gamma |
| `salient_pos_weight` | None | BCEWithLogitsLoss pos_weight |

### Salient Model Training

To train the new dual-head salient segmentation model, run `train.py` with the `--model fast_scnn_salient` argument:

```bash
# Train on your custom dataset (data/) with custom 16:9 aspect ratio (e.g. 540x960)
python train.py --model fast_scnn_salient --profile project --data-root data --train-height 540 --train-width 960 --val-height 540 --val-width 960

# Train on the DUTS dataset (requires --allow-threshold)
python train.py --model fast_scnn_salient --profile project --data-root duts_data --allow-threshold

# Transfer Learning: Fine-tune using pre-trained weights
python train.py --model fast_scnn_salient --profile project --data-root data --weights checkpoints/TIMESTAMP/best_miou.pt

# Transfer Learning: Freeze the backbone (Feature Extraction)
python train.py --model fast_scnn_salient --profile project --data-root data --weights checkpoints/TIMESTAMP/best_miou.pt --freeze-backbone

# Run a quick synthetic data smoke test
python train.py --model fast_scnn_salient --smoke-test
```

### Salient Model Test & Benchmark

Run the built-in shape test, gradient flow verification, odd-size test, and FPS benchmark:
```bash
# Default resolution (512×1024)
python models/fast_scnn_salient.py --device cuda

# Full TV resolution (1080×1920)
python models/fast_scnn_salient.py --device cuda --full-res

# CPU-only test
python models/fast_scnn_salient.py --device cpu --iterations 5
```

---

## Troubleshooting

### 1920×1080 Training VRAM Issues
The original 1920×1080 resolution is memory-intensive. For optimal VRAM usage:
- Use smaller crop sizes: default training uses `512×1024` crops (`--train-height 512 --train-width 1024`).
- Decrease batch size: `--batch-size 2` or `4`.
- Enable mixed-precision (AMP): `--profile project` turns AMP on by default.

### BatchNorm + Small Batch Size
- Due to PPM's 1×1 average pooling branch, training with `batch_size=1` causes BatchNorm instability.
- **Always train with `batch_size ≥ 2`**.

### Windows `num_workers`
On Windows, `num_workers > 0` requires code to be executed inside `if __name__ == '__main__':`. Set `--num-workers 0` if you encounter multiprocessing errors.
