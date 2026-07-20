# Fast-SCNN: Fast Semantic Segmentation Network

A complete PyTorch implementation of [Fast-SCNN](https://arxiv.org/abs/1902.04502) for **binary semantic segmentation** (2 classes: background / foreground) on 1920×1080 TV screen images.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Network Architecture Table](#network-architecture-table)
- [Module Details](#module-details)
- [Project vs Paper Settings](#project-vs-paper-settings)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset Format](#dataset-format)
- [Training](#training)
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
├── Pipfile               # Python 3.10.2 + dependencies
├── README.md
├── config.py             # Centralized configuration
├── dataset.py            # Dataset + transforms + DataLoader
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── inference.py          # Single/folder inference
├── export.py             # ONNX export + validation
├── models/
│   ├── __init__.py
│   └── fast_scnn.py      # Complete Fast-SCNN implementation
├── utils/
│   ├── __init__.py
│   ├── losses.py         # CE, Dice, Focal, Combined loss
│   ├── metrics.py        # Confusion matrix metrics
│   ├── scheduler.py      # PolyLR, CosineAnnealing
│   ├── checkpoint.py     # Save/load checkpoints
│   ├── visualization.py  # Training curves + segmentation vis
│   └── seed.py           # Reproducibility
├── tests/
│   ├── test_model.py
│   ├── test_dataset.py
│   └── test_metrics.py
├── checkpoints/          # Saved model checkpoints
├── training_results/     # Training curve plots
├── runs/                 # TensorBoard logs
└── exports/              # ONNX models
```

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

### CPU-only PyTorch

The default `Pipfile` installs CPU PyTorch from PyPI. This works out of the box:

```bash
pipenv install --dev
```

### CUDA PyTorch

For NVIDIA GPU support, install the CUDA version of PyTorch manually:

```bash
# Activate pipenv shell first
pipenv shell

# Install CUDA 12.1 PyTorch (adjust URL for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Or CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Verify:
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Dataset Format

```
data/
├── train/
│   ├── images/    (*.jpg, *.jpeg, *.png)
│   └── masks/     (*.png, single-channel)
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

**Pairing**: Image and mask match by file stem (e.g. `images/frame_0001.jpg` ↔ `masks/frame_0001.png`).

### Mask format

- Values `{0, 1}`: used directly (0 = background, 1 = foreground)
- Values `{0, 255}`: automatically converted (255 → 1)
- Other values: **raises error** by default. Set `allow_threshold=True` to threshold at 127.

---

## Training

### Project profile (recommended for this dataset)

```bash
python train.py --profile project --epochs 200 --batch-size 4
```

### Paper profile

```bash
python train.py --profile paper --epochs 1000 --batch-size 12
```

### Resume training

```bash
python train.py --resume checkpoints/latest.pt
```

### Common options

```bash
python train.py \
    --profile project \
    --epochs 100 \
    --batch-size 8 \
    --lr 0.001 \
    --optimizer adamw \
    --scheduler poly \
    --train-height 512 \
    --train-width 1024 \
    --device auto \
    --seed 42 \
    --early-stopping 50
```

### Smoke test (no real data needed)

```bash
python train.py --smoke-test
```

---

## Evaluation

```bash
# Evaluate on validation set
python evaluate.py --checkpoint checkpoints/best_miou.pt --split val

# Evaluate on test set with visualizations
python evaluate.py --checkpoint checkpoints/best_miou.pt --split test --save-vis
```

---

## Inference

### Single image

```bash
python inference.py --checkpoint checkpoints/best_miou.pt --input image.jpg --output-dir results/
```

### Folder inference

```bash
python inference.py --checkpoint checkpoints/best_miou.pt --input data/test/images/ --output-dir results/
```

Outputs per image: `_class.png` (0/1), `_binary.png` (0/255), `_prob.jpg` (heatmap), `_overlay.jpg`.

Timing breakdown reports model-only latency and end-to-end latency separately.

---

## ONNX Export

### Fixed-size export

```bash
python export.py --checkpoint checkpoints/best_miou.pt --height 512 --width 1024
```

### Dynamic axes export

```bash
python export.py --checkpoint checkpoints/best_miou.pt --height 512 --width 1024 --dynamic
```

### Custom opset

```bash
python export.py --checkpoint checkpoints/best_miou.pt --opset 17
```

ONNX Runtime validation runs automatically after export: checks model structure, output shape, and numerical accuracy vs PyTorch. Dynamic exports are tested with multiple input sizes.

### ONNX/TensorRT Deployment Notes

- Default opset 17 provides good compatibility with TensorRT 8.6+
- `align_corners=False` is used for all bilinear interpolation — verify your TensorRT version handles this correctly
- Dynamic axes support batch, height, and width dimensions
- If dynamic spatial dims cause issues with specific TensorRT versions, use fixed-size export

---

## TensorBoard

```bash
tensorboard --logdir runs/
```

Logged metrics: `Loss/train`, `Loss/validation`, `Loss/aux_downsample`, `Loss/aux_global`, `Metrics/pixel_accuracy`, `Metrics/miou`, `Metrics/foreground_iou`, `Metrics/foreground_dice`, `LearningRate`.

---

## Testing

### Run all tests

```bash
pytest tests/ -v
```

### Run specific test file

```bash
pytest tests/test_model.py -v
pytest tests/test_dataset.py -v
pytest tests/test_metrics.py -v
```

### Smoke test

```bash
python train.py --smoke-test
```

### Model benchmark

```bash
# Default 512×1024
python models/fast_scnn.py

# Full resolution 1080×1920
python models/fast_scnn.py --full-res

# With auxiliary heads
python models/fast_scnn.py --aux
```

---

## Troubleshooting

### 1920×1080 Training VRAM Issues

The original 1920×1080 resolution requires significant GPU memory. Recommendations:

- **Reduce crop size**: Default training uses `512×1024` crops (configurable via `--train-height` / `--train-width`)
- **Reduce batch size**: `--batch-size 2` or even 1
- **Enable AMP**: `--profile project` enables mixed precision by default
- Typical VRAM: ~4 GB for batch_size=4 @ 512×1024 with AMP

### Suggested Crop Size / Batch Size

| GPU VRAM | Crop Size | Batch Size |
|---|---|---|
| 4 GB | 256×512 | 4 |
| 8 GB | 512×1024 | 4 |
| 12 GB | 512×1024 | 8 |
| 24 GB | 768×1536 | 8-12 |

### BatchNorm + Small Batch Size

- PPM's 1×1 pooling branch produces single-value features, causing BatchNorm instability with batch_size=1
- **Use batch_size ≥ 2 for training**
- For batch_size=1 inference, use `model.eval()` (BN uses running statistics)

### Windows `num_workers`

On Windows, `num_workers > 0` requires code to run inside `if __name__ == '__main__':`. Set `--num-workers 0` if you encounter multiprocessing errors.

### Model Latency vs End-to-End Latency

- **Model-only latency**: Pure neural network forward pass time
- **End-to-end latency**: Includes image loading, preprocessing, inference, and postprocessing
- These are reported separately in `inference.py`
- CUDA timing uses `torch.cuda.synchronize()` for accurate measurement

### FPS Measurement

```
FPS = total_images_processed / total_inference_seconds
```

Not `iterations_per_second` — each iteration may process a full batch of images.

### Deterministic Mode

Setting `--seed 42` with `deterministic=True` in config enables reproducible training but **may slow down training** due to deterministic CUDA kernel fallbacks. Some CUDA operations may still introduce tiny differences.
