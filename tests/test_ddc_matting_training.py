"""
End-to-end integration and smoke test for DDC matting training.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from config import Config
from dataset import MattingDataset, build_dataloader, build_matting_train_transform
from models.fast_scnn_salient import FastSCNNSalient
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.ddc_loss import KnownRegionL1Loss, DirectionalDistanceConsistencyLoss
from train import train_one_epoch_matting

DEVICE = torch.device("cpu")


def create_synthetic_matting_dataset(data_dir: Path, num_samples: int = 4) -> None:
    images_dir = data_dir / "images"
    masks_dir = data_dir / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        # Synthetic RGB image
        img = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        # Synthetic binary mask
        mask = np.zeros((128, 128), dtype=np.uint8)
        mask[32:96, 32:96] = 255  # FG square

        cv2.imwrite(str(images_dir / f"sample_{i:04d}.jpg"), img)
        cv2.imwrite(str(masks_dir / f"sample_{i:04d}.png"), mask)


def test_ddc_matting_e2e_training_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        create_synthetic_matting_dataset(data_dir, num_samples=4)

        # 1. Config Setup
        cfg = Config(
            task_mode="ddc_matting",
            loss_profile="ddc_matting",
            batch_size=2,
            epochs=2,
            matting_crop_height=64,
            matting_crop_width=64,
            ddc_window_size=3,
            ddc_num_neighbors=4,
            ddc_chunk_size=1024,
            ddc_lambda=5.0,
        )

        # 2. Dataset & DataLoader Setup
        transform = build_matting_train_transform(
            height=cfg.matting_crop_height,
            width=cfg.matting_crop_width,
            scale_min=0.8,
            scale_max=1.2,
        )
        dataset = MattingDataset(
            data_dir,
            transform=transform,
            trimap_source="binary_mask",
            trimap_kernel_min=2,
            trimap_kernel_max=6,
        )
        dataloader = build_dataloader(
            dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=0,
        )

        # 3. Model Setup
        model = FastSCNNSalient(
            ppm_pool_sizes=(1, 2),
            coarse_channels=16,
            refinement_channels=16,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        known_loss_fn = KnownRegionL1Loss().to(DEVICE)
        ddc_loss_fn = DirectionalDistanceConsistencyLoss(
            window_size=cfg.ddc_window_size,
            num_neighbors=cfg.ddc_num_neighbors,
            chunk_size=cfg.ddc_chunk_size,
        ).to(DEVICE)

        # 4. Train one epoch
        losses, global_step = train_one_epoch_matting(
            model=model,
            dataloader=dataloader,
            known_loss_fn=known_loss_fn,
            ddc_loss_fn=ddc_loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=None,
            device=DEVICE,
            cfg=cfg,
            global_step=0,
            epoch=0,
            writer=None,
            use_amp=False,
        )

        assert "total" in losses
        assert "coarse_known" in losses
        assert "fine_known" in losses
        assert "ddc" in losses
        assert losses["total"] > 0
        assert global_step == 2  # 4 samples, batch size 2 -> 2 steps

        # 5. Checkpoint test
        ckpt_path = data_dir / "latest.pt"
        save_checkpoint(
            path=ckpt_path,
            epoch=0,
            global_step=global_step,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            best_miou=0.0,
            history={"train_loss": [losses["total"]]},
            config=cfg.__dict__,
        )

        assert ckpt_path.exists()

        # Load checkpoint
        new_model = FastSCNNSalient(
            ppm_pool_sizes=(1, 2),
            coarse_channels=16,
            refinement_channels=16,
        ).to(DEVICE)
        ckpt = load_checkpoint(ckpt_path, new_model, map_location=DEVICE, weights_only=True)
        assert ckpt["global_step"] == 2
        assert ckpt["config"]["task_mode"] == "ddc_matting"


def test_stop_gradient_spatial_prompt_gradient_flow() -> None:
    # Verify that gradients from DDC Loss or Fine L1 Loss do NOT backpropagate to the Coarse Head parameters
    model = FastSCNNSalient(
        ppm_pool_sizes=(1, 2),
        coarse_channels=16,
        refinement_channels=16,
    ).to(DEVICE)

    # Put in train mode
    model.train()

    # Zero grads
    model.zero_grad()

    # Inputs
    image = torch.randn(1, 3, 64, 64, device=DEVICE)
    trimap = torch.zeros(1, 1, 64, 64, device=DEVICE)
    trimap[:, :, 16:48, 16:48] = 0.5  # unknown region

    # Forward
    output = model(image)
    fine_alpha = output["fine_prob"]

    # Compute a dummy loss ONLY on fine prediction (e.g., L1 against target)
    loss = torch.abs(fine_alpha - 1.0).mean()
    loss.backward()

    # Coarse head parameters should have NO gradients
    for name, param in model.coarse_head.named_parameters():
        assert param.grad is None or torch.all(param.grad == 0.0), f"Coarse Head param {name} received gradient!"

    # Refinement head and backbone parameters SHOULD have gradients
    refinement_has_grad = False
    for param in model.refinement_head.parameters():
        if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
            refinement_has_grad = True
            break
    assert refinement_has_grad, "Refinement Head did not receive gradient!"

    backbone_has_grad = False
    for param in model.backbone.parameters():
        if param.grad is not None and torch.sum(torch.abs(param.grad)) > 0:
            backbone_has_grad = True
            break
    assert backbone_has_grad, "Shared Backbone did not receive gradient!"
