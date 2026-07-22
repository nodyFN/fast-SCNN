"""
Unit tests for DDC alpha-free matting losses.
"""

from __future__ import annotations

import pytest
import torch

from utils.ddc_loss import KnownRegionL1Loss, DirectionalDistanceConsistencyLoss

DEVICE = torch.device("cpu")


class TestKnownRegionL1Loss:
    def test_l1_zero_when_equal(self) -> None:
        loss_fn = KnownRegionL1Loss()
        pred = torch.tensor([[[[0.2, 0.5, 0.8]]]], dtype=torch.float32, device=DEVICE)
        trimap = torch.tensor([[[[0.2, 0.5, 0.8]]]], dtype=torch.float32, device=DEVICE)
        # Note: 0.5 is unknown region so its value in L1 calculation should be ignored.
        # But 0.2 and 0.8 are known, and they match exactly.
        loss = loss_fn(pred, trimap)
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_l1_only_computed_on_known(self) -> None:
        loss_fn = KnownRegionL1Loss()
        # Known region (0.0 or 1.0) values differ; unknown region (0.5) has massive difference
        pred = torch.tensor([[[[0.1, 0.9, 0.9]]]], dtype=torch.float32, device=DEVICE)   # error: 0.1 at 0.0, 0.1 at 1.0
        trimap = torch.tensor([[[[0.0, 0.5, 1.0]]]], dtype=torch.float32, device=DEVICE)
        
        loss = loss_fn(pred, trimap)
        # Known pixels: [0, 0] (error 0.1) and [0, 2] (error 0.1). Total known = 2.
        # Sum of absolute differences = 0.1 + 0.1 = 0.2.
        # Mean over known = 0.2 / 2 = 0.1.
        assert torch.allclose(loss, torch.tensor(0.1))

    def test_no_known_pixels_graceful_handling(self) -> None:
        loss_fn = KnownRegionL1Loss()
        pred = torch.tensor([[[[0.1, 0.2]]]], dtype=torch.float32, device=DEVICE)
        trimap = torch.tensor([[[[0.5, 0.5]]]], dtype=torch.float32, device=DEVICE)
        
        with pytest.warns(RuntimeWarning, match="no known pixels"):
            loss = loss_fn(pred, trimap)
        assert torch.allclose(loss, torch.tensor(0.0))


class TestDirectionalDistanceConsistencyLoss:
    def test_ddc_gradient_flow_and_shapes(self) -> None:
        alpha = torch.randn(1, 1, 32, 32, dtype=torch.float32, device=DEVICE).sigmoid()
        alpha.requires_grad = True
        
        # Raw RGB image
        image = torch.rand(1, 3, 32, 32, dtype=torch.float32, device=DEVICE)
        
        # window size 5, 5 neighbors
        loss_fn = DirectionalDistanceConsistencyLoss(
            window_size=5, num_neighbors=5, chunk_size=0,
        )
        
        loss = loss_fn(alpha, image)
        assert loss.shape == ()
        assert loss.device == DEVICE
        assert torch.isfinite(loss)
        
        loss.backward()
        assert alpha.grad is not None
        assert alpha.grad.shape == alpha.shape
        assert torch.isfinite(alpha.grad).all()

    def test_chunked_vs_full_consistency(self) -> None:
        # Generate random inputs
        alpha = torch.rand(2, 1, 32, 32, dtype=torch.float32, device=DEVICE)
        image = torch.rand(2, 3, 32, 32, dtype=torch.float32, device=DEVICE)
        
        loss_fn_full = DirectionalDistanceConsistencyLoss(
            window_size=7, num_neighbors=12, chunk_size=0,
        )
        loss_fn_chunked = DirectionalDistanceConsistencyLoss(
            window_size=7, num_neighbors=12, chunk_size=128,  # small chunk size to trigger chunking
        )
        
        loss_full = loss_fn_full(alpha, image)
        loss_chunked = loss_fn_chunked(alpha, image)
        
        assert torch.allclose(loss_full, loss_chunked, atol=1e-6)

    def test_downsample_factor_consistency(self) -> None:
        alpha = torch.rand(1, 1, 16, 16, dtype=torch.float32, device=DEVICE)
        image = torch.rand(1, 3, 16, 16, dtype=torch.float32, device=DEVICE)
        
        loss_fn_ds = DirectionalDistanceConsistencyLoss(
            window_size=3, num_neighbors=4, downsample_factor=2,
        )
        loss = loss_fn_ds(alpha, image)
        assert torch.isfinite(loss)
