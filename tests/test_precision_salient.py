import pytest
import torch
import torch.nn as nn
from models.fast_scnn_salient import FastSCNNSalient
from utils.precision_losses import BinaryTverskyLoss, BoundaryWeightedBCELoss, HardNegativeBCELoss
from utils.losses import PrecisionSalientLoss
from config import Config

def test_fast_scnn_salient_architectures():
    # 1. Test legacy head
    model_legacy = FastSCNNSalient(refinement_head="legacy_h8")
    x = torch.randn(1, 3, 256, 512)
    out_legacy = model_legacy(x)
    assert "fine_logits" in out_legacy
    assert out_legacy["fine_logits"].shape == (1, 1, 256, 512)
    
    # 2. Test multiscale head with bidirectional prompt gate
    model_ms = FastSCNNSalient(
        refinement_head="multiscale",
        prompt_gate_mode="bidirectional",
        prompt_gate_strength=0.5
    )
    out_ms = model_ms(x)
    assert "fine_logits" in out_ms
    assert out_ms["fine_logits"].shape == (1, 1, 256, 512)
    assert out_ms["coarse_logits"].shape == (1, 1, 256, 512)


def test_precision_losses():
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randint(0, 2, (2, 1, 64, 64)).float()
    
    # 1. BinaryTverskyLoss
    tversky = BinaryTverskyLoss(fp_weight=0.7, fn_weight=0.3)
    loss_t = tversky(pred, target)
    assert loss_t.shape == ()
    assert loss_t.item() >= 0.0
    
    # 2. BoundaryWeightedBCELoss
    boundary_bce = BoundaryWeightedBCELoss(boundary_kernel_size=7, boundary_extra_weight=4.0)
    loss_b = boundary_bce(pred, target.long())
    assert loss_b.shape == ()
    assert loss_b.item() >= 0.0
    
    # 3. HardNegativeBCELoss
    hard_bce = HardNegativeBCELoss(hard_negative_ratio=0.10, hard_negative_min_pixels=64)
    loss_h = hard_bce(pred, target.long())
    assert loss_h.shape == ()
    assert loss_h.item() >= 0.0


def test_precision_salient_loss_wrapper():
    cfg = Config()
    cfg.loss_profile = "precision_salient"
    cfg.salient_lambda_coarse = 1.0
    cfg.salient_lambda_fine = 1.0
    cfg.coarse_bce_weight = 1.0
    cfg.coarse_dice_weight = 1.0
    cfg.fine_bce_weight = 0.5
    cfg.fine_tversky_weight = 1.0
    cfg.fine_boundary_weight = 0.25
    cfg.fine_hard_negative_weight = 0.25
    
    loss_fn = PrecisionSalientLoss(cfg)
    
    coarse_logits = torch.randn(2, 1, 64, 64)
    fine_logits = torch.randn(2, 1, 64, 64)
    targets = torch.randint(0, 2, (2, 1, 64, 64))
    
    losses = loss_fn(coarse_logits, fine_logits, targets)
    
    expected_keys = [
        "total", "coarse_total", "coarse_bce", "coarse_dice",
        "fine_total", "fine_bce", "fine_tversky", "fine_boundary_bce",
        "fine_hard_negative"
    ]
    for key in expected_keys:
        assert key in losses, f"Missing expected loss component key: {key}"
        assert losses[key].shape == ()
