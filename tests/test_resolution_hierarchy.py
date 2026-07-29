import pytest
import torch
from models.fast_scnn_salient import FastSCNNSalient

def test_fast_scnn_salient_resolution_hierarchy():
    model = FastSCNNSalient(
        ppm_pool_sizes=(1, 2, 3, 6),
        coarse_channels=16,
        refinement_channels=16,
        dropout_p=0.1,
        refinement_head="multiscale",
        prompt_gate_mode="bidirectional",
        prompt_gate_strength=0.5,
        refine_h8_channels=16,
        h4_skip_channels=8,
        refine_h4_channels=16,
        h2_skip_channels=4,
        refine_h2_channels=8,
        fine_output_channels=8,
        fine_dropout=0.1,
        prompt_detach=True,
        uncertainty_floor=0.15,
    )
    model.train()
    
    # Input size 1920*1080 -> 16:9 ratio
    dummy_input = torch.randn(2, 3, 576, 1024)
    
    output = model(dummy_input)
    
    # Verify outputs dictionary keys
    required_keys = {
        "coarse_logits",
        "coarse_prob",
        "fine_logits",
        "fine_prob",
        "coarse_logits_lowres",
        "coarse_prompt",
        "residual_logits",
        "alpha_prompt",
        "uncertainty",
        "boundary",
        "detail_gate",
    }
    for key in required_keys:
        assert key in output, f"Key '{key}' missing from model outputs"
        
    # Verify spatial resolutions
    expected_full_shape = (2, 1, 576, 1024)
    assert output["coarse_logits"].shape == expected_full_shape
    assert output["fine_logits"].shape == expected_full_shape
    assert output["alpha_prompt"].shape == expected_full_shape
    assert output["uncertainty"].shape == expected_full_shape
    assert output["boundary"].shape == expected_full_shape
    assert output["detail_gate"].shape == expected_full_shape
    
    # Low resolution coarse checks (Stage 0 input is 1/2 size -> 288x512 -> lowres feature is 1/8 of that -> 36x64)
    expected_lowres_shape = (2, 1, 36, 64)
    assert output["coarse_logits_lowres"].shape == expected_lowres_shape
    assert output["coarse_prompt"].shape == (2, 1, 72, 128)  # H/8 of full resolution (576/8=72, 1024/8=128)
    
    # Verify detail gate constraints
    detail_gate_val = output["detail_gate"]
    assert detail_gate_val.min() >= 0.15
    assert detail_gate_val.max() <= 1.0
    
    # Verify backward pass / gradient flow
    loss = output["fine_prob"].sum()
    loss.backward()
    
    # Check that parameters in all components have gradients
    backbone_has_grad = any(p.grad is not None for p in model.backbone.parameters())
    coarse_has_grad = any(p.grad is not None for p in model.coarse_head.parameters())
    refinement_has_grad = any(p.grad is not None for p in model.refinement_head.parameters())
    
    assert backbone_has_grad, "Backbone parameters did not receive gradients"
    assert coarse_has_grad, "Coarse head parameters did not receive gradients"
    assert refinement_has_grad, "Refinement head parameters did not receive gradients"
