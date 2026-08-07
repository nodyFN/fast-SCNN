import pytest
import torch
import torch.nn.functional as F
from utils.losses import binary_erode, binary_dilate, masked_balanced_bce_with_logits

def compute_ranking_loss(student_fine_prob, inner_fg, outer_bg, margin, radius):
    bg_score = student_fine_prob.masked_fill(outer_bg < 0.5, -1e9)
    rank_kernel_size = 2 * radius + 1
    local_hard_bg = F.max_pool2d(
        bg_score,
        kernel_size=rank_kernel_size,
        stride=1,
        padding=radius,
    )
    has_local_bg = F.max_pool2d(
        outer_bg,
        kernel_size=rank_kernel_size,
        stride=1,
        padding=radius,
    ) > 0.5
    
    valid_mask = (inner_fg >= 0.5) & has_local_bg
    pixel_rank_loss = F.relu(margin - (student_fine_prob - local_hard_bg))
    
    rank_sum = (pixel_rank_loss * valid_mask.float()).sum(dim=(1, 2, 3))
    valid_count = valid_mask.float().sum(dim=(1, 2, 3))
    ranking_loss = torch.where(
        valid_count > 0,
        rank_sum / valid_count.clamp_min(1.0),
        torch.zeros_like(rank_sum)
    ).mean()
    
    return ranking_loss, valid_count.mean().item()

def test_boundary_losses_good_prediction():
    # Set seed
    torch.manual_seed(42)
    
    # Create a simple synthetic mask (circle in the center)
    B, C, H, W = 2, 1, 32, 32
    gt_binary = torch.zeros((B, C, H, W), dtype=torch.float32)
    gt_binary[:, :, 8:24, 8:24] = 1.0
    
    # Morphology
    eroded = binary_erode(gt_binary, kernel_size=3)
    dilated = binary_dilate(gt_binary, radius=1)
    inner_fg = (gt_binary - eroded).clamp(0.0, 1.0)
    outer_bg = (dilated - gt_binary).clamp(0.0, 1.0)
    
    # 1. GOOD PREDICTION SCENARIO: inner_fg has high probability, outer_bg has low probability
    # We construct logits
    logits = torch.zeros((B, C, H, W), dtype=torch.float32)
    # Background interior: very negative
    logits.fill_(-10.0)
    # Foreground interior: very positive
    logits[gt_binary > 0.5] = 10.0
    
    # Close to boundary:
    # Inner FG: highly positive (e.g. 2.2 -> sigmoid = 0.90)
    logits[inner_fg > 0.5] = 2.2
    # Outer BG: highly negative (e.g. -2.2 -> sigmoid = 0.10)
    logits[outer_bg > 0.5] = -2.2
    
    # Compute losses
    boundary_side_loss = masked_balanced_bce_with_logits(
        logits=logits,
        targets=gt_binary,
        fg_mask=inner_fg,
        bg_mask=outer_bg,
    )
    
    prob = torch.sigmoid(logits)
    ranking_loss, valid_cnt = compute_ranking_loss(
        student_fine_prob=prob,
        inner_fg=inner_fg,
        outer_bg=outer_bg,
        margin=0.3,
        radius=2,
    )
    
    # For good prediction, BCE loss on boundary should be low (around BCE of 0.9 and 0.1, which is ~0.325)
    assert boundary_side_loss.item() < 0.4
    
    # Difference is 0.9 - 0.1 = 0.8, which is > margin (0.3). Thus, ranking loss should be exactly 0.0
    assert ranking_loss.item() == 0.0
    assert valid_cnt > 0

def test_boundary_losses_bad_prediction():
    B, C, H, W = 2, 1, 32, 32
    gt_binary = torch.zeros((B, C, H, W), dtype=torch.float32)
    gt_binary[:, :, 8:24, 8:24] = 1.0
    
    eroded = binary_erode(gt_binary, kernel_size=3)
    dilated = binary_dilate(gt_binary, radius=1)
    inner_fg = (gt_binary - eroded).clamp(0.0, 1.0)
    outer_bg = (dilated - gt_binary).clamp(0.0, 1.0)
    
    # 2. BAD PREDICTION SCENARIO: confused probabilities on boundary (inner_fg = 0.55, outer_bg = 0.60)
    logits = torch.zeros((B, C, H, W), dtype=torch.float32)
    # Background interior: very negative
    logits.fill_(-10.0)
    # Foreground interior: very positive
    logits[gt_binary > 0.5] = 10.0
    
    # Inner FG: logits = 0.2 (sigmoid = 0.55)
    logits[inner_fg > 0.5] = 0.2
    # Outer BG: logits = 0.4 (sigmoid = 0.60)
    logits[outer_bg > 0.5] = 0.4
    
    boundary_side_loss = masked_balanced_bce_with_logits(
        logits=logits,
        targets=gt_binary,
        fg_mask=inner_fg,
        bg_mask=outer_bg,
    )
    
    prob = torch.sigmoid(logits)
    ranking_loss, valid_cnt = compute_ranking_loss(
        student_fine_prob=prob,
        inner_fg=inner_fg,
        outer_bg=outer_bg,
        margin=0.3,
        radius=2,
    )
    
    # For bad prediction, BCE loss on boundary should be high (BCE of 0.55/0.60 is > 0.6)
    assert boundary_side_loss.item() > 0.5
    
    # Difference is 0.55 - 0.60 = -0.05. Margin is 0.3.
    # ranking loss = relu(0.3 - (-0.05)) = 0.35
    assert ranking_loss.item() > 0.3
    assert valid_cnt > 0

def test_boundary_losses_gradient_propagation():
    # Require grad on logits to test backward pass
    B, C, H, W = 1, 1, 16, 16
    logits = torch.zeros((B, C, H, W), requires_grad=True)
    gt_binary = torch.zeros((B, C, H, W))
    gt_binary[:, :, 4:12, 4:12] = 1.0
    
    eroded = binary_erode(gt_binary, kernel_size=3)
    dilated = binary_dilate(gt_binary, radius=1)
    inner_fg = (gt_binary - eroded).clamp(0.0, 1.0)
    outer_bg = (dilated - gt_binary).clamp(0.0, 1.0)
    
    # Mix values
    logits_data = logits.detach().clone()
    logits_data[inner_fg > 0.5] = 0.1
    logits_data[outer_bg > 0.5] = 0.5
    logits.data.copy_(logits_data)
    
    # Forward
    boundary_side_loss = masked_balanced_bce_with_logits(
        logits=logits,
        targets=gt_binary,
        fg_mask=inner_fg,
        bg_mask=outer_bg,
    )
    
    prob = torch.sigmoid(logits)
    ranking_loss, _ = compute_ranking_loss(
        student_fine_prob=prob,
        inner_fg=inner_fg,
        outer_bg=outer_bg,
        margin=0.3,
        radius=2,
    )
    
    total_loss = 0.5 * boundary_side_loss + 0.25 * ranking_loss
    total_loss.backward()
    
    # Verify gradients are non-zero and finite
    assert logits.grad is not None
    assert torch.all(torch.isfinite(logits.grad))
    assert not torch.all(logits.grad == 0.0)

def test_boundary_losses_empty_mask():
    # Empty mask checks for numerical safety
    B, C, H, W = 2, 1, 16, 16
    logits = torch.zeros((B, C, H, W), requires_grad=True)
    gt_binary = torch.zeros((B, C, H, W)) # completely empty!
    
    eroded = binary_erode(gt_binary, kernel_size=3)
    dilated = binary_dilate(gt_binary, radius=1)
    inner_fg = (gt_binary - eroded).clamp(0.0, 1.0)
    outer_bg = (dilated - gt_binary).clamp(0.0, 1.0)
    
    boundary_side_loss = masked_balanced_bce_with_logits(
        logits=logits,
        targets=gt_binary,
        fg_mask=inner_fg,
        bg_mask=outer_bg,
    )
    
    prob = torch.sigmoid(logits)
    ranking_loss, valid_cnt = compute_ranking_loss(
        student_fine_prob=prob,
        inner_fg=inner_fg,
        outer_bg=outer_bg,
        margin=0.3,
        radius=2,
    )
    
    assert boundary_side_loss.item() == 0.0
    assert ranking_loss.item() == 0.0
    assert valid_cnt == 0
    
    # Backprop should still succeed without NaN
    total_loss = boundary_side_loss + ranking_loss
    total_loss.backward()
    assert logits.grad is not None
    assert torch.all(logits.grad == 0.0)
