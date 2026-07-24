import os
import tempfile
import pytest
import torch
import torch.nn as nn
from models.unet import UNet
from models.fast_scnn_salient import FastSCNNSalient
from utils.losses import compute_kd_loss
from utils.checkpoint import save_checkpoint, load_checkpoint

def test_unet_shapes():
    # Test UNet input/output shape consistency
    model = UNet(in_channels=3, out_channels=1, init_features=8)
    model.eval()
    
    x = torch.randn(1, 3, 256, 512)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 1, 256, 512)


def test_kd_loss_gradient_flow():
    device = torch.device("cpu")
    
    # 1. Instantiate student and teacher models
    student = FastSCNNSalient(
        ppm_pool_sizes=(1, 2, 3, 6),
        coarse_channels=16,
        refinement_channels=16,
        dropout_p=0.1
    ).to(device)
    
    teacher = UNet(in_channels=3, out_channels=1, init_features=8).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
        
    # 2. Forward pass
    images = torch.randn(2, 3, 128, 256, device=device)
    
    student.train()
    student_out = student(images)
    
    with torch.no_grad():
        teacher_out = teacher(images)
        
    # 3. Compute KD Loss
    kd_loss = compute_kd_loss(
        student_out=student_out,
        teacher_out=teacher_out,
        loss_type="mse",
        temp=1.5,
        is_salient=True
    )
    
    assert kd_loss.shape == ()
    assert kd_loss.item() >= 0.0
    
    # 4. Backward pass
    kd_loss.backward()
    
    # Verify student has gradients
    student_has_grad = False
    for p in student.parameters():
        if p.grad is not None:
            student_has_grad = True
            break
    assert student_has_grad, "Student parameters should have gradients from KD loss."
    
    # Verify teacher has NO gradients
    for p in teacher.parameters():
        assert p.grad is None, "Teacher parameters should NOT have gradients."


def test_kd_save_load_teacher():
    # Save a dummy teacher checkpoint and load it
    teacher = UNet(in_channels=3, out_channels=1, init_features=8)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "teacher.pt")
        # Save dummy weights
        save_checkpoint(
            ckpt_path,
            epoch=0,
            global_step=0,
            model=teacher,
            optimizer=None,
            scheduler=None,
            scaler=None,
            best_miou=0.0,
            history={},
            config={},
            class_names=[],
            num_classes=1,
            seed=42
        )
        
        # Load weights into a new teacher model
        new_teacher = UNet(in_channels=3, out_channels=1, init_features=8)
        load_checkpoint(ckpt_path, new_teacher, weights_only=True)
        
        # Verify weight equivalence
        for p1, p2 in zip(teacher.parameters(), new_teacher.parameters()):
            assert torch.allclose(p1, p2)


def test_unet_salient_adapter():
    from models.unet import UNetSalientAdapter
    unet = UNet(in_channels=3, out_channels=1, init_features=8)
    adapter = UNetSalientAdapter(unet)
    adapter.train()
    
    x = torch.randn(2, 3, 64, 128)
    out = adapter(x)
    
    assert isinstance(out, dict)
    assert "coarse_logits" in out
    assert "fine_logits" in out
    assert out["coarse_logits"].shape == (2, 1, 64, 128)
    assert out["fine_logits"].shape == (2, 1, 64, 128)

