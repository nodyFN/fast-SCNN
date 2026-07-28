import os
import tempfile
import pytest
import torch
import numpy as np
import cv2
from pathlib import Path
from models.unet import UNet
from utils.birefnet_loader import load_birefnet_teacher
from dataset import SegmentationDataset

def test_load_birefnet_teacher():
    # Save original sys.path
    import sys
    original_path = list(sys.path)
    
    # Temporarily append BiRefNet directory to instantiate BiRefNet for saving dummy weights
    birefnet_path = str(Path(__file__).resolve().parent.parent / "BiRefNet")
    if birefnet_path not in sys.path:
        sys.path.insert(0, birefnet_path)
        
    try:
        from models.birefnet import BiRefNet
        # Create a tiny version or use the standard class
        model = BiRefNet(bb_pretrained=False)
        state_dict = model.state_dict()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "birefnet_dummy.pth")
            # Save state dict
            torch.save({"state_dict": state_dict}, ckpt_path)
            
            # Now load using our safe loader
            loaded_model = load_birefnet_teacher(ckpt_path, torch.device("cpu"))
            
            assert isinstance(loaded_model, BiRefNet)
            assert not any(p.requires_grad for p in loaded_model.parameters())
            assert not loaded_model.training
    finally:
        # Restore sys.path
        sys.path = original_path


def test_segmentation_dataset_load_as_alpha():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create directory layout
        images_dir = tmp_path / "images"
        masks_dir = tmp_path / "alpha_masks"
        images_dir.mkdir(parents=True)
        masks_dir.mkdir(parents=True)
        
        # Create dummy image (RGB) and mask (continuous values)
        dummy_img = np.random.randint(0, 256, (128, 256, 3), dtype=np.uint8)
        dummy_mask = np.random.randint(0, 256, (128, 256), dtype=np.uint8)
        
        cv2.imwrite(str(images_dir / "frame_0001.png"), dummy_img)
        cv2.imwrite(str(masks_dir / "frame_0001.png"), dummy_mask)
        
        # Instantiate dataset with load_as_alpha=True
        dataset = SegmentationDataset(
            root=tmp_path,
            transform=None,
            allow_threshold=False,
            mask_subdir="alpha_masks",
            load_as_alpha=True
        )
        
        assert len(dataset) == 1
        item = dataset[0]
        
        assert "image" in item
        assert "mask" in item
        assert item["mask"].dtype == torch.float32
        assert item["mask"].shape == (128, 256)
        
        # Values should be scaled to [0.0, 1.0]
        mask_np = item["mask"].numpy()
        assert np.all(mask_np >= 0.0)
        assert np.all(mask_np <= 1.0)
        assert np.allclose(mask_np, dummy_mask / 255.0)
