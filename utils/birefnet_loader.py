import sys
from pathlib import Path
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_birefnet_teacher(weights_path: str, device: torch.device) -> nn.Module:
    """Load pre-trained BiRefNet model from local weights path."""
    birefnet_path = str(PROJECT_ROOT / "BiRefNet")
    
    # Save original sys.path
    original_path = list(sys.path)
    try:
        # Prepend BiRefNet path to avoid namespace conflicts
        if birefnet_path not in sys.path:
            sys.path.insert(0, birefnet_path)
            
        from models.birefnet import BiRefNet
        from utils import check_state_dict
        
        # Instantiate model without backbone pretraining downloads (weights contain everything)
        model = BiRefNet(bb_pretrained=False)
        
        # Load weights
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict):
            if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
                checkpoint = checkpoint["state_dict"]
            elif "model" in checkpoint and isinstance(checkpoint["model"], dict):
                checkpoint = checkpoint["model"]
                
        state_dict = check_state_dict(checkpoint)
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        
        # Disable gradients
        for param in model.parameters():
            param.requires_grad = False
            
        model.to(device)
        return model
        
    finally:
        # Restore original sys.path
        sys.path = original_path
