import os
import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path

from config import Config
from models.fast_scnn import FastSCNN
from models.fast_scnn_salient import FastSCNNSalient

class MACProfiler:
    def __init__(self, model, resolution_hierarchy_enabled=False):
        self.model = model
        self.resolution_hierarchy_enabled = resolution_hierarchy_enabled
        self.records = []
        self.hooks = []
        self.register_hooks(self.model, "")

    def register_hooks(self, module, prefix):
        for name, child in module.named_children():
            child_name = f"{prefix}.{name}" if prefix else name
            # Register hooks on layers that do computations
            if len(list(child.children())) == 0 or isinstance(
                child, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d)
            ):
                h = child.register_forward_hook(self.make_hook(child_name))
                self.hooks.append(h)
            else:
                self.register_hooks(child, child_name)

    def make_hook(self, name):
        def hook(module, input_args, output):
            # Parse input/output shapes
            in_shape = tuple(input_args[0].shape) if input_args and len(input_args) > 0 and isinstance(input_args[0], torch.Tensor) else None
            out_shape = tuple(output.shape) if isinstance(output, torch.Tensor) else (tuple(output[0].shape) if isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], torch.Tensor) else None)
            
            macs = 0
            kernel_size = None
            stride = None
            
            if isinstance(module, nn.Conv2d):
                kernel_size = module.kernel_size
                stride = module.stride
                groups = module.groups
                in_ch = module.in_channels
                out_ch = module.out_channels
                out_h, out_w = out_shape[-2:]
                macs = out_h * out_w * out_ch * (in_ch // groups) * kernel_size[0] * kernel_size[1]
            elif isinstance(module, nn.Linear):
                in_features = module.in_features
                out_features = module.out_features
                batch_size = out_shape[0] if out_shape else 1
                macs = in_features * out_features * batch_size

            # Assign Stage label if resolution hierarchy is enabled
            stage = "N/A"
            if self.resolution_hierarchy_enabled:
                if in_shape:
                    h = in_shape[-2]
                    # If input height is 64 (or 0.5x of 128), it's Stage 0 (Coarse)
                    if h <= 64:
                        stage = "Stage 0 (Coarse)"
                    else:
                        stage = "Stage 1 (Fine)"
            else:
                # For single-stage models
                stage = "Stage 1 (Fine)"

            self.records.append({
                "Stage": stage,
                "Layer Name": name,
                "Layer Type": module.__class__.__name__,
                "Input Shape": str(in_shape) if in_shape else "",
                "Output Shape": str(out_shape) if out_shape else "",
                "Kernel Size": str(kernel_size) if kernel_size else "",
                "Stride": str(stride) if stride else "",
                "MACs": macs
            })
        return hook

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []


def main():
    device = torch.device("cpu")
    cfg = Config()
    
    # Target resolution: w=224, h=128
    input_size = (1, 3, 128, 224)
    x = torch.randn(input_size)
    
    print("Initializing models...")
    
    # 1. Original Fast-SCNN
    # Fast-SCNN in models/fast_scnn.py uses num_classes and aux
    model_orig = FastSCNN(num_classes=2, aux=True).to(device)
    model_orig.eval()
    
    # 2. Fast-SCNN Dual Head (Single-Stage, resolution_hierarchy = False)
    model_dh = FastSCNNSalient(
        ppm_pool_sizes=cfg.ppm_pool_sizes,
        coarse_channels=cfg.coarse_channels,
        refinement_channels=cfg.refinement_channels,
        dropout_p=cfg.dropout_p,
        refinement_head=cfg.refinement_head,
        prompt_gate_mode=cfg.prompt_gate_mode,
        prompt_gate_strength=cfg.prompt_gate_strength,
        refine_h8_channels=cfg.refine_h8_channels,
        h4_skip_channels=cfg.h4_skip_channels,
        refine_h4_channels=cfg.refine_h4_channels,
        h2_skip_channels=cfg.h2_skip_channels,
        refine_h2_channels=cfg.refine_h2_channels,
        fine_dropout=cfg.fine_dropout,
        prompt_detach=cfg.prompt_detach,
        uncertainty_floor=cfg.uncertainty_floor,
        resolution_hierarchy=False, # Force Single-Stage
    ).to(device)
    model_dh.eval()
    
    # 3. Fast-SCNN Two-Stage (Multi-Stage, resolution_hierarchy = True)
    model_ts = FastSCNNSalient(
        ppm_pool_sizes=cfg.ppm_pool_sizes,
        coarse_channels=cfg.coarse_channels,
        refinement_channels=cfg.refinement_channels,
        dropout_p=cfg.dropout_p,
        refinement_head=cfg.refinement_head,
        prompt_gate_mode=cfg.prompt_gate_mode,
        prompt_gate_strength=cfg.prompt_gate_strength,
        refine_h8_channels=cfg.refine_h8_channels,
        h4_skip_channels=cfg.h4_skip_channels,
        refine_h4_channels=cfg.refine_h4_channels,
        h2_skip_channels=cfg.h2_skip_channels,
        refine_h2_channels=cfg.refine_h2_channels,
        fine_dropout=cfg.fine_dropout,
        prompt_detach=cfg.prompt_detach,
        uncertainty_floor=cfg.uncertainty_floor,
        resolution_hierarchy=True, # Force Two-Stage
    ).to(device)
    model_ts.eval()
    
    print("Profiling Original Fast-SCNN...")
    profiler_orig = MACProfiler(model_orig, resolution_hierarchy_enabled=False)
    with torch.no_grad():
        model_orig(x)
    profiler_orig.remove_hooks()
    df_orig = pd.DataFrame(profiler_orig.records)
    
    print("Profiling Fast-SCNN Dual Head...")
    profiler_dh = MACProfiler(model_dh, resolution_hierarchy_enabled=False)
    with torch.no_grad():
        model_dh(x)
    profiler_dh.remove_hooks()
    df_dh = pd.DataFrame(profiler_dh.records)
    
    print("Profiling Fast-SCNN Two-Stage...")
    profiler_ts = MACProfiler(model_ts, resolution_hierarchy_enabled=True)
    with torch.no_grad():
        model_ts(x)
    profiler_ts.remove_hooks()
    df_ts = pd.DataFrame(profiler_ts.records)
    
    # Add summary rows
    def append_summary(df):
        total_macs = df["MACs"].sum()
        summary_row = {
            "Stage": "Total",
            "Layer Name": "",
            "Layer Type": "",
            "Input Shape": "",
            "Output Shape": "",
            "Kernel Size": "",
            "Stride": "",
            "MACs": total_macs
        }
        return pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
        
    df_orig_sum = append_summary(df_orig)
    df_dh_sum = append_summary(df_dh)
    df_ts_sum = append_summary(df_ts)
    
    # Output to Excel file
    output_excel = "fast_scnn_macs_224x128.xlsx"
    print(f"Saving results to {output_excel}...")
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        df_orig_sum.to_excel(writer, sheet_name="original_fast_scnn", index=False)
        df_dh_sum.to_excel(writer, sheet_name="fast_scnn_dual_head", index=False)
        df_ts_sum.to_excel(writer, sheet_name="fast_scnn_two_stage", index=False)
        
    print("Done! Excel file generated successfully.")
    
    # Print high-level summaries
    print("\nHigh-Level MACs Summary (Input Size: 224x128):")
    print(f"1. Original Fast-SCNN          : {df_orig['MACs'].sum():,} MACs")
    print(f"2. Fast-SCNN Dual Head (1-stage): {df_dh['MACs'].sum():,} MACs")
    print(f"3. Fast-SCNN Two-Stage (2-stage): {df_ts['MACs'].sum():,} MACs")

if __name__ == "__main__":
    main()
