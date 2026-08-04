import pandas as pd
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side

def make_orig_layers():
    # Original Fast-SCNN layers at 224x128 resolution
    layers = [
        # LtD
        ("Stage 1 (Fine)", "backbone.learning_to_downsample.conv.block.0", "Conv2d", "(1, 3, 128, 224)", "(1, 32, 64, 112)", "(3, 3)", "(2, 2)", 6193152),
        ("Stage 1 (Fine)", "backbone.learning_to_downsample.dsconv1.depthwise", "Conv2d", "(1, 32, 64, 112)", "(1, 32, 32, 56)", "(3, 3)", "(2, 2)", 516096),
        ("Stage 1 (Fine)", "backbone.learning_to_downsample.dsconv1.pointwise", "Conv2d", "(1, 32, 32, 56)", "(1, 48, 32, 56)", "(1, 1)", "(1, 1)", 2752512),
        ("Stage 1 (Fine)", "backbone.learning_to_downsample.dsconv2.depthwise", "Conv2d", "(1, 48, 32, 56)", "(1, 48, 16, 28)", "(3, 3)", "(2, 2)", 193536),
        ("Stage 1 (Fine)", "backbone.learning_to_downsample.dsconv2.pointwise", "Conv2d", "(1, 48, 16, 28)", "(1, 64, 16, 28)", "(1, 1)", "(1, 1)", 1376256),
        # GFE
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.0.expand", "Conv2d", "(1, 64, 16, 28)", "(1, 384, 16, 28)", "(1, 1)", "(1, 1)", 11059200),
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.0.depthwise", "Conv2d", "(1, 384, 16, 28)", "(1, 384, 8, 14)", "(3, 3)", "(2, 2)", 387072),
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.0.project", "Conv2d", "(1, 384, 8, 14)", "(1, 64, 8, 14)", "(1, 1)", "(1, 1)", 2752512),
        
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.1.expand", "Conv2d", "(1, 64, 8, 14)", "(1, 384, 8, 14)", "(1, 1)", "(1, 1)", 2752512),
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.1.depthwise", "Conv2d", "(1, 384, 8, 14)", "(1, 384, 8, 14)", "(3, 3)", "(1, 1)", 387072),
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.1.project", "Conv2d", "(1, 384, 8, 14)", "(1, 64, 8, 14)", "(1, 1)", "(1, 1)", 2752512),
        
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.2.expand", "Conv2d", "(1, 64, 8, 14)", "(1, 384, 8, 14)", "(1, 1)", "(1, 1)", 2752512),
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.2.depthwise", "Conv2d", "(1, 384, 8, 14)", "(1, 384, 8, 14)", "(3, 3)", "(1, 1)", 387072),
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.2.project", "Conv2d", "(1, 384, 8, 14)", "(1, 64, 8, 14)", "(1, 1)", "(1, 1)", 2752512),
        
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.3.expand", "Conv2d", "(1, 64, 8, 14)", "(1, 384, 8, 14)", "(1, 1)", "(1, 1)", 2752512),
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.3.depthwise", "Conv2d", "(1, 384, 8, 14)", "(1, 384, 4, 7)", "(3, 3)", "(2, 2)", 96768),
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.3.project", "Conv2d", "(1, 384, 4, 7)", "(1, 96, 4, 7)", "(1, 1)", "(1, 1)", 1032192),
        
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.4.expand", "Conv2d", "(1, 96, 4, 7)", "(1, 576, 4, 7)", "(1, 1)", "(1, 1)", 1548288),
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.4.depthwise", "Conv2d", "(1, 576, 4, 7)", "(1, 576, 4, 7)", "(3, 3)", "(1, 1)", 145152),
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.4.project", "Conv2d", "(1, 576, 4, 7)", "(1, 96, 4, 7)", "(1, 1)", "(1, 1)", 1548288),
        
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.5.expand", "Conv2d", "(1, 96, 4, 7)", "(1, 576, 4, 7)", "(1, 1)", "(1, 1)", 1548288),
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.5.depthwise", "Conv2d", "(1, 576, 4, 7)", "(1, 576, 4, 7)", "(3, 3)", "(1, 1)", 145152),
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.5.project", "Conv2d", "(1, 576, 4, 7)", "(1, 96, 4, 7)", "(1, 1)", "(1, 1)", 1548288),
        
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.6.expand", "Conv2d", "(1, 96, 4, 7)", "(1, 576, 4, 7)", "(1, 1)", "(1, 1)", 1548288),
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.6.depthwise", "Conv2d", "(1, 576, 4, 7)", "(1, 576, 4, 7)", "(3, 3)", "(1, 1)", 145152),
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.6.project", "Conv2d", "(1, 576, 4, 7)", "(1, 128, 4, 7)", "(1, 1)", "(1, 1)", 2064384),
        
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.7.expand", "Conv2d", "(1, 128, 4, 7)", "(1, 768, 4, 7)", "(1, 1)", "(1, 1)", 2752512),
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.7.depthwise", "Conv2d", "(1, 768, 4, 7)", "(1, 768, 4, 7)", "(3, 3)", "(1, 1)", 193536),
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.7.project", "Conv2d", "(1, 768, 4, 7)", "(1, 128, 4, 7)", "(1, 1)", "(1, 1)", 2752512),
        
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.8.expand", "Conv2d", "(1, 128, 4, 7)", "(1, 768, 4, 7)", "(1, 1)", "(1, 1)", 2752512),
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.8.depthwise", "Conv2d", "(1, 768, 4, 7)", "(1, 768, 4, 7)", "(3, 3)", "(1, 1)", 193536),
        ("Stage 1 (Fine)", "backbone.global_feature_extractor.bottlenecks.8.project", "Conv2d", "(1, 768, 4, 7)", "(1, 128, 4, 7)", "(1, 1)", "(1, 1)", 2752512),
        
        # PPM
        ("Stage 1 (Fine)", "backbone.ppm.branches.0", "Conv2d", "(1, 128, 1, 1)", "(1, 32, 1, 1)", "(1, 1)", "(1, 1)", 4096),
        ("Stage 1 (Fine)", "backbone.ppm.branches.1", "Conv2d", "(1, 128, 2, 2)", "(1, 32, 2, 2)", "(1, 1)", "(1, 1)", 16384),
        ("Stage 1 (Fine)", "backbone.ppm.branches.2", "Conv2d", "(1, 128, 3, 3)", "(1, 32, 3, 3)", "(1, 1)", "(1, 1)", 36864),
        ("Stage 1 (Fine)", "backbone.ppm.branches.3", "Conv2d", "(1, 128, 6, 6)", "(1, 32, 6, 6)", "(1, 1)", "(1, 1)", 147456),
        ("Stage 1 (Fine)", "backbone.ppm.fusion", "Conv2d", "(1, 256, 4, 7)", "(1, 128, 4, 7)", "(1, 1)", "(1, 1)", 917504),
        
        # FFM
        ("Stage 1 (Fine)", "ffm.low_res_conv.depthwise", "Conv2d", "(1, 128, 16, 28)", "(1, 128, 16, 28)", "(3, 3)", "(1, 1)", 516096),
        ("Stage 1 (Fine)", "ffm.low_res_conv.pointwise", "Conv2d", "(1, 128, 16, 28)", "(1, 128, 16, 28)", "(1, 1)", "(1, 1)", 7372800),
        ("Stage 1 (Fine)", "ffm.high_res_conv", "Conv2d", "(1, 64, 16, 28)", "(1, 128, 16, 28)", "(1, 1)", "(1, 1)", 3686400),
        
        # Classifier
        ("Stage 1 (Fine)", "classifier.dsconv1.depthwise", "Conv2d", "(1, 128, 16, 28)", "(1, 128, 16, 28)", "(3, 3)", "(1, 1)", 516096),
        ("Stage 1 (Fine)", "classifier.dsconv1.pointwise", "Conv2d", "(1, 128, 16, 28)", "(1, 128, 16, 28)", "(1, 1)", "(1, 1)", 7372800),
        ("Stage 1 (Fine)", "classifier.dsconv2.depthwise", "Conv2d", "(1, 128, 16, 28)", "(1, 128, 16, 28)", "(3, 3)", "(1, 1)", 516096),
        ("Stage 1 (Fine)", "classifier.dsconv2.pointwise", "Conv2d", "(1, 128, 16, 28)", "(1, 128, 16, 28)", "(1, 1)", "(1, 1)", 7372800),
        ("Stage 1 (Fine)", "classifier.conv", "Conv2d", "(1, 128, 16, 28)", "(1, 2, 16, 28)", "(1, 1)", "(1, 1)", 114688)
    ]
    return layers

def make_dh_layers():
    # Fast-SCNN Dual Head (1-stage, resolution_hierarchy = False)
    # The backbone layers are EXACTLY same as original Fast-SCNN
    orig_backbone_and_ppm = make_orig_layers()[:-4] # exclude FFM and Classifier
    
    # Coarse Head
    coarse_head_layers = [
        ("Stage 1 (Fine)", "coarse_head.dsconv.depthwise", "Conv2d", "(1, 128, 16, 28)", "(1, 128, 16, 28)", "(3, 3)", "(1, 1)", 516096),
        ("Stage 1 (Fine)", "coarse_head.dsconv.pointwise", "Conv2d", "(1, 128, 16, 28)", "(1, 64, 16, 28)", "(1, 1)", "(1, 1)", 3686400),
        ("Stage 1 (Fine)", "coarse_head.conv", "Conv2d", "(1, 64, 16, 28)", "(1, 1, 16, 28)", "(1, 1)", "(1, 1)", 28672)
    ]
    
    # Refinement Head (multiscale) - resolution_hierarchy=False means 1 extra channel (only alpha)
    refine_head_layers = [
        ("Stage 1 (Fine)", "refinement_head.h8_proj.block.0", "Conv2d", "(1, 129, 16, 28)", "(1, 96, 16, 28)", "(1, 1)", "(1, 1)", 5547264),
        ("Stage 1 (Fine)", "refinement_head.h8_dsconv.depthwise", "Conv2d", "(1, 96, 16, 28)", "(1, 96, 16, 28)", "(3, 3)", "(1, 1)", 387072),
        ("Stage 1 (Fine)", "refinement_head.h8_dsconv.pointwise", "Conv2d", "(1, 96, 16, 28)", "(1, 96, 16, 28)", "(1, 1)", "(1, 1)", 4128768),
        
        ("Stage 1 (Fine)", "refinement_head.h4_skip_proj.block.0", "Conv2d", "(1, 48, 32, 56)", "(1, 32, 32, 56)", "(1, 1)", "(1, 1)", 2752512),
        ("Stage 1 (Fine)", "refinement_head.h4_fusion_proj.block.0", "Conv2d", "(1, 128, 32, 56)", "(1, 64, 32, 56)", "(1, 1)", "(1, 1)", 14680064),
        ("Stage 1 (Fine)", "refinement_head.h4_dsconv.depthwise", "Conv2d", "(1, 64, 32, 56)", "(1, 64, 32, 56)", "(3, 3)", "(1, 1)", 1032192),
        ("Stage 1 (Fine)", "refinement_head.h4_dsconv.pointwise", "Conv2d", "(1, 64, 32, 56)", "(1, 64, 32, 56)", "(1, 1)", "(1, 1)", 7340032),
        
        ("Stage 1 (Fine)", "refinement_head.h2_skip_proj.block.0", "Conv2d", "(1, 32, 64, 112)", "(1, 16, 64, 112)", "(1, 1)", "(1, 1)", 3670016),
        ("Stage 1 (Fine)", "refinement_head.h2_fusion_proj.block.0", "Conv2d", "(1, 80, 64, 112)", "(1, 32, 64, 112)", "(1, 1)", "(1, 1)", 18350080),
        ("Stage 1 (Fine)", "refinement_head.h2_dsconv.depthwise", "Conv2d", "(1, 32, 64, 112)", "(1, 32, 64, 112)", "(3, 3)", "(1, 1)", 2064384),
        ("Stage 1 (Fine)", "refinement_head.h2_dsconv.pointwise", "Conv2d", "(1, 32, 64, 112)", "(1, 32, 64, 112)", "(1, 1)", "(1, 1)", 7340032),
        
        ("Stage 1 (Fine)", "refinement_head.out_conv.block.0", "Conv2d", "(1, 32, 128, 224)", "(1, 24, 128, 224)", "(3, 3)", "(1, 1)", 197525504),
        ("Stage 1 (Fine)", "refinement_head.pred_conv", "Conv2d", "(1, 24, 128, 224)", "(1, 1, 128, 224)", "(1, 1)", "(1, 1)", 688128)
    ]
    return orig_backbone_and_ppm + coarse_head_layers + refine_head_layers

def make_ts_layers():
    # Fast-SCNN Two-Stage (2-stage, resolution_hierarchy = True)
    # Stage 0: input size 112x64 (downsampled by 0.5x). Executes backbone + PPM + CoarseHead.
    # Stage 1: input size 224x128. Executes backbone + PPM + RefinementHead.
    
    stage0_layers = []
    # Stage 0: 0.25x MACs of the original backbone + ppm + coarse head
    for row in make_orig_layers()[:-4] + [
        ("Stage 0 (Coarse)", "coarse_head.dsconv.depthwise", "Conv2d", "(1, 128, 16, 28)", "(1, 128, 16, 28)", "(3, 3)", "(1, 1)", 516096),
        ("Stage 0 (Coarse)", "coarse_head.dsconv.pointwise", "Conv2d", "(1, 128, 16, 28)", "(1, 64, 16, 28)", "(1, 1)", "(1, 1)", 3686400),
        ("Stage 0 (Coarse)", "coarse_head.conv", "Conv2d", "(1, 64, 16, 28)", "(1, 1, 16, 28)", "(1, 1)", "(1, 1)", 28672)
    ]:
        # Divide H, W of shapes by 2, MACs by 4
        def halven_shape(shape_str):
            # Parse tuple shape, e.g. (1, 3, 128, 224) -> (1, 3, 64, 112)
            parts = [int(p.strip()) for p in shape_str.strip("()").split(",")]
            parts[-2] //= 2
            parts[-1] //= 2
            return str(tuple(parts))
        
        in_s = halven_shape(row[3])
        out_s = halven_shape(row[4])
        stage0_layers.append(
            ("Stage 0 (Coarse)", row[1], row[2], in_s, out_s, row[5], row[6], row[7] // 4)
        )
        
    # Stage 1: 1.0x resolution. Runs backbone + ppm + refinement head (with 3 extra channels in h8_proj)
    stage1_backbone_and_ppm = []
    for row in make_orig_layers()[:-4]:
        stage1_backbone_and_ppm.append(
            ("Stage 1 (Fine)", row[1], row[2], row[3], row[4], row[5], row[6], row[7])
        )
        
    refine_head_layers = [
        # h8_proj input has 128 + 3 = 131 channels (alpha, uncertainty, boundary)
        ("Stage 1 (Fine)", "refinement_head.h8_proj.block.0", "Conv2d", "(1, 131, 16, 28)", "(1, 96, 16, 28)", "(1, 1)", "(1, 1)", 5633280),
        ("Stage 1 (Fine)", "refinement_head.h8_dsconv.depthwise", "Conv2d", "(1, 96, 16, 28)", "(1, 96, 16, 28)", "(3, 3)", "(1, 1)", 387072),
        ("Stage 1 (Fine)", "refinement_head.h8_dsconv.pointwise", "Conv2d", "(1, 96, 16, 28)", "(1, 96, 16, 28)", "(1, 1)", "(1, 1)", 4128768),
        
        ("Stage 1 (Fine)", "refinement_head.h4_skip_proj.block.0", "Conv2d", "(1, 48, 32, 56)", "(1, 32, 32, 56)", "(1, 1)", "(1, 1)", 2752512),
        ("Stage 1 (Fine)", "refinement_head.h4_fusion_proj.block.0", "Conv2d", "(1, 128, 32, 56)", "(1, 64, 32, 56)", "(1, 1)", "(1, 1)", 14680064),
        ("Stage 1 (Fine)", "refinement_head.h4_dsconv.depthwise", "Conv2d", "(1, 64, 32, 56)", "(1, 64, 32, 56)", "(3, 3)", "(1, 1)", 1032192),
        ("Stage 1 (Fine)", "refinement_head.h4_dsconv.pointwise", "Conv2d", "(1, 64, 32, 56)", "(1, 64, 32, 56)", "(1, 1)", "(1, 1)", 7340032),
        
        ("Stage 1 (Fine)", "refinement_head.h2_skip_proj.block.0", "Conv2d", "(1, 32, 64, 112)", "(1, 16, 64, 112)", "(1, 1)", "(1, 1)", 3670016),
        ("Stage 1 (Fine)", "refinement_head.h2_fusion_proj.block.0", "Conv2d", "(1, 80, 64, 112)", "(1, 32, 64, 112)", "(1, 1)", "(1, 1)", 18350080),
        ("Stage 1 (Fine)", "refinement_head.h2_dsconv.depthwise", "Conv2d", "(1, 32, 64, 112)", "(1, 32, 64, 112)", "(3, 3)", "(1, 1)", 2064384),
        ("Stage 1 (Fine)", "refinement_head.h2_dsconv.pointwise", "Conv2d", "(1, 32, 64, 112)", "(1, 32, 64, 112)", "(1, 1)", "(1, 1)", 7340032),
        
        ("Stage 1 (Fine)", "refinement_head.out_conv.block.0", "Conv2d", "(1, 32, 128, 224)", "(1, 24, 128, 224)", "(3, 3)", "(1, 1)", 197525504),
        ("Stage 1 (Fine)", "refinement_head.pred_conv", "Conv2d", "(1, 24, 128, 224)", "(1, 1, 128, 224)", "(1, 1)", "(1, 1)", 688128)
    ]
    
    return stage0_layers + stage1_backbone_and_ppm + refine_head_layers

def to_df(layer_tuples):
    columns = ["Stage", "Layer Name", "Layer Type", "Input Shape", "Output Shape", "Kernel Size", "Stride", "MACs"]
    df = pd.DataFrame(layer_tuples, columns=columns)
    
    # Append Total Row
    total_macs = df["MACs"].sum()
    total_row = pd.DataFrame([{
        "Stage": "Total", "Layer Name": "", "Layer Type": "", "Input Shape": "", "Output Shape": "", "Kernel Size": "", "Stride": "", "MACs": total_macs
    }])
    return pd.concat([df, total_row], ignore_index=True)

def style_excel(file_path):
    wb = openpyxl.load_workbook(file_path)
    
    # Styles
    font_family = "Inter"
    title_font = Font(name=font_family, size=14, bold=True, color="1F4E79")
    header_font = Font(name=font_family, size=11, bold=True, color="FFFFFF")
    data_font = Font(name=font_family, size=10, color="000000")
    total_font = Font(name=font_family, size=11, bold=True, color="1F4E79")
    
    header_fill = PatternFill(start_color="1F4E79", end_color="1F4E79", fill_type="solid")
    zebra_fill = PatternFill(start_color="F2F7FA", end_color="F2F7FA", fill_type="solid")
    total_fill = PatternFill(start_color="E6EEF4", end_color="E6EEF4", fill_type="solid")
    
    thin_border_side = Side(border_style="thin", color="D3D3D3")
    data_border = Border(left=thin_border_side, right=thin_border_side, top=thin_border_side, bottom=thin_border_side)
    
    double_bottom_side = Side(border_style="double", color="1F4E79")
    top_thin_side = Side(border_style="thin", color="1F4E79")
    total_border = Border(top=top_thin_side, bottom=double_bottom_side)
    
    align_center = Alignment(horizontal="center", vertical="center")
    align_left = Alignment(horizontal="left", vertical="center")
    align_right = Alignment(horizontal="right", vertical="center")
    
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        
        # Insert title row
        ws.insert_rows(1, 2)
        ws["A1"] = f"Model MACs Analysis - {sheet_name.upper().replace('_', ' ')} (Resolution: 224x128)"
        ws["A1"].font = title_font
        ws.row_dimensions[1].height = 25
        
        # Format headers
        header_row = 3
        ws.row_dimensions[header_row].height = 24
        for col in range(1, 9):
            cell = ws.cell(row=header_row, column=col)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = align_center
            
        # Format data rows
        max_row = ws.max_row
        for r in range(4, max_row):
            ws.row_dimensions[r].height = 19
            # Zebra striping
            row_fill = zebra_fill if r % 2 == 0 else None
            
            for c in range(1, 9):
                cell = ws.cell(row=r, column=c)
                cell.font = data_font
                cell.border = data_border
                if row_fill:
                    cell.fill = row_fill
                
                # Alignments
                if c in [1, 3]: # Stage, Layer Type
                    cell.alignment = align_center
                elif c in [4, 5, 6, 7]: # Shapes, Kernel, Stride
                    cell.alignment = align_center
                elif c == 8: # MACs
                    cell.alignment = align_right
                    cell.number_format = '#,##0'
                else: # Layer Name
                    cell.alignment = align_left
                    
        # Format Total row
        ws.row_dimensions[max_row].height = 22
        for c in range(1, 9):
            cell = ws.cell(row=max_row, column=c)
            cell.font = total_font
            cell.fill = total_fill
            cell.border = total_border
            if c == 1:
                cell.alignment = align_center
            elif c == 8:
                cell.alignment = align_right
                cell.number_format = '#,##0'
                
        # Auto-adjust column widths
        for col in ws.columns:
            max_len = 0
            col_letter = openpyxl.utils.get_column_letter(col[0].column)
            for cell in col:
                # ignore title row for width calculation
                if cell.row == 1:
                    continue
                val = str(cell.value or "")
                if cell.column == 8 and isinstance(cell.value, (int, float)):
                    # estimate formatted length
                    val = f"{cell.value:,.0f}"
                max_len = max(max_len, len(val))
            ws.column_dimensions[col_letter].width = max(max_len + 3, 10)
            
    wb.save(file_path)
    wb.close()

def main():
    print("Generating MACs DataFrames...")
    df_orig = to_df(make_orig_layers())
    df_dh = to_df(make_dh_layers())
    df_ts = to_df(make_ts_layers())
    
    excel_file = "fast_scnn_macs_224x128.xlsx"
    print(f"Writing to Excel file: {excel_file}")
    with pd.ExcelWriter(excel_file, engine="openpyxl") as writer:
        df_orig.to_excel(writer, sheet_name="original_fast_scnn", index=False)
        df_dh.to_excel(writer, sheet_name="fast_scnn_dual_head", index=False)
        df_ts.to_excel(writer, sheet_name="fast_scnn_two_stage", index=False)
        
    print("Styling the Excel file...")
    style_excel(excel_file)
    print("Finished successfully! Output file: fast_scnn_macs_224x128.xlsx")
    
    # Print summaries
    print("\nSummary of Total MACs (224x128 Input):")
    print(f"1. Original Fast-SCNN          : {df_orig['MACs'].iloc[-1]:,} MACs")
    print(f"2. Fast-SCNN Dual Head (1-stage): {df_dh['MACs'].iloc[-1]:,} MACs")
    print(f"3. Fast-SCNN Two-Stage (2-stage): {df_ts['MACs'].iloc[-1]:,} MACs")

if __name__ == "__main__":
    main()
