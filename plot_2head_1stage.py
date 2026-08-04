import sys
import os

# 本腳本旨在放置於 PlotNeuralNet 的 pyexamples/ 目錄下執行
# 例如: python pyexamples/plot_2head_1stage.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pycore.tikzeng import *
from pycore.blocks import *

def my_Conv(name, s_filer=256, n_filer=64, offset="(0,0,0)", to="(0,0,0)", width=1, height=40, depth=40, caption=" ", fill="\\ConvColor"):
    return r"""
\pic[shift={"""+ offset +"""}] at """+ to +""" 
    {Box={
        name=""" + name +""",
        caption=""" + caption +""",
        xlabel={{" """ + str(n_filer) +""" ", }},
        zlabel=""" + str(s_filer) +""",
        fill=""" + fill +""",
        height=""" + str(height) +""",
        width=""" + str(width) +""",
        depth=""" + str(depth) +"""
        }
    };
"""

arch = [
    to_head('..'),
    to_cor(),
    to_begin(),
    
    # ────────────────────────────────────────────────────────────────────────
    # 0. INPUT IMAGE
    # ────────────────────────────────────────────────────────────────────────
    # 可放一張輸入鳥或鸚鵡的圖片，此處預設路徑為 ../pyexamples/input.png
    to_input('../pyexamples/input.png', width=8, height=8, name="temp_input"),
    
    # ────────────────────────────────────────────────────────────────────────
    # 1. LEARNING TO DOWNSAMPLE (LtD) - 共享主幹起手
    # ────────────────────────────────────────────────────────────────────────
    # Conv2D: 128x224x3 -> 64x112x32 (Red)
    my_Conv("ltd_conv1", 64, 32, offset="(1.5,0,0)", to="(0,0,0)", width=2, height=16, depth=28, caption="Conv2D (s2)"),
    
    # DSConv1 (DWConv + PWConv): 64x112x32 -> 32x56x48 (Blue)
    my_Conv("ltd_ds1", 32, 48, offset="(1.5,0,0)", to="(ltd_conv1-east)", width=3, height=12, depth=20, caption="DSConv (s2)"),
    
    # DSConv2 (DWConv + PWConv): 32x56x48 -> 16x28x64 (Blue)
    my_Conv("ltd_ds2", 16, 64, offset="(1.5,0,0)", to="(ltd_ds1-east)", width=4, height=8, depth=14, caption="DSConv (s2)"),
    
    # 分支連接線：從 LtD 尾端分出兩路 (GFE 與 FFM)
    to_connection("ltd_ds2", "gfe_b0"),
    to_connection("ltd_ds2", "ffm_high_res_proj"),
 
    # ────────────────────────────────────────────────────────────────────────
    # 2. GLOBAL FEATURE EXTRACTOR (GFE) - 深層語義分支
    # ────────────────────────────────────────────────────────────────────────
    # 8x Bottlenecks (Green blocks): 16x28x64 -> 4x7x128
    # 我們以一組緊密的綠色 Box 陣列代表 8 個 Bottlenecks
    my_Conv("gfe_b0", 16, 64, offset="(1.8,2.5,0)", to="(ltd_ds2-east)", width=1.5, height=8, depth=14, caption="Bottlenecks", fill="\\color{green!70!black}"),
    my_Conv("gfe_b1", 16, 64, offset="(0.2,0,0)", to="(gfe_b0-east)", width=1.5, height=8, depth=14, fill="\\color{green!70!black}"),
    my_Conv("gfe_b2", 16, 64, offset="(0.2,0,0)", to="(gfe_b1-east)", width=1.5, height=8, depth=14, fill="\\color{green!70!black}"),
    my_Conv("gfe_b3", 8, 96, offset="(0.3,0,0)", to="(gfe_b2-east)", width=2.0, height=6, depth=10, fill="\\color{green!70!black}"),
    my_Conv("gfe_b4", 4, 96, offset="(0.2,0,0)", to="(gfe_b3-east)", width=2.0, height=4, depth=7, fill="\\color{green!70!black}"),
    my_Conv("gfe_b5", 4, 96, offset="(0.2,0,0)", to="(gfe_b4-east)", width=2.0, height=4, depth=7, fill="\\color{green!70!black}"),
    my_Conv("gfe_b6", 4, 128, offset="(0.2,0,0)", to="(gfe_b5-east)", width=2.5, height=4, depth=7, fill="\\color{green!70!black}"),
    my_Conv("gfe_b7", 4, 128, offset="(0.2,0,0)", to="(gfe_b6-east)", width=2.5, height=4, depth=7, fill="\\color{green!70!black}"),
    my_Conv("gfe_b8", 4, 128, offset="(0.2,0,0)", to="(gfe_b7-east)", width=2.5, height=4, depth=7, fill="\\color{green!70!black}"),
    
    # Pyramid Pooling Module (PPM) (Purple)
    my_Conv("gfe_ppm", 4, 128, offset="(1.2,0,0)", to="(gfe_b8-east)", width=4, height=4, depth=7, caption="PPM", fill="\\color{purple!70}"),
    
    # ────────────────────────────────────────────────────────────────────────
    # 3. FEATURE FUSION MODULE (FFM) - 特徵融合
    # ────────────────────────────────────────────────────────────────────────
    # 來自 GFE 的 Upsample (Yellow): 4x7x128 -> 16x28x128
    my_Conv("ffm_low_res_up", 16, 128, offset="(2.0,0,0)", to="(gfe_ppm-east)", width=4, height=8, depth=14, caption="Upsample 4x", fill="\\color{yellow!80!orange}"),
    # FFM Low-Res Conv: 16x28x128 -> 16x28x128 (DWConv + PWConv) (Grey + Red)
    my_Conv("ffm_low_res_dw", 16, 128, offset="(1.2,0,0)", to="(ffm_low_res_up-east)", width=4, height=8, depth=14, caption="DWConv", fill="\\color{gray!30}"),
    my_Conv("ffm_low_res_pw", 16, 128, offset="(0.2,0,0)", to="(ffm_low_res_dw-east)", width=4, height=8, depth=14, caption="1x1 Conv"),
    
    # 來自 LtD 的 High-Res Conv (Red): 16x28x64 -> 16x28x128
    my_Conv("ffm_high_res_proj", 16, 128, offset="(8.0,-2.5,0)", to="(ltd_ds2-east)", width=4, height=8, depth=14, caption="1x1 Conv"),
    
    # Summation Node (+)
    to_Sum("ffm_sum", offset="(1.5,-1.25,0)", to="(ffm_low_res_pw-east)"),
    to_connection("ffm_low_res_pw", "ffm_sum"),
    to_connection("ffm_high_res_proj", "ffm_sum"),
    
    # FFM Output features (Shared Features): 16x28x128 (Blue)
    my_Conv("ffm_out", 16, 128, offset="(1.5,0,0)", to="(ffm_sum-east)", width=5, height=8, depth=14, caption="FFM Output"),
    
    # ────────────────────────────────────────────────────────────────────────
    # 4. HEAD 1: COARSE HEAD (粗糙定位分支)
    # ────────────────────────────────────────────────────────────────────────
    # DSConv: 16x28x128 -> 16x28x64 (Blue)
    my_Conv("coarse_ds", 16, 64, offset="(2.0,2.5,0)", to="(ffm_out-east)", width=3, height=8, depth=14, caption="Coarse DSConv"),
    # Conv2D: 16x28x64 -> 16x28x1 (Red)
    my_Conv("coarse_conv", 16, 1, offset="(1.2,0,0)", to="(coarse_ds-east)", width=0.5, height=8, depth=14, caption="1x1 Conv"),
    # 8x Upsample (Yellow): 16x28x1 -> 128x224x1
    my_Conv("coarse_up", 128, 1, offset="(1.5,0,0)", to="(coarse_conv-east)", width=0.5, height=16, depth=28, caption="Upsample 8x", fill="\\color{yellow!80!orange}"),
    # Sigmoid Output Mask (Purple image slice)
    my_Conv("coarse_out", 128, 1, offset="(1.5,0,0)", to="(coarse_up-east)", width=0.1, height=16, depth=28, caption="Coarse Mask", fill="\\color{purple!80!black}"),
    
    to_connection("ffm_out", "coarse_ds"),
 
    # ────────────────────────────────────────────────────────────────────────
    # 5. HEAD 2: MULTISCALE REFINEMENT HEAD (精細化雙頭分支)
    # ────────────────────────────────────────────────────────────────────────
    # 為了版面美觀，我們將精細頭垂直偏移向下畫。
    # 該頭會引入 Coarse Head 的 Prompt 以及 LtD 的 Skip 特徵
    
    # H8 Proj (Concatenated 128+1 -> 96): 16x28x96 (Blue)
    my_Conv("refine_h8", 16, 96, offset="(2.0,-3.5,0)", to="(ffm_out-east)", width=4, height=8, depth=14, caption="H/8 Fusion"),
    
    # 提示與 Skip 連接線 (示意圖的特色箭頭)：
    # 1. 粗糙預測反饋至精細頭起點
    to_connection("coarse_out", "refine_h8"),
    
    # 2. H4 Stage: Upsample 2x + Skip (from ltd_ds1) -> 32x56x128
    my_Conv("refine_h4", 32, 128, offset="(2.0,0,0)", to="(refine_h8-east)", width=5, height=12, depth=20, caption="H/4 Fusion"),
    # Skip link 示意
    to_connection("ltd_ds1", "refine_h4"),
    
    # 3. H2 Stage: Upsample 2x + Skip (from ltd_conv1) -> 64x112x80
    my_Conv("refine_h2", 64, 80, offset="(2.0,0,0)", to="(refine_h4-east)", width=4, height=14, depth=24, caption="H/2 Fusion"),
    # Skip link 示意
    to_connection("ltd_conv1", "refine_h2"),
    
    # 4. Out Conv: Upsample 2x + 3x3 Conv -> 128x224x24 (Red)
    my_Conv("refine_out_conv", 128, 24, offset="(2.0,0,0)", to="(refine_h2-east)", width=2, height=16, depth=28, caption="3x3 Conv (s1)"),
    
    # 5. Pred Conv: 1x1 Conv -> 128x224x1 (Red)
    my_Conv("refine_pred", 128, 1, offset="(1.2,0,0)", to="(refine_out_conv-east)", width=0.5, height=16, depth=28, caption="1x1 Conv"),
    
    # Sigmoid Output Mask (Purple image slice) - 最終高品質精細分割圖
    my_Conv("refine_out", 128, 1, offset="(1.5,0,0)", to="(refine_pred-east)", width=0.1, height=16, depth=28, caption="Fine Mask (Sigmoid)", fill="\\color{purple!80!black}"),
    
    to_connection("ffm_out", "refine_h8"),
 
    to_end()
]

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    to_generate(arch, namefile + '.tex')
    print(f"Successfully generated TikZ LaTeX source code in: {namefile}.tex")

if __name__ == "__main__":
    main()
