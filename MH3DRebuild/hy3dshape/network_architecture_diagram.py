#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hunyuan3D + 深度图特征提取 + LoRA微调 网络架构图生成器
Scientific Style Neural Network Architecture Visualization
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch, Rectangle
import numpy as np

# Set scientific style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'cm'

def create_network_diagram():
    """创建Hunyuan3D + 深度LoRA的完整网络架构图"""
    
    # Create figure with higher DPI for publication quality
    fig = plt.figure(figsize=(16, 12), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Color scheme - scientific and professional
    colors = {
        'input': '#E8F4FD',      # Light blue for inputs
        'encoder': '#D4E6F1',     # Medium blue for encoders  
        'feature': '#A9CCE3',     # Blue for features
        'attention': '#7FB3D3',   # Darker blue for attention
        'dit': '#5DADE2',         # DiT blue
        'lora': '#FF6B6B',        # Red for LoRA modifications
        'decoder': '#85C1E9',     # Light blue for decoder
        'output': '#AED6F1',      # Very light blue for output
        'controlnet': '#58D68D',  # Green for ControlNet
        'flow': '#F7DC6F'         # Yellow for flow matching
    }
    
    # === Title ===
    ax.text(8, 11.5, 'Hunyuan3D with Depth-Conditioned LoRA Fine-tuning Architecture', 
            ha='center', va='center', fontsize=16, fontweight='bold')
    
    # === Input Layer ===
    # RGB Image Input
    rgb_box = FancyBboxPatch((0.5, 9.5), 2, 1, 
                            boxstyle="round,pad=0.1", 
                            facecolor=colors['input'], 
                            edgecolor='black', linewidth=1.5)
    ax.add_patch(rgb_box)
    ax.text(1.5, 10, 'RGB Image\n(B, 3, 518, 518)', ha='center', va='center', fontweight='bold')
    
    # Depth Map Input  
    depth_box = FancyBboxPatch((0.5, 7.8), 2, 1,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['controlnet'],
                              edgecolor='black', linewidth=1.5)
    ax.add_patch(depth_box)
    ax.text(1.5, 8.3, 'Depth Map\n(B, 1, H, W)', ha='center', va='center', fontweight='bold')
    
    # Point Cloud Input
    pc_box = FancyBboxPatch((0.5, 6.1), 2, 1,
                           boxstyle="round,pad=0.1",
                           facecolor=colors['input'],
                           edgecolor='black', linewidth=1.5)
    ax.add_patch(pc_box)
    ax.text(1.5, 6.6, 'Point Cloud\n(B, 81920, 6)', ha='center', va='center', fontweight='bold')
    
    # === Feature Extraction Layer ===
    # DINO-v2 Encoder
    dino_box = FancyBboxPatch((3.5, 9.5), 2.5, 1,
                             boxstyle="round,pad=0.1", 
                             facecolor=colors['encoder'],
                             edgecolor='black', linewidth=1.5)
    ax.add_patch(dino_box)
    ax.text(4.75, 10, 'DINO-v2 Large\nImage Encoder\n1024D Features', 
            ha='center', va='center', fontweight='bold')
    
    # Depth ControlNet
    controlnet_box = FancyBboxPatch((3.5, 7.3), 2.5, 1.4,
                                   boxstyle="round,pad=0.1",
                                   facecolor=colors['controlnet'],
                                   edgecolor='black', linewidth=2)
    ax.add_patch(controlnet_box)
    ax.text(4.75, 8, 'Depth ControlNet\n• Conv2d(1→64→128→256)\n• AdaptiveAvgPool2d\n• Linear(256→768)', 
            ha='center', va='center', fontweight='bold', fontsize=9)
    
    # ShapeVAE Encoder
    vae_enc_box = FancyBboxPatch((3.5, 5.6), 2.5, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['encoder'],
                                edgecolor='black', linewidth=1.5)
    ax.add_patch(vae_enc_box)
    ax.text(4.75, 6.1, 'ShapeVAE\nEncoder\n4096D Latents', 
            ha='center', va='center', fontweight='bold')
    
    # === Condition Processing ===
    # Main Condition
    main_cond_box = FancyBboxPatch((7, 9.5), 2, 1,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['feature'],
                                  edgecolor='black', linewidth=1.5)
    ax.add_patch(main_cond_box)
    ax.text(8, 10, 'Main Context\n(B, L, 1024)', ha='center', va='center', fontweight='bold')
    
    # Additional Condition Processing
    add_cond_box = FancyBboxPatch((7, 7.8), 2, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['controlnet'],
                                 edgecolor='black', linewidth=1.5)
    ax.add_patch(add_cond_box)
    ax.text(8, 8.3, 'Additional Proj\nLinear(768→4H→1024)', ha='center', va='center', fontweight='bold')
    
    # === HunYuanDiT Main Model ===
    # DiT Block representation
    dit_main_box = FancyBboxPatch((10.5, 7.5), 4, 3,
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['dit'],
                                 edgecolor='black', linewidth=2)
    ax.add_patch(dit_main_box)
    
    # DiT internal structure
    ax.text(12.5, 9.8, 'HunYuanDiT (24 Layers)', ha='center', va='center', 
            fontweight='bold', fontsize=12)
    
    # Individual DiT blocks
    for i, y_pos in enumerate([9.4, 8.9, 8.4, 7.9]):
        if i < 3:
            block_box = Rectangle((10.7, y_pos-0.15), 3.6, 0.25, 
                                facecolor=colors['attention'], 
                                edgecolor='black', linewidth=0.8)
            ax.add_patch(block_box)
            if i == 0:
                ax.text(12.5, y_pos, f'DiT Block {i+1}: Self-Attn + Cross-Attn + FFN', 
                       ha='center', va='center', fontsize=8)
            else:
                ax.text(12.5, y_pos, f'DiT Block {i+1}: ... (22 more layers)', 
                       ha='center', va='center', fontsize=8)
        else:
            ax.text(12.5, y_pos, '⋮', ha='center', va='center', fontsize=16)
    
    # === LoRA Annotations ===
    # LoRA indicators on DiT blocks
    lora_indicators = []
    for y_pos in [9.4, 8.9, 8.4]:
        for x_pos in [11.0, 11.8, 12.6, 13.4]:
            lora_dot = plt.Circle((x_pos, y_pos), 0.05, 
                                color=colors['lora'], zorder=10)
            ax.add_patch(lora_dot)
    
    # LoRA Legend
    lora_legend_box = FancyBboxPatch((10.5, 6.5), 4, 0.8,
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['lora'],
                                    edgecolor='black', linewidth=1.5, alpha=0.3)
    ax.add_patch(lora_legend_box)
    ax.text(12.5, 6.9, 'LoRA Fine-tuning (Rank=8)', ha='center', va='center', fontweight='bold')
    ax.text(12.5, 6.6, 'Target: [to_q, to_k, to_v, to_out.0]', ha='center', va='center', fontsize=9)
    
    # === Flow Matching & Scheduler ===
    flow_box = FancyBboxPatch((10.5, 5.0), 4, 1,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['flow'],
                             edgecolor='black', linewidth=1.5)
    ax.add_patch(flow_box)
    ax.text(12.5, 5.5, 'Flow Matching Scheduler\nEuler ODE Sampler (50 steps)', 
            ha='center', va='center', fontweight='bold')
    
    # === Decoder ===
    vae_dec_box = FancyBboxPatch((10.5, 3.5), 4, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['decoder'],
                                edgecolor='black', linewidth=1.5)
    ax.add_patch(vae_dec_box)
    ax.text(12.5, 4, 'ShapeVAE Decoder\nLatents → 3D Mesh', 
            ha='center', va='center', fontweight='bold')
    
    # === Output ===
    output_box = FancyBboxPatch((10.5, 2.0), 4, 1,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['output'],
                               edgecolor='black', linewidth=1.5)
    ax.add_patch(output_box)
    ax.text(12.5, 2.5, '3D Mesh Output\nTriMesh Format', 
            ha='center', va='center', fontweight='bold')
    
    # === Arrows and Data Flow ===
    arrow_props = dict(arrowstyle='->', connectionstyle='arc3', 
                      linewidth=2, color='black')
    
    # RGB → DINO
    ax.annotate('', xy=(3.5, 10), xytext=(2.5, 10), arrowprops=arrow_props)
    
    # Depth → ControlNet  
    ax.annotate('', xy=(3.5, 8), xytext=(2.5, 8.3), arrowprops=arrow_props)
    
    # Point Cloud → VAE Encoder
    ax.annotate('', xy=(3.5, 6.1), xytext=(2.5, 6.6), arrowprops=arrow_props)
    
    # DINO → Main Context
    ax.annotate('', xy=(7, 10), xytext=(6, 10), arrowprops=arrow_props)
    
    # ControlNet → Additional Proj
    ax.annotate('', xy=(7, 8.3), xytext=(6, 8), arrowprops=arrow_props)
    
    # Context → DiT
    ax.annotate('', xy=(10.5, 9.5), xytext=(9, 10), 
               arrowprops=dict(arrowstyle='->', linewidth=2, color='blue'))
    ax.annotate('', xy=(10.5, 8.5), xytext=(9, 8.3), 
               arrowprops=dict(arrowstyle='->', linewidth=2, color='green'))
    
    # VAE Encoder → DiT
    ax.annotate('', xy=(10.5, 7.8), xytext=(6, 6.1), 
               arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', 
                              linewidth=2, color='purple'))
    
    # DiT → Flow Matching
    ax.annotate('', xy=(12.5, 6.0), xytext=(12.5, 7.5), arrowprops=arrow_props)
    
    # Flow Matching → VAE Decoder
    ax.annotate('', xy=(12.5, 4.5), xytext=(12.5, 5.0), arrowprops=arrow_props)
    
    # VAE Decoder → Output
    ax.annotate('', xy=(12.5, 3.0), xytext=(12.5, 3.5), arrowprops=arrow_props)
    
    # === Technical Annotations ===
    # Training strategy annotation
    training_box = FancyBboxPatch((0.5, 1.0), 6, 2,
                                 boxstyle="round,pad=0.1",
                                 facecolor='lightgray',
                                 edgecolor='black', linewidth=1, alpha=0.7)
    ax.add_patch(training_box)
    ax.text(3.5, 2.3, 'Training Strategy', ha='center', va='center', fontweight='bold', fontsize=11)
    ax.text(3.5, 1.9, '• Freeze: DINO-v2, ShapeVAE', ha='center', va='center', fontsize=9)
    ax.text(3.5, 1.6, '• Train: LoRA adapters (Rank=8)', ha='center', va='center', fontsize=9)  
    ax.text(3.5, 1.3, '• Train: DepthControlNet (full parameters)', ha='center', va='center', fontsize=9)
    
    # Architecture details
    arch_box = FancyBboxPatch((7.5, 1.0), 6, 2,
                             boxstyle="round,pad=0.1",
                             facecolor='lightgray',
                             edgecolor='black', linewidth=1, alpha=0.7)
    ax.add_patch(arch_box)
    ax.text(10.5, 2.3, 'Architecture Details', ha='center', va='center', fontweight='bold', fontsize=11)
    ax.text(10.5, 1.9, '• Flow Matching: Linear path, Velocity prediction', ha='center', va='center', fontsize=9)
    ax.text(10.5, 1.6, '• Attention: Self + Cross with LoRA adaptation', ha='center', va='center', fontsize=9)
    ax.text(10.5, 1.3, '• Depth injection: Additional condition pathway', ha='center', va='center', fontsize=9)
    
    # === Legend for Colors ===
    legend_elements = [
        patches.Patch(color=colors['input'], label='Input Data'),
        patches.Patch(color=colors['encoder'], label='Encoders (Frozen)'),
        patches.Patch(color=colors['controlnet'], label='Depth Processing'),
        patches.Patch(color=colors['dit'], label='DiT Backbone'),
        patches.Patch(color=colors['lora'], label='LoRA Adaptations'),
        patches.Patch(color=colors['flow'], label='Flow Matching'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
             frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    return fig

def save_diagram():
    """保存网络架构图"""
    fig = create_network_diagram()
    
    # Save in multiple formats
    fig.savefig('/mnt/e/vscode/git_project/MV3DHuman/hy3dshape/hunyuan3d_depth_lora_architecture.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig('/mnt/e/vscode/git_project/MV3DHuman/hy3dshape/hunyuan3d_depth_lora_architecture.pdf', 
                dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig('/mnt/e/vscode/git_project/MV3DHuman/hy3dshape/hunyuan3d_depth_lora_architecture.svg', 
                dpi=300, bbox_inches='tight', facecolor='white')
    
    print("✅ 网络架构图已保存:")
    print("   - PNG格式: hunyuan3d_depth_lora_architecture.png")  
    print("   - PDF格式: hunyuan3d_depth_lora_architecture.pdf")
    print("   - SVG格式: hunyuan3d_depth_lora_architecture.svg")
    
    # Close the figure to free memory
    plt.close(fig)

if __name__ == "__main__":
    save_diagram()
