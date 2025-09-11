# -*- coding: utf-8 -*-
# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossModalFusion(nn.Module):
    """
    跨模态融合模块
    用于融合RGB特征和深度特征
    """
    
    def __init__(
        self,
        rgb_dim=1024,      # RGB特征维度
        depth_dim=1024,    # 深度特征维度
        hidden_dim=1024,   # 隐藏层维度
        output_dim=1024,   # 输出特征维度
        num_heads=16,      # 注意力头数
        num_layers=4,      # Transformer层数
        dropout=0.1,       # dropout率
        fusion_type='cross_attention',  # 融合类型
        **kwargs
    ):
        super().__init__()
        
        self.rgb_dim = rgb_dim
        self.depth_dim = depth_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.fusion_type = fusion_type
        
        # 输入投影层
        self.rgb_proj = nn.Linear(rgb_dim, hidden_dim)
        self.depth_proj = nn.Linear(depth_dim, hidden_dim)
        
        if fusion_type == 'cross_attention':
            self._build_cross_attention_fusion()
        elif fusion_type == 'concat_mlp':
            self._build_concat_mlp_fusion()
        elif fusion_type == 'gated_fusion':
            self._build_gated_fusion()
        elif fusion_type == 'adaptive_fusion':
            self._build_adaptive_fusion()
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")
        
        # 输出投影层
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _build_cross_attention_fusion(self):
        """构建交叉注意力融合模块"""
        # RGB到深度的交叉注意力
        self.rgb_to_depth_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 深度到RGB的交叉注意力
        self.depth_to_rgb_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # 自注意力层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=self.num_layers
        )
        
        # 层归一化
        self.rgb_norm = nn.LayerNorm(self.hidden_dim)
        self.depth_norm = nn.LayerNorm(self.hidden_dim)
        self.fusion_norm = nn.LayerNorm(self.hidden_dim)
    
    def _build_concat_mlp_fusion(self):
        """构建拼接+MLP融合模块"""
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
    
    def _build_gated_fusion(self):
        """构建门控融合模块"""
        # 门控机制
        self.gate_rgb = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid()
        )
        self.gate_depth = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Sigmoid()
        )
        
        # 融合MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim)
        )
    
    def _build_adaptive_fusion(self):
        """构建自适应融合模块"""
        # 注意力权重计算
        self.attention_weights = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # 特征变换
        self.rgb_transform = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )
        self.depth_transform = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.hidden_dim)
        )
    
    def _init_weights(self):
        """初始化模型权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
    
    def forward(self, rgb_features, depth_features, mask=None, **kwargs):
        """
        前向传播
        
        Args:
            rgb_features: RGB特征 [B, N, rgb_dim]
            depth_features: 深度特征 [B, N, depth_dim]
            mask: 可选的注意力mask
            
        Returns:
            fused_features: 融合后的特征 [B, N, output_dim]
        """
        # 输入投影
        rgb_proj = self.rgb_proj(rgb_features)    # [B, N, hidden_dim]
        depth_proj = self.depth_proj(depth_features)  # [B, N, hidden_dim]
        
        if self.fusion_type == 'cross_attention':
            fused = self._cross_attention_fusion(rgb_proj, depth_proj, mask)
        elif self.fusion_type == 'concat_mlp':
            fused = self._concat_mlp_fusion(rgb_proj, depth_proj)
        elif self.fusion_type == 'gated_fusion':
            fused = self._gated_fusion(rgb_proj, depth_proj)
        elif self.fusion_type == 'adaptive_fusion':
            fused = self._adaptive_fusion(rgb_proj, depth_proj)
        
        # 输出投影和归一化
        output = self.output_proj(fused)
        output = self.output_norm(output)
        
        return output
    
    def _cross_attention_fusion(self, rgb_features, depth_features, mask=None):
        """交叉注意力融合"""
        # RGB特征通过深度特征增强
        rgb_enhanced, _ = self.rgb_to_depth_attention(
            query=rgb_features,
            key=depth_features,
            value=depth_features,
            attn_mask=mask
        )
        rgb_enhanced = self.rgb_norm(rgb_enhanced + rgb_features)
        
        # 深度特征通过RGB特征增强
        depth_enhanced, _ = self.depth_to_rgb_attention(
            query=depth_features,
            key=rgb_features,
            value=rgb_features,
            attn_mask=mask
        )
        depth_enhanced = self.depth_norm(depth_enhanced + depth_features)
        
        # 拼接并通过Transformer进一步融合
        combined = torch.cat([rgb_enhanced, depth_enhanced], dim=1)  # [B, 2N, hidden_dim]
        fused = self.fusion_transformer(combined)
        
        # 取前N个token作为输出（或者可以用其他策略）
        N = rgb_features.shape[1]
        fused = fused[:, :N, :]  # [B, N, hidden_dim]
        fused = self.fusion_norm(fused)
        
        return fused
    
    def _concat_mlp_fusion(self, rgb_features, depth_features):
        """拼接+MLP融合"""
        # 拼接特征
        combined = torch.cat([rgb_features, depth_features], dim=-1)  # [B, N, 2*hidden_dim]
        
        # 通过MLP融合
        fused = self.fusion_mlp(combined)  # [B, N, hidden_dim]
        
        return fused
    
    def _gated_fusion(self, rgb_features, depth_features):
        """门控融合"""
        # 计算门控权重
        combined = torch.cat([rgb_features, depth_features], dim=-1)
        gate_rgb = self.gate_rgb(combined)    # [B, N, hidden_dim]
        gate_depth = self.gate_depth(combined)  # [B, N, hidden_dim]
        
        # 门控融合
        gated_rgb = gate_rgb * rgb_features
        gated_depth = gate_depth * depth_features
        fused = gated_rgb + gated_depth
        
        # 通过MLP进一步处理
        fused = self.fusion_mlp(fused)
        
        return fused
    
    def _adaptive_fusion(self, rgb_features, depth_features):
        """自适应融合"""
        B, N, D = rgb_features.shape
        
        # 特征变换
        rgb_transformed = self.rgb_transform(rgb_features)
        depth_transformed = self.depth_transform(depth_features)
        
        # 计算自适应权重
        combined = torch.cat([rgb_transformed, depth_transformed], dim=-1)
        weights = self.attention_weights(combined)  # [B, N, 2]
        
        # 加权融合
        rgb_weight = weights[:, :, 0:1]    # [B, N, 1]
        depth_weight = weights[:, :, 1:2]  # [B, N, 1]
        
        fused = rgb_weight * rgb_transformed + depth_weight * depth_transformed
        
        return fused
    
    def unconditional_embedding(self, batch_size, num_patches, **kwargs):
        """生成无条件嵌入"""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        zero_embedding = torch.zeros(
            batch_size,
            num_patches,
            self.output_dim,
            device=device,
            dtype=dtype
        )
        
        return zero_embedding


class SimpleCrossModalFusion(nn.Module):
    """
    简化版跨模态融合模块
    使用更少的参数和计算量
    """
    
    def __init__(
        self,
        rgb_dim=1024,
        depth_dim=1024,
        output_dim=1024,
        **kwargs
    ):
        super().__init__()
        
        self.output_dim = output_dim
        
        # 简单的线性融合
        self.rgb_proj = nn.Linear(rgb_dim, output_dim)
        self.depth_proj = nn.Linear(depth_dim, output_dim)
        
        # 融合权重
        self.fusion_weight = nn.Parameter(torch.tensor([0.7, 0.3]))  # RGB权重更高
        
        # 输出层
        self.output_norm = nn.LayerNorm(output_dim)
        
    def forward(self, rgb_features, depth_features, **kwargs):
        """简单的加权融合"""
        # 投影到相同维度
        rgb_proj = self.rgb_proj(rgb_features)
        depth_proj = self.depth_proj(depth_features)
        
        # 归一化融合权重
        weights = F.softmax(self.fusion_weight, dim=0)
        
        # 加权融合
        fused = weights[0] * rgb_proj + weights[1] * depth_proj
        
        # 输出归一化
        output = self.output_norm(fused)
        
        return output
    
    def unconditional_embedding(self, batch_size, num_patches, **kwargs):
        """生成无条件嵌入"""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        zero_embedding = torch.zeros(
            batch_size,
            num_patches,
            self.output_dim,
            device=device,
            dtype=dtype
        )
        
        return zero_embedding