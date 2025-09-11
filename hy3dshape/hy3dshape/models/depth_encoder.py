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
from torchvision import transforms
import numpy as np


class DepthEncoder(nn.Module):
    """
    深度特征提取器模块
    用于从深度图中提取特征，与RGB特征进行融合
    """
    
    def __init__(
        self,
        input_channels=1,  # 深度图通道数
        hidden_dim=1024,   # 隐藏层维度
        output_dim=1024,   # 输出特征维度
        image_size=518,    # 输入图像尺寸
        patch_size=14,     # patch大小，与DINO保持一致
        num_layers=6,      # Transformer层数
        num_heads=16,      # 注意力头数
        mlp_ratio=4.0,     # MLP扩展比例
        dropout=0.1,       # dropout率
        **kwargs
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # 深度图预处理
        self.depth_transform = transforms.Compose([
            transforms.Resize(image_size, transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(image_size),
        ])
        
        # Patch embedding层
        self.patch_embed = nn.Conv2d(
            input_channels, hidden_dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, hidden_dim)
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=int(hidden_dim * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # 输出投影层
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Layer normalization
        self.norm = nn.LayerNorm(hidden_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        # 初始化位置编码
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 初始化patch embedding
        torch.nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            torch.nn.init.zeros_(self.patch_embed.bias)
        
        # 初始化输出投影层
        torch.nn.init.trunc_normal_(self.output_proj.weight, std=0.02)
        torch.nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, depth_image, mask=None, value_range=None, **kwargs):
        """
        前向传播
        
        Args:
            depth_image: 深度图 [B, C, H, W]
            mask: 可选的mask
            value_range: 深度值范围，用于归一化
            
        Returns:
            depth_features: 深度特征 [B, num_patches+1, output_dim]
        """
        B = depth_image.shape[0]
        
        # 深度图预处理
        if value_range is not None:
            low, high = value_range
            depth_image = (depth_image - low) / (high - low)
        
        # 应用变换
        depth_image = self.depth_transform(depth_image)
        
        # Patch embedding
        x = self.patch_embed(depth_image)  # [B, hidden_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)   # [B, num_patches, hidden_dim]
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, hidden_dim]
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # Transformer编码
        x = self.transformer(x)
        
        # Layer normalization
        x = self.norm(x)
        
        # 输出投影
        x = self.output_proj(x)
        
        return x
    
    def unconditional_embedding(self, batch_size, **kwargs):
        """生成无条件嵌入（用于classifier-free guidance）"""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        zero_embedding = torch.zeros(
            batch_size,
            self.num_patches + 1,
            self.output_dim,
            device=device,
            dtype=dtype
        )
        
        return zero_embedding


class LightweightDepthEncoder(nn.Module):
    """
    轻量级深度特征提取器
    使用更少的参数和计算量
    """
    
    def __init__(
        self,
        input_channels=1,
        hidden_dim=512,
        output_dim=1024,
        image_size=518,
        **kwargs
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.image_size = image_size
        
        # 深度图预处理
        self.depth_transform = transforms.Compose([
            transforms.Resize(image_size, transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(image_size),
        ])
        
        # 简单的CNN特征提取器
        self.feature_extractor = nn.Sequential(
            # 第一层
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 第二层
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 第三层
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 第四层
            nn.Conv2d(256, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # 为了与DINO特征维度匹配，我们需要扩展到patch数量
        patch_size = 14
        self.num_patches = (image_size // patch_size) ** 2 + 1  # +1 for CLS token
        
        # 特征扩展层，将全局特征扩展到patch级别
        self.feature_expand = nn.Linear(output_dim, output_dim * self.num_patches)
    
    def forward(self, depth_image, mask=None, value_range=None, **kwargs):
        """
        前向传播
        
        Args:
            depth_image: 深度图 [B, C, H, W]
            
        Returns:
            depth_features: 深度特征 [B, num_patches, output_dim]
        """
        B = depth_image.shape[0]
        
        # 深度图预处理
        if value_range is not None:
            low, high = value_range
            depth_image = (depth_image - low) / (high - low)
        
        # 应用变换
        depth_image = self.depth_transform(depth_image)
        
        # 特征提取
        features = self.feature_extractor(depth_image)  # [B, hidden_dim, 1, 1]
        features = features.flatten(1)  # [B, hidden_dim]
        
        # 输出投影
        features = self.output_proj(features)  # [B, output_dim]
        
        # 扩展到patch级别
        expanded_features = self.feature_expand(features)  # [B, output_dim * num_patches]
        expanded_features = expanded_features.view(B, self.num_patches, self.output_dim)
        
        return expanded_features
    
    def unconditional_embedding(self, batch_size, **kwargs):
        """生成无条件嵌入"""
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        
        zero_embedding = torch.zeros(
            batch_size,
            self.num_patches,
            self.output_dim,
            device=device,
            dtype=dtype
        )
        
        return zero_embedding