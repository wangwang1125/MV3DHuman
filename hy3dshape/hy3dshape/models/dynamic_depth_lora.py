# -*- coding: utf-8 -*-
# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any


class DynamicLoRALayer(nn.Module):
    """
    动态LoRA层 - 根据深度特征动态生成LoRA权重
    
    这是真正的LoRA实现：
    output = Linear(input) + LoRA_B(LoRA_A(input)) * depth_scale
    其中LoRA_A和LoRA_B的权重由深度特征动态生成
    """
    
    def __init__(
        self,
        original_layer: nn.Linear,
        depth_feature_dim: int,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.original_layer = original_layer
        self.in_features = original_layer.in_features
        self.out_features = original_layer.out_features
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank
        
        # 冻结原始层参数
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # 权重生成器：从深度特征生成LoRA权重
        # 使用更小的隐藏层和更简单的结构
        weight_gen_hidden = max(16, depth_feature_dim // 16)
        self.weight_generator = nn.Sequential(
            nn.Linear(depth_feature_dim, weight_gen_hidden),
            nn.ReLU(),
            nn.Linear(weight_gen_hidden, lora_rank * self.in_features + lora_rank * self.out_features)
        )
        
        # 深度缩放因子生成 - 简化结构
        self.depth_scale_gen = nn.Sequential(
            nn.Linear(depth_feature_dim, 1),
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
        # 权重生成器的最后一层初始化为0，确保初始时LoRA输出为0
        nn.init.zeros_(self.weight_generator[-1].weight)
        nn.init.zeros_(self.weight_generator[-1].bias)
    
    def forward(self, x: torch.Tensor, depth_features: torch.Tensor = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [B, N, in_features]
            depth_features: 深度特征 [B, depth_feature_dim] (可选，如果为None则使用存储的特征)
            
        Returns:
            output: 输出特征 [B, N, out_features]
        """
        batch_size, seq_len = x.shape[:2]
        
        # 原始线性变换
        original_output = self.original_layer(x)  # [B, N, out_features]
        
        # 获取深度特征
        if depth_features is None:
            if hasattr(self, 'current_depth_features'):
                depth_features = self.current_depth_features
            else:
                # 如果没有深度特征，返回原始输出
                return original_output
        
        # 从深度特征生成LoRA权重
        weights = self.weight_generator(depth_features)  # [B, lora_rank * (in_features + out_features)]
        
        # 分离A和B权重
        lora_A_size = self.lora_rank * self.in_features
        lora_B_size = self.lora_rank * self.out_features
        
        lora_A_weights = weights[:, :lora_A_size]  # [B, lora_rank * in_features]
        lora_B_weights = weights[:, lora_A_size:lora_A_size + lora_B_size]  # [B, lora_rank * out_features]
        
        # 重塑权重
        lora_A_weights = lora_A_weights.view(batch_size, self.in_features, self.lora_rank)
        lora_B_weights = lora_B_weights.view(batch_size, self.lora_rank, self.out_features)
        
        # 深度缩放因子
        depth_scale = self.depth_scale_gen(depth_features)  # [B, 1]
        
        # LoRA前向传播: x @ A @ B
        # x: [B, N, in_features], A: [B, in_features, lora_rank], B: [B, lora_rank, out_features]
        
        # 第一步: x @ A
        # [B, N, in_features] @ [B, in_features, lora_rank] -> [B, N, lora_rank]
        lora_A_output = torch.bmm(x, lora_A_weights)
        
        # 第二步: (x @ A) @ B
        # [B, N, lora_rank] @ [B, lora_rank, out_features] -> [B, N, out_features]
        lora_output = torch.bmm(lora_A_output, lora_B_weights)
        
        # 应用缩放和深度调制
        # depth_scale: [B, 1] -> [B, 1, 1] for broadcasting
        depth_scale = depth_scale.unsqueeze(1)  # [B, 1, 1]
        lora_output = lora_output * self.scaling * depth_scale
        
        # 最终输出: 原始输出 + LoRA输出
        final_output = original_output + lora_output
        
        return final_output


class DepthGuidedDinoLoRA(nn.Module):
    """
    深度引导的DINO LoRA模块
    
    将动态LoRA层插入到预训练的DINO模型中的特定层
    """
    
    def __init__(
        self,
        dino_model: nn.Module,
        depth_encoder: nn.Module,
        lora_layers: list = None,  # 要插入LoRA的层索引
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.dino_model = dino_model
        self.depth_encoder = depth_encoder
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        
        # 默认在所有attention层的qkv上添加LoRA
        if lora_layers is None:
            lora_layers = list(range(len(dino_model.encoder.layer)))
        self.lora_layers = lora_layers
        
        # 获取深度特征维度
        depth_feature_dim = depth_encoder.output_dim
        
        # 为指定层添加动态LoRA
        self.dynamic_lora_modules = nn.ModuleDict()
        
        for layer_idx in lora_layers:
            if layer_idx < len(dino_model.encoder.layer):
                layer = dino_model.encoder.layer[layer_idx]
                
                # 为query层添加动态LoRA
                original_query = layer.attention.attention.query
                dynamic_lora_query = DynamicLoRALayer(
                    original_layer=original_query,
                    depth_feature_dim=depth_feature_dim,
                    lora_rank=lora_rank,
                    lora_alpha=lora_alpha,
                    dropout=dropout
                )
                
                # 替换原始query层
                layer.attention.attention.query = dynamic_lora_query
                self.dynamic_lora_modules[f'layer_{layer_idx}_query'] = dynamic_lora_query
        
        # 冻结DINO的其他参数
        for name, param in dino_model.named_parameters():
            if 'qkv' not in name:  # 只有LoRA相关参数可训练
                param.requires_grad = False
    
    def forward(
        self, 
        rgb_image: torch.Tensor, 
        depth_image: torch.Tensor,
        return_intermediate: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            rgb_image: RGB图像 [B, 3, H, W]
            depth_image: 深度图 [B, 1, H, W]
            return_intermediate: 是否返回中间特征
            
        Returns:
            features: 融合后的特征
        """
        # 获取深度特征
        depth_features = self.depth_encoder(depth_image)  # [B, N+1, depth_dim]
        
        # 提取CLS token作为全局深度特征
        depth_features = depth_features[:, 0, :]  # [B, depth_dim]
        
        # 将深度特征存储到所有LoRA模块中
        for lora_module in self.dynamic_lora_modules.values():
            lora_module.current_depth_features = depth_features
        
        # 使用DINO模型的forward方法
        outputs = self.dino_model(rgb_image)
        
        # 获取最后隐藏状态
        x = outputs.last_hidden_state
        
        return x
    
    def get_lora_parameters(self):
        """获取LoRA相关的参数"""
        lora_params = []
        for module in self.dynamic_lora_modules.values():
            lora_params.extend(list(module.parameters()))
        return lora_params
    
    def save_lora_weights(self, filepath: str):
        """保存LoRA权重"""
        lora_state_dict = {}
        for name, module in self.dynamic_lora_modules.items():
            lora_state_dict[name] = module.state_dict()
        torch.save(lora_state_dict, filepath)
    
    def load_lora_weights(self, filepath: str):
        """加载LoRA权重"""
        lora_state_dict = torch.load(filepath, map_location='cpu')
        for name, module in self.dynamic_lora_modules.items():
            if name in lora_state_dict:
                module.load_state_dict(lora_state_dict[name])


class DepthFeatureExtractor(nn.Module):
    """
    深度特征提取器
    专门用于生成LoRA权重的深度特征
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        hidden_dim: int = 512,
        output_dim: int = 1024,
        image_size: int = 518,
        patch_size: int = 14,
        num_layers: int = 4,
        num_heads: int = 8,
        **kwargs
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            input_channels, hidden_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, hidden_dim)
        )
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
        torch.nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        if self.patch_embed.bias is not None:
            torch.nn.init.zeros_(self.patch_embed.bias)
        
        torch.nn.init.trunc_normal_(self.output_proj.weight, std=0.02)
        torch.nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, depth_image: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        前向传播
        
        Args:
            depth_image: 深度图 [B, 1, H, W]
            
        Returns:
            depth_features: 深度特征 [B, num_patches+1, output_dim]
        """
        B = depth_image.shape[0]
        
        # Patch embedding
        x = self.patch_embed(depth_image)  # [B, hidden_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)   # [B, num_patches, hidden_dim]
        
        # 添加位置编码
        x = x + self.pos_embed.to(x.device)
        
        # 添加CLS token
        cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        cls_tokens = cls_token.expand(B, -1, -1).to(x.device)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer编码
        x = self.transformer(x)
        
        # 输出投影和归一化
        x = self.output_proj(x)
        x = self.norm(x)
        
        return x