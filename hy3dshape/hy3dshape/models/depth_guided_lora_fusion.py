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


class DepthGuidedLoRAFusion(nn.Module):
    """
    深度引导的LoRA风格特征融合模块
    
    该模块使用深度特征来引导RGB特征的融合，采用低秩分解来减少参数量，
    同时保持高效的特征融合能力。
    
    核心思想：
    1. 使用深度特征生成动态的LoRA权重
    2. 通过低秩分解减少参数量
    3. 自适应地调整融合强度
    """
    
    def __init__(
        self,
        rgb_dim=1024,           # RGB特征维度
        depth_dim=1024,         # 深度特征维度
        hidden_dim=1024,        # 隐藏层维度
        output_dim=1024,        # 输出特征维度
        lora_rank=64,           # LoRA秩
        lora_alpha=16,          # LoRA缩放因子
        num_guidance_layers=3,  # 引导层数
        dropout=0.1,            # dropout率
        use_bias=True,          # 是否使用偏置
        **kwargs
    ):
        super().__init__()
        
        self.rgb_dim = rgb_dim
        self.depth_dim = depth_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.num_guidance_layers = num_guidance_layers
        self.scaling = lora_alpha / lora_rank
        
        # 输入投影层
        self.rgb_proj = nn.Linear(rgb_dim, hidden_dim, bias=use_bias)
        self.depth_proj = nn.Linear(depth_dim, hidden_dim, bias=use_bias)
        
        # 深度引导网络 - 生成LoRA权重
        self.depth_guidance_net = self._build_guidance_network()
        
        # LoRA分解层
        self.lora_A = nn.Linear(hidden_dim, lora_rank, bias=False)
        self.lora_B = nn.Linear(lora_rank, hidden_dim, bias=False)
        
        # 动态权重生成器
        self.weight_generator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, lora_rank),
            nn.Tanh()
        )
        
        # 自适应融合权重
        self.adaptive_weight = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # 特征增强模块
        self.feature_enhancer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # 输出投影
        self.output_proj = nn.Linear(hidden_dim, output_dim, bias=use_bias)
        self.output_norm = nn.LayerNorm(output_dim)
        
        # 残差连接权重
        self.residual_weight = nn.Parameter(torch.ones(1))
        
        # 初始化权重
        self._init_weights()
    
    def _build_guidance_network(self):
        """构建深度引导网络"""
        layers = []
        
        for i in range(self.num_guidance_layers):
            if i == 0:
                layers.extend([
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1)
                ])
            else:
                layers.extend([
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.1)
                ])
        
        layers.append(nn.LayerNorm(self.hidden_dim))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """初始化模型权重"""
        # 初始化LoRA层
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        # 初始化其他线性层
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module != self.lora_A and module != self.lora_B:
                    nn.init.trunc_normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
    
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
        batch_size, seq_len = rgb_features.shape[:2]
        
        # 输入投影
        rgb_proj = self.rgb_proj(rgb_features)      # [B, N, hidden_dim]
        depth_proj = self.depth_proj(depth_features) # [B, N, hidden_dim]
        
        # 深度引导特征生成
        depth_guidance = self.depth_guidance_net(depth_proj)  # [B, N, hidden_dim]
        
        # 生成动态LoRA权重
        dynamic_weights = self.weight_generator(depth_guidance)  # [B, N, lora_rank]
        
        # LoRA变换
        lora_down = self.lora_A(rgb_proj)  # [B, N, lora_rank]
        
        # 应用动态权重
        lora_weighted = lora_down * dynamic_weights  # [B, N, lora_rank]
        
        # LoRA上采样
        lora_up = self.lora_B(lora_weighted)  # [B, N, hidden_dim]
        
        # 缩放LoRA输出
        lora_output = lora_up * self.scaling
        
        # 计算自适应融合权重
        concat_features = torch.cat([rgb_proj, depth_guidance], dim=-1)
        adaptive_alpha = self.adaptive_weight(concat_features)  # [B, N, 1]
        
        # 融合RGB特征和LoRA输出
        enhanced_rgb = rgb_proj + lora_output * adaptive_alpha
        
        # 与深度特征进行最终融合
        fused_features = enhanced_rgb + depth_guidance * (1 - adaptive_alpha)
        
        # 特征增强
        enhanced_features = self.feature_enhancer(fused_features)
        
        # 残差连接
        final_features = fused_features + enhanced_features * self.residual_weight
        
        # 输出投影和归一化
        output = self.output_proj(final_features)
        output = self.output_norm(output)
        
        return output
    
    def unconditional_embedding(self, batch_size, num_patches, device=None, **kwargs):
        """
        生成无条件嵌入（用于classifier-free guidance）
        
        Args:
            batch_size: 批次大小
            num_patches: patch数量
            device: 设备
            
        Returns:
            zero_embedding: 零嵌入 [batch_size, num_patches, output_dim]
        """
        if device is None:
            device = next(self.parameters()).device
        
        zero_embedding = torch.zeros(
            batch_size, num_patches, self.output_dim,
            device=device, dtype=next(self.parameters()).dtype
        )
        return zero_embedding


class AdaptiveDepthGuidedLoRAFusion(nn.Module):
    """
    自适应深度引导LoRA融合模块
    
    在DepthGuidedLoRAFusion基础上增加了：
    1. 多尺度特征融合
    2. 注意力机制
    3. 更复杂的自适应权重计算
    """
    
    def __init__(
        self,
        rgb_dim=1024,
        depth_dim=1024,
        hidden_dim=1024,
        output_dim=1024,
        lora_rank=64,
        lora_alpha=16,
        num_heads=16,
        num_guidance_layers=3,
        num_scales=3,
        dropout=0.1,
        **kwargs
    ):
        super().__init__()
        
        self.rgb_dim = rgb_dim
        self.depth_dim = depth_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.num_heads = num_heads
        self.num_scales = num_scales
        self.scaling = lora_alpha / lora_rank
        
        # 输入投影
        self.rgb_proj = nn.Linear(rgb_dim, hidden_dim)
        self.depth_proj = nn.Linear(depth_dim, hidden_dim)
        
        # 多尺度LoRA模块
        self.multi_scale_lora = nn.ModuleList([
            self._create_lora_module(hidden_dim, lora_rank // (2**i))
            for i in range(num_scales)
        ])
        
        # 跨模态注意力
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # 深度引导网络
        self.depth_guidance_net = self._build_guidance_network(num_guidance_layers)
        
        # 自适应权重网络
        self.adaptive_weight_net = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_scales),
            nn.Softmax(dim=-1)
        )
        
        # 输出层
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        
        self._init_weights()
    
    def _create_lora_module(self, hidden_dim, rank):
        """创建LoRA模块"""
        return nn.ModuleDict({
            'lora_A': nn.Linear(hidden_dim, rank, bias=False),
            'lora_B': nn.Linear(rank, hidden_dim, bias=False),
            'weight_gen': nn.Sequential(
                nn.Linear(hidden_dim, rank),
                nn.Tanh()
            )
        })
    
    def _build_guidance_network(self, num_layers):
        """构建深度引导网络"""
        layers = []
        for i in range(num_layers):
            layers.extend([
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1)
            ])
        layers.append(nn.LayerNorm(self.hidden_dim))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias)
                nn.init.ones_(module.weight)
        
        # 特殊初始化LoRA层
        for lora_module in self.multi_scale_lora:
            nn.init.kaiming_uniform_(lora_module['lora_A'].weight, a=math.sqrt(5))
            nn.init.zeros_(lora_module['lora_B'].weight)
    
    def forward(self, rgb_features, depth_features, mask=None, **kwargs):
        """前向传播"""
        # 输入投影
        rgb_proj = self.rgb_proj(rgb_features)
        depth_proj = self.depth_proj(depth_features)
        
        # 深度引导
        depth_guidance = self.depth_guidance_net(depth_proj)
        
        # 跨模态注意力
        attended_rgb, _ = self.cross_attention(
            rgb_proj, depth_guidance, depth_guidance, 
            key_padding_mask=mask
        )
        
        # 多尺度LoRA变换
        lora_outputs = []
        for lora_module in self.multi_scale_lora:
            # 生成动态权重
            dynamic_weights = lora_module['weight_gen'](depth_guidance)
            
            # LoRA变换
            lora_down = lora_module['lora_A'](attended_rgb)
            lora_weighted = lora_down * dynamic_weights
            lora_up = lora_module['lora_B'](lora_weighted)
            
            lora_outputs.append(lora_up * self.scaling)
        
        # 计算自适应权重
        concat_features = torch.cat([rgb_proj, depth_guidance, attended_rgb], dim=-1)
        scale_weights = self.adaptive_weight_net(concat_features)  # [B, N, num_scales]
        
        # 加权融合多尺度LoRA输出
        weighted_lora = sum(
            weight.unsqueeze(-1) * lora_out 
            for weight, lora_out in zip(scale_weights.unbind(-1), lora_outputs)
        )
        
        # 最终融合
        fused_features = attended_rgb + weighted_lora + depth_guidance
        
        # 输出投影
        output = self.output_proj(fused_features)
        output = self.output_norm(output)
        
        return output
    
    def unconditional_embedding(self, batch_size, num_patches, device=None, **kwargs):
        """生成无条件嵌入"""
        if device is None:
            device = next(self.parameters()).device
        
        zero_embedding = torch.zeros(
            batch_size, num_patches, self.output_dim,
            device=device, dtype=next(self.parameters()).dtype
        )
        return zero_embedding