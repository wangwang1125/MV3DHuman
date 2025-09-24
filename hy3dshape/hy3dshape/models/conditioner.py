# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

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

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import (
    CLIPVisionModelWithProjection,
    CLIPVisionConfig,
    Dinov2Model,
    Dinov2Config,
)
from transformers import AutoImageProcessor, AutoModel
from .depth_encoder import DepthEncoder, LightweightDepthEncoder
from .cross_modal_fusion import CrossModalFusion, SimpleCrossModalFusion
from .dynamic_depth_lora import DepthGuidedDinoLoRA, DepthFeatureExtractor
from .depth_guided_lora_fusion import DepthGuidedLoRAFusion, AdaptiveDepthGuidedLoRAFusion


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    return np.concatenate([emb_sin, emb_cos], axis=1)


class ImageEncoder(nn.Module):
    def __init__(
        self,
        version=None,
        config=None,
        use_cls_token=True,
        image_size=224,
        **kwargs,
    ):
        super().__init__()

        if config is None:
            self.model = AutoModel.from_pretrained(version)
        else:
            self.model = self.MODEL_CLASS(self.MODEL_CONFIG_CLASS.from_dict(config))
            
        self.model.eval()
        self.model.requires_grad_(False)
        self.use_cls_token = use_cls_token
        self.size = image_size // 14
        self.num_patches = (image_size // 14) ** 2
        if self.use_cls_token:
            self.num_patches += 1

        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size, transforms.InterpolationMode.BILINEAR, antialias=True),
                transforms.CenterCrop(image_size),
                transforms.Normalize(
                    mean=self.mean,
                    std=self.std,
                ),
            ]
        )

    def forward(self, image, mask=None, value_range=(-1, 1), **kwargs):
        if value_range is not None:
            low, high = value_range
            image = (image - low) / (high - low)

        image = image.to(self.model.device, dtype=self.model.dtype)
        inputs = self.transform(image)
        outputs = self.model(inputs)

        last_hidden_state = outputs.last_hidden_state
        if not self.use_cls_token:
            last_hidden_state = last_hidden_state[:, 1:, :]

        return last_hidden_state

    def unconditional_embedding(self, batch_size, **kwargs):
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        zero = torch.zeros(
            batch_size,
            self.num_patches,
            self.model.config.hidden_size,
            device=device,
            dtype=dtype,
        )

        return zero


class CLIPImageEncoder(ImageEncoder):
    MODEL_CLASS = CLIPVisionModelWithProjection
    MODEL_CONFIG_CLASS = CLIPVisionConfig
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]


class DinoImageEncoder(ImageEncoder):
    MODEL_CLASS = Dinov2Model
    MODEL_CONFIG_CLASS = Dinov2Config
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


class DinoImageEncoderMV(DinoImageEncoder):
    def __init__(
        self,
        version=None,
        config=None,
        use_cls_token=True,
        image_size=224,
        view_num=4,
        **kwargs,
    ):
        super().__init__(version, config, use_cls_token, image_size, **kwargs)
        self.view_num = view_num
        self.num_patches = self.num_patches
        pos = np.arange(self.view_num, dtype=np.float32)
        view_embedding = torch.from_numpy(
            get_1d_sincos_pos_embed_from_grid(self.model.config.hidden_size, pos)).float()

        view_embedding = view_embedding.unsqueeze(1).repeat(1, self.num_patches, 1)
        self.view_embed = view_embedding.unsqueeze(0)

    def forward(self, image, mask=None, value_range=(-1, 1), view_idxs=None):
        if value_range is not None:
            low, high = value_range
            image = (image - low) / (high - low)

        image = image.to(self.model.device, dtype=self.model.dtype)

        bs, num_views, c, h, w = image.shape
        image = image.view(bs * num_views, c, h, w)

        inputs = self.transform(image)
        outputs = self.model(inputs)

        last_hidden_state = outputs.last_hidden_state
        last_hidden_state = last_hidden_state.view(
            bs, num_views, last_hidden_state.shape[-2],
            last_hidden_state.shape[-1]
        )

        view_embedding = self.view_embed.to(last_hidden_state.dtype).to(last_hidden_state.device)
        if view_idxs is not None:
            assert len(view_idxs) == bs
            view_embeddings = []
            for i in range(bs):
                view_idx = view_idxs[i]
                assert num_views == len(view_idx)
                view_embeddings.append(self.view_embed[:, view_idx, ...])
            view_embedding = torch.cat(view_embeddings, 0).to(last_hidden_state.dtype).to(last_hidden_state.device)

        if num_views != self.view_num:
            view_embedding = view_embedding[:, :num_views, ...]
        last_hidden_state = last_hidden_state + view_embedding
        last_hidden_state = last_hidden_state.view(bs, num_views * last_hidden_state.shape[-2],
                                                   last_hidden_state.shape[-1])
        return last_hidden_state

    def unconditional_embedding(self, batch_size, view_idxs=None, **kwargs):
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        zero = torch.zeros(
            batch_size,
            self.num_patches * len(view_idxs[0]),
            self.model.config.hidden_size,
            device=device,
            dtype=dtype,
        )
        return zero


def build_image_encoder(config):
    if config['type'] == 'CLIPImageEncoder':
        return CLIPImageEncoder(**config['kwargs'])
    elif config['type'] == 'DinoImageEncoder':
        return DinoImageEncoder(**config['kwargs'])
    elif config['type'] == 'DinoImageEncoderMV':
        return DinoImageEncoderMV(**config['kwargs'])
    else:
        raise ValueError(f'Unknown image encoder type: {config["type"]}')


class DualImageEncoder(nn.Module):
    def __init__(
        self,
        main_image_encoder,
        additional_image_encoder,
    ):
        super().__init__()
        self.main_image_encoder = build_image_encoder(main_image_encoder)
        self.additional_image_encoder = build_image_encoder(additional_image_encoder)

    def forward(self, image, mask=None, **kwargs):
        outputs = {
            'main': self.main_image_encoder(image, mask=mask, **kwargs),
            'additional': self.additional_image_encoder(image, mask=mask, **kwargs),
        }
        return outputs

    def unconditional_embedding(self, batch_size, **kwargs):
        outputs = {
            'main': self.main_image_encoder.unconditional_embedding(batch_size, **kwargs),
            'additional': self.additional_image_encoder.unconditional_embedding(batch_size, **kwargs),
        }
        return outputs


class SingleImageEncoder(nn.Module):
    def __init__(
        self,
        main_image_encoder,
        drop_ratio=0.0
    ):
        super().__init__()
        self.main_image_encoder = build_image_encoder(main_image_encoder)
        self.drop_ratio = drop_ratio
        self.disable_drop = True

    def forward(self, image, mask=None, **kwargs):
        outputs = {
            'main': self.main_image_encoder(image, mask=mask, **kwargs),
        }
        if self.disable_drop:
            return outputs
        else:
            random_p = torch.rand(len(image), device='cuda')
            remain_bool_tensor = random_p > self.drop_ratio
            outputs['main'] *= remain_bool_tensor.view(-1,1,1)
        return outputs

        
        outputs = {
            'main': self.main_image_encoder(image, mask=mask, **kwargs),
        }
        return outputs

    def unconditional_embedding(self, batch_size, **kwargs):
        outputs = {
            'main': self.main_image_encoder.unconditional_embedding(batch_size, **kwargs),
        }
        return outputs


class RGBDImageEncoder(nn.Module):
    """RGB+深度图像编码器，支持微调训练"""
    def __init__(
        self,
        main_image_encoder,
        depth_encoder_config=None,
        fusion_config=None,
        drop_ratio=0.0,
        freeze_rgb_encoder=True
    ):
        super().__init__()
        # RGB编码器（预训练，可选择冻结）
        self.main_image_encoder = build_image_encoder(main_image_encoder)
        if freeze_rgb_encoder:
            for param in self.main_image_encoder.parameters():
                param.requires_grad = False
        
        # 深度编码器（新增，需要训练）
        depth_config = depth_encoder_config or {
            'type': 'lightweight',
            'input_channels': 1,
            'hidden_dim': 768,
            'num_layers': 4
        }
        
        if depth_config['type'] == 'lightweight':
            self.depth_encoder = LightweightDepthEncoder(
                input_channels=depth_config['input_channels'],
                hidden_dim=depth_config['hidden_dim'],
                num_layers=depth_config['num_layers']
            )
        else:
            self.depth_encoder = DepthEncoder(
                input_channels=depth_config['input_channels'],
                hidden_dim=depth_config['hidden_dim'],
                num_heads=depth_config.get('num_heads', 8),
                num_layers=depth_config.get('num_layers', 6)
            )
        
        # 跨模态融合模块（新增，需要训练）
        fusion_config = fusion_config or {
            'type': 'cross_attention',
            'hidden_dim': 768,
            'num_heads': 8
        }
        
        if fusion_config['type'] == 'simple':
            self.fusion_module = SimpleCrossModalFusion(
                rgb_dim=self.main_image_encoder.model.config.hidden_size,
                depth_dim=depth_config['hidden_dim'],
                output_dim=fusion_config['hidden_dim']
            )
        elif fusion_config['type'] == 'depth_guided_lora':
            # 使用深度引导LoRA融合
            self.fusion_module = DepthGuidedLoRAFusion(
                rgb_dim=fusion_config.get('rgb_dim', 1024),
                depth_dim=fusion_config.get('depth_dim', 768),
                hidden_dim=fusion_config.get('hidden_dim', 1024),
                output_dim=fusion_config.get('output_dim', 1024),
                lora_rank=fusion_config.get('lora_rank', 64),
                lora_alpha=fusion_config.get('lora_alpha', 16),
                num_guidance_layers=fusion_config.get('num_guidance_layers', 3),
                dropout=fusion_config.get('dropout', 0.1),
                use_bias=fusion_config.get('use_bias', True)
            )
        elif fusion_config['type'] == 'adaptive_depth_guided_lora':
            # 使用自适应深度引导LoRA融合
            adaptive_config = fusion_config.get('adaptive_config', {})
            self.fusion_module = AdaptiveDepthGuidedLoRAFusion(
                rgb_dim=fusion_config.get('rgb_dim', 1024),
                depth_dim=fusion_config.get('depth_dim', 768),
                hidden_dim=fusion_config.get('hidden_dim', 1024),
                output_dim=fusion_config.get('output_dim', 1024),
                lora_rank=fusion_config.get('lora_rank', 64),
                lora_alpha=fusion_config.get('lora_alpha', 16),
                num_guidance_layers=fusion_config.get('num_guidance_layers', 3),
                dropout=fusion_config.get('dropout', 0.1),
                use_bias=fusion_config.get('use_bias', True),
                weight_dim=adaptive_config.get('weight_dim', 512),
                num_weight_layers=adaptive_config.get('num_weight_layers', 2),
                temperature=adaptive_config.get('temperature', 1.0),
                use_gating=adaptive_config.get('use_gating', True)
            )
        else:
            self.fusion_module = CrossModalFusion(
                rgb_dim=self.main_image_encoder.model.config.hidden_size,
                depth_dim=depth_config['hidden_dim'],
                hidden_dim=fusion_config['hidden_dim'],
                num_heads=fusion_config.get('num_heads', 8),
                fusion_type=fusion_config['type']
            )
        
        self.drop_ratio = drop_ratio
        self.disable_drop = True
    
    def forward(self, image, depth_image=None, mask=None, **kwargs):
        # RGB特征提取（预训练模型）
        rgb_features = self.main_image_encoder(image, mask=mask, **kwargs)
        
        if depth_image is not None:
            # 深度特征提取（新训练模块）
            depth_features = self.depth_encoder(depth_image)
            
            # 跨模态融合（新训练模块）
            fused_features = self.fusion_module(rgb_features, depth_features)
            
            outputs = {
                'main': fused_features,
                'rgb': rgb_features,
                'depth': depth_features
            }
        else:
            # 仅RGB输入的情况
            outputs = {
                'main': rgb_features,
                'rgb': rgb_features
            }
        
        # Dropout处理
        if not self.disable_drop and self.drop_ratio > 0:
            random_p = torch.rand(len(image), device=image.device)
            remain_bool_tensor = random_p > self.drop_ratio
            outputs['main'] = outputs['main'] * remain_bool_tensor.view(-1, 1, 1)
        
        return outputs
    
    def unconditional_embedding(self, batch_size, **kwargs):
        # 获取RGB编码器的无条件嵌入
        rgb_unconditional = self.main_image_encoder.unconditional_embedding(batch_size, **kwargs)
        
        # 创建深度特征的零嵌入
        device = next(self.depth_encoder.parameters()).device
        dtype = next(self.depth_encoder.parameters()).dtype
        depth_unconditional = torch.zeros(
            batch_size,
            rgb_unconditional.shape[1],  # 保持序列长度一致
            self.depth_encoder.hidden_dim,
            device=device,
            dtype=dtype
        )
        
        # 融合无条件嵌入
        fused_unconditional = self.fusion_module(rgb_unconditional, depth_unconditional)
        
        outputs = {
            'main': fused_unconditional,
            'rgb': rgb_unconditional,
            'depth': depth_unconditional
        }
        return outputs
    
    def get_trainable_parameters(self):
        """获取可训练参数（仅深度编码器和融合模块）"""
        trainable_params = []
        trainable_params.extend(list(self.depth_encoder.parameters()))
        trainable_params.extend(list(self.fusion_module.parameters()))
        return trainable_params


class DynamicLoRAImageEncoder(nn.Module):
    """
    动态LoRA图像编码器
    
    使用深度特征动态生成LoRA权重，直接插入到DINO模型的attention层中
    这是真正的LoRA实现，符合LoRA的设计理念
    """
    
    def __init__(
        self,
        main_image_encoder,
        depth_encoder_config=None,
        lora_config=None,
        drop_ratio=0.0,
        freeze_rgb_encoder=True,
        **kwargs
    ):
        super().__init__()
        
        # 构建RGB编码器对象
        self.main_image_encoder = build_image_encoder(main_image_encoder)
        self.drop_ratio = drop_ratio
        self.disable_drop = True
        
        # 冻结RGB编码器
        if freeze_rgb_encoder:
            for param in self.main_image_encoder.parameters():
                param.requires_grad = False
        
        # 深度特征提取器配置
        depth_config = depth_encoder_config or {
            'input_channels': 1,
            'hidden_dim': 512,
            'output_dim': 1024,
            'image_size': 518,
            'patch_size': 14,
            'num_layers': 4,
            'num_heads': 8
        }
        
        # 创建深度特征提取器
        self.depth_feature_extractor = DepthFeatureExtractor(
            input_channels=depth_config['input_channels'],
            hidden_dim=depth_config['hidden_dim'],
            output_dim=depth_config['output_dim'],
            image_size=depth_config['image_size'],
            patch_size=depth_config['patch_size'],
            num_layers=depth_config['num_layers'],
            num_heads=depth_config['num_heads']
        )
        
        # LoRA配置
        lora_config = lora_config or {
            'lora_layers': None,  # None表示所有层
            'lora_rank': 16,
            'lora_alpha': 16.0,
            'dropout': 0.1
        }
        
        # 创建深度引导的DINO LoRA模型
        self.dino_lora_model = DepthGuidedDinoLoRA(
            dino_model=self.main_image_encoder.model,
            depth_encoder=self.depth_feature_extractor,
            lora_layers=lora_config['lora_layers'],
            lora_rank=lora_config['lora_rank'],
            lora_alpha=lora_config['lora_alpha'],
            dropout=lora_config['dropout']
        )
        
        # 如果使用传统的融合方式作为补充
        fusion_config = kwargs.get('fusion_config', None)
        if fusion_config:
            if fusion_config['type'] == 'depth_guided_lora':
                self.fusion_module = DepthGuidedLoRAFusion(
                    rgb_dim=fusion_config.get('rgb_dim', 1024),
                    depth_dim=fusion_config.get('depth_dim', 1024),
                    hidden_dim=fusion_config.get('hidden_dim', 1024),
                    output_dim=fusion_config.get('output_dim', 1024),
                    lora_rank=fusion_config.get('lora_rank', 64),
                    lora_alpha=fusion_config.get('lora_alpha', 16),
                    num_guidance_layers=fusion_config.get('num_guidance_layers', 3),
                    dropout=fusion_config.get('dropout', 0.1),
                    use_bias=fusion_config.get('use_bias', True)
                )
            elif fusion_config['type'] == 'adaptive_depth_guided_lora':
                self.fusion_module = AdaptiveDepthGuidedLoRAFusion(
                    rgb_dim=fusion_config.get('rgb_dim', 1024),
                    depth_dim=fusion_config.get('depth_dim', 1024),
                    hidden_dim=fusion_config.get('hidden_dim', 1024),
                    output_dim=fusion_config.get('output_dim', 1024),
                    lora_rank=fusion_config.get('lora_rank', 64),
                    lora_alpha=fusion_config.get('lora_alpha', 16),
                    num_heads=fusion_config.get('num_heads', 16),
                    num_guidance_layers=fusion_config.get('num_guidance_layers', 3),
                    num_scales=fusion_config.get('num_scales', 3),
                    dropout=fusion_config.get('dropout', 0.1)
                )
            else:
                self.fusion_module = None
        else:
            self.fusion_module = None
    
    def forward(self, image, depth_image=None, mask=None, **kwargs):
        """
        前向传播
        
        Args:
            image: RGB图像 [B, 3, H, W]
            depth_image: 深度图 [B, 1, H, W]
            mask: 可选的mask
            
        Returns:
            outputs: 包含融合特征的字典
        """
        if depth_image is not None:
            # 使用动态LoRA进行RGB-深度融合
            fused_features = self.dino_lora_model(image, depth_image, **kwargs)
            
            # 如果有额外的融合模块，可以进一步处理
            if self.fusion_module is not None:
                # 获取原始RGB特征用于融合模块
                rgb_features = self.main_image_encoder(image, mask=mask, **kwargs)
                depth_features = self.depth_feature_extractor(depth_image)
                
                # 使用传统融合模块进行额外处理
                additional_fused = self.fusion_module(rgb_features, depth_features, mask=mask)
                
                # 可以选择如何组合两种融合结果
                # 这里简单地使用动态LoRA的结果作为主要输出
                outputs = {
                    'main': fused_features,
                    'lora_fused': fused_features,
                    'traditional_fused': additional_fused,
                    'depth': depth_features
                }
            else:
                outputs = {
                    'main': fused_features,
                    'lora_fused': fused_features
                }
        else:
            # 仅RGB输入的情况，使用原始DINO模型
            rgb_features = self.main_image_encoder(image, mask=mask, **kwargs)
            outputs = {
                'main': rgb_features,
                'rgb': rgb_features
            }
        
        # Dropout处理
        if not self.disable_drop and self.drop_ratio > 0:
            random_p = torch.rand(len(image), device=image.device)
            remain_bool_tensor = random_p > self.drop_ratio
            outputs['main'] = outputs['main'] * remain_bool_tensor.view(-1, 1, 1)
        
        return outputs
    
    def unconditional_embedding(self, batch_size, **kwargs):
        """生成无条件嵌入"""
        # 获取RGB编码器的无条件嵌入
        rgb_unconditional = self.main_image_encoder.unconditional_embedding(batch_size, **kwargs)
        
        # 创建深度特征的零嵌入
        device = next(self.depth_feature_extractor.parameters()).device
        dtype = next(self.depth_feature_extractor.parameters()).dtype
        
        # 创建零深度图
        zero_depth = torch.zeros(
            batch_size, 1, 518, 518,  # 假设深度图尺寸为518x518
            device=device, dtype=dtype
        )
        
        # 通过深度特征提取器获取零特征
        depth_unconditional = self.depth_feature_extractor(zero_depth)
        
        # 如果有融合模块，使用它生成无条件嵌入
        if self.fusion_module is not None:
            fused_unconditional = self.fusion_module.unconditional_embedding(
                batch_size, rgb_unconditional.shape[1], device=device
            )
            outputs = {
                'main': fused_unconditional,
                'rgb': rgb_unconditional,
                'depth': depth_unconditional
            }
        else:
            # 否则使用RGB的无条件嵌入
            outputs = {
                'main': rgb_unconditional,
                'rgb': rgb_unconditional,
                'depth': depth_unconditional
            }
        
        return outputs
    
    def get_trainable_parameters(self):
        """获取可训练参数（深度特征提取器和LoRA参数）"""
        trainable_params = []
        
        # 深度特征提取器参数
        trainable_params.extend(list(self.depth_feature_extractor.parameters()))
        
        # LoRA参数
        trainable_params.extend(self.dino_lora_model.get_lora_parameters())
        
        # 如果有融合模块，也包含其参数
        if self.fusion_module is not None:
            trainable_params.extend(list(self.fusion_module.parameters()))
        
        return trainable_params
    
    def save_lora_weights(self, filepath: str):
        """保存LoRA权重"""
        self.dino_lora_model.save_lora_weights(filepath)
    
    def load_lora_weights(self, filepath: str):
        """加载LoRA权重"""
        self.dino_lora_model.load_lora_weights(filepath)
    
    def get_lora_parameter_count(self):
        """
        获取LoRA参数的详细统计信息
        
        Returns:
            dict: 包含LoRA参数统计的字典
        """
        lora_params = 0
        depth_encoder_params = 0
        fusion_params = 0
        total_trainable_params = 0
        
        # 统计LoRA参数
        for param in self.dino_lora_model.get_lora_parameters():
            if param.requires_grad:
                lora_params += param.numel()
                total_trainable_params += param.numel()
        
        # 统计深度编码器参数
        for param in self.depth_feature_extractor.parameters():
            if param.requires_grad:
                depth_encoder_params += param.numel()
                total_trainable_params += param.numel()
        
        # 统计融合模块参数（如果存在）
        if self.fusion_module is not None:
            for param in self.fusion_module.parameters():
                if param.requires_grad:
                    fusion_params += param.numel()
                    total_trainable_params += param.numel()
        
        # 统计主编码器的总参数（包括冻结的）
        main_encoder_total_params = sum(p.numel() for p in self.main_image_encoder.parameters())
        main_encoder_trainable_params = sum(p.numel() for p in self.main_image_encoder.parameters() if p.requires_grad)
        
        return {
            'lora_parameters': lora_params,
            'depth_encoder_parameters': depth_encoder_params,
            'fusion_module_parameters': fusion_params,
            'total_trainable_parameters': total_trainable_params,
            'main_encoder_total_parameters': main_encoder_total_params,
            'main_encoder_trainable_parameters': main_encoder_trainable_params,
            'parameter_efficiency': lora_params / main_encoder_total_params if main_encoder_total_params > 0 else 0.0
        }
    
    def print_lora_parameter_summary(self):
        """打印LoRA参数的详细摘要"""
        stats = self.get_lora_parameter_count()
        
        print("=" * 80)
        print("LoRA Parameter Summary for DynamicLoRAImageEncoder")
        print("=" * 80)
        print(f"LoRA Parameters:                {stats['lora_parameters']:,}")
        print(f"Depth Encoder Parameters:       {stats['depth_encoder_parameters']:,}")
        print(f"Fusion Module Parameters:       {stats['fusion_module_parameters']:,}")
        print(f"Total Trainable Parameters:     {stats['total_trainable_parameters']:,}")
        print(f"Main Encoder Total Parameters:  {stats['main_encoder_total_parameters']:,}")
        print(f"Main Encoder Trainable Params:  {stats['main_encoder_trainable_parameters']:,}")
        print(f"Parameter Efficiency:           {stats['parameter_efficiency']:.4%}")
        print("=" * 80)
        
        return stats
