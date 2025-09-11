# -*- coding: utf-8 -*-

# RGBD数据加载器，支持RGB+深度图的微调训练
# 基于dit_asl.py扩展，添加深度图处理功能

import os
import io
import sys
import time
import random
import traceback
from typing import Optional, Union, List, Tuple, Dict

import json
import glob
import cv2
import numpy as np
import trimesh

import torch
import torchvision.transforms as transforms
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_info

from .dit_asl import (
    ResampledShards, read_npz, read_json, padding, viz_pc,
    AlignedShapeLatentDataset, worker_init_fn, pytorch_worker_seed, make_seed
)


def load_depth_image(depth_path, depth_clip_range=[0.0, 10.0]):
    """
    加载深度图像
    
    Args:
        depth_path (str): 深度图路径
        depth_clip_range (list): 深度值裁剪范围 [min, max]
    
    Returns:
        depth (np.ndarray): 归一化的深度图 (H, W, 1)
    """
    if depth_path.endswith('.npy'):
        depth = np.load(depth_path)
    elif depth_path.endswith('.npz'):
        depth_data = np.load(depth_path)
        depth = depth_data['depth'] if 'depth' in depth_data else depth_data['arr_0']
    else:
        # 假设是图像格式 (PNG, EXR等)
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth is None:
            raise ValueError(f"无法加载深度图: {depth_path}")
    
    # 确保是单通道
    if len(depth.shape) == 3:
        depth = depth[:, :, 0]
    
    # 裁剪深度值
    depth = np.clip(depth, depth_clip_range[0], depth_clip_range[1])
    
    # 归一化到 [0, 1]
    depth_min, depth_max = depth_clip_range
    depth = (depth - depth_min) / (depth_max - depth_min)
    
    # 添加通道维度
    depth = depth[:, :, np.newaxis]
    
    return depth.astype(np.float32)


def depth_augmentation(depth, noise_std=0.01, dropout_prob=0.1):
    """
    深度图数据增强
    
    Args:
        depth (np.ndarray): 输入深度图 (H, W, 1)
        noise_std (float): 高斯噪声标准差
        dropout_prob (float): 随机置零概率
    
    Returns:
        depth (np.ndarray): 增强后的深度图
    """
    depth = depth.copy()
    
    # 添加高斯噪声
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, depth.shape)
        depth = depth + noise
        depth = np.clip(depth, 0, 1)
    
    # 随机dropout
    if dropout_prob > 0:
        dropout_mask = np.random.random(depth.shape[:2]) > dropout_prob
        depth = depth * dropout_mask[:, :, np.newaxis]
    
    return depth


class RGBDAlignedShapeLatentDataset(AlignedShapeLatentDataset):
    """
    RGBD数据集，支持RGB图像和深度图的联合加载
    """
    
    def __init__(
        self,
        data_list: str = None,
        cond_stage_key: str = "image",
        depth_stage_key: str = "depth",
        image_transform=None,
        depth_transform=None,
        pc_size: int = 2048,
        pc_sharpedge_size: int = 2048,
        sharpedge_label: bool = False,
        return_normal: bool = False,
        deterministic=False,
        worker_seed=None,
        padding=True,
        padding_ratio_range=[1.15, 1.15],
        # 深度图相关参数
        depth_clip_range=[0.0, 10.0],
        depth_normalize=True,
        depth_augmentation_config=None,
        require_depth=True,  # 是否必须有深度图
    ):
        super().__init__(
            data_list=data_list,
            cond_stage_key=cond_stage_key,
            image_transform=image_transform,
            pc_size=pc_size,
            pc_sharpedge_size=pc_sharpedge_size,
            sharpedge_label=sharpedge_label,
            return_normal=return_normal,
            deterministic=deterministic,
            worker_seed=worker_seed,
            padding=padding,
            padding_ratio_range=padding_ratio_range
        )
        
        self.depth_stage_key = depth_stage_key
        self.depth_transform = depth_transform
        self.depth_clip_range = depth_clip_range
        self.depth_normalize = depth_normalize
        self.require_depth = require_depth
        
        # 深度图增强配置
        self.depth_aug_config = depth_augmentation_config or {
            'noise_std': 0.01,
            'dropout_prob': 0.1
        }
    
    def load_depth(self, depth_path):
        """
        加载深度图
        
        Args:
            depth_path (str): 深度图路径
        
        Returns:
            depth (np.ndarray): 处理后的深度图
        """
        try:
            depth = load_depth_image(depth_path, self.depth_clip_range)
            
            # 数据增强
            if self.depth_aug_config:
                depth = depth_augmentation(
                    depth,
                    noise_std=self.depth_aug_config.get('noise_std', 0.01),
                    dropout_prob=self.depth_aug_config.get('dropout_prob', 0.1)
                )
            
            return depth
            
        except Exception as e:
            if self.require_depth:
                raise e
            else:
                # 如果不要求深度图，返回零深度图
                print(f"警告: 无法加载深度图 {depth_path}, 使用零深度图")
                return np.zeros((224, 224, 1), dtype=np.float32)
    
    def decode(self, item):
        """
        解码数据项，添加深度图处理
        """
        # 调用父类的decode方法
        sample = super().decode(item)
        
        # 自动构建深度图路径
        uid = item.split('/')[-1]
        render_cond_dir = os.path.join(item, 'render_cond')
        
        # 为每个RGB图像构建对应的深度图路径
        depth_images = []
        rgb_images = sample[self.cond_stage_key]  # 这是RGB图像路径列表
        
        for rgb_path in rgb_images:
            # 从RGB路径构建深度图路径 (例如: 000.png -> 000_depth.png)
            rgb_filename = os.path.basename(rgb_path)
            depth_filename = rgb_filename.replace('.png', '_depth.png')
            depth_path = os.path.join(render_cond_dir, depth_filename)
            
            if os.path.exists(depth_path):
                try:
                    depth = self.load_depth(depth_path)
                    depth_images.append(depth)
                except Exception as e:
                    if self.require_depth:
                        raise ValueError(f"无法加载深度图 {depth_path}: {e}")
                    else:
                        # 创建零深度图
                        depth_images.append(np.zeros((224, 224, 1), dtype=np.float32))
            else:
                if self.require_depth:
                    raise ValueError(f"深度图不存在: {depth_path}")
                else:
                    # 创建零深度图
                    depth_images.append(np.zeros((224, 224, 1), dtype=np.float32))
        
        sample[self.depth_stage_key] = depth_images
        return sample
    
    def transform(self, sample):
        """
        应用变换，包括深度图变换
        """
        # 获取RGB图像和深度图
        image = sample[self.cond_stage_key]
        depth = sample[self.depth_stage_key]
        
        # 确保RGB和深度图尺寸一致
        if image.shape[:2] != depth.shape[:2]:
            depth = cv2.resize(depth, (image.shape[1], image.shape[0]))
            if len(depth.shape) == 2:
                depth = depth[:, :, np.newaxis]
        
        # 应用padding（同时处理RGB和深度图）
        if self.padding:
            # 为深度图创建mask（非零区域）
            depth_mask = (depth[:, :, 0] > 0).astype(np.uint8) * 255
            
            # 对RGB图像应用padding
            image, image_mask = padding(
                image, 
                np.ones(image.shape[:2], dtype=np.uint8) * 255,
                padding_ratio_range=self.padding_ratio_range
            )
            
            # 对深度图应用相同的padding
            depth, depth_mask = padding(
                depth,
                depth_mask,
                padding_ratio_range=self.padding_ratio_range
            )
        
        # 应用图像变换
        if self.image_transform is not None:
            image = self.image_transform(image)
        
        # 应用深度图变换
        if self.depth_transform is not None:
            depth = self.depth_transform(depth)
        else:
            # 默认深度图变换：转换为tensor并归一化
            depth = torch.from_numpy(depth).permute(2, 0, 1).float()
            if self.depth_normalize:
                # 归一化到[-1, 1]
                depth = depth * 2.0 - 1.0
        
        # 更新sample
        sample[self.cond_stage_key] = image
        sample[self.depth_stage_key] = depth
        
        return sample


class RGBDAlignedShapeLatentModule(LightningDataModule):
    """
    RGBD数据模块，支持RGB+深度图的训练
    """
    
    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 4,
        val_num_workers: int = 2,
        train_data_list: str = None,
        val_data_list: str = None,
        cond_stage_key: str = "image",
        depth_stage_key: str = "depth",
        image_size: int = 224,
        mean: Union[List[float], Tuple[float]] = (0.485, 0.456, 0.406),
        std: Union[List[float], Tuple[float]] = (0.229, 0.224, 0.225),
        depth_mean: Union[List[float], Tuple[float]] = (0.5,),
        depth_std: Union[List[float], Tuple[float]] = (0.5,),
        pc_size: int = 2048,
        pc_sharpedge_size: int = 2048,
        sharpedge_label: bool = False,
        return_normal: bool = False,
        padding=True,
        padding_ratio_range=[1.15, 1.15],
        # 深度图相关参数
        depth_clip_range=[0.0, 10.0],
        depth_normalize=True,
        depth_augmentation=None,
        require_depth=True,
    ):
        super().__init__()
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_num_workers = val_num_workers
        self.train_data_list = train_data_list
        self.val_data_list = val_data_list
        self.cond_stage_key = cond_stage_key
        self.depth_stage_key = depth_stage_key
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.depth_mean = depth_mean
        self.depth_std = depth_std
        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size
        self.sharpedge_label = sharpedge_label
        self.return_normal = return_normal
        self.padding = padding
        self.padding_ratio_range = padding_ratio_range
        self.depth_clip_range = depth_clip_range
        self.depth_normalize = depth_normalize
        self.depth_augmentation = depth_augmentation
        self.require_depth = require_depth
        
        # 图像变换
        self.image_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # 深度图变换
        self.depth_transform = transforms.Compose([
            transforms.ToPILImage(mode='F'),
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.depth_mean, std=self.depth_std)
        ])
    
    def train_dataloader(self):
        dataset = RGBDAlignedShapeLatentDataset(
            data_list=self.train_data_list,
            cond_stage_key=self.cond_stage_key,
            depth_stage_key=self.depth_stage_key,
            image_transform=self.image_transform,
            depth_transform=self.depth_transform,
            pc_size=self.pc_size,
            pc_sharpedge_size=self.pc_sharpedge_size,
            sharpedge_label=self.sharpedge_label,
            return_normal=self.return_normal,
            padding=self.padding,
            padding_ratio_range=self.padding_ratio_range,
            depth_clip_range=self.depth_clip_range,
            depth_normalize=self.depth_normalize,
            depth_augmentation_config=self.depth_augmentation,
            require_depth=self.require_depth,
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def val_dataloader(self):
        dataset = RGBDAlignedShapeLatentDataset(
            data_list=self.val_data_list,
            cond_stage_key=self.cond_stage_key,
            depth_stage_key=self.depth_stage_key,
            image_transform=self.image_transform,
            depth_transform=self.depth_transform,
            pc_size=self.pc_size,
            pc_sharpedge_size=self.pc_sharpedge_size,
            sharpedge_label=self.sharpedge_label,
            return_normal=self.return_normal,
            padding=self.padding,
            padding_ratio_range=self.padding_ratio_range,
            depth_clip_range=self.depth_clip_range,
            depth_normalize=self.depth_normalize,
            depth_augmentation_config=None,  # 验证时不使用数据增强
            require_depth=self.require_depth,
            deterministic=True,  # 验证时使用确定性采样
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.val_num_workers,
            worker_init_fn=worker_init_fn,
            persistent_workers=True if self.val_num_workers > 0 else False,
        )