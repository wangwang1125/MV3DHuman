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
    elif depth_path.endswith('.exr'):
        # 读取EXR格式深度图
        import OpenEXR
        import Imath
        try:
            exr_file = OpenEXR.InputFile(depth_path)
            dw = exr_file.header()['dataWindow']
            size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
            
            # 读取深度通道 (通常是R通道)
            depth_str = exr_file.channel('R', Imath.PixelType(Imath.PixelType.FLOAT))
            depth = np.frombuffer(depth_str, dtype=np.float32)
            depth = depth.reshape(size[1], size[0])
        except Exception as e:
            raise ValueError(f"无法加载EXR深度图 {depth_path}: {e}")
    else:
        # 假设是PNG等其他图像格式
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
        # 深度图归一化参数
        depth_mean=0.5,
        depth_std=0.5,
        image_size=224,
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
        self.depth_mean = depth_mean
        self.depth_std = depth_std
        self.image_size = image_size
        
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
            # 从RGB路径构建深度图路径 (例如: 000.png -> 000_depth.exr)
            rgb_filename = os.path.basename(rgb_path)
            depth_filename = rgb_filename.replace('.png', '_depth.exr')
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
        # 首先处理深度图数据
        rgb_images = sample[self.cond_stage_key]  # 这是RGB图像路径列表
        depth_images = sample[self.depth_stage_key]  # 这是深度图数组列表
        
        # 选择一个RGB图像和对应的深度图
        selected_idx = self.rng.randint(0, len(rgb_images) - 1)
        selected_rgb_path = rgb_images[selected_idx]
        selected_depth = depth_images[selected_idx]
        
        # 加载RGB图像（使用父类的load_render方法）
        image_input, mask_input = self.load_render([selected_rgb_path])
        
        # 处理深度图
        depth = selected_depth
        
        # 确保RGB和深度图尺寸一致
        if len(image_input.shape) == 4:  # (1, C, H, W)
            rgb_h, rgb_w = image_input.shape[2], image_input.shape[3]
        else:  # (C, H, W)
            rgb_h, rgb_w = image_input.shape[1], image_input.shape[2]
            
        if depth.shape[:2] != (rgb_h, rgb_w):
            depth = cv2.resize(depth, (rgb_w, rgb_h))
            if len(depth.shape) == 2:
                depth = depth[:, :, np.newaxis]
        
        # 安全的深度图变换处理
        # 确保深度图是numpy数组
        if isinstance(depth, torch.Tensor):
            depth = depth.numpy()
        
        # 确保深度图形状正确 (H, W, 1)
        if len(depth.shape) == 2:
            depth = depth[:, :, np.newaxis]
        elif len(depth.shape) == 3 and depth.shape[2] != 1:
            depth = depth[:, :, 0:1]  # 只取第一个通道
        
        # 调整尺寸到目标大小
        target_size = getattr(self, 'image_size', 224)
        if depth.shape[:2] != (target_size, target_size):
            depth = cv2.resize(depth, (target_size, target_size))
            if len(depth.shape) == 2:
                depth = depth[:, :, np.newaxis]
        
        # 转换为tensor
        depth = torch.from_numpy(depth.copy()).permute(2, 0, 1).float()
        
        # 归一化处理
        if self.depth_normalize:
            # 使用配置的均值和标准差进行归一化
            depth_mean = getattr(self, 'depth_mean', 0.5)
            depth_std = getattr(self, 'depth_std', 0.5)
            
            # 处理ListConfig类型（来自Hydra配置）
            if hasattr(depth_mean, '__iter__') and not isinstance(depth_mean, (str, torch.Tensor)):
                depth_mean = float(depth_mean[0]) if len(depth_mean) > 0 else 0.5
            else:
                depth_mean = float(depth_mean)
                
            if hasattr(depth_std, '__iter__') and not isinstance(depth_std, (str, torch.Tensor)):
                depth_std = float(depth_std[0]) if len(depth_std) > 0 else 0.5
            else:
                depth_std = float(depth_std)
            
            depth = (depth - depth_mean) / depth_std
        
        # 处理点云数据（调用父类方法）
        rng = np.random.default_rng()
        random_surface = sample.get("random_surface", 0)
        sharpedge_surface = sample.get("sharpedge_surface", 0)
        surface, geo_points = self.load_surface_sdf_points(rng, random_surface, sharpedge_surface)
        
        # 构建最终的sample
        final_sample = {
            "surface": surface,
            "geo_points": geo_points,
            "image": image_input,
            "mask": mask_input,
            self.depth_stage_key: depth,
        }
        
        return final_sample


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
        
        # 图像变换（与父类保持一致）
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
        
        # 深度图变换（避免PIL模式问题）
        self.depth_transform = None  # 在transform方法中手动处理
    
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
            depth_mean=self.depth_mean[0] if isinstance(self.depth_mean, (list, tuple)) else self.depth_mean,
            depth_std=self.depth_std[0] if isinstance(self.depth_std, (list, tuple)) else self.depth_std,
            image_size=self.image_size,
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
            depth_mean=self.depth_mean[0] if isinstance(self.depth_mean, (list, tuple)) else self.depth_mean,
            depth_std=self.depth_std[0] if isinstance(self.depth_std, (list, tuple)) else self.depth_std,
            image_size=self.image_size,
        )
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.val_num_workers,
            worker_init_fn=worker_init_fn,
            persistent_workers=True if self.val_num_workers > 0 else False,
        )


def test_depth_transform():
    """
    测试深度图转换功能
    """
    print("=" * 60)
    print("开始深度图转换测试")
    print("=" * 60)
    
    # 测试1: 创建模拟深度图数据
    print("\n1. 测试深度图数据创建...")
    try:
        # 创建模拟深度图 (224, 224, 1)
        test_depth = np.random.rand(224, 224, 1).astype(np.float32)
        print(f"✓ 创建模拟深度图成功: shape={test_depth.shape}, dtype={test_depth.dtype}")
        print(f"  深度值范围: [{test_depth.min():.3f}, {test_depth.max():.3f}]")
    except Exception as e:
        print(f"✗ 创建模拟深度图失败: {e}")
        return False
    
    # 测试2: 测试深度图加载函数
    print("\n2. 测试深度图加载函数...")
    try:
        # 创建临时深度图文件
        temp_depth_path = "/tmp/test_depth.npy"
        np.save(temp_depth_path, test_depth[:, :, 0])  # 保存为2D数组
        
        # 测试加载
        loaded_depth = load_depth_image(temp_depth_path, depth_clip_range=[0.0, 1.0])
        print(f"✓ 深度图加载成功: shape={loaded_depth.shape}, dtype={loaded_depth.dtype}")
        print(f"  加载后深度值范围: [{loaded_depth.min():.3f}, {loaded_depth.max():.3f}]")
        
        # 清理临时文件
        os.remove(temp_depth_path)
    except Exception as e:
        print(f"✗ 深度图加载测试失败: {e}")
        return False
    
    # 测试3: 测试深度图数据增强
    print("\n3. 测试深度图数据增强...")
    try:
        augmented_depth = depth_augmentation(
            test_depth, 
            noise_std=0.01, 
            dropout_prob=0.1
        )
        print(f"✓ 深度图数据增强成功: shape={augmented_depth.shape}")
        print(f"  增强后深度值范围: [{augmented_depth.min():.3f}, {augmented_depth.max():.3f}]")
    except Exception as e:
        print(f"✗ 深度图数据增强测试失败: {e}")
        return False
    
    # 测试4: 测试深度图变换处理（模拟transform方法中的逻辑）
    print("\n4. 测试深度图变换处理...")
    try:
        depth = test_depth.copy()
        
        # 确保深度图是numpy数组
        if isinstance(depth, torch.Tensor):
            depth = depth.numpy()
        
        # 确保深度图形状正确 (H, W, 1)
        if len(depth.shape) == 2:
            depth = depth[:, :, np.newaxis]
        elif len(depth.shape) == 3 and depth.shape[2] != 1:
            depth = depth[:, :, 0:1]
        
        # 调整尺寸到目标大小
        target_size = 224
        if depth.shape[:2] != (target_size, target_size):
            depth = cv2.resize(depth, (target_size, target_size))
            if len(depth.shape) == 2:
                depth = depth[:, :, np.newaxis]
        
        # 转换为tensor
        depth_tensor = torch.from_numpy(depth.copy()).permute(2, 0, 1).float()
        print(f"✓ 深度图转换为tensor成功: shape={depth_tensor.shape}")
        
        # 归一化处理
        depth_mean = 0.5
        depth_std = 0.5
        
        # 处理ListConfig类型（模拟配置情况）
        if hasattr(depth_mean, '__iter__') and not isinstance(depth_mean, (str, torch.Tensor)):
            depth_mean = float(depth_mean[0]) if len(depth_mean) > 0 else 0.5
        else:
            depth_mean = float(depth_mean)
            
        if hasattr(depth_std, '__iter__') and not isinstance(depth_std, (str, torch.Tensor)):
            depth_std = float(depth_std[0]) if len(depth_std) > 0 else 0.5
        else:
            depth_std = float(depth_std)
        
        normalized_depth = (depth_tensor - depth_mean) / depth_std
        print(f"✓ 深度图归一化成功: shape={normalized_depth.shape}")
        print(f"  归一化后深度值范围: [{normalized_depth.min():.3f}, {normalized_depth.max():.3f}]")
        print(f"  使用的归一化参数: mean={depth_mean}, std={depth_std}")
        
    except Exception as e:
        print(f"✗ 深度图变换处理测试失败: {e}")
        traceback.print_exc()
        return False
    
    # 测试5: 测试RGBDAlignedShapeLatentDataset初始化
    print("\n5. 测试RGBDAlignedShapeLatentDataset初始化...")
    try:
        # 创建测试数据集（不需要实际数据文件）
        dataset = RGBDAlignedShapeLatentDataset(
            data_list=None,  # 不提供数据列表，只测试初始化
            depth_mean=0.5,
            depth_std=0.5,
            image_size=224,
            require_depth=False  # 不要求深度图，避免文件不存在的问题
        )
        print(f"✓ RGBDAlignedShapeLatentDataset初始化成功")
        print(f"  depth_mean: {dataset.depth_mean}")
        print(f"  depth_std: {dataset.depth_std}")
        print(f"  image_size: {dataset.image_size}")
        
    except Exception as e:
        print(f"✗ RGBDAlignedShapeLatentDataset初始化测试失败: {e}")
        traceback.print_exc()
        return False
    
    # 测试6: 测试RGBDAlignedShapeLatentModule初始化
    print("\n6. 测试RGBDAlignedShapeLatentModule初始化...")
    try:
        # 创建测试数据模块
        data_module = RGBDAlignedShapeLatentModule(
            batch_size=1,
            depth_mean=[0.5],
            depth_std=[0.5],
            image_size=224,
            require_depth=False
        )
        print(f"✓ RGBDAlignedShapeLatentModule初始化成功")
        print(f"  depth_mean: {data_module.depth_mean}")
        print(f"  depth_std: {data_module.depth_std}")
        print(f"  image_size: {data_module.image_size}")
        
    except Exception as e:
        print(f"✗ RGBDAlignedShapeLatentModule初始化测试失败: {e}")
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ 所有深度图转换测试通过！")
    print("=" * 60)
    return True


if __name__ == "__main__":
    # 运行深度图转换测试
    test_depth_transform()