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

"""
Multi-view data loader for training with front/back/left/right views and corresponding depth maps
"""

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

from .utils import worker_init_fn, pytorch_worker_seed, make_seed
from .dit_asl import ResampledShards, read_npz, read_json, padding, viz_pc


class AlignedShapeLatentMultiViewDataset(torch.utils.data.dataset.IterableDataset):
    def __init__(
        self,
        data_list: str = None,
        cond_stage_key: str = "image",
        image_transform = None,
        pc_size: int = 2048,
        pc_sharpedge_size: int = 2048,
        sharpedge_label: bool = False,
        return_normal: bool = False,
        deterministic = False,
        worker_seed = None,
        padding = True,
        padding_ratio_range=[1.15, 1.15],
        load_depth: bool = False,
        multiview_indices: List[int] = [0, 6, 12, 18],  # 前后左右四个视图的索引 (0°, 90°, 180°, 270°)
        depth_fusion_strategy: str = "multiview"  # "multiview" (keep all views), "average", "max", "weighted"
    ):
        """
        Multi-view dataset for loading front/back/left/right views with depth maps
        
        Args:
            multiview_indices: List of camera view indices to use. Default [0, 6, 12, 18] 
                              corresponds to front, right, back, left views (every 90 degrees)
                              Total 24 views means 360/24 = 15 degrees per view
        """
        super().__init__()
        if isinstance(data_list, str) and data_list.endswith('.json'):
            self.data_list = read_json(data_list)
        elif isinstance(data_list, str) and os.path.isdir(data_list):
            self.data_list = glob.glob(data_list + '/*')
        else:
            self.data_list = data_list
        assert isinstance(self.data_list, list)
        self.rng = random.Random(0)
        
        self.cond_stage_key = cond_stage_key
        self.image_transform = image_transform
        
        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size
        self.sharpedge_label = sharpedge_label
        self.return_normal = return_normal

        self.padding = padding
        self.padding_ratio_range = padding_ratio_range
        self.load_depth = load_depth
        self.multiview_indices = multiview_indices
        self.depth_fusion_strategy = depth_fusion_strategy
        
        rank_zero_info(f'*' * 50)
        rank_zero_info(f'Multi-View Dataset Infos:')
        rank_zero_info(f'# of 3D file: {len(self.data_list)}')
        rank_zero_info(f'# of Surface Points: {self.pc_size}')
        rank_zero_info(f'# of Sharpedge Surface Points: {self.pc_sharpedge_size}')
        rank_zero_info(f'Using sharp edge label: {self.sharpedge_label}')
        rank_zero_info(f'Load depth maps: {self.load_depth}')
        rank_zero_info(f'Multi-view indices: {self.multiview_indices} (total {len(self.multiview_indices)} views)')
        rank_zero_info(f'*' * 50)

    def load_surface_sdf_points(self, rng, random_surface, sharpedge_surface):
        """Same as original implementation"""
        surface_normal = []
        if self.pc_size > 0:
            ind = rng.choice(random_surface.shape[0], self.pc_size, replace=False)
            random_surface = random_surface[ind]
            if self.sharpedge_label:
                sharpedge_label = np.zeros((self.pc_size, 1))
                random_surface = np.concatenate((random_surface, sharpedge_label), axis=1)
            surface_normal.append(random_surface)
            
        if self.pc_sharpedge_size > 0:
            ind_sharpedge = rng.choice(sharpedge_surface.shape[0], self.pc_sharpedge_size, replace=False)
            sharpedge_surface = sharpedge_surface[ind_sharpedge]
            if self.sharpedge_label:
                sharpedge_label = np.ones((self.pc_sharpedge_size, 1))
                sharpedge_surface = np.concatenate((sharpedge_surface, sharpedge_label), axis=1)
            surface_normal.append(sharpedge_surface)
            
        surface_normal = np.concatenate(surface_normal, axis=0)
        surface_normal = torch.FloatTensor(surface_normal)
        surface = surface_normal[:, 0:3]
        normal = surface_normal[:, 3:6]
        assert surface.shape[0] == self.pc_size + self.pc_sharpedge_size
        
        geo_points = 0.0
        normal = torch.nn.functional.normalize(normal, p=2, dim=1)
        if self.return_normal:
            surface = torch.cat([surface, normal], dim=-1)
        if self.sharpedge_label:
            surface = torch.cat([surface, surface_normal[:, -1:]], dim=-1)
        return surface, geo_points

    def load_multiview_render(self, imgs_path):
        """Load multi-view images based on specified indices"""
        # 选择指定的多视图索引
        imgs_choice = [imgs_path[i] for i in self.multiview_indices]
        images, masks = [], []
        
        for image_path in imgs_choice:
            if not os.path.exists(image_path):
                rank_zero_info(f"Warning: Image path does not exist: {image_path}")
                continue
                
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            assert image.shape[2] == 4
            alpha = image[:, :, 3:4].astype(np.float32) / 255
            forground = image[:, :, :3]
            background = np.ones_like(forground) * 255
            img_new = forground * alpha + background * (1 - alpha)
            image = img_new.astype(np.uint8)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = (alpha[:, :, 0] * 255).astype(np.uint8)

            if self.padding:
                h, w = image.shape[:2]
                binary = mask > 0.3
                if np.any(binary):
                    non_zero_coords = np.argwhere(binary)
                    x_min, y_min = non_zero_coords.min(axis=0)
                    x_max, y_max = non_zero_coords.max(axis=0)
                    image, mask = padding(
                        image[max(x_min - 5, 0):min(x_max + 5, h), max(y_min - 5, 0):min(y_max + 5, w)],
                        mask[max(x_min - 5, 0):min(x_max + 5, h), max(y_min - 5, 0):min(y_max + 5, w)],
                        center=True, padding_ratio_range=self.padding_ratio_range)
            
            if self.image_transform:
                image = self.image_transform(image)
                mask = np.stack((mask, mask, mask), axis=-1)
                mask = self.image_transform(mask)
                
            images.append(image)
            masks.append(mask)
            
        # 将多个视图沿batch维度拼接
        images = torch.stack(images, dim=0)  # Shape: (num_views, C, H, W)
        masks = torch.stack(masks, dim=0)[:, :1, ...]  # Shape: (num_views, 1, H, W)
        return images, masks

    def load_multiview_depth_maps(self, depth_paths):
        """Load multi-view depth maps based on specified indices"""
        # 选择指定的多视图深度图索引
        depth_choice = [depth_paths[i] for i in self.multiview_indices]
        depths = []
        
        for depth_path in depth_choice:
            if not os.path.exists(depth_path):
                rank_zero_info(f"Warning: Depth path does not exist: {depth_path}")
                # 创建零深度图作为占位符
                depths.append(torch.zeros(1, 512, 512))  # Default size, will be resized by transform
                continue
                
            # Read EXR depth map using OpenEXR or OpenCV
            try:
                import OpenEXR
                import Imath
                
                # Open EXR file
                exr_file = OpenEXR.InputFile(depth_path)
                header = exr_file.header()
                
                # Get image dimensions
                dw = header['dataWindow']
                width = dw.max.x - dw.min.x + 1
                height = dw.max.y - dw.min.y + 1
                
                # Read depth channel (try different channel names)
                channel_names = ['R', 'G', 'B', 'Y', 'Z', 'depth']
                depth_channel = None
                
                for channel_name in channel_names:
                    if channel_name in header['channels']:
                        depth_channel = channel_name
                        break
                
                if depth_channel is None:
                    # If no standard channel found, use the first available channel
                    available_channels = list(header['channels'].keys())
                    if available_channels:
                        depth_channel = available_channels[0]
                    else:
                        raise ValueError(f"No channels found in EXR file: {depth_path}")
                
                # Read the depth channel
                depth_str = exr_file.channel(depth_channel, Imath.PixelType(Imath.PixelType.FLOAT))
                depth = np.frombuffer(depth_str, dtype=np.float32)
                depth = depth.reshape((height, width))
                # Create a writable copy to avoid "assignment destination is read-only" error
                depth = depth.copy()
                
                exr_file.close()
                
            except ImportError:
                # Fallback to OpenCV if OpenEXR is not available
                depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
                if depth is None:
                    rank_zero_info(f"Warning: Failed to load depth map: {depth_path}")
                    depths.append(torch.zeros(1, 512, 512))  # Default size
                    continue
                
                # Extract depth channel (assuming single channel or first channel)
                if len(depth.shape) == 3:
                    depth = depth[:, :, 0]
            
            # Filter invalid depth values (very large values)
            depth[depth > 1e9] = 0
            
            # Normalize depth to [0, 1] range
            if depth.max() > depth.min():
                depth = (depth - depth.min()) / (depth.max() - depth.min())
            
            # Apply same padding as images if enabled
            if self.padding:
                # For depth, we don't have a mask, so we create one from non-zero values
                mask = (depth > 0).astype(np.uint8) * 255
                h, w = depth.shape[:2]
                binary = mask > 0.3
                if np.any(binary):
                    non_zero_coords = np.argwhere(binary)
                    x_min, y_min = non_zero_coords.min(axis=0)
                    x_max, y_max = non_zero_coords.max(axis=0)
                    # Crop the depth and mask
                    cropped_depth = depth[max(x_min - 5, 0):min(x_max + 5, h), max(y_min - 5, 0):min(y_max + 5, w)]
                    cropped_mask = mask[max(x_min - 5, 0):min(x_max + 5, h), max(y_min - 5, 0):min(y_max + 5, w)]
                    
                    # Convert single-channel depth to 3-channel for padding function compatibility
                    cropped_depth_3ch = np.stack([cropped_depth, cropped_depth, cropped_depth], axis=-1)
                    
                    # Apply padding
                    padded_depth_3ch, _ = padding(
                        cropped_depth_3ch,
                        cropped_mask,
                        padding_ratio_range=self.padding_ratio_range
                    )
                    
                    # Convert back to single channel (take first channel)
                    depth = padded_depth_3ch[:, :, 0].astype(np.float32)
            
            # Apply image transform if available (resize, normalize) - do this on numpy array
            if self.image_transform:
                # Convert single-channel depth to 3-channel numpy array for transform compatibility
                depth_3ch = np.stack([depth, depth, depth], axis=-1)
                # Convert to PIL Image for transform
                from PIL import Image
                depth_pil = Image.fromarray((depth_3ch * 255).astype(np.uint8))
                depth_pil = self.image_transform(depth_pil)
                # Convert back to numpy and extract single channel
                depth_np = np.array(depth_pil).astype(np.float32) / 255.0
                if len(depth_np.shape) == 3:
                    depth = depth_np[:, :, 0]  # Take first channel
                else:
                    depth = depth_np
            
            # Convert to tensor and add channel dimension
            depth = torch.FloatTensor(depth).unsqueeze(0)  # Shape: (1, H, W)
            depths.append(depth)
        
        # 将多个深度图沿batch维度拼接，保持多视图格式
        depths = torch.stack(depths, dim=0)  # Shape: (num_views, 1, H, W)
        
        # 根据融合策略处理深度数据
        if self.depth_fusion_strategy == "multiview":
            # 保持多视图格式，供MultiViewDepthControlNet处理
            return depths  # Shape: (num_views, 1, H, W)
        elif self.depth_fusion_strategy == "average":
            # 平均融合多个视图
            return depths.mean(dim=0, keepdim=True)  # Shape: (1, 1, H, W)
        elif self.depth_fusion_strategy == "max":
            # 取最大值融合
            return depths.max(dim=0, keepdim=True)[0]  # Shape: (1, 1, H, W)
        elif self.depth_fusion_strategy == "weighted":
            # 简单的加权融合（后续可以改为学习的权重）
            weights = torch.tensor([0.3, 0.2, 0.3, 0.2], device=depths.device).view(-1, 1, 1, 1)
            weighted = (depths * weights[:len(depths)]).sum(dim=0, keepdim=True)
            return weighted  # Shape: (1, 1, H, W)
        else:
            # 默认返回多视图格式
            return depths

    def decode(self, item):
        """Same as original but load all 24 views for selection"""
        uid = item.split('/')[-1]
        render_img_paths = [os.path.join(item, f'render_cond/{i:03d}.png') for i in range(24)]
        surface_npz_path = os.path.join(item, f'geo_data/{uid}_surface.npz')
        
        sample = {}
        sample["image"] = render_img_paths
        
        # Load depth maps if enabled
        if self.load_depth:
            depth_img_paths = [os.path.join(item, f'render_cond/{i:03d}_depth.exr') for i in range(24)]
            sample["depth"] = depth_img_paths
        
        surface_data = read_npz(surface_npz_path)
        sample["random_surface"] = surface_data['random_surface']
        sample["sharpedge_surface"] = surface_data['sharp_surface']
        return sample

    def transform(self, sample):
        """Transform sample with multi-view loading"""
        rng = np.random.default_rng()
        random_surface = sample.get("random_surface", 0)
        sharpedge_surface = sample.get("sharpedge_surface", 0)
        
        # Load multi-view images
        image_input, mask_input = self.load_multiview_render(sample['image'])
        surface, geo_points = self.load_surface_sdf_points(rng, random_surface, sharpedge_surface)
        
        result_sample = {
            "surface": surface,
            "geo_points": geo_points,
            "image": image_input,  # Shape: (num_views, C, H, W)
            "mask": mask_input,    # Shape: (num_views, 1, H, W)
        }
        
        # Load and process multi-view depth maps if enabled
        if self.load_depth and "depth" in sample:
            depth_input = self.load_multiview_depth_maps(sample['depth'])
            result_sample["depth"] = depth_input  # Shape: (num_views, 1, H, W) - multi-view depths
        
        return result_sample

    def __iter__(self):
        total_num = 0
        failed_num = 0
        for data in ResampledShards(self.data_list):
            total_num += 1
            if total_num % 1000 == 0:
                print(f"Current failure rate of multi-view data loading:")
                print(f"{failed_num}/{total_num}={failed_num/total_num}")
            try:
                sample = self.decode(data)
                sample = self.transform(sample)
            except Exception as err:
                print(f"Multi-view loading error: {err}")
                failed_num += 1
                continue
            yield sample


class AlignedShapeLatentMultiViewModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 4,
        val_num_workers: int = 2,
        train_data_list: str = None,
        val_data_list: str = None,
        cond_stage_key: str = "all",
        image_size: int = 224,
        mean: Union[List[float], Tuple[float]] = (0.485, 0.456, 0.406),
        std: Union[List[float], Tuple[float]] = (0.229, 0.224, 0.225),
        pc_size: int = 2048,
        pc_sharpedge_size: int = 2048,
        sharpedge_label: bool = False,
        return_normal: bool = False, 
        padding = True,
        padding_ratio_range=[1.15, 1.15],
        load_depth: bool = False,
        multiview_indices: List[int] = [0, 6, 12, 18],  # 前后左右四个视图
        depth_fusion_strategy: str = "multiview"  # 深度融合策略
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_num_workers = val_num_workers

        self.train_data_list = train_data_list
        self.val_data_list = val_data_list
        
        self.cond_stage_key = cond_stage_key
        self.image_size = image_size
        self.mean = mean
        self.std = std
        self.train_image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size),
            transforms.Normalize(mean=self.mean, std=self.std)])
        self.val_image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.image_size),
            transforms.Normalize(mean=self.mean, std=self.std)])

        self.pc_size = pc_size
        self.pc_sharpedge_size = pc_sharpedge_size
        self.sharpedge_label = sharpedge_label
        self.return_normal = return_normal

        self.padding = padding
        self.padding_ratio_range = padding_ratio_range
        self.load_depth = load_depth
        self.multiview_indices = multiview_indices
        self.depth_fusion_strategy = depth_fusion_strategy
        
    def train_dataloader(self):
        asl_params = {
            "data_list": self.train_data_list,
            "cond_stage_key": self.cond_stage_key,
            "image_transform": self.train_image_transform,
            "pc_size": self.pc_size,
            "pc_sharpedge_size": self.pc_sharpedge_size,
            "sharpedge_label": self.sharpedge_label,
            "return_normal": self.return_normal,
            "padding": self.padding,
            "padding_ratio_range": self.padding_ratio_range,
            "load_depth": self.load_depth,
            "multiview_indices": self.multiview_indices,
            "depth_fusion_strategy": self.depth_fusion_strategy
        }
        dataset = AlignedShapeLatentMultiViewDataset(**asl_params)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )

    def val_dataloader(self):
        asl_params = {
            "data_list": self.val_data_list,
            "cond_stage_key": self.cond_stage_key,
            "image_transform": self.val_image_transform,
            "pc_size": self.pc_size,
            "pc_sharpedge_size": self.pc_sharpedge_size,
            "sharpedge_label": self.sharpedge_label,
            "return_normal": self.return_normal, 
            "padding": self.padding,
            "padding_ratio_range": self.padding_ratio_range,
            "load_depth": self.load_depth,
            "multiview_indices": self.multiview_indices,
            "depth_fusion_strategy": self.depth_fusion_strategy
        }
        dataset = AlignedShapeLatentMultiViewDataset(**asl_params)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.val_num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )
