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


# 与 MVImageProcessorV2 / view_order 一致，用于 collate 时统一视图子集
_COLLATE_VIEW_ORDER = ['front', 'left', 'back', 'right']


def multiview_collate_fn(batch):
    """
    自定义 collate_fn，用于处理多视图数据。
    将多个样本的 image_dict 和 normal_dict 合并为 list of dicts。
    当启用 sample_num_views 时：先为本 batch 确定目标视图数 k（在 1～4 中随机），
    再将该 batch 内所有样本统一为 k 个视图（多于 k 的随机裁到 k；若 batch 中最小视图数 < k，则 k 取该最小值，避免填充）。
    视角位置编码仍使用原始 view 索引（如 0=front, 2=back），不会变成“左视图”。
    """
    if len(batch) == 0:
        return {}
    
    # 先为本 batch 确定目标视图数 k（1～4 随机），再与 batch 内最小视图数取 min，保证不需填充
    num_views_per_sample = [len(s.get('image') or {}) for s in batch]
    min_views_in_batch = min(num_views_per_sample)
    if min_views_in_batch <= 0:
        raise ValueError("batch 中存在无有效 image 的样本")
    k = min(random.choice([1, 2, 3, 4]), min_views_in_batch)
    
    for sample in batch:
        img_dict = sample.get('image') or {}
        if len(img_dict) <= k:
            continue
        # 当前样本视图（保持插入顺序，与 depth 维度对应）
        original_views = [v for v in _COLLATE_VIEW_ORDER if v in img_dict]
        if len(original_views) <= k:
            continue
        # 随机保留 k 个视图，并保持 view_order 顺序
        chosen = sorted(random.sample(range(len(original_views)), k))
        views_to_keep = [original_views[i] for i in chosen]
        sample['image'] = {v: img_dict[v] for v in views_to_keep}
        # 同步截取 depth：depth 的 dim0 与 original_views 顺序一致
        if 'depth' in sample and isinstance(sample['depth'], torch.Tensor):
            indices_to_keep = [original_views.index(v) for v in views_to_keep]
            sample['depth'] = sample['depth'][indices_to_keep]
        if 'normal' in sample and isinstance(sample['normal'], dict):
            sample['normal'] = {v: sample['normal'][v] for v in views_to_keep if v in sample['normal']}
    
    # 收集所有字段
    collated = {}
    for key in batch[0].keys():
        values = [sample.get(key) for sample in batch]
        
        # 过滤 None 值
        values = [v for v in values if v is not None]
        if len(values) == 0:
            continue
        
        # 对于 image 和 normal（如果是 dict），保留为 list of dicts
        if key in ['image', 'normal'] and isinstance(values[0], dict):
            collated[key] = values  # 保留为 list of dicts
        # 对于其他字段（tensor），使用默认的 stack/cat
        elif isinstance(values[0], torch.Tensor):
            if values[0].dim() == 0:
                collated[key] = torch.stack(values, dim=0)
            else:
                collated[key] = torch.stack(values, dim=0) if values[0].dim() > 1 else torch.cat(values, dim=0)
        else:
            collated[key] = values
    
    return collated


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
        padding_ratio_range=[1, 1],
        load_depth: bool = False,
        load_normal: bool = False,
        multiview_indices: List[int] = [0, 6, 12, 18],  # 前后左右四个视图的索引 (0°, 90°, 180°, 270°)
        depth_fusion_strategy: str = "multiview",  # "multiview" (keep all views), "average", "max", "weighted"
        normal_fusion_strategy: str = "multiview",  # For normal maps, keep all views for DinoImageEncoderMV
        sample_num_views: Optional[Union[str, List[int]]] = None,  # None/4: 固定4视图; "random" 或 [1,2,3,4]: 每样本随机1~4视图
    ):
        """
        Multi-view dataset for loading front/back/left/right views with depth maps and normal maps
        
        Args:
            multiview_indices: List of camera view indices to use. Default [0, 6, 12, 18] 
                              corresponds to front, right, back, left views (every 90 degrees)
                              Total 24 views means 360/24 = 15 degrees per view
            sample_num_views: If None or 4, use all 4 views. If "random" or [1,2,3,4], each sample
                              randomly uses 1, 2, 3, or 4 views (view position encoding is preserved).
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
        self.load_normal = load_normal
        self.multiview_indices = multiview_indices
        self.depth_fusion_strategy = depth_fusion_strategy
        self.normal_fusion_strategy = normal_fusion_strategy
        self.sample_num_views = sample_num_views  # None/4: 全4视图; "random" 或 [1,2,3,4]: 随机1~4视图
        
        # 映射视图索引到视图名称（与 MVImageProcessorV2 的 view2idx 一致）
        # 假设 multiview_indices 的顺序是 [front_idx, left_idx, back_idx, right_idx]
        # 如果 multiview_indices 是 [0, 1, 2, 3]，则对应 front, left, back, right
        self.idx2view = {
            0: 'front',
            1: 'left',
            2: 'back',
            3: 'right'
        }
        # 如果 multiview_indices 不是 [0,1,2,3]，需要根据实际索引映射
        # 这里假设配置中的 multiview_indices 顺序就是 front, left, back, right
        self.view_order = ['front', 'left', 'back', 'right']
        
        rank_zero_info(f'*' * 50)
        rank_zero_info(f'Multi-View Dataset Infos:')
        rank_zero_info(f'# of 3D file: {len(self.data_list)}')
        rank_zero_info(f'# of Surface Points: {self.pc_size}')
        rank_zero_info(f'# of Sharpedge Surface Points: {self.pc_sharpedge_size}')
        rank_zero_info(f'Using sharp edge label: {self.sharpedge_label}')
        rank_zero_info(f'Load depth maps: {self.load_depth}')
        rank_zero_info(f'Load normal maps: {self.load_normal}')
        rank_zero_info(f'Multi-view indices: {self.multiview_indices} (total {len(self.multiview_indices)} views)')
        rank_zero_info(f'Sample num_views: {self.sample_num_views} (random 1~4 views when "random" or [1,2,3,4])')
        rank_zero_info(f'*' * 50)

    def _sample_view_subset(self) -> List[str]:
        """每个样本随机选取 1/2/3/4 个视图（保持 view_order 顺序，便于位置编码一致）。"""
        if self.sample_num_views is None:
            return list(self.view_order)
        if self.sample_num_views == 4 or (isinstance(self.sample_num_views, int) and self.sample_num_views >= 4):
            return list(self.view_order)
        # "random" 或 [1,2,3,4]：随机选 k 个视图，k 在 1~4 之间
        if self.sample_num_views == "random":
            k = self.rng.randint(1, len(self.view_order))
        elif isinstance(self.sample_num_views, (list, tuple)) and len(self.sample_num_views) > 0:
            k = self.rng.choice(self.sample_num_views)
            k = max(1, min(k, len(self.view_order)))
        else:
            return list(self.view_order)
        # 在 view_order 中随机选 k 个，并保持原有顺序（保证 view_idxs 与 MVImageProcessorV2 一致）
        indices = sorted(self.rng.sample(range(len(self.view_order)), k))
        return [self.view_order[i] for i in indices]

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
        """Load multi-view images from provided paths"""
        # 直接使用传入的图片路径（已经在decode函数中根据multiview_indices筛选过）
        images, masks = [], []
        
        for image_path in imgs_path:
            if not os.path.exists(image_path):
                rank_zero_info(f"Warning: Image path does not exist: {image_path}")
                continue
                
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            
            # Handle both 3-channel (RGB) and 4-channel (RGBA) images
            if image.shape[2] == 4:
                # RGBA image - handle alpha channel
                alpha = image[:, :, 3:4].astype(np.float32) / 255
                forground = image[:, :, :3]
                background = np.ones_like(forground) * 255
                img_new = forground * alpha + background * (1 - alpha)
                image = img_new.astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mask = (alpha[:, :, 0] * 255).astype(np.uint8)
            elif image.shape[2] == 3:
                # RGB image (e.g., normal maps) - no alpha channel
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                # Create a default mask (all ones) for RGB images
                mask = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
            else:
                raise ValueError(f"Unsupported image channels: {image.shape[2]} in {image_path}")

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
        """Load multi-view depth maps from provided paths"""
        # 直接使用传入的深度图路径（已经在decode函数中根据multiview_indices筛选过）
        depths = []
        
        for depth_path in depth_paths:
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
            # 动态加权融合，根据实际视图数量生成权重
            num_views = depths.shape[0]
            # 生成均匀权重，也可以根据需要自定义权重
            weights = torch.ones(num_views, device=depths.device) / num_views
            weights = weights.view(-1, 1, 1, 1)
            weighted = (depths * weights).sum(dim=0, keepdim=True)
            return weighted  # Shape: (1, 1, H, W)
        else:
            # 默认返回多视图格式
            return depths

    def decode(self, item):
        """Generate image paths based on multiview_indices configuration"""
        uid = item.split('/')[-1]
        
        # 根据multiview_indices生成对应的图片路径，并构建 image_dict
        # multiview_indices 的顺序应该对应 view_order: [front, left, back, right]
        if self.multiview_indices is not None and len(self.multiview_indices) > 0:
            # 构建 image_dict，键为视图名称，值为图片路径
            image_dict = {}
            for idx, view_name in enumerate(self.view_order):
                if idx < len(self.multiview_indices):
                    view_idx = self.multiview_indices[idx]
                    image_dict[view_name] = os.path.join(item, f'render_cond/{view_idx:03d}.png')
        else:
            # 如果没有设置multiview_indices，使用默认的24个视图的前4个
            image_dict = {
                'front': os.path.join(item, f'render_cond/000.png'),
                'left': os.path.join(item, f'render_cond/006.png'),
                'back': os.path.join(item, f'render_cond/012.png'),
                'right': os.path.join(item, f'render_cond/018.png')
            }
        
        surface_npz_path = os.path.join(item, f'geo_data/{uid}_surface.npz')
        
        sample = {}
        sample["image"] = image_dict  # 返回 image_dict 而不是路径列表
        
        # Load depth maps if enabled
        if self.load_depth:
            if self.multiview_indices is not None and len(self.multiview_indices) > 0:
                # 构建 depth_dict
                depth_dict = {}
                for idx, view_name in enumerate(self.view_order):
                    if idx < len(self.multiview_indices):
                        view_idx = self.multiview_indices[idx]
                        depth_dict[view_name] = os.path.join(item, f'render_cond/{view_idx:03d}_depth.exr')
                sample["depth"] = depth_dict
            else:
                depth_dict = {
                    'front': os.path.join(item, f'render_cond/000_depth.exr'),
                    'left': os.path.join(item, f'render_cond/006_depth.exr'),
                    'back': os.path.join(item, f'render_cond/012_depth.exr'),
                    'right': os.path.join(item, f'render_cond/018_depth.exr')
                }
                sample["depth"] = depth_dict
        
        # Load normal maps if enabled
        if self.load_normal:
            if self.multiview_indices is not None and len(self.multiview_indices) > 0:
                # 构建 normal_dict
                normal_dict = {}
                for idx, view_name in enumerate(self.view_order):
                    if idx < len(self.multiview_indices):
                        view_idx = self.multiview_indices[idx]
                        normal_dict[view_name] = os.path.join(item, f'render_cond/{view_idx:03d}_normal.png')
                sample["normal"] = normal_dict
            else:
                normal_dict = {
                    'front': os.path.join(item, f'render_cond/000_normal.png'),
                    'left': os.path.join(item, f'render_cond/006_normal.png'),
                    'back': os.path.join(item, f'render_cond/012_normal.png'),
                    'right': os.path.join(item, f'render_cond/018_normal.png')
                }
                sample["normal"] = normal_dict
        
        surface_data = read_npz(surface_npz_path)
        sample["random_surface"] = surface_data['random_surface']
        sample["sharpedge_surface"] = surface_data['sharp_surface']
        return sample

    def transform(self, sample):
        """Transform sample with multi-view loading - returns image_dict for MVImageProcessorV2"""
        rng = np.random.default_rng()
        random_surface = sample.get("random_surface", 0)
        sharpedge_surface = sample.get("sharpedge_surface", 0)
        
        # 随机选取本样本使用的视图子集（1/2/3/4 视图），保持 view_order 顺序以匹配位置编码
        selected_views = self._sample_view_subset()
        image_dict_full = sample['image']
        image_dict = {v: image_dict_full[v] for v in selected_views if v in image_dict_full}
        
        surface, geo_points = self.load_surface_sdf_points(rng, random_surface, sharpedge_surface)
        
        result_sample = {
            "surface": surface,
            "geo_points": geo_points,
            "image": image_dict,  # 仅包含 selected_views，顺序已保持
        }
        
        # Load depth maps if enabled - 仅加载选中视图的 depth
        if self.load_depth and "depth" in sample:
            depth_paths = [sample['depth'][view_name] for view_name in selected_views if view_name in sample.get('depth', {})]
            depth_input = self.load_multiview_depth_maps(depth_paths)
            result_sample["depth"] = depth_input  # Shape: (num_views, 1, H, W)
        
        # Load normal maps if enabled - 仅返回选中视图的 normal_dict
        if self.load_normal and "normal" in sample:
            result_sample["normal"] = {v: sample['normal'][v] for v in selected_views if v in sample.get('normal', {})}
        
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
        padding_ratio_range=[1, 1],
        load_depth: bool = False,
        load_normal: bool = False,
        multiview_indices: List[int] = [0, 6, 12, 18],  # 前后左右四个视图
        depth_fusion_strategy: str = "multiview",  # 深度融合策略
        normal_fusion_strategy: str = "multiview",  # 法线图融合策略
        sample_num_views: Optional[Union[str, List[int]]] = None,  # None/4: 固定4视图; "random" 或 [1,2,3,4]: 每样本随机1~4视图
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
        self.load_normal = load_normal
        self.multiview_indices = multiview_indices
        self.depth_fusion_strategy = depth_fusion_strategy
        self.normal_fusion_strategy = normal_fusion_strategy
        self.sample_num_views = sample_num_views
        
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
            "load_normal": self.load_normal,
            "multiview_indices": self.multiview_indices,
            "depth_fusion_strategy": self.depth_fusion_strategy,
            "normal_fusion_strategy": self.normal_fusion_strategy,
            "sample_num_views": self.sample_num_views,
        }
        dataset = AlignedShapeLatentMultiViewDataset(**asl_params)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            collate_fn=multiview_collate_fn,  # 使用自定义 collate_fn 处理 image_dict
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
            "load_normal": self.load_normal,
            "multiview_indices": self.multiview_indices,
            "depth_fusion_strategy": self.depth_fusion_strategy,
            "normal_fusion_strategy": self.normal_fusion_strategy,
            "sample_num_views": self.sample_num_views,
        }
        dataset = AlignedShapeLatentMultiViewDataset(**asl_params)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.val_num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
            collate_fn=multiview_collate_fn,  # 使用自定义 collate_fn 处理 image_dict
        )
