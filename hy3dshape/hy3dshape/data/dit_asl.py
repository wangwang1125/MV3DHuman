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


class ResampledShards(torch.utils.data.dataset.IterableDataset):
    def __init__(self, datalist, nshards=sys.maxsize, worker_seed=None, deterministic=False):
        super().__init__()
        self.datalist = datalist
        self.nshards = nshards
        # If no worker_seed provided, use pytorch_worker_seed function; else use given seed
        self.worker_seed = pytorch_worker_seed if worker_seed is None else worker_seed
        self.deterministic = deterministic
        self.epoch = -1

    def __iter__(self):
        self.epoch += 1
        if self.deterministic:
            seed = make_seed(self.worker_seed(), self.epoch)
        else:
            seed = make_seed(self.worker_seed(), self.epoch, 
                             os.getpid(), time.time_ns(), os.urandom(4))
        self.rng = random.Random(seed)
        for _ in range(self.nshards):
            index = self.rng.randint(0, len(self.datalist) - 1)
            yield self.datalist[index]

            
def read_npz(data):
    # Load a numpy .npz file from a file path or file-like object
    # The commented line shows how to load from bytes in memory
    # return np.load(io.BytesIO(data))
    return np.load(data)


def read_json(path):
    # Read and parse a JSON file from the given file path
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def padding(image, mask, center=True, padding_ratio_range=[1, 1]):
    """
    Pad the input image and mask to a square shape with padding ratio.

    Args:
        image (np.ndarray): Input image array of shape (H, W, C).
        mask (np.ndarray): Corresponding mask array of shape (H, W).
        center (bool): Whether to center the original image in the padded output.
        padding_ratio_range (list): Range [min, max] to randomly select padding ratio.

    Returns:
        newimg (np.ndarray): Padded image of shape (resize_side, resize_side, 3).
        newmask (np.ndarray): Padded mask of shape (resize_side, resize_side).
    """
    h, w = image.shape[:2]
    max_side = max(h, w)

    # Select padding ratio either fixed or randomly within the given range
    if padding_ratio_range[0] == padding_ratio_range[1]:
        padding_ratio = padding_ratio_range[0]
    else:
        padding_ratio = random.uniform(padding_ratio_range[0], padding_ratio_range[1])
    resize_side = int(max_side * padding_ratio)
    # resize_side = int(max_side * 1.15)

    pad_h = resize_side - h
    pad_w = resize_side - w
    if center:
        start_h = pad_h // 2
    else:
        start_h = pad_h - resize_side // 20
        
    start_w = pad_w // 2

    # Create new white image and black mask with padded size
    newimg = np.ones((resize_side, resize_side, 3), dtype=np.uint8) * 255
    newmask = np.zeros((resize_side, resize_side), dtype=np.uint8)
    
    # Place original image and mask into the padded canvas
    newimg[start_h:start_h + h, start_w:start_w + w] = image
    newmask[start_h:start_h + h, start_w:start_w + w] = mask
    
    return newimg, newmask


def viz_pc(surface, normal, image_input, name):
    image_input = image_input.cpu().numpy()
    image_input = image_input.transpose(1, 2, 0) * 0.5 + 0.5
    image_input = (image_input * 255).astype(np.uint8)
    cv2.imwrite(name + '.png', cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR))
    surface = surface.cpu().numpy()
    normal = normal.cpu().numpy()
    surface_mesh = trimesh.Trimesh(surface, vertex_colors=(normal + 1) / 2)
    surface_mesh.export(name + '.obj')


class AlignedShapeLatentDataset(torch.utils.data.dataset.IterableDataset):
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
        load_depth: bool = False
    ):
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
        
        rank_zero_info(f'*' * 50)
        rank_zero_info(f'Dataset Infos:')
        rank_zero_info(f'# of 3D file: {len(self.data_list)}')
        rank_zero_info(f'# of Surface Points: {self.pc_size}')
        rank_zero_info(f'# of Sharpedge Surface Points: {self.pc_sharpedge_size}')
        rank_zero_info(f'Using sharp edge label: {self.sharpedge_label}')
        rank_zero_info(f'Load depth maps: {self.load_depth}')
        rank_zero_info(f'*' * 50)


    def load_surface_sdf_points(self, rng, random_surface, sharpedge_surface):
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

    def load_render(self, imgs_path):
        imgs_choice = self.rng.sample(imgs_path, 1)
        images, masks = [], []
        for image_path in imgs_choice:
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
            
        images = torch.cat(images, dim=0)
        masks = torch.cat(masks, dim=0)[:1, ...]
        return images, masks

    def load_depth_maps(self, depth_paths):
        """Load and preprocess depth maps from EXR files"""
        depth_choice = self.rng.sample(depth_paths, 1)
        depths = []
        for depth_path in depth_choice:
            # Read EXR depth map using OpenEXR
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
                    raise ValueError(f"Failed to load depth map: {depth_path}")
                
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
        
        depths = torch.cat(depths, dim=0)  # Shape: (1, H, W)
        return depths

    def decode(self, item):
        uid = item.split('/')[-1]
        render_img_paths = [os.path.join(item, f'render_cond/{i:03d}.png') for i in range(24)]
        # transforms_json_path = os.path.join(item, 'render_cond/transforms.json')
        surface_npz_path = os.path.join(item, f'geo_data/{uid}_surface.npz')
        # sdf_npz_path = os.path.join(item, f'geo_data/{uid}_sdf.npz')
        # watertight_obj_path = os.path.join(item, f'geo_data/{uid}_watertight.obj')
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
        rng = np.random.default_rng()
        random_surface = sample.get("random_surface", 0)
        sharpedge_surface = sample.get("sharpedge_surface", 0)
        image_input, mask_input = self.load_render(sample['image'])
        surface, geo_points = self.load_surface_sdf_points(rng, random_surface, sharpedge_surface)
        
        result_sample = {
            "surface": surface,
            "geo_points": geo_points,
            "image": image_input,
            "mask": mask_input,
        }
        
        # Load and process depth maps if enabled
        if self.load_depth and "depth" in sample:
            depth_input = self.load_depth_maps(sample['depth'])
            result_sample["depth"] = depth_input
        
        return result_sample

    def __iter__(self):
        total_num = 0
        failed_num = 0
        for data in ResampledShards(self.data_list):
            total_num += 1
            if total_num % 1000 == 0:
                print(f"Current failure rate of data loading:")
                print(f"{failed_num}/{total_num}={failed_num/total_num}")
            try:
                sample = self.decode(data)
                sample = self.transform(sample)
            except Exception as err:
                print(err)
                failed_num += 1
                continue
            yield sample


class AlignedShapeLatentModule(LightningDataModule):
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
        load_depth: bool = False
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
            "load_depth": self.load_depth
        }
        dataset = AlignedShapeLatentDataset(**asl_params)
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
            "load_depth": self.load_depth
        }
        dataset = AlignedShapeLatentDataset(**asl_params)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.val_num_workers,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=worker_init_fn,
        )
