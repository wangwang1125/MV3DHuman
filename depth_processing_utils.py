#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度图处理工具模块
确保gradio_app.py和hy3dshape中深度图处理的一致性
"""

import numpy as np
import cv2
from PIL import Image
import torch
from typing import Tuple, Optional, Union


def detect_depth_unit(depth_array: np.ndarray) -> str:
    """
    检测深度图的单位，支持16位深度图
    
    Args:
        depth_array (np.ndarray): 深度图数组
    
    Returns:
        str: 深度图单位类型，默认为'mm'
    """
    # 简化逻辑：默认PNG深度图都是毫米单位
    return 'mm'


def load_depth_image(depth_path: str, depth_clip_range: list = [0.0, 10.0]) -> np.ndarray:
    """
    加载深度图像，支持多种格式
    
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
    if depth_max > depth_min:
        depth = (depth - depth_min) / (depth_max - depth_min)
    
    # 添加通道维度
    depth = depth[:, :, np.newaxis]
    
    return depth.astype(np.float32)


def normalize_depth_for_model(depth: np.ndarray, 
                             depth_mean: float = 0.5, 
                             depth_std: float = 0.5) -> np.ndarray:
    """
    对深度图进行模型输入标准化
    
    Args:
        depth (np.ndarray): 输入深度图，值域[0,1]
        depth_mean (float): 标准化均值
        depth_std (float): 标准化标准差
    
    Returns:
        normalized_depth (np.ndarray): 标准化后的深度图
    """
    return (depth - depth_mean) / depth_std


def denormalize_depth_for_display(depth: np.ndarray, 
                                 depth_mean: float = 0.5, 
                                 depth_std: float = 0.5) -> np.ndarray:
    """
    反标准化深度图用于显示
    
    Args:
        depth (np.ndarray): 标准化的深度图
        depth_mean (float): 标准化均值
        depth_std (float): 标准化标准差
    
    Returns:
        denormalized_depth (np.ndarray): 反标准化后的深度图，值域[0,255]
    """
    # 反标准化到[0,1]
    depth_01 = depth * depth_std + depth_mean
    # 转换到[0,255]
    depth_255 = np.clip(depth_01 * 255, 0, 255).astype(np.uint8)
    return depth_255


def apply_rgb_mask_to_depth(depth: np.ndarray, 
                           rgb_mask: np.ndarray, 
                           fill_value: float = 0.0) -> np.ndarray:
    """
    将RGB掩码应用到深度图
    
    Args:
        depth (np.ndarray): 深度图
        rgb_mask (np.ndarray): RGB掩码 (0-1或0-255)
        fill_value (float): 掩码外区域的填充值
    
    Returns:
        masked_depth (np.ndarray): 应用掩码后的深度图
    """
    # 确保掩码是0-1范围
    if rgb_mask.max() > 1:
        rgb_mask = rgb_mask / 255.0
    
    # 确保尺寸匹配
    if rgb_mask.shape[:2] != depth.shape[:2]:
        rgb_mask = cv2.resize(rgb_mask, (depth.shape[1], depth.shape[0]))
    
    # 如果掩码是3通道，取第一个通道
    if len(rgb_mask.shape) == 3:
        rgb_mask = rgb_mask[:, :, 0]
    
    # 应用掩码
    masked_depth = depth.copy()
    if len(depth.shape) == 3:
        for c in range(depth.shape[2]):
            masked_depth[:, :, c] = depth[:, :, c] * rgb_mask + fill_value * (1 - rgb_mask)
    else:
        masked_depth = depth * rgb_mask + fill_value * (1 - rgb_mask)
    
    return masked_depth


def process_depth_for_gradio(depth_image: Image.Image, 
                           rgb_mask: Optional[np.ndarray] = None,
                           depth_clip_range: list = [0.0, 10.0],
                           depth_mean: float = 0.5,
                           depth_std: float = 0.5,
                           input_unit: str = 'auto') -> Tuple[Image.Image, np.ndarray]:
    """
    为gradio_app.py处理深度图的统一接口，与hy3dshape训练时的处理保持一致
    
    Args:
        depth_image (PIL.Image): 输入深度图
        rgb_mask (np.ndarray, optional): RGB掩码
        depth_clip_range (list): 深度值裁剪范围（单位：米）
        depth_mean (float): 标准化均值
        depth_std (float): 标准化标准差
        input_unit (str): 输入深度图的单位，'auto'自动检测、'normalized'已归一化、'mm'毫米、'm'米
    
    Returns:
        processed_image (PIL.Image): 处理后的深度图（用于显示）
        depth_array (np.ndarray): 标准化后的深度数组（用于模型，范围[-1,1]）
    """
    # 转换为numpy数组
    depth_array = np.array(depth_image, dtype=np.float32)
    

    # PNG深度图通常以毫米为单位，需要转换为米
    original_range = f"{depth_array.min():.1f}mm - {depth_array.max():.1f}mm"
    depth_array = depth_array / 1000.0  # mm -> m
    converted_range = f"{depth_array.min():.3f}m - {depth_array.max():.3f}m"
    print(f"深度图单位转换: {original_range} -> {converted_range}")
    
    # 裁剪到合理范围并归一化到[0,1]
    depth_array = np.clip(depth_array, depth_clip_range[0], depth_clip_range[1])
    depth_min, depth_max = depth_clip_range
    if depth_max > depth_min:
        depth_array = (depth_array - depth_min) / (depth_max - depth_min)

    
    # 应用RGB掩码（如果提供）
    if rgb_mask is not None:
        depth_array = apply_rgb_mask_to_depth(depth_array, rgb_mask)
    
    # 标准化到[-1,1]范围（与训练时一致）
    # 训练时使用: (depth - 0.5) / 0.5，将[0,1]映射到[-1,1]
    depth_normalized = (depth_array - depth_mean) / depth_std
    
    print(f"最终深度值范围: [{depth_normalized.min():.3f}, {depth_normalized.max():.3f}]")
    
    # 转换回PIL图像（用于显示，将[-1,1]映射回[0,255]）
    depth_display = ((depth_normalized * depth_std + depth_mean) * 255).astype(np.uint8)
    depth_display = np.clip(depth_display, 0, 255)
    processed_image = Image.fromarray(depth_display, mode='L')
    
    return processed_image, depth_normalized


def detect_depth_unit(depth_array: np.ndarray) -> str:
    """
    自动检测深度图的单位
    
    Args:
        depth_array (np.ndarray): 深度图数组
    
    Returns:
        str: 检测到的单位 ('mm' 或 'm')
    """
    max_depth = np.max(depth_array)
    mean_depth = np.mean(depth_array[depth_array > 0])
    
    # 启发式规则：
    # - 如果最大深度值 > 100，很可能是毫米单位
    # - 如果平均深度值 > 50，很可能是毫米单位
    if max_depth > 100 or mean_depth > 50:
        return 'mm'
    else:
        return 'm'


def validate_depth_processing(depth_array: np.ndarray, 
                             expected_range: Tuple[float, float] = (-1.0, 1.0)) -> bool:
    """
    验证深度图处理是否正确
    
    Args:
        depth_array (np.ndarray): 处理后的深度数组
        expected_range (tuple): 期望的值域范围
    
    Returns:
        is_valid (bool): 是否有效
    """
    min_val, max_val = expected_range
    actual_min, actual_max = depth_array.min(), depth_array.max()
    
    # 检查值域是否在期望范围内（允许小的误差）
    tolerance = 0.1
    is_valid = (actual_min >= min_val - tolerance and 
                actual_max <= max_val + tolerance)
    
    if not is_valid:
        print(f"Warning: Depth values out of expected range. "
              f"Expected: {expected_range}, Actual: ({actual_min:.3f}, {actual_max:.3f})")
    
    return is_valid


if __name__ == "__main__":
    # 测试代码
    print("深度图处理工具模块测试")
    
    # 创建测试深度图
    test_depth = np.random.rand(256, 256) * 10  # 0-10范围的随机深度
    test_image = Image.fromarray((test_depth * 25.5).astype(np.uint8), mode='L')
    
    # 测试处理流程
    processed_image, depth_array = process_depth_for_gradio(test_image)
    
    # 验证结果
    is_valid = validate_depth_processing(depth_array)
    print(f"处理结果验证: {'通过' if is_valid else '失败'}")
    print(f"原始深度范围: {test_depth.min():.3f} - {test_depth.max():.3f}")
    print(f"处理后范围: {depth_array.min():.3f} - {depth_array.max():.3f}")