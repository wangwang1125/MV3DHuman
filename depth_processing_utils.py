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


def process_depth_for_gradio(depth_image: Union[str, object], 
                           rgb_mask: Optional[np.ndarray] = None,
                           depth_clip_range: list = [0.0, 10.0],
                           depth_mean: float = 0.5,
                           depth_std: float = 0.5,
                           invalid_depth_value: float = -2.0) -> Tuple[Image.Image, np.ndarray]:
    """
    为gradio_app.py处理深度图的统一接口，与hy3dshape训练时的处理保持一致
    
    Args:
        depth_image (str or file object): 输入深度图文件路径或gradio文件对象
        rgb_mask (np.ndarray, optional): RGB掩码
        depth_clip_range (list): 深度值裁剪范围（与训练时保持一致）
        depth_mean (float): 标准化均值
        depth_std (float): 标准化标准差
        invalid_depth_value (float): 无效深度值标识，与训练时保持一致
    
    Returns:
        processed_image (PIL.Image): 处理后的深度图（用于显示）
        depth_array (np.ndarray): 标准化后的深度数组（用于模型，范围[-1,1]，无效区域为-2.0）
    """
    # 处理gradio文件对象
    if hasattr(depth_image, 'name'):
        # gradio文件对象，获取文件路径
        depth_path = depth_image.name
        print(f"从gradio文件对象获取路径: {depth_path}")
    elif isinstance(depth_image, str):
        # 直接的文件路径
        depth_path = depth_image
        print(f"使用文件路径: {depth_path}")
    else:
        raise ValueError(f"不支持的深度图输入类型: {type(depth_image)}")
    
    # 使用cv2读取深度图，与训练时保持一致
    try:
        # 使用cv2.IMREAD_ANYDEPTH保持原始位深度，与训练时一致
        depth_array = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
        if depth_array is None:
            raise ValueError(f"无法读取深度图文件: {depth_path}")
        
        # 转换为float32
        depth_array = depth_array.astype(np.float32)
        print(f"使用cv2读取深度图，原始值范围: {depth_array.min():.1f} - {depth_array.max():.1f}")
        
        # 确保是单通道
        if len(depth_array.shape) == 3:
            depth_array = depth_array[:, :, 0]
            
    except Exception as e:
        raise ValueError(f"无法读取深度图文件: {depth_path}, 错误: {e}")
    
    depth_array = depth_array / 1000.0
    print(f"深度图值范围（与训练时一致）: {depth_array.min():.1f} - {depth_array.max():.1f}")
    
    # === 与训练时一致的无效值处理 ===
    # 1. 创建有效深度掩码（与rgbd_dit_asl.py中的load_depth_image一致）
    valid_mask = np.isfinite(depth_array) & (depth_array > 0) & (depth_array <= depth_clip_range[1])
    
    # 2. 处理深度图，标记无效区域
    depth_processed = depth_array.copy()
    depth_processed[~valid_mask] = invalid_depth_value  # 标记无效区域为-2.0
    
    # 3. 对有效区域进行第一次归一化到[0,1]
    valid_depth = depth_array[valid_mask]
    if len(valid_depth) > 0:
        # 裁剪有效深度值
        valid_depth_clipped = np.clip(valid_depth, depth_clip_range[0], depth_clip_range[1])
        depth_min, depth_max = depth_clip_range
        if depth_max > depth_min:
            valid_depth_normalized = (valid_depth_clipped - depth_min) / (depth_max - depth_min)
            depth_processed[valid_mask] = valid_depth_normalized
    
    # 4. 应用RGB掩码（如果提供）
    if rgb_mask is not None:
        # 将RGB掩码为0的区域也标记为无效
        rgb_invalid_mask = rgb_mask < 0.5  # 假设掩码值小于0.5为无效
        depth_processed[rgb_invalid_mask] = invalid_depth_value
        # 更新有效掩码
        valid_mask = valid_mask & (~rgb_invalid_mask)
    
    # 5. 对有效区域进行第二次归一化到[-1,1]（与训练时一致）
    depth_final = depth_processed.copy()
    final_valid_mask = depth_processed != invalid_depth_value
    if np.any(final_valid_mask):
        depth_final[final_valid_mask] = (depth_processed[final_valid_mask] - depth_mean) / depth_std
    
    print(f"有效像素数量: {np.sum(final_valid_mask)}")
    print(f"无效像素数量: {np.sum(~final_valid_mask)}")
    if np.any(final_valid_mask):
        valid_values = depth_final[final_valid_mask]
        print(f"有效深度值范围: [{valid_values.min():.3f}, {valid_values.max():.3f}]")
    print(f"无效深度值: {invalid_depth_value}")
    
    # 转换回PIL图像（用于显示）
    # 只显示有效区域，无效区域显示为黑色
    depth_display = np.zeros_like(depth_final, dtype=np.uint8)
    if np.any(final_valid_mask):
        # 将有效区域的[-1,1]值映射回[0,255]用于显示
        valid_display = ((depth_final[final_valid_mask] * depth_std + depth_mean) * 255)
        valid_display = np.clip(valid_display, 0, 255).astype(np.uint8)
        depth_display[final_valid_mask] = valid_display
    
    processed_image = Image.fromarray(depth_display, mode='L')
    
    return processed_image, depth_final


def validate_depth_processing(depth_array: np.ndarray, 
                             expected_range: Tuple[float, float] = (-1.0, 1.0),
                             invalid_depth_value: float = -2.0) -> bool:
    """
    验证深度图处理结果是否符合预期
    
    Args:
        depth_array (np.ndarray): 处理后的深度数组
        expected_range (tuple): 期望的有效值范围
        invalid_depth_value (float): 无效深度值标识
    
    Returns:
        bool: 验证是否通过
    """
    # 分离有效值和无效值
    valid_mask = depth_array != invalid_depth_value
    invalid_mask = depth_array == invalid_depth_value
    
    # 检查是否有NaN或无穷值
    if not np.isfinite(depth_array).all():
        print("Warning: Depth array contains NaN or infinite values")
        return False
    
    # 检查无效值是否正确
    if np.any(invalid_mask):
        invalid_values = depth_array[invalid_mask]
        if not np.all(invalid_values == invalid_depth_value):
            print(f"Warning: Invalid depth values are not consistent. Expected: {invalid_depth_value}")
            return False
    
    # 检查有效值范围
    if np.any(valid_mask):
        valid_values = depth_array[valid_mask]
        min_val, max_val = valid_values.min(), valid_values.max()
        expected_min, expected_max = expected_range
        
        if min_val < expected_min or max_val > expected_max:
            print(f"Warning: Valid depth values out of expected range. Expected: {expected_range}, Actual: ({min_val:.3f}, {max_val:.3f})")
            return False
    
    return True


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