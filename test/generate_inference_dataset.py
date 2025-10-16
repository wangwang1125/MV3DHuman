#!/usr/bin/env python3
"""
生成推理数据集
从预处理数据集中提取RGB图像和深度图，并转换深度图格式

输入数据集格式:
|obj名称
|--render_cond
|----000.png (RGB图像)
|----000_depth.exr (深度图，单位：米)
|----001.png
|----001_depth.exr
|----...

输出数据集格式:
|mesh名称
|--000.png (复制的RGB图像)
|--000_depth.png (转换后的深度图，单位：毫米)
|--001.png
|--001_depth.png
|--...
"""

import os
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm
import cv2

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("⚠️  警告: imageio未安装，将尝试使用OpenCV读取EXR文件")


def read_exr_depth(exr_path: str, max_valid_depth: float = 100.0, debug: bool = False) -> np.ndarray:
    """
    读取EXR深度图，并处理无效值
    
    Args:
        exr_path: EXR文件路径
        max_valid_depth: 最大有效深度值（米），超过此值视为无效
        debug: 是否打印调试信息
    
    Returns:
        深度图数组 (单位：米)
    """
    depth = None
    
    if HAS_IMAGEIO:
        try:
            # 使用imageio读取EXR
            depth = imageio.imread(exr_path, format='EXR-FI')
            # 如果是多通道，取第一个通道
            if len(depth.shape) == 3:
                depth = depth[:, :, 0]
        except Exception as e:
            if debug:
                print(f"   ⚠️  imageio读取失败: {e}，尝试OpenCV...")
    
    # 如果imageio失败，尝试使用OpenCV读取
    if depth is None:
        try:
            depth = cv2.imread(exr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if depth is None:
                raise ValueError(f"无法读取EXR文件: {exr_path}")
            # 如果是多通道，取第一个通道
            if len(depth.shape) == 3:
                depth = depth[:, :, 0]
        except Exception as e:
            raise RuntimeError(f"读取EXR文件失败: {exr_path}, 错误: {e}")
    
    # 处理无效值
    # 1. 替换inf和nan为0
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    
    # 2. 将超过最大有效深度的值设为0（标记为无效/背景）
    depth[depth > max_valid_depth] = 0.0
    
    # 3. 将负值设为0
    depth[depth < 0] = 0.0
    
    if debug:
        valid_mask = depth > 0
        if np.any(valid_mask):
            print(f"   深度范围: [{np.min(depth[valid_mask]):.3f}, {np.max(depth[valid_mask]):.3f}] 米")
            print(f"   有效像素: {np.sum(valid_mask)} / {depth.size} ({np.sum(valid_mask)/depth.size*100:.2f}%)")
        else:
            print(f"   ⚠️  警告: 没有有效深度值！")
    
    return depth


def convert_depth_to_png(depth_meters: np.ndarray, output_path: str, 
                        normalize: bool = False, 
                        max_depth_mm: float = None,
                        debug: bool = False) -> bool:
    """
    将深度图从米转换为毫米并保存为PNG
    
    Args:
        depth_meters: 深度数组（单位：米）
        output_path: 输出PNG路径
        normalize: 是否归一化到0-65535范围
        max_depth_mm: 最大深度值（毫米），用于归一化
        debug: 是否打印调试信息
    
    Returns:
        是否成功
    """
    try:
        # 转换为毫米
        depth_mm = depth_meters * 1000.0
        
        # 处理无效值（inf, nan）已经在read_exr_depth中处理过了，这里再做一次保险
        depth_mm = np.nan_to_num(depth_mm, nan=0.0, posinf=0.0, neginf=0.0)
        
        if debug:
            valid_mask = depth_mm > 0
            if np.any(valid_mask):
                print(f"   深度(mm)范围: [{np.min(depth_mm[valid_mask]):.1f}, {np.max(depth_mm[valid_mask]):.1f}]")
        
        if normalize:
            # 归一化模式：映射到0-65535（16位PNG）
            if max_depth_mm is None:
                max_depth_mm = np.max(depth_mm[depth_mm > 0]) if np.any(depth_mm > 0) else 10000.0
            
            depth_normalized = np.clip(depth_mm / max_depth_mm * 65535.0, 0, 65535)
            depth_uint16 = depth_normalized.astype(np.uint16)
            
            if debug:
                print(f"   归一化：max_depth={max_depth_mm:.1f}mm, 输出范围=[0, {np.max(depth_uint16)}]")
        else:
            # 直接模式：保存毫米值（可能超过65535）
            # 限制在16位范围内
            depth_uint16 = np.clip(depth_mm, 0, 65535).astype(np.uint16)
            
            if debug:
                clipped = np.sum(depth_mm > 65535)
                if clipped > 0:
                    print(f"   ⚠️  警告: {clipped} 个像素深度超过65535mm，已截断")
        
        # 保存为PNG
        cv2.imwrite(output_path, depth_uint16)
        return True
    
    except Exception as e:
        print(f"   ❌ 转换深度图失败: {e}")
        return False


def find_valid_samples(dataset_path: str) -> List[Tuple[str, Path]]:
    """
    查找数据集中的有效样本
    
    Args:
        dataset_path: 数据集根目录
    
    Returns:
        [(样本名, 样本目录路径), ...]
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ 错误: 数据集路径不存在: {dataset_path}")
        return []
    
    valid_samples = []
    
    # 遍历所有子目录
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    for obj_dir in sorted(subdirs):
        obj_name = obj_dir.name
        render_cond_dir = obj_dir / "render_cond"
        
        # 检查是否有render_cond目录和必需的文件
        if not render_cond_dir.exists():
            continue
        
        # 检查是否至少有一对图像+深度图
        has_files = False
        for i in range(4):
            idx = f"{i:03d}"
            png_file = render_cond_dir / f"{idx}.png"
            depth_file = render_cond_dir / f"{idx}_depth.exr"
            if png_file.exists() and depth_file.exists():
                has_files = True
                break
        
        if has_files:
            valid_samples.append((obj_name, obj_dir))
    
    return valid_samples


def generate_inference_dataset(source_dataset_path: str,
                               output_path: str,
                               normalize_depth: bool = False,
                               max_depth_mm: float = None,
                               max_valid_depth: float = 100.0,
                               overwrite: bool = False,
                               debug: bool = False) -> dict:
    """
    生成推理数据集
    
    Args:
        source_dataset_path: 源数据集路径
        output_path: 输出路径
        normalize_depth: 是否归一化深度图
        max_depth_mm: 最大深度值（毫米）
        max_valid_depth: 读取EXR时的最大有效深度（米），默认100米
        overwrite: 是否覆盖已存在的文件
        debug: 是否打印调试信息
    
    Returns:
        统计信息字典
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找有效样本
    print("🔍 扫描数据集...")
    valid_samples = find_valid_samples(source_dataset_path)
    
    if not valid_samples:
        print("❌ 错误: 未找到有效样本")
        return None
    
    print(f"✅ 找到 {len(valid_samples)} 个有效样本\n")
    
    # 统计信息
    stats = {
        'total_samples': len(valid_samples),
        'processed_samples': 0,
        'failed_samples': 0,
        'total_images': 0,
        'total_depths': 0,
        'failed_images': [],
        'failed_depths': []
    }
    
    # 处理每个样本
    for obj_name, obj_dir in tqdm(valid_samples, desc="生成推理数据集"):
        render_cond_dir = obj_dir / "render_cond"
        
        # 创建输出目录
        output_sample_dir = output_path / obj_name
        output_sample_dir.mkdir(parents=True, exist_ok=True)
        
        sample_success = True
        
        # 处理4个视角
        for i in range(4):
            idx = f"{i:03d}"
            
            # RGB图像路径
            src_png = render_cond_dir / f"{idx}.png"
            dst_png = output_sample_dir / f"{idx}.png"
            
            # 深度图路径
            src_depth_exr = render_cond_dir / f"{idx}_depth.exr"
            dst_depth_png = output_sample_dir / f"{idx}_depth.png"
            
            # 复制RGB图像
            if src_png.exists():
                try:
                    if overwrite or not dst_png.exists():
                        shutil.copy2(src_png, dst_png)
                        stats['total_images'] += 1
                except Exception as e:
                    print(f"\n❌ 复制RGB图像失败: {obj_name}/{idx}.png")
                    print(f"   错误: {e}")
                    stats['failed_images'].append(f"{obj_name}/{idx}.png")
                    sample_success = False
            
            # 转换深度图
            if src_depth_exr.exists():
                try:
                    if overwrite or not dst_depth_png.exists():
                        depth = read_exr_depth(str(src_depth_exr), max_valid_depth=max_valid_depth, debug=debug)
                        success = convert_depth_to_png(
                            depth, str(dst_depth_png),
                            normalize=normalize_depth,
                            max_depth_mm=max_depth_mm,
                            debug=debug
                        )
                        if success:
                            stats['total_depths'] += 1
                        else:
                            stats['failed_depths'].append(f"{obj_name}/{idx}_depth.png")
                            sample_success = False
                except Exception as e:
                    print(f"\n❌ 转换深度图失败: {obj_name}/{idx}_depth.exr")
                    print(f"   错误: {e}")
                    stats['failed_depths'].append(f"{obj_name}/{idx}_depth.png")
                    sample_success = False
        
        if sample_success:
            stats['processed_samples'] += 1
        else:
            stats['failed_samples'] += 1
    
    return stats


def print_statistics(stats: dict):
    """打印统计信息"""
    print("\n" + "=" * 80)
    print("生成统计")
    print("=" * 80)
    
    print(f"\n📊 样本统计:")
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  成功处理: {stats['processed_samples']}")
    print(f"  失败样本: {stats['failed_samples']}")
    
    print(f"\n📁 文件统计:")
    print(f"  RGB图像: {stats['total_images']} 个")
    print(f"  深度图: {stats['total_depths']} 个")
    
    if stats['failed_images']:
        print(f"\n❌ 失败的RGB图像 ({len(stats['failed_images'])}个):")
        for img in stats['failed_images'][:10]:  # 最多显示10个
            print(f"  - {img}")
        if len(stats['failed_images']) > 10:
            print(f"  ... 还有 {len(stats['failed_images']) - 10} 个")
    
    if stats['failed_depths']:
        print(f"\n❌ 失败的深度图 ({len(stats['failed_depths'])}个):")
        for depth in stats['failed_depths'][:10]:
            print(f"  - {depth}")
        if len(stats['failed_depths']) > 10:
            print(f"  ... 还有 {len(stats['failed_depths']) - 10} 个")
    
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='生成推理数据集（RGB + 深度图PNG）')
    parser.add_argument('--source', '-s', type=str,
                        default='/mnt/g/mini_mv_depth_testset/preprocessed/',
                        help='源数据集路径')
    parser.add_argument('--output', '-o', type=str,
                        default='/mnt/g/inference_dataset/',
                        help='输出路径')
    parser.add_argument('--normalize', '-n', action='store_true',
                        help='归一化深度图到0-65535范围')
    parser.add_argument('--max_depth', '-m', type=float, default=None,
                        help='最大深度值（毫米），用于归一化（默认：自动计算）')
    parser.add_argument('--max_valid', '-v', type=float, default=100.0,
                        help='EXR读取时的最大有效深度（米），超过此值视为无效（默认：100米）')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已存在的文件')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='打印调试信息')
    
    args = parser.parse_args()
    
    print("🚀 开始生成推理数据集...")
    print(f"源数据集: {args.source}")
    print(f"输出路径: {args.output}")
    print(f"归一化深度: {args.normalize}")
    if args.normalize and args.max_depth:
        print(f"最大深度: {args.max_depth} mm")
    print(f"最大有效深度: {args.max_valid} m")
    print(f"覆盖模式: {args.overwrite}")
    print(f"调试模式: {args.debug}\n")
    
    stats = generate_inference_dataset(
        source_dataset_path=args.source,
        output_path=args.output,
        normalize_depth=args.normalize,
        max_depth_mm=args.max_depth,
        max_valid_depth=args.max_valid,
        overwrite=args.overwrite,
        debug=args.debug
    )
    
    if stats:
        print_statistics(stats)
        
        if stats['failed_samples'] == 0:
            print("\n✅ 推理数据集生成完成！")
            return 0
        else:
            print(f"\n⚠️  推理数据集生成完成，但有 {stats['failed_samples']} 个样本处理失败")
            return 1
    else:
        print("\n❌ 推理数据集生成失败！")
        return 1


if __name__ == "__main__":
    exit(main())

