#!/usr/bin/env python3
"""
EXR深度图可视化工具

功能：
1. 读取EXR深度图
2. 应用颜色映射可视化
3. 显示深度统计信息
4. 保存可视化结果
5. 支持批量处理
"""

import os
import argparse
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("⚠️  警告: imageio未安装，将使用OpenCV读取EXR文件")


def read_exr_depth(exr_path: str, max_valid_depth: float = 100.0) -> np.ndarray:
    """
    读取EXR深度图
    
    Args:
        exr_path: EXR文件路径
        max_valid_depth: 最大有效深度值（米）
    
    Returns:
        深度图数组 (单位：米)
    """
    depth = None
    
    if HAS_IMAGEIO:
        try:
            depth = imageio.imread(exr_path, format='EXR-FI')
            if len(depth.shape) == 3:
                depth = depth[:, :, 0]
        except Exception as e:
            print(f"⚠️  imageio读取失败: {e}，尝试OpenCV...")
    
    if depth is None:
        try:
            depth = cv2.imread(exr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if depth is None:
                raise ValueError(f"无法读取EXR文件: {exr_path}")
            if len(depth.shape) == 3:
                depth = depth[:, :, 0]
        except Exception as e:
            raise RuntimeError(f"读取EXR文件失败: {exr_path}, 错误: {e}")
    
    # 处理无效值
    depth[depth > 1e9] = 0
    depth = np.nan_to_num(depth, nan=0.0, posinf=max_valid_depth, neginf=0.0)
    depth = np.clip(depth, 0.0, max_valid_depth)
    
    return depth


def compute_depth_statistics(depth: np.ndarray) -> dict:
    """
    计算深度图统计信息
    
    Args:
        depth: 深度图数组
    
    Returns:
        统计信息字典
    """
    valid_mask = depth > 0
    
    if not np.any(valid_mask):
        return {
            'min': 0.0,
            'max': 0.0,
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'valid_pixels': 0,
            'total_pixels': depth.size,
            'valid_ratio': 0.0
        }
    
    valid_depth = depth[valid_mask]
    
    return {
        'min': float(np.min(valid_depth)),
        'max': float(np.max(valid_depth)),
        'mean': float(np.mean(valid_depth)),
        'median': float(np.median(valid_depth)),
        'std': float(np.std(valid_depth)),
        'valid_pixels': int(np.sum(valid_mask)),
        'total_pixels': int(depth.size),
        'valid_ratio': float(np.sum(valid_mask) / depth.size)
    }


def visualize_depth(depth: np.ndarray,
                   colormap: str = 'turbo',
                   min_depth: Optional[float] = None,
                   max_depth: Optional[float] = None,
                   invert: bool = False) -> np.ndarray:
    """
    可视化深度图
    
    Args:
        depth: 深度图数组（米）
        colormap: matplotlib颜色映射名称
        min_depth: 最小深度值（用于归一化），None表示自动计算
        max_depth: 最大深度值（用于归一化），None表示自动计算
        invert: 是否反转颜色映射（近处亮，远处暗）
    
    Returns:
        RGB可视化图像 (H, W, 3) uint8
    """
    # 获取有效深度范围
    
    depth[depth > 100] = 0
    valid_mask = depth > 0
    
    if not np.any(valid_mask):
        # 全零深度图，返回黑色图像
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    
    if min_depth is None:
        min_depth = np.min(depth[valid_mask])
    if max_depth is None:
        max_depth = np.max(depth[valid_mask])
    
    # 归一化深度到[0, 1]
    depth_normalized = np.zeros_like(depth, dtype=np.float32)
    depth_normalized[valid_mask] = (depth[valid_mask] - min_depth) / (max_depth - min_depth + 1e-8)
    depth_normalized = np.clip(depth_normalized, 0, 1)
    
    # 反转颜色映射（让近处更亮）
    if invert:
        depth_normalized[valid_mask] = 1.0 - depth_normalized[valid_mask]
    
    # 应用颜色映射
    cmap = cm.get_cmap(colormap)
    depth_colored = cmap(depth_normalized)  # (H, W, 4) with RGBA
    
    # 转换为RGB uint8
    depth_rgb = (depth_colored[:, :, :3] * 255).astype(np.uint8)
    
    # 将无效区域设为黑色
    depth_rgb[~valid_mask] = 0
    
    return depth_rgb


def create_visualization_with_info(depth: np.ndarray,
                                   stats: dict,
                                   colormap: str = 'turbo',
                                   filename: str = "",
                                   invert: bool = False) -> np.ndarray:
    """
    创建带统计信息的可视化图像
    
    Args:
        depth: 深度图数组
        stats: 统计信息字典
        colormap: 颜色映射名称
        filename: 文件名（用于显示）
        invert: 是否反转颜色映射
    
    Returns:
        可视化图像
    """
    # 创建figure
    fig = plt.figure(figsize=(12, 8))
    
    # 主深度图
    ax1 = plt.subplot(2, 2, (1, 2))
    depth_vis = visualize_depth(depth, colormap=colormap, invert=invert)
    ax1.imshow(depth_vis)
    ax1.set_title(f'深度图可视化\n{filename}', fontsize=12)
    ax1.axis('off')
    
    # 深度直方图
    ax2 = plt.subplot(2, 2, 3)
    valid_depth = depth[depth > 0]
    if len(valid_depth) > 0:
        ax2.hist(valid_depth, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax2.axvline(stats['mean'], color='red', linestyle='--', linewidth=2, label=f"均值: {stats['mean']:.2f}m")
        ax2.axvline(stats['median'], color='green', linestyle='--', linewidth=2, label=f"中位数: {stats['median']:.2f}m")
        ax2.set_xlabel('深度 (米)', fontsize=10)
        ax2.set_ylabel('像素数量', fontsize=10)
        ax2.set_title('深度分布', fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
    
    # 统计信息文本
    ax3 = plt.subplot(2, 2, 4)
    ax3.axis('off')
    
    info_text = f"""
深度统计信息
{'='*40}

深度范围:
  最小值: {stats['min']:.3f} m  ({stats['min']*1000:.1f} mm)
  最大值: {stats['max']:.3f} m  ({stats['max']*1000:.1f} mm)
  
中心趋势:
  平均值: {stats['mean']:.3f} m  ({stats['mean']*1000:.1f} mm)
  中位数: {stats['median']:.3f} m  ({stats['median']*1000:.1f} mm)
  标准差: {stats['std']:.3f} m
  
像素信息:
  有效像素: {stats['valid_pixels']:,} / {stats['total_pixels']:,}
  有效比例: {stats['valid_ratio']*100:.2f}%
  
图像尺寸: {depth.shape[1]} × {depth.shape[0]}
颜色映射: {colormap}
    """
    
    ax3.text(0.05, 0.95, info_text.strip(), 
             transform=ax3.transAxes,
             fontsize=10,
             verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # 转换为numpy数组
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    
    return img


def visualize_single_depth(exr_path: str,
                          output_path: Optional[str] = None,
                          colormap: str = 'turbo',
                          max_valid_depth: float = 100.0,
                          show: bool = False,
                          save_simple: bool = False,
                          invert: bool = False):
    """
    可视化单个深度图
    
    Args:
        exr_path: EXR文件路径
        output_path: 输出图像路径
        colormap: 颜色映射
        max_valid_depth: 最大有效深度
        show: 是否显示图像
        save_simple: 是否保存简单版本（无统计信息）
        invert: 是否反转颜色映射
    """
    print(f"\n{'='*60}")
    print(f"处理: {exr_path}")
    print(f"{'='*60}")
    
    # 读取深度图
    depth = read_exr_depth(exr_path, max_valid_depth)
    
    # 计算统计信息
    stats = compute_depth_statistics(depth)
    
    # 打印统计信息
    print(f"\n深度范围: [{stats['min']:.3f}, {stats['max']:.3f}] 米")
    print(f"          [{stats['min']*1000:.1f}, {stats['max']*1000:.1f}] 毫米")
    print(f"平均深度: {stats['mean']:.3f} 米 ({stats['mean']*1000:.1f} 毫米)")
    print(f"中位深度: {stats['median']:.3f} 米 ({stats['median']*1000:.1f} 毫米)")
    print(f"标准差:   {stats['std']:.3f} 米")
    print(f"有效像素: {stats['valid_pixels']:,} / {stats['total_pixels']:,} ({stats['valid_ratio']*100:.2f}%)")
    print(f"图像尺寸: {depth.shape[1]} × {depth.shape[0]}")
    
    # 生成可视化
    if save_simple:
        # 简单版本：只有深度图
        vis_img = visualize_depth(depth, colormap=colormap, invert=invert)
    else:
        # 完整版本：包含统计信息
        filename = Path(exr_path).name
        vis_img = create_visualization_with_info(depth, stats, colormap, filename, invert)
    
    # 保存
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))
        print(f"\n✅ 已保存到: {output_path}")
    
    # 显示
    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(vis_img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


def batch_visualize_depth(input_dir: str,
                         output_dir: str,
                         colormap: str = 'turbo',
                         max_valid_depth: float = 100.0,
                         save_simple: bool = False,
                         invert: bool = False,
                         pattern: str = "**/*_depth.exr"):
    """
    批量可视化深度图
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        colormap: 颜色映射
        max_valid_depth: 最大有效深度
        save_simple: 是否保存简单版本
        invert: 是否反转颜色映射
        pattern: 文件匹配模式
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有EXR文件
    exr_files = list(input_path.glob(pattern))
    
    if not exr_files:
        print(f"❌ 错误: 在 {input_dir} 下未找到匹配 {pattern} 的文件")
        return
    
    print(f"📊 找到 {len(exr_files)} 个深度图文件\n")
    
    # 处理每个文件
    for exr_file in exr_files:
        try:
            # 构建输出路径（保持相对路径结构）
            rel_path = exr_file.relative_to(input_path)
            output_file = output_path / rel_path.with_suffix('.png')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 可视化
            visualize_single_depth(
                str(exr_file),
                str(output_file),
                colormap=colormap,
                max_valid_depth=max_valid_depth,
                show=False,
                save_simple=save_simple,
                invert=invert
            )
        except Exception as e:
            print(f"\n❌ 处理失败: {exr_file}")
            print(f"   错误: {e}\n")
    
    print(f"\n{'='*60}")
    print(f"✅ 批量处理完成！")
    print(f"   输出目录: {output_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description='EXR深度图可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 可视化单个深度图
  python visualize_depth.py input.exr -o output.png

  # 显示交互式窗口
  python visualize_depth.py input.exr --show

  # 使用不同的颜色映射
  python visualize_depth.py input.exr -o output.png -c viridis

  # 反转颜色（近处亮，远处暗）
  python visualize_depth.py input.exr -o output.png --invert

  # 批量处理
  python visualize_depth.py --batch /path/to/dataset/preprocessed/ -o ./visualizations/

  # 保存简单版本（无统计信息）
  python visualize_depth.py input.exr -o output.png --simple

可用颜色映射: turbo, jet, viridis, plasma, inferno, magma, hot, cool, rainbow
        """
    )
    
    parser.add_argument('input', type=str, nargs='?',
                        help='输入EXR文件路径（单文件模式）')
    parser.add_argument('--batch', '-b', type=str,
                        help='批量处理模式：输入目录路径')
    parser.add_argument('--output', '-o', type=str,
                        help='输出图像路径或目录')
    parser.add_argument('--colormap', '-c', type=str, default='turbo',
                        help='颜色映射 (默认: turbo)')
    parser.add_argument('--max_valid', '-v', type=float, default=100.0,
                        help='最大有效深度（米） (默认: 100)')
    parser.add_argument('--show', '-s', action='store_true',
                        help='显示可视化结果')
    parser.add_argument('--simple', action='store_true',
                        help='保存简单版本（仅深度图，无统计信息）')
    parser.add_argument('--invert', '-i', action='store_true',
                        help='反转颜色映射（近处亮，远处暗）')
    parser.add_argument('--pattern', '-p', type=str, default='**/*_depth.exr',
                        help='批量模式下的文件匹配模式 (默认: **/*_depth.exr)')
    
    args = parser.parse_args()
    
    # 检查模式
    if args.batch:
        # 批量模式
        if not args.output:
            print("❌ 错误: 批量模式需要指定输出目录 (--output)")
            return 1
        
        batch_visualize_depth(
            input_dir=args.batch,
            output_dir=args.output,
            colormap=args.colormap,
            max_valid_depth=args.max_valid,
            save_simple=args.simple,
            invert=args.invert,
            pattern=args.pattern
        )
    elif args.input:
        # 单文件模式
        if not args.output and not args.show:
            print("❌ 错误: 请指定输出路径 (--output) 或显示选项 (--show)")
            return 1
        
        visualize_single_depth(
            exr_path=args.input,
            output_path=args.output,
            colormap=args.colormap,
            max_valid_depth=args.max_valid,
            show=args.show,
            save_simple=args.simple,
            invert=args.invert
        )
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

