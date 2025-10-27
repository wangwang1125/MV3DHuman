#!/usr/bin/env python3
"""
EXR法线图可视化工具

功能：
1. 读取EXR法线图
2. 应用颜色映射可视化
3. 显示法线统计信息
4. 保存可视化结果
5. 支持批量处理
"""

import os
import argparse
from pathlib import Path
from typing import Optional, Tuple
import numpy as np

# 设置matplotlib后端，避免Qt问题
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as mcolors

import cv2

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False
    print("⚠️  警告: imageio未安装，将使用OpenCV读取EXR文件")

try:
    import OpenEXR
    import Imath
    HAS_OPENEXR = True
except ImportError:
    HAS_OPENEXR = False
    print("⚠️  警告: OpenEXR库未安装，将尝试其他方法读取EXR文件")


def read_exr_normal(exr_path: str, debug: bool = False) -> np.ndarray:
    """
    读取EXR法线图
    
    Args:
        exr_path: EXR文件路径
        debug: 是否打印调试信息
    
    Returns:
        法线图数组 (H, W, 3) 范围 [-1, 1]
    """
    normal = None
    
    if HAS_IMAGEIO:
        try:
            # 尝试不同的imageio格式
            try:
                normal = imageio.imread(exr_path, format='EXR-FI')
            except:
                # 如果EXR-FI失败，尝试直接读取
                normal = imageio.imread(exr_path)
            
            if debug:
                print(f"   ✅ imageio成功读取: {normal.shape}, dtype: {normal.dtype}")
                print(f"   📊 原始数据范围: [{normal.min():.6f}, {normal.max():.6f}]")
                
        except Exception as e:
            if debug:
                print(f"   ⚠️  imageio读取失败: {e}，尝试OpenCV...")
    
    # 如果imageio失败，尝试使用OpenEXR库读取
    if normal is None and HAS_OPENEXR:
        try:
            exr_file = OpenEXR.InputFile(exr_path)
            header = exr_file.header()
            
            # 获取图像尺寸
            dw = header['dataWindow']
            width = dw.max.x - dw.min.x + 1
            height = dw.max.y - dw.min.y + 1
            
            # 读取法线通道 (X, Y, Z)
            channels = ['X', 'Y', 'Z']
            channel_data = {}
            
            for channel in channels:
                if channel in header['channels']:
                    channel_str = exr_file.channel(channel, Imath.PixelType(Imath.PixelType.FLOAT))
                    channel_data[channel] = np.frombuffer(channel_str, dtype=np.float32)
                    channel_data[channel] = channel_data[channel].reshape((height, width))
                else:
                    # 如果某个通道不存在，用0填充
                    channel_data[channel] = np.zeros((height, width), dtype=np.float32)
            
            # 组合成3通道图像 (X, Y, Z)
            normal = np.stack([channel_data['X'], channel_data['Y'], channel_data['Z']], axis=2)
            exr_file.close()
            
            if debug:
                print(f"   ✅ OpenEXR成功读取: {normal.shape}, dtype: {normal.dtype}")
                
        except Exception as e:
            if debug:
                print(f"   ⚠️  OpenEXR读取失败: {e}，尝试OpenCV...")
    
    # 如果OpenEXR也失败，尝试使用OpenCV读取
    if normal is None:
        try:
            normal = cv2.imread(exr_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if normal is None:
                raise ValueError(f"无法读取EXR文件: {exr_path}")
            if debug:
                print(f"   ✅ OpenCV成功读取: {normal.shape}, dtype: {normal.dtype}")
        except Exception as e:
            raise RuntimeError(f"读取EXR文件失败: {exr_path}, 错误: {e}")
    
    # 确保是3通道
    if len(normal.shape) == 2:
        # 单通道，复制为3通道
        normal = np.stack([normal] * 3, axis=-1)
    elif normal.shape[2] > 3:
        # 超过3通道，取前3个
        normal = normal[:, :, :3]
    
    # 处理无效值
    normal = np.nan_to_num(normal, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # 检查数据范围，判断是否需要从[0,1]转换到[-1,1]
    data_min = normal.min()
    data_max = normal.max()
    
    if debug:
        print(f"   📊 处理前数据范围: [{data_min:.6f}, {data_max:.6f}]")
    
    # 检查是否有有效数据（非零值）
    non_zero_mask = np.any(normal != 0, axis=2)
    non_zero_count = np.sum(non_zero_mask)
    
    if debug:
        print(f"   📊 非零像素: {non_zero_count} / {normal.size//3} ({non_zero_count/(normal.size//3)*100:.2f}%)")
    
    if non_zero_count > 0:
        # 如果有非零数据，检查数据范围
        if data_min >= 0.0 and data_max <= 1.0:
            if debug:
                print(f"   🔄 检测到[0,1]范围数据，转换到[-1,1]")
            normal = normal * 2.0 - 1.0
        elif data_min >= -1.0 and data_max <= 1.0:
            if debug:
                print(f"   ✅ 数据已在[-1,1]范围内")
        else:
            if debug:
                print(f"   ⚠️  数据范围异常: [{data_min:.6f}, {data_max:.6f}]")
                print(f"   🔄 尝试归一化到[-1,1]范围")
            # 尝试归一化到[-1,1]范围
            if data_max > data_min:
                normal = 2.0 * (normal - data_min) / (data_max - data_min) - 1.0
    else:
        if debug:
            print(f"   ⚠️  警告: 没有检测到有效数据，所有像素都是0")
    
    # 最终裁剪到[-1,1]范围
    normal = np.clip(normal, -1.0, 1.0)
    
    if debug:
        print(f"   📊 法线范围: X[{normal[:,:,0].min():.3f}, {normal[:,:,0].max():.3f}], "
              f"Y[{normal[:,:,1].min():.3f}, {normal[:,:,1].max():.3f}], "
              f"Z[{normal[:,:,2].min():.3f}, {normal[:,:,2].max():.3f}]")
    
    return normal


def compute_normal_statistics(normal: np.ndarray) -> dict:
    """
    计算法线图统计信息
    
    Args:
        normal: 法线图数组 (H, W, 3)
    
    Returns:
        统计信息字典
    """
    # 计算每个像素的法线长度
    normal_lengths = np.linalg.norm(normal, axis=2)
    
    # 有效法线（非零值，长度合理）
    valid_mask = (normal_lengths > 0.001) & (normal_lengths < 10.0)
    
    if not np.any(valid_mask):
        return {
            'x_min': -1.0, 'x_max': 1.0, 'x_mean': 0.0, 'x_std': 0.0,
            'y_min': -1.0, 'y_max': 1.0, 'y_mean': 0.0, 'y_std': 0.0,
            'z_min': -1.0, 'z_max': 1.0, 'z_mean': 0.0, 'z_std': 0.0,
            'length_min': 0.0, 'length_max': 0.0, 'length_mean': 0.0, 'length_std': 0.0,
            'valid_pixels': 0, 'total_pixels': normal.shape[0] * normal.shape[1],
            'valid_ratio': 0.0
        }
    
    valid_normal = normal[valid_mask]
    valid_lengths = normal_lengths[valid_mask]
    
    return {
        'x_min': float(np.min(valid_normal[:, 0])),
        'x_max': float(np.max(valid_normal[:, 0])),
        'x_mean': float(np.mean(valid_normal[:, 0])),
        'x_std': float(np.std(valid_normal[:, 0])),
        'y_min': float(np.min(valid_normal[:, 1])),
        'y_max': float(np.max(valid_normal[:, 1])),
        'y_mean': float(np.mean(valid_normal[:, 1])),
        'y_std': float(np.std(valid_normal[:, 1])),
        'z_min': float(np.min(valid_normal[:, 2])),
        'z_max': float(np.max(valid_normal[:, 2])),
        'z_mean': float(np.mean(valid_normal[:, 2])),
        'z_std': float(np.std(valid_normal[:, 2])),
        'length_min': float(np.min(valid_lengths)),
        'length_max': float(np.max(valid_lengths)),
        'length_mean': float(np.mean(valid_lengths)),
        'length_std': float(np.std(valid_lengths)),
        'valid_pixels': int(np.sum(valid_mask)),
        'total_pixels': int(normal.shape[0] * normal.shape[1]),
        'valid_ratio': float(np.sum(valid_mask) / (normal.shape[0] * normal.shape[1]))
    }


def visualize_normal(normal: np.ndarray, 
                    mode: str = 'rgb',
                    colormap: str = 'viridis') -> np.ndarray:
    """
    可视化法线图
    
    Args:
        normal: 法线图数组 (H, W, 3) 范围 [-1, 1]
        mode: 可视化模式 ('rgb', 'components', 'length')
        colormap: matplotlib颜色映射名称（仅用于length模式）
    
    Returns:
        RGB可视化图像 (H, W, 3) uint8
    """
    if mode == 'rgb':
        # RGB模式：直接映射XYZ到RGB
        # 将[-1,1]映射到[0,1]，然后映射到[0,255]
        normal_rgb = ((normal + 1.0) * 0.5 * 255).astype(np.uint8)
        return normal_rgb
    
    elif mode == 'components':
        # 分量模式：分别显示XYZ分量
        # 将[-1,1]映射到[0,1]
        normal_vis = (normal + 1.0) * 0.5
        # 转换为RGB uint8
        normal_rgb = (normal_vis * 255).astype(np.uint8)
        return normal_rgb
    
    elif mode == 'length':
        # 长度模式：显示法线长度
        normal_lengths = np.linalg.norm(normal, axis=2)
        # 归一化到[0,1]
        length_normalized = np.clip(normal_lengths, 0, 2.0) / 2.0
        # 应用颜色映射
        cmap = cm.get_cmap(colormap)
        length_colored = cmap(length_normalized)  # (H, W, 4) with RGBA
        # 转换为RGB uint8
        length_rgb = (length_colored[:, :, :3] * 255).astype(np.uint8)
        return length_rgb
    
    else:
        raise ValueError(f"不支持的可视化模式: {mode}")


def create_visualization_with_info(normal: np.ndarray,
                                  stats: dict,
                                  mode: str = 'rgb',
                                  colormap: str = 'viridis',
                                  filename: str = "") -> np.ndarray:
    """
    创建带统计信息的可视化图像
    
    Args:
        normal: 法线图数组
        stats: 统计信息字典
        mode: 可视化模式
        colormap: 颜色映射名称
        filename: 文件名（用于显示）
    
    Returns:
        可视化图像
    """
    # 创建figure
    fig = plt.figure(figsize=(15, 10))
    
    # 主法线图
    ax1 = plt.subplot(2, 3, (1, 2))
    normal_vis = visualize_normal(normal, mode=mode, colormap=colormap)
    ax1.imshow(normal_vis)
    ax1.set_title(f'法线图可视化 ({mode.upper()})\n{filename}', fontsize=12)
    ax1.axis('off')
    
    # XYZ分量分别显示
    for i, (component, color) in enumerate([('X', 'Reds'), ('Y', 'Greens'), ('Z', 'Blues')]):
        ax = plt.subplot(2, 3, 3 + i)
        component_data = normal[:, :, i]
        im = ax.imshow(component_data, cmap=color, vmin=-1, vmax=1)
        ax.set_title(f'{component} 分量', fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 统计信息文本
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    info_text = f"""
法线统计信息
{'='*40}

X分量范围:
  最小值: {stats['x_min']:.3f}
  最大值: {stats['x_max']:.3f}
  平均值: {stats['x_mean']:.3f}
  标准差: {stats['x_std']:.3f}

Y分量范围:
  最小值: {stats['y_min']:.3f}
  最大值: {stats['y_max']:.3f}
  平均值: {stats['y_mean']:.3f}
  标准差: {stats['y_std']:.3f}

Z分量范围:
  最小值: {stats['z_min']:.3f}
  最大值: {stats['z_max']:.3f}
  平均值: {stats['z_mean']:.3f}
  标准差: {stats['z_std']:.3f}

法线长度:
  最小值: {stats['length_min']:.3f}
  最大值: {stats['length_max']:.3f}
  平均值: {stats['length_mean']:.3f}
  标准差: {stats['length_std']:.3f}

像素信息:
  有效像素: {stats['valid_pixels']:,} / {stats['total_pixels']:,}
  有效比例: {stats['valid_ratio']*100:.2f}%

图像尺寸: {normal.shape[1]} × {normal.shape[0]}
可视化模式: {mode.upper()}
    """
    
    ax6.text(0.05, 0.95, info_text.strip(), 
             transform=ax6.transAxes,
             fontsize=9,
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


def visualize_single_normal(exr_path: str,
                           output_path: Optional[str] = None,
                           mode: str = 'rgb',
                           colormap: str = 'viridis',
                           show: bool = False,
                           save_simple: bool = False,
                           debug: bool = False):
    """
    可视化单个法线图
    
    Args:
        exr_path: EXR文件路径
        output_path: 输出图像路径
        mode: 可视化模式
        colormap: 颜色映射
        show: 是否显示图像
        save_simple: 是否保存简单版本（无统计信息）
    """
    print(f"\n{'='*60}")
    print(f"处理: {exr_path}")
    print(f"{'='*60}")
    
    # 读取法线图
    normal = read_exr_normal(exr_path, debug=debug)
    
    # 计算统计信息
    stats = compute_normal_statistics(normal)
    
    # 打印统计信息
    print(f"\nX分量范围: [{stats['x_min']:.3f}, {stats['x_max']:.3f}]")
    print(f"Y分量范围: [{stats['y_min']:.3f}, {stats['y_max']:.3f}]")
    print(f"Z分量范围: [{stats['z_min']:.3f}, {stats['z_max']:.3f}]")
    print(f"法线长度: [{stats['length_min']:.3f}, {stats['length_max']:.3f}] (平均: {stats['length_mean']:.3f})")
    print(f"有效像素: {stats['valid_pixels']:,} / {stats['total_pixels']:,} ({stats['valid_ratio']*100:.2f}%)")
    print(f"图像尺寸: {normal.shape[1]} × {normal.shape[0]}")
    
    # 生成可视化
    if save_simple:
        # 简单版本：只有法线图
        vis_img = visualize_normal(normal, mode=mode, colormap=colormap)
    else:
        # 完整版本：包含统计信息
        filename = Path(exr_path).name
        vis_img = create_visualization_with_info(normal, stats, mode, colormap, filename)
    
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


def batch_visualize_normal(input_dir: str,
                          output_dir: str,
                          mode: str = 'rgb',
                          colormap: str = 'viridis',
                          save_simple: bool = False,
                          pattern: str = "**/*_normal.exr",
                          debug: bool = False):
    """
    批量可视化法线图
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        mode: 可视化模式
        colormap: 颜色映射
        save_simple: 是否保存简单版本
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
    
    print(f"📊 找到 {len(exr_files)} 个法线图文件\n")
    
    # 处理每个文件
    for exr_file in exr_files:
        try:
            # 构建输出路径（保持相对路径结构）
            rel_path = exr_file.relative_to(input_path)
            output_file = output_path / rel_path.with_suffix('.png')
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 可视化
            visualize_single_normal(
                str(exr_file),
                str(output_file),
                mode=mode,
                colormap=colormap,
                show=False,
                save_simple=save_simple,
                debug=debug
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
        description='EXR法线图可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:

  # 可视化单个法线图
  python visualize_normal.py input.exr -o output.png

  # 显示交互式窗口
  python visualize_normal.py input.exr --show

  # 使用不同的可视化模式
  python visualize_normal.py input.exr -o output.png -m components
  python visualize_normal.py input.exr -o output.png -m length

  # 使用不同的颜色映射（length模式）
  python visualize_normal.py input.exr -o output.png -m length -c plasma

  # 批量处理
  python visualize_normal.py --batch /path/to/dataset/ -o ./visualizations/

  # 保存简单版本（无统计信息）
  python visualize_normal.py input.exr -o output.png --simple

可视化模式:
  rgb       - 直接映射XYZ到RGB (默认)
  components - 分别显示XYZ分量
  length     - 显示法线长度

可用颜色映射: turbo, jet, viridis, plasma, inferno, magma, hot, cool, rainbow
        """
    )
    
    parser.add_argument('input', type=str, nargs='?',
                        help='输入EXR文件路径（单文件模式）')
    parser.add_argument('--batch', '-b', type=str,
                        help='批量处理模式：输入目录路径')
    parser.add_argument('--output', '-o', type=str,
                        help='输出图像路径或目录')
    parser.add_argument('--mode', '-m', type=str, default='rgb',
                        choices=['rgb', 'components', 'length'],
                        help='可视化模式 (默认: rgb)')
    parser.add_argument('--colormap', '-c', type=str, default='viridis',
                        help='颜色映射 (默认: viridis)')
    parser.add_argument('--show', '-s', action='store_true',
                        help='显示可视化结果')
    parser.add_argument('--simple', action='store_true',
                        help='保存简单版本（仅法线图，无统计信息）')
    parser.add_argument('--pattern', '-p', type=str, default='**/*_normal.exr',
                        help='批量模式下的文件匹配模式 (默认: **/*_normal.exr)')
    parser.add_argument('--debug', '-d', action='store_true',
                        help='启用调试模式，显示详细信息')
    
    args = parser.parse_args()
    
    # 检查模式
    if args.batch:
        # 批量模式
        if not args.output:
            print("❌ 错误: 批量模式需要指定输出目录 (--output)")
            return 1
        
        batch_visualize_normal(
            input_dir=args.batch,
            output_dir=args.output,
            mode=args.mode,
            colormap=args.colormap,
            save_simple=args.simple,
            pattern=args.pattern,
            debug=args.debug
        )
    elif args.input:
        # 单文件模式
        if not args.output and not args.show:
            print("❌ 错误: 请指定输出路径 (--output) 或显示选项 (--show)")
            return 1
        
        visualize_single_normal(
            exr_path=args.input,
            output_path=args.output,
            mode=args.mode,
            colormap=args.colormap,
            show=args.show,
            save_simple=args.simple,
            debug=args.debug
        )
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
