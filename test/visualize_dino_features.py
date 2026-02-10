#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINO特征可视化工具

功能：
1. 提取DINO特征
2. 多种可视化方法：
   - PCA降维到RGB可视化
   - 特征图可视化（映射回原图位置）
   - t-SNE/UMAP降维可视化
   - 特征统计信息
3. 支持单视图和多视图
4. 保存可视化结果用于论文
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional, List, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("⚠️  警告: umap-learn未安装，将无法使用UMAP可视化")

# 添加项目路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hy3dshape'))

from hy3dshape.models.conditioner import DinoImageEncoderMV, DinoImageEncoder


def load_image(image_path: str, target_size: Optional[int] = None) -> np.ndarray:
    """
    加载图像
    
    Args:
        image_path: 图像路径
        target_size: 目标尺寸（可选）
    
    Returns:
        图像数组 (H, W, 3)，值范围[0, 255]
    """
    img = Image.open(image_path).convert('RGB')
    if target_size:
        img = img.resize((target_size, target_size), Image.BILINEAR)
    img_array = np.array(img)
    return img_array


def load_multiview_images(image_paths: List[str], target_size: Optional[int] = None) -> np.ndarray:
    """
    加载多视图图像
    
    Args:
        image_paths: 图像路径列表
        target_size: 目标尺寸（可选）
    
    Returns:
        图像数组 (num_views, H, W, 3)，值范围[0, 255]
    """
    images = []
    for path in image_paths:
        img = load_image(path, target_size)
        images.append(img)
    return np.stack(images, axis=0)


def preprocess_image_for_dino(img: np.ndarray, image_size: int = 518) -> torch.Tensor:
    """
    预处理图像用于DINO编码器
    
    Args:
        img: 图像数组 (H, W, 3)，值范围[0, 255]
        image_size: DINO输入尺寸
    
    Returns:
        预处理后的tensor (1, 3, image_size, image_size)，值范围[-1, 1]
    """
    # 转换为PIL Image
    img_pil = Image.fromarray(img.astype(np.uint8))
    
    # 调整大小和中心裁剪
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize(image_size, transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    tensor = transform(img_pil)
    return tensor.unsqueeze(0)  # 添加batch维度


def extract_dino_features(
    encoder: nn.Module,
    images: Union[np.ndarray, List[np.ndarray]],
    image_size: int = 518,
    device: str = 'cuda'
) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    提取DINO特征
    
    Args:
        encoder: DINO编码器
        images: 图像数组或图像列表
        image_size: 图像尺寸
        device: 设备
    
    Returns:
        features: 特征tensor (batch_size, num_patches, hidden_size)
        patch_grid: patch网格尺寸 (h_patches, w_patches)
    """
    encoder.eval()
    encoder = encoder.to(device)
    
    # 处理输入
    if isinstance(images, list):
        images = np.stack(images, axis=0)
    
    if len(images.shape) == 3:  # 单图像 (H, W, 3)
        images = images[np.newaxis, ...]  # (1, H, W, 3)
    
    if len(images.shape) == 4 and images.shape[-1] == 3:  # (num_views, H, W, 3) 或 (batch, H, W, 3)
        # 转换为tensor并预处理
        batch_tensors = []
        for i in range(images.shape[0]):
            tensor = preprocess_image_for_dino(images[i], image_size)
            batch_tensors.append(tensor)
        image_tensor = torch.cat(batch_tensors, dim=0)  # (batch, 3, H, W)
    else:
        raise ValueError(f"不支持的图像形状: {images.shape}")
    
    # 判断是单视图还是多视图编码器
    is_multiview = isinstance(encoder, DinoImageEncoderMV)
    
    if is_multiview:
        # 多视图：需要 (batch=1, num_views, 3, H, W)
        # image_tensor当前是 (num_views, 3, H, W)，需要添加batch维度并调整顺序
        num_views = image_tensor.shape[0]
        image_tensor = image_tensor.unsqueeze(0)  # (1, num_views, 3, H, W)
    else:
        # 单视图：已经是 (batch, 3, H, W)
        pass
    
    image_tensor = image_tensor.to(device)
    
    # 提取特征
    with torch.no_grad():
        features = encoder(image_tensor, value_range=(-1, 1))
    
    # 计算patch网格尺寸
    patch_size = 14  # DINO的patch size
    h_patches = image_size // patch_size
    w_patches = image_size // patch_size
    patch_grid = (h_patches, w_patches)
    
    return features, patch_grid


def visualize_features_pca(
    features: np.ndarray,
    patch_grid: Tuple[int, int],
    original_image: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    title: str = "DINO特征PCA可视化",
    create_figure: bool = True
) -> np.ndarray:
    """
    使用PCA将特征降维到3维并可视化
    
    Args:
        features: 特征数组 (num_patches, hidden_size)
        patch_grid: patch网格尺寸 (h_patches, w_patches)
        original_image: 原始图像（可选，用于叠加）
        save_path: 保存路径
        title: 标题
        create_figure: 是否创建matplotlib图形
    
    Returns:
        可视化图像
    """
    num_patches, hidden_size = features.shape
    h_patches, w_patches = patch_grid
    
    # PCA降维到3维
    pca = PCA(n_components=3)
    features_3d = pca.fit_transform(features)
    
    # 归一化到[0, 1]
    features_3d = (features_3d - features_3d.min(axis=0)) / (features_3d.max(axis=0) - features_3d.min(axis=0) + 1e-8)
    
    # 重塑为图像形状
    feature_map = features_3d.reshape(h_patches, w_patches, 3)
    
    # 上采样到原图尺寸
    if original_image is not None:
        target_h, target_w = original_image.shape[:2]
    else:
        target_h, target_w = h_patches * 14, w_patches * 14
    
    feature_map_upsampled = cv2.resize(
        feature_map,
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST
    )
    
    # 创建可视化
    if create_figure:
        fig, axes = plt.subplots(1, 2 if original_image is not None else 1, figsize=(15, 8))
        if original_image is None:
            axes = [axes]
        
        # 显示特征图
        axes[0].imshow(feature_map_upsampled)
        axes[0].set_title(f'{title}\nPCA解释方差比: {pca.explained_variance_ratio_.sum():.3f}', fontsize=12)
        axes[0].axis('off')
        
        # 如果提供了原图，显示叠加效果
        if original_image is not None:
            # 叠加显示
            overlay = (0.6 * original_image / 255.0 + 0.4 * feature_map_upsampled)
            overlay = np.clip(overlay, 0, 1)
            axes[1].imshow(overlay)
            axes[1].set_title('特征图叠加原图', fontsize=12)
            axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 已保存PCA可视化到: {save_path}")
            plt.close()
        else:
            plt.close()
    
    return feature_map_upsampled


def visualize_features_tsne(
    features: np.ndarray,
    patch_grid: Tuple[int, int],
    original_image: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    create_figure: bool = True
) -> np.ndarray:
    """
    使用t-SNE将特征降维到2维并可视化
    
    Args:
        features: 特征数组 (num_patches, hidden_size)
        patch_grid: patch网格尺寸 (h_patches, w_patches)
        original_image: 原始图像（可选）
        save_path: 保存路径
        perplexity: t-SNE的perplexity参数
        n_iter: 迭代次数
    
    Returns:
        可视化图像
    """
    num_patches, hidden_size = features.shape
    h_patches, w_patches = patch_grid
    
    # 如果patch太多，先降维
    if num_patches > 1000:
        print(f"⚠️  Patch数量较多({num_patches})，先使用PCA降维...")
        pca_pre = PCA(n_components=50)
        features_reduced = pca_pre.fit_transform(features)
        print(f"✅ PCA降维完成，解释方差比: {pca_pre.explained_variance_ratio_.sum():.3f}")
    else:
        features_reduced = features
    
    # t-SNE降维到2维
    print("🔄 运行t-SNE降维（可能需要一些时间）...")
    tsne = TSNE(n_components=2, perplexity=min(perplexity, num_patches - 1), 
                n_iter=n_iter, random_state=42, verbose=1)
    features_2d = tsne.fit_transform(features_reduced)
    
    # 归一化到[0, 1]
    features_2d = (features_2d - features_2d.min(axis=0)) / (features_2d.max(axis=0) - features_2d.min(axis=0) + 1e-8)
    
    # 重塑为图像形状（需要映射回2D）
    feature_map = features_2d.reshape(h_patches, w_patches, 2)
    
    # 转换为RGB（使用2个通道映射到RGB）
    feature_map_rgb = np.zeros((h_patches, w_patches, 3))
    feature_map_rgb[:, :, 0] = feature_map[:, :, 0]  # R
    feature_map_rgb[:, :, 1] = feature_map[:, :, 1]  # G
    feature_map_rgb[:, :, 2] = (feature_map[:, :, 0] + feature_map[:, :, 1]) / 2  # B
    
    # 上采样
    if original_image is not None:
        target_h, target_w = original_image.shape[:2]
    else:
        target_h, target_w = h_patches * 14, w_patches * 14
    
    feature_map_upsampled = cv2.resize(
        feature_map_rgb,
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST
    )
    
    # 创建可视化
    if create_figure:
        fig, axes = plt.subplots(1, 2 if original_image is not None else 1, figsize=(15, 8))
        if original_image is None:
            axes = [axes]
        
        axes[0].imshow(feature_map_upsampled)
        axes[0].set_title('DINO特征t-SNE可视化', fontsize=12)
        axes[0].axis('off')
        
        if original_image is not None:
            overlay = (0.6 * original_image / 255.0 + 0.4 * feature_map_upsampled)
            overlay = np.clip(overlay, 0, 1)
            axes[1].imshow(overlay)
            axes[1].set_title('特征图叠加原图', fontsize=12)
            axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 已保存t-SNE可视化到: {save_path}")
            plt.close()
        else:
            plt.close()
    
    return feature_map_upsampled


def visualize_features_umap(
    features: np.ndarray,
    patch_grid: Tuple[int, int],
    original_image: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    create_figure: bool = True
) -> np.ndarray:
    """
    使用UMAP将特征降维到2维并可视化
    
    Args:
        features: 特征数组 (num_patches, hidden_size)
        patch_grid: patch网格尺寸 (h_patches, w_patches)
        original_image: 原始图像（可选）
        save_path: 保存路径
        n_neighbors: UMAP的邻居数
        min_dist: UMAP的最小距离
    
    Returns:
        可视化图像
    """
    if not HAS_UMAP:
        raise ImportError("需要安装umap-learn: pip install umap-learn")
    
    num_patches, hidden_size = features.shape
    h_patches, w_patches = patch_grid
    
    # UMAP降维到2维
    print("🔄 运行UMAP降维...")
    umap_model = UMAP(n_components=2, n_neighbors=min(n_neighbors, num_patches - 1),
                      min_dist=min_dist, random_state=42, verbose=True)
    features_2d = umap_model.fit_transform(features)
    
    # 归一化到[0, 1]
    features_2d = (features_2d - features_2d.min(axis=0)) / (features_2d.max(axis=0) - features_2d.min(axis=0) + 1e-8)
    
    # 重塑为图像形状
    feature_map = features_2d.reshape(h_patches, w_patches, 2)
    
    # 转换为RGB
    feature_map_rgb = np.zeros((h_patches, w_patches, 3))
    feature_map_rgb[:, :, 0] = feature_map[:, :, 0]
    feature_map_rgb[:, :, 1] = feature_map[:, :, 1]
    feature_map_rgb[:, :, 2] = (feature_map[:, :, 0] + feature_map[:, :, 1]) / 2
    
    # 上采样
    if original_image is not None:
        target_h, target_w = original_image.shape[:2]
    else:
        target_h, target_w = h_patches * 14, w_patches * 14
    
    feature_map_upsampled = cv2.resize(
        feature_map_rgb,
        (target_w, target_h),
        interpolation=cv2.INTER_NEAREST
    )
    
    # 创建可视化
    if create_figure:
        fig, axes = plt.subplots(1, 2 if original_image is not None else 1, figsize=(15, 8))
        if original_image is None:
            axes = [axes]
        
        axes[0].imshow(feature_map_upsampled)
        axes[0].set_title('DINO特征UMAP可视化', fontsize=12)
        axes[0].axis('off')
        
        if original_image is not None:
            overlay = (0.6 * original_image / 255.0 + 0.4 * feature_map_upsampled)
            overlay = np.clip(overlay, 0, 1)
            axes[1].imshow(overlay)
            axes[1].set_title('特征图叠加原图', fontsize=12)
            axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 已保存UMAP可视化到: {save_path}")
            plt.close()
        else:
            plt.close()
    
    return feature_map_upsampled


def visualize_feature_statistics(
    features: np.ndarray,
    patch_grid: Tuple[int, int],
    save_path: Optional[str] = None
):
    """
    可视化特征统计信息
    
    Args:
        features: 特征数组 (num_patches, hidden_size)
        patch_grid: patch网格尺寸 (h_patches, w_patches)
        save_path: 保存路径
    """
    num_patches, hidden_size = features.shape
    h_patches, w_patches = patch_grid
    
    # 计算统计信息
    feature_mean = features.mean(axis=0)  # (hidden_size,)
    feature_std = features.std(axis=0)   # (hidden_size,)
    feature_norm = np.linalg.norm(features, axis=1)  # (num_patches,)
    
    # 创建可视化
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 特征均值分布
    ax1 = plt.subplot(2, 3, 1)
    ax1.hist(feature_mean, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_title(f'特征均值分布\n均值: {feature_mean.mean():.4f}, 标准差: {feature_mean.std():.4f}')
    ax1.set_xlabel('特征值')
    ax1.set_ylabel('频数')
    ax1.grid(True, alpha=0.3)
    
    # 2. 特征标准差分布
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(feature_std, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_title(f'特征标准差分布\n均值: {feature_std.mean():.4f}')
    ax2.set_xlabel('标准差')
    ax2.set_ylabel('频数')
    ax2.grid(True, alpha=0.3)
    
    # 3. Patch特征范数分布
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(feature_norm, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax3.set_title(f'Patch特征范数分布\n均值: {feature_norm.mean():.4f}, 标准差: {feature_norm.std():.4f}')
    ax3.set_xlabel('L2范数')
    ax3.set_ylabel('频数')
    ax3.grid(True, alpha=0.3)
    
    # 4. 特征范数热图
    ax4 = plt.subplot(2, 3, 4)
    norm_map = feature_norm.reshape(h_patches, w_patches)
    im = ax4.imshow(norm_map, cmap='hot', interpolation='nearest')
    ax4.set_title('特征范数热图')
    ax4.axis('off')
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
    
    # 5. 前几个主成分的贡献
    ax5 = plt.subplot(2, 3, 5)
    pca = PCA()
    pca.fit(features)
    explained_var = pca.explained_variance_ratio_[:20]  # 前20个主成分
    ax5.bar(range(len(explained_var)), explained_var, alpha=0.7, color='purple', edgecolor='black')
    ax5.set_title(f'PCA主成分贡献（前20个）\n累计解释方差: {explained_var.sum():.4f}')
    ax5.set_xlabel('主成分索引')
    ax5.set_ylabel('解释方差比')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. 特征相关性矩阵（采样）
    ax6 = plt.subplot(2, 3, 6)
    # 采样部分特征维度计算相关性
    sample_size = min(50, hidden_size)
    sample_indices = np.linspace(0, hidden_size - 1, sample_size, dtype=int)
    feature_sample = features[:, sample_indices]
    corr_matrix = np.corrcoef(feature_sample.T)
    im = ax6.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    ax6.set_title(f'特征维度相关性矩阵（采样{sample_size}维）')
    ax6.set_xlabel('特征维度索引')
    ax6.set_ylabel('特征维度索引')
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 已保存特征统计可视化到: {save_path}")
    else:
        plt.show()


def visualize_multiview_features(
    features: torch.Tensor,
    patch_grid: Tuple[int, int],
    original_images: List[np.ndarray],
    view_names: List[str],
    save_path: Optional[str] = None,
    method: str = 'pca'
):
    """
    可视化多视图特征
    
    Args:
        features: 特征tensor (batch_size, num_views * num_patches, hidden_size)
        patch_grid: patch网格尺寸
        original_images: 原始图像列表
        view_names: 视图名称列表
        save_path: 保存路径
        method: 可视化方法 ('pca', 'tsne', 'umap')
    """
    features_np = features.cpu().numpy()
    batch_size, total_patches, hidden_size = features_np.shape
    
    h_patches, w_patches = patch_grid
    num_views = len(original_images)
    patches_per_view = h_patches * w_patches
    
    # 分离每个视图的特征
    fig, axes = plt.subplots(num_views, 2, figsize=(16, 4 * num_views))
    if num_views == 1:
        axes = axes[np.newaxis, :]
    
    for view_idx in range(num_views):
        start_idx = view_idx * patches_per_view
        end_idx = (view_idx + 1) * patches_per_view
        view_features = features_np[0, start_idx:end_idx, :]  # (patches_per_view, hidden_size)
        original_img = original_images[view_idx]
        
        # 根据方法选择可视化
        if method == 'pca':
            vis_img = visualize_features_pca(
                view_features, patch_grid, original_img, save_path=None, create_figure=False
            )
            title = f'{view_names[view_idx]} - PCA可视化'
        elif method == 'tsne':
            vis_img = visualize_features_tsne(
                view_features, patch_grid, original_img, save_path=None, create_figure=False
            )
            title = f'{view_names[view_idx]} - t-SNE可视化'
        elif method == 'umap':
            vis_img = visualize_features_umap(
                view_features, patch_grid, original_img, save_path=None, create_figure=False
            )
            title = f'{view_names[view_idx]} - UMAP可视化'
        else:
            raise ValueError(f"未知的可视化方法: {method}")
        
        # 显示原图和特征图
        axes[view_idx, 0].imshow(original_img)
        axes[view_idx, 0].set_title(f'{view_names[view_idx]} - 原图')
        axes[view_idx, 0].axis('off')
        
        axes[view_idx, 1].imshow(vis_img)
        axes[view_idx, 1].set_title(title)
        axes[view_idx, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 已保存多视图可视化到: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='DINO特征可视化工具')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='输入图像路径（单图像）或包含多个图像路径的目录')
    parser.add_argument('--output', '-o', type=str, default='./dino_visualizations',
                       help='输出目录')
    parser.add_argument('--method', '-m', type=str, default='pca',
                       choices=['pca', 'tsne', 'umap', 'stats', 'all'],
                       help='可视化方法')
    parser.add_argument('--image-size', type=int, default=518,
                       help='DINO输入图像尺寸')
    parser.add_argument('--model-version', type=str, default='facebook/dinov2-large',
                       help='DINO模型版本')
    parser.add_argument('--multiview', action='store_true',
                       help='是否使用多视图模式')
    parser.add_argument('--view-num', type=int, default=4,
                       help='多视图模式下的视图数量')
    parser.add_argument('--device', type=str, default='cuda',
                       help='计算设备 (cuda/cpu)')
    parser.add_argument('--overlay', action='store_true',
                       help='是否叠加原图显示')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载图像
    input_path = Path(args.input)
    if input_path.is_file():
        # 单图像
        print(f"📷 加载图像: {input_path}")
        images = [load_image(str(input_path))]
        view_names = [input_path.stem]
        is_multiview = False
    elif input_path.is_dir():
        # 目录：查找图像文件
        image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        image_files = sorted([f for f in input_path.iterdir() 
                             if f.suffix in image_extensions])
        if not image_files:
            raise ValueError(f"目录中没有找到图像文件: {input_path}")
        
        if args.multiview:
            # 多视图模式：使用所有图像
            images = [load_image(str(f)) for f in image_files[:args.view_num]]
            view_names = [f.stem for f in image_files[:args.view_num]]
            is_multiview = True
        else:
            # 单视图模式：只使用第一张
            images = [load_image(str(image_files[0]))]
            view_names = [image_files[0].stem]
            is_multiview = False
    else:
        raise ValueError(f"无效的输入路径: {input_path}")
    
    print(f"✅ 加载了 {len(images)} 张图像")
    
    # 创建编码器
    print(f"🔧 创建DINO编码器: {args.model_version}")
    if is_multiview:
        encoder = DinoImageEncoderMV(
            version=args.model_version,
            image_size=args.image_size,
            view_num=args.view_num
        )
    else:
        encoder = DinoImageEncoder(
            version=args.model_version,
            image_size=args.image_size
        )
    
    # 提取特征
    print("🔄 提取DINO特征...")
    if is_multiview:
        images_array = np.stack(images, axis=0)  # (num_views, H, W, 3)
    else:
        images_array = images[0]  # (H, W, 3)
    
    features, patch_grid = extract_dino_features(
        encoder, images_array, args.image_size, args.device
    )
    
    print(f"✅ 特征提取完成")
    print(f"   特征形状: {features.shape}")
    print(f"   Patch网格: {patch_grid}")
    
    # 可视化
    methods = ['pca', 'tsne', 'umap', 'stats'] if args.method == 'all' else [args.method]
    
    for method in methods:
        if method == 'stats':
            # 统计信息可视化
            if is_multiview:
                # 合并所有视图的特征
                features_combined = features[0].cpu().numpy()  # (total_patches, hidden_size)
            else:
                features_combined = features[0].cpu().numpy()  # (num_patches, hidden_size)
            
            save_path = output_dir / f'dino_features_stats_{input_path.stem}.png'
            visualize_feature_statistics(features_combined, patch_grid, str(save_path))
        
        elif method == 'pca':
            if is_multiview:
                save_path = output_dir / f'dino_features_multiview_pca_{input_path.stem}.png'
                visualize_multiview_features(
                    features, patch_grid, images, view_names, str(save_path), 'pca'
                )
            else:
                save_path = output_dir / f'dino_features_pca_{input_path.stem}.png'
                visualize_features_pca(
                    features[0].cpu().numpy(), patch_grid,
                    images[0] if args.overlay else None,
                    str(save_path)
                )
        
        elif method == 'tsne':
            if is_multiview:
                save_path = output_dir / f'dino_features_multiview_tsne_{input_path.stem}.png'
                visualize_multiview_features(
                    features, patch_grid, images, view_names, str(save_path), 'tsne'
                )
            else:
                save_path = output_dir / f'dino_features_tsne_{input_path.stem}.png'
                visualize_features_tsne(
                    features[0].cpu().numpy(), patch_grid,
                    images[0] if args.overlay else None,
                    str(save_path)
                )
        
        elif method == 'umap':
            if not HAS_UMAP:
                print("⚠️  跳过UMAP可视化（需要安装umap-learn）")
                continue
            
            if is_multiview:
                save_path = output_dir / f'dino_features_multiview_umap_{input_path.stem}.png'
                visualize_multiview_features(
                    features, patch_grid, images, view_names, str(save_path), 'umap'
                )
            else:
                save_path = output_dir / f'dino_features_umap_{input_path.stem}.png'
                visualize_features_umap(
                    features[0].cpu().numpy(), patch_grid,
                    images[0] if args.overlay else None,
                    str(save_path)
                )
    
    print(f"\n✅ 所有可视化完成！结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
