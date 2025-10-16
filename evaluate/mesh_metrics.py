#!/usr/bin/env python3
"""
3D Mesh评估指标
包括Chamfer Distance (CD)、Earth Mover's Distance (EMD)和F-Score

处理生成mesh与原始mesh位置、大小不确定的问题：
1. 归一化：将点云缩放到单位球内
2. 中心化：将点云中心移到原点
3. 可选ICP对齐
"""

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from typing import Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


def load_mesh(mesh_path: str) -> trimesh.Trimesh:
    """
    加载mesh文件
    
    Args:
        mesh_path: mesh文件路径 (.obj, .ply等)
    
    Returns:
        trimesh.Trimesh对象
    """
    mesh = trimesh.load(mesh_path, force='mesh', process=False)
    return mesh


def sample_points_from_mesh(mesh: trimesh.Trimesh, n_points: int = 10000, seed: int = 42) -> np.ndarray:
    """
    从mesh表面均匀采样点云
    
    Args:
        mesh: trimesh对象
        n_points: 采样点数
        seed: 随机种子
    
    Returns:
        采样的点云 (n_points, 3)
    """
    np.random.seed(seed)
    points, _ = trimesh.sample.sample_surface(mesh, n_points)
    return points


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """
    归一化点云：中心化并缩放到单位球内
    
    Args:
        points: 输入点云 (N, 3)
    
    Returns:
        归一化后的点云 (N, 3)
    """
    # 中心化
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid
    
    # 缩放到单位球
    max_dist = np.max(np.linalg.norm(points_centered, axis=1))
    if max_dist > 0:
        points_normalized = points_centered / max_dist
    else:
        points_normalized = points_centered
    
    return points_normalized


def align_point_clouds_icp(source: np.ndarray, target: np.ndarray, 
                            max_iterations: int = 50, tolerance: float = 1e-6) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用ICP算法对齐两个点云（仅旋转和平移，保持归一化的缩放）
    
    Args:
        source: 源点云 (N, 3)，待对齐的点云
        target: 目标点云 (M, 3)，参考点云
        max_iterations: 最大迭代次数
        tolerance: 收敛阈值
    
    Returns:
        (对齐后的源点云, 旋转矩阵, 平移向量)
    """
    aligned = source.copy()
    prev_error = float('inf')
    R_total = np.eye(3)
    t_total = np.zeros(3)
    
    for i in range(max_iterations):
        # 构建KD树找最近邻
        tree = cKDTree(target)
        distances, indices = tree.query(aligned)
        
        # 计算当前误差
        current_error = np.mean(distances)
        
        # 检查收敛
        if abs(prev_error - current_error) < tolerance:
            break
        prev_error = current_error
        
        # 计算质心
        source_centroid = np.mean(aligned, axis=0)
        target_centroid = np.mean(target[indices], axis=0)
        
        # 中心化
        source_centered = aligned - source_centroid
        target_centered = target[indices] - target_centroid
        
        # 计算旋转矩阵 (SVD)
        H = source_centered.T @ target_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # 确保是旋转矩阵（行列式为1）
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        # 计算平移
        t = target_centroid - R @ source_centroid
        
        # 应用变换
        aligned = (R @ aligned.T).T + t
        
        # 累积变换
        R_total = R @ R_total
        t_total = R @ t_total + t
    
    return aligned, R_total, t_total


def apply_transform_to_mesh(mesh: trimesh.Trimesh, rotation: np.ndarray, translation: np.ndarray) -> trimesh.Trimesh:
    """
    应用旋转和平移变换到mesh
    
    Args:
        mesh: 输入mesh
        rotation: 3x3旋转矩阵
        translation: 3维平移向量
    
    Returns:
        变换后的mesh（新副本）
    """
    # 创建mesh副本
    mesh_transformed = mesh.copy()
    
    # 应用旋转和平移到顶点
    mesh_transformed.vertices = (rotation @ mesh_transformed.vertices.T).T + translation
    
    # 应用旋转到法向量（法向量不受平移影响）
    if hasattr(mesh_transformed, 'vertex_normals'):
        mesh_transformed.vertex_normals = (rotation @ mesh_transformed.vertex_normals.T).T
    
    # 重新计算面法向量
    mesh_transformed.face_normals = (rotation @ mesh_transformed.face_normals.T).T
    
    return mesh_transformed


def chamfer_distance(points1: np.ndarray, points2: np.ndarray) -> float:
    """
    计算Chamfer Distance (CD)
    
    CD = mean(min_dist(P1->P2)) + mean(min_dist(P2->P1))
    
    Args:
        points1: 第一个点云 (N, 3)
        points2: 第二个点云 (M, 3)
    
    Returns:
        Chamfer Distance值（越小越好）
    """
    # 构建KD树
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    
    # P1到P2的最近邻距离
    dist1, _ = tree2.query(points1)
    # P2到P1的最近邻距离
    dist2, _ = tree1.query(points2)
    
    # CD是双向平均距离之和
    cd = np.mean(dist1) + np.mean(dist2)
    
    return cd


def earth_movers_distance(points1: np.ndarray, points2: np.ndarray, use_1d_approximation: bool = False) -> float:
    """
    计算Earth Mover's Distance (EMD) / Wasserstein距离的近似
    
    Args:
        points1: 第一个点云 (N, 3)
        points2: 第二个点云 (M, 3)
        use_1d_approximation: 是否使用1D近似（分维度计算）
    
    Returns:
        EMD值（越小越好）
    
    注意：
        - 默认使用基于最近邻的3D近似，更符合几何直觉
        - 1D近似模式计算更快但可能不准确
    """
    if use_1d_approximation:
        try:
            # 尝试导入scipy.stats中的wasserstein_distance
            from scipy.stats import wasserstein_distance
            
            # 对于3D点云，我们计算每个维度的1D EMD并求和
            emd = 0.0
            for dim in range(3):
                emd += wasserstein_distance(points1[:, dim], points2[:, dim])
            
            return emd / 3.0  # 平均
        except ImportError:
            pass
    
    # 使用基于最近邻的3D近似（更准确）
    # EMD近似为单向平均距离（比CD更关注分布）
    tree = cKDTree(points2)
    dist, _ = tree.query(points1)
    return np.mean(dist)


def f_score(points1: np.ndarray, points2: np.ndarray, threshold: float = 0.01) -> Dict[str, float]:
    """
    计算F-Score（精度、召回率和F1分数）
    
    Args:
        points1: 生成的点云 (N, 3)
        points2: 真实的点云 (M, 3)
        threshold: 距离阈值，用于判断点是否匹配
    
    Returns:
        包含precision、recall、f_score的字典
    """
    # 构建KD树
    tree1 = cKDTree(points1)
    tree2 = cKDTree(points2)
    
    # 计算精度：生成点云中有多少点在阈值内能找到真实点
    dist1, _ = tree2.query(points1)
    precision = np.mean(dist1 < threshold)
    
    # 计算召回率：真实点云中有多少点在阈值内能找到生成点
    dist2, _ = tree1.query(points2)
    recall = np.mean(dist2 < threshold)
    
    # 计算F-Score (调和平均)
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f_score': float(f1)
    }


def mean_normal_error(mesh1: trimesh.Trimesh, mesh2: trimesh.Trimesh, n_points: int = 10000) -> float:
    """
    计算Mean Normal Error (MNE) - 平均法向误差
    
    度量两个mesh在表面法向量上的差异，值越小表示法向一致性越好
    
    Args:
        mesh1: 生成的mesh
        mesh2: 真实的mesh
        n_points: 采样点数
    
    Returns:
        MNE值（越小越好），范围[0, 1]，0表示法向完全一致，1表示法向完全相反
    """
    # 从两个mesh采样点和法向量
    points1, face_indices1 = trimesh.sample.sample_surface(mesh1, n_points)
    normals1 = mesh1.face_normals[face_indices1]
    
    points2, face_indices2 = trimesh.sample.sample_surface(mesh2, n_points)
    normals2 = mesh2.face_normals[face_indices2]
    
    # 为每个生成点找到真实mesh上的最近邻点
    tree = cKDTree(points2)
    distances, indices = tree.query(points1)
    
    # 获取对应的法向量
    normals2_matched = normals2[indices]
    
    # 计算法向量的余弦相似度（点积）
    # dot product范围[-1, 1]，1表示方向相同，-1表示方向相反
    cosine_similarity = np.sum(normals1 * normals2_matched, axis=1)
    
    # 裁剪到[-1, 1]范围（避免数值误差）
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    
    # 计算角度误差（弧度）
    angular_error = np.arccos(np.abs(cosine_similarity))  # abs考虑法向反向的情况
    
    # 归一化到[0, 1]范围（0度对应0，90度对应1）
    normalized_error = angular_error / (np.pi / 2)
    
    # 返回平均误差
    mne = np.mean(normalized_error)
    
    return float(mne)


def evaluate_mesh_pair(generated_mesh_path: str, 
                       ground_truth_mesh_path: str,
                       n_points: int = 10000,
                       use_icp: bool = True,
                       f_score_threshold: float = 0.02) -> Dict[str, float]:
    """
    评估一对mesh（生成mesh vs 真实mesh）
    
    Args:
        generated_mesh_path: 生成的mesh文件路径
        ground_truth_mesh_path: 真实的mesh文件路径
        n_points: 采样点数
        use_icp: 是否使用ICP对齐
        f_score_threshold: F-Score的距离阈值
    
    Returns:
        包含所有指标的字典
    """
    # 1. 加载mesh
    gen_mesh = load_mesh(generated_mesh_path)
    gt_mesh = load_mesh(ground_truth_mesh_path)
    
    # 2. 采样点云
    gen_points = sample_points_from_mesh(gen_mesh, n_points)
    gt_points = sample_points_from_mesh(gt_mesh, n_points)
    
    # 3. 归一化点云（解决大小不一致问题）
    gen_points_norm = normalize_point_cloud(gen_points)
    gt_points_norm = normalize_point_cloud(gt_points)
    
    # 4. 归一化mesh（用于MNE计算）
    # 计算归一化参数
    gen_centroid = np.mean(gen_points, axis=0)
    gen_max_dist = np.max(np.linalg.norm(gen_points - gen_centroid, axis=1))
    gt_centroid = np.mean(gt_points, axis=0)
    gt_max_dist = np.max(np.linalg.norm(gt_points - gt_centroid, axis=1))
    
    # 归一化生成mesh
    gen_mesh_norm = gen_mesh.copy()
    gen_mesh_norm.vertices = (gen_mesh_norm.vertices - gen_centroid) / gen_max_dist if gen_max_dist > 0 else gen_mesh_norm.vertices - gen_centroid
    
    # 归一化真实mesh
    gt_mesh_norm = gt_mesh.copy()
    gt_mesh_norm.vertices = (gt_mesh_norm.vertices - gt_centroid) / gt_max_dist if gt_max_dist > 0 else gt_mesh_norm.vertices - gt_centroid
    
    # 5. ICP对齐（解决位置和姿态不一致问题）
    if use_icp:
        gen_points_aligned, rotation, translation = align_point_clouds_icp(gen_points_norm, gt_points_norm)
        # 应用相同的变换到生成mesh
        gen_mesh_aligned = apply_transform_to_mesh(gen_mesh_norm, rotation, translation)
    else:
        gen_points_aligned = gen_points_norm
        gen_mesh_aligned = gen_mesh_norm
    
    # 6. 计算评估指标
    cd = chamfer_distance(gen_points_aligned, gt_points_norm)
    emd = earth_movers_distance(gen_points_aligned, gt_points_norm)
    f_score_metrics = f_score(gen_points_aligned, gt_points_norm, threshold=f_score_threshold)
    
    # 7. 计算MNE（平均法向误差） - 在对齐后的mesh上计算
    mne = mean_normal_error(gen_mesh_aligned, gt_mesh_norm, n_points)
    
    # 8. 汇总结果
    results = {
        'chamfer_distance': float(cd),
        'emd': float(emd),
        'precision': f_score_metrics['precision'],
        'recall': f_score_metrics['recall'],
        'f_score': f_score_metrics['f_score'],
        'mne': float(mne)
    }
    
    return results


def print_evaluation_results(results: Dict[str, float], name: str = ""):
    """打印评估结果"""
    if name:
        print(f"\n{'='*60}")
        print(f"评估结果: {name}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("评估结果")
        print(f"{'='*60}")
    
    print(f"Chamfer Distance (CD):         {results['chamfer_distance']:.6f}  (越小越好)")
    print(f"Earth Mover's Distance (EMD):  {results['emd']:.6f}  (越小越好)")
    print(f"Mean Normal Error (MNE):       {results['mne']:.6f}  (越小越好)")
    print(f"Precision:                     {results['precision']:.4f}")
    print(f"Recall:                        {results['recall']:.4f}")
    print(f"F-Score:                       {results['f_score']:.4f}  (越大越好)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # 示例用法
    import sys
    
    if len(sys.argv) < 3:
        print("用法: python mesh_metrics.py <生成的mesh路径> <真实mesh路径> [采样点数] [是否使用ICP]")
        print("示例: python mesh_metrics.py generated.obj ground_truth.obj 10000 1")
        sys.exit(1)
    
    gen_path = sys.argv[1]
    gt_path = sys.argv[2]
    n_points = int(sys.argv[3]) if len(sys.argv) > 3 else 10000
    use_icp = bool(int(sys.argv[4])) if len(sys.argv) > 4 else True
    
    print(f"🔍 开始评估...")
    print(f"生成mesh: {gen_path}")
    print(f"真实mesh: {gt_path}")
    print(f"采样点数: {n_points}")
    print(f"使用ICP对齐: {use_icp}")
    
    results = evaluate_mesh_pair(gen_path, gt_path, n_points, use_icp)
    print_evaluation_results(results)

