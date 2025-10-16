#!/usr/bin/env python3
"""
数据集验证脚本
检查数据集中每个样本是否包含所有必需的文件
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

def check_dataset_structure(dataset_path: str) -> Tuple[List[str], List[Dict]]:
    """
    检查数据集结构的完整性
    
    Args:
        dataset_path: 数据集根目录路径
    
    Returns:
        (valid_samples, invalid_samples): 有效样本列表和无效样本信息列表
    """
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"❌ 错误: 数据集路径不存在: {dataset_path}")
        return [], []
    
    valid_samples = []
    invalid_samples = []
    
    # 遍历所有子目录
    subdirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if not subdirs:
        print(f"⚠️  警告: 在 {dataset_path} 下没有找到任何子目录")
        return [], []
    
    print(f"📁 开始检查数据集: {dataset_path}")
    print(f"📊 发现 {len(subdirs)} 个样本目录\n")
    
    for obj_dir in sorted(subdirs):
        obj_name = obj_dir.name
        missing_files = []
        
        # 检查 geo_data 目录
        geo_data_dir = obj_dir / "geo_data"
        if not geo_data_dir.exists():
            missing_files.append("geo_data/")
        else:
            # 检查 geo_data 下的文件
            required_geo_files = [
                f"{obj_name}_sdf.npz",
                f"{obj_name}_surface.npz",
                f"{obj_name}_watertight.obj"
            ]
            for file_name in required_geo_files:
                file_path = geo_data_dir / file_name
                if not file_path.exists():
                    missing_files.append(f"geo_data/{file_name}")
        
        # 检查 render_cond 目录
        render_cond_dir = obj_dir / "render_cond"
        if not render_cond_dir.exists():
            missing_files.append("render_cond/")
        else:
            # 检查 render_cond 下的文件
            # 4个视角的图像和深度图 (000-003)
            for i in range(4):
                idx = f"{i:03d}"
                png_file = render_cond_dir / f"{idx}.png"
                depth_file = render_cond_dir / f"{idx}_depth.exr"
                
                if not png_file.exists():
                    missing_files.append(f"render_cond/{idx}.png")
                if not depth_file.exists():
                    missing_files.append(f"render_cond/{idx}_depth.exr")
            
            # 检查 mesh.ply
            mesh_file = render_cond_dir / "mesh.ply"
            if not mesh_file.exists():
                missing_files.append("render_cond/mesh.ply")
        
        # 记录结果
        if missing_files:
            invalid_samples.append({
                'name': obj_name,
                'path': str(obj_dir),
                'missing': missing_files
            })
        else:
            valid_samples.append(obj_name)
    
    return valid_samples, invalid_samples


def print_results(valid_samples: List[str], invalid_samples: List[Dict]):
    """打印验证结果"""
    total = len(valid_samples) + len(invalid_samples)
    
    print("=" * 80)
    print("验证结果汇总")
    print("=" * 80)
    print(f"✅ 有效样本: {len(valid_samples)}/{total}")
    print(f"❌ 无效样本: {len(invalid_samples)}/{total}")
    print()
    
    if invalid_samples:
        print("🔍 无效样本详情:")
        print("-" * 80)
        for i, sample in enumerate(invalid_samples, 1):
            print(f"\n{i}. 样本名称: {sample['name']}")
            print(f"   路径: {sample['path']}")
            print(f"   缺失文件 ({len(sample['missing'])}个):")
            for missing in sample['missing']:
                print(f"     - {missing}")
        print()
    
    if valid_samples:
        print("✨ 所有有效样本:")
        print("-" * 80)
        for i, sample in enumerate(valid_samples, 1):
            print(f"{i:3d}. {sample}")
        print()
    
    # 统计缺失文件类型
    if invalid_samples:
        missing_stats = defaultdict(int)
        for sample in invalid_samples:
            for missing in sample['missing']:
                missing_stats[missing] += 1
        
        print("\n📊 缺失文件统计:")
        print("-" * 80)
        for file_type, count in sorted(missing_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {file_type}: {count} 次")
    
    print("\n" + "=" * 80)
    
    # 返回状态码
    return 0 if not invalid_samples else 1


def main():
    """主函数"""
    dataset_path = "/mnt/g/mini_mv_depth_trainset/preprocessed/"
    
    print("🔍 数据集验证工具")
    print("=" * 80)
    print()
    
    valid_samples, invalid_samples = check_dataset_structure(dataset_path)
    exit_code = print_results(valid_samples, invalid_samples)
    
    if exit_code == 0:
        print("🎉 所有数据都符合要求！")
    else:
        print("⚠️  发现不完整的数据，请检查上述列表")
    
    return exit_code


if __name__ == "__main__":
    exit(main())

