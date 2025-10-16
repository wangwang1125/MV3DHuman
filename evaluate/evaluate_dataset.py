#!/usr/bin/env python3
"""
针对特定数据集格式的评估脚本

数据集格式:
|obj名称
|--geo_data
|----obj名称_watertight.obj  (真实mesh)
|--render_cond
|----mesh.ply  (生成mesh)

比较每个样本的生成mesh (mesh.ply) 和真实mesh (watertight.obj)
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import argparse

from mesh_metrics import evaluate_mesh_pair, print_evaluation_results


def find_dataset_samples(dataset_dir: str) -> List[Tuple[str, str, str]]:
    """
    查找数据集中的样本
    
    数据格式:
    - 真实mesh: obj名称/geo_data/obj名称_watertight.obj
    - 生成mesh: obj名称/render_cond/mesh.ply
    
    Args:
        dataset_dir: 数据集根目录
    
    Returns:
        [(样本名, 生成mesh路径, 真实mesh路径), ...]
    """
    dataset_dir = Path(dataset_dir)
    
    if not dataset_dir.exists():
        print(f"❌ 错误: 数据集目录不存在: {dataset_dir}")
        return []
    
    samples = []
    
    # 遍历所有子目录
    for obj_dir in sorted(dataset_dir.iterdir()):
        if not obj_dir.is_dir():
            continue
        
        obj_name = obj_dir.name
        
        # 查找生成mesh
        gen_mesh = obj_dir / "render_cond" / "mesh.ply"
        
        # 查找真实mesh
        gt_mesh = obj_dir / "geo_data" / f"{obj_name}_watertight.obj"
        
        # 检查文件是否存在
        if not gen_mesh.exists():
            print(f"⚠️  警告: 未找到生成mesh: {gen_mesh}")
            continue
        
        if not gt_mesh.exists():
            print(f"⚠️  警告: 未找到真实mesh: {gt_mesh}")
            continue
        
        samples.append((obj_name, str(gen_mesh), str(gt_mesh)))
    
    return samples


def evaluate_dataset(dataset_dir: str,
                     output_json: str = None,
                     n_points: int = 10000,
                     use_icp: bool = True,
                     f_score_threshold: float = 0.02) -> Dict:
    """
    评估整个数据集
    
    Args:
        dataset_dir: 数据集根目录
        output_json: 输出JSON文件路径
        n_points: 采样点数
        use_icp: 是否使用ICP对齐
        f_score_threshold: F-Score阈值
    
    Returns:
        评估结果字典
    """
    # 查找样本
    samples = find_dataset_samples(dataset_dir)
    
    if not samples:
        print("❌ 错误: 未找到任何有效样本")
        return None
    
    print(f"📊 找到 {len(samples)} 个有效样本\n")
    
    # 批量评估
    all_results = {}
    failed = []
    
    for sample_name, gen_path, gt_path in tqdm(samples, desc="评估进度"):
        try:
            results = evaluate_mesh_pair(
                gen_path, gt_path,
                n_points=n_points,
                use_icp=use_icp,
                f_score_threshold=f_score_threshold
            )
            all_results[sample_name] = results
        except Exception as e:
            print(f"\n❌ 评估失败: {sample_name}")
            print(f"   生成mesh: {gen_path}")
            print(f"   真实mesh: {gt_path}")
            print(f"   错误: {str(e)}\n")
            failed.append(sample_name)
    
    # 计算统计信息
    if all_results:
        metrics = ['chamfer_distance', 'emd', 'mne', 'precision', 'recall', 'f_score']
        stats = {}
        
        for metric in metrics:
            values = [result[metric] for result in all_results.values()]
            stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        # 汇总结果
        summary = {
            'dataset_dir': str(dataset_dir),
            'total_samples': len(samples),
            'successful': len(all_results),
            'failed': len(failed),
            'failed_samples': failed,
            'statistics': stats,
            'individual_results': all_results,
            'config': {
                'n_points': n_points,
                'use_icp': use_icp,
                'f_score_threshold': f_score_threshold
            }
        }
        
        # 保存结果
        if output_json:
            output_path = Path(output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"\n💾 结果已保存到: {output_json}")
        
        return summary
    
    return None


def print_summary(summary: Dict):
    """打印汇总统计"""
    print("\n" + "="*80)
    print("评估汇总统计")
    print("="*80)
    
    print(f"\n📊 样本统计:")
    print(f"  数据集目录: {summary['dataset_dir']}")
    print(f"  总样本数: {summary['total_samples']}")
    print(f"  成功评估: {summary['successful']}")
    print(f"  失败评估: {summary['failed']}")
    
    if summary['failed'] > 0:
        print(f"\n❌ 失败样本:")
        for name in summary['failed_samples']:
            print(f"  - {name}")
    
    print(f"\n📈 指标统计:")
    print(f"\n{'指标':<30} {'均值':<12} {'标准差':<12} {'中位数':<12} {'最小值':<12} {'最大值':<12}")
    print("-"*90)
    
    stats = summary['statistics']
    
    metric_names = {
        'chamfer_distance': 'Chamfer Distance',
        'emd': 'EMD',
        'mne': 'Mean Normal Error',
        'precision': 'Precision',
        'recall': 'Recall',
        'f_score': 'F-Score'
    }
    
    for metric, display_name in metric_names.items():
        s = stats[metric]
        print(f"{display_name:<30} {s['mean']:<12.6f} {s['std']:<12.6f} {s['median']:<12.6f} {s['min']:<12.6f} {s['max']:<12.6f}")
    
    print("\n" + "="*80)
    
    # 打印前5个最好和最差的样本
    results = summary['individual_results']
    
    if len(results) > 0:
        # 按CD排序
        sorted_by_cd = sorted(results.items(), key=lambda x: x[1]['chamfer_distance'])
        
        n_show = min(5, len(results))
        
        print(f"\n🏆 Chamfer Distance 最好的{n_show}个样本 (越小越好):")
        for i, (name, result) in enumerate(sorted_by_cd[:n_show], 1):
            print(f"{i}. {name}")
            print(f"   CD={result['chamfer_distance']:.6f}, EMD={result['emd']:.6f}, F-Score={result['f_score']:.4f}")
        
        if len(results) > 5:
            print(f"\n⚠️  Chamfer Distance 最差的{n_show}个样本:")
            for i, (name, result) in enumerate(sorted_by_cd[-n_show:][::-1], 1):
                print(f"{i}. {name}")
                print(f"   CD={result['chamfer_distance']:.6f}, EMD={result['emd']:.6f}, F-Score={result['f_score']:.4f}")
        
        # 按F-Score排序
        sorted_by_fscore = sorted(results.items(), key=lambda x: x[1]['f_score'], reverse=True)
        
        print(f"\n🏆 F-Score 最好的{n_show}个样本 (越大越好):")
        for i, (name, result) in enumerate(sorted_by_fscore[:n_show], 1):
            print(f"{i}. {name}")
            print(f"   F-Score={result['f_score']:.4f}, CD={result['chamfer_distance']:.6f}, EMD={result['emd']:.6f}")
        
        if len(results) > 5:
            print(f"\n⚠️  F-Score 最差的{n_show}个样本:")
            for i, (name, result) in enumerate(sorted_by_fscore[-n_show:][::-1], 1):
                print(f"{i}. {name}")
                print(f"   F-Score={result['f_score']:.4f}, CD={result['chamfer_distance']:.6f}, EMD={result['emd']:.6f}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='评估数据集中生成mesh vs 真实mesh')
    parser.add_argument('--dataset_dir', '-d', type=str, 
                        default='/mnt/g/mini_mv_depth_trainset/preprocessed/',
                        help='数据集根目录')
    parser.add_argument('--output', '-o', type=str, 
                        default='dataset_evaluation_results.json',
                        help='输出JSON文件路径')
    parser.add_argument('--n_points', '-n', type=int, default=10000,
                        help='采样点数 (默认: 10000)')
    parser.add_argument('--no_icp', action='store_true',
                        help='禁用ICP对齐')
    parser.add_argument('--f_threshold', '-f', type=float, default=0.01,
                        help='F-Score距离阈值 (默认: 0.01)')
    
    args = parser.parse_args()
    
    print("🚀 开始评估数据集...")
    print(f"数据集目录: {args.dataset_dir}")
    print(f"输出文件: {args.output}")
    print(f"采样点数: {args.n_points}")
    print(f"使用ICP对齐: {not args.no_icp}")
    print(f"F-Score阈值: {args.f_threshold}\n")
    
    summary = evaluate_dataset(
        dataset_dir=args.dataset_dir,
        output_json=args.output,
        n_points=args.n_points,
        use_icp=not args.no_icp,
        f_score_threshold=args.f_threshold
    )
    
    if summary:
        print_summary(summary)
        print("\n✅ 评估完成!")
        return 0
    else:
        print("\n❌ 评估失败!")
        return 1


if __name__ == "__main__":
    exit(main())

