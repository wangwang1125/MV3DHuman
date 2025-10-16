#!/usr/bin/env python3
"""
批量评估mesh质量
支持批量评估多个生成mesh与真实mesh的配对
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from tqdm import tqdm
import argparse

from mesh_metrics import evaluate_mesh_pair, print_evaluation_results


def find_mesh_pairs(generated_dir: str, 
                    ground_truth_dir: str,
                    generated_suffix: str = ".obj",
                    gt_suffix: str = ".obj") -> List[Tuple[str, str, str]]:
    """
    查找生成mesh和真实mesh的配对
    
    Args:
        generated_dir: 生成mesh目录
        ground_truth_dir: 真实mesh目录
        generated_suffix: 生成mesh文件后缀
        gt_suffix: 真实mesh文件后缀
    
    Returns:
        [(样本名, 生成mesh路径, 真实mesh路径), ...]
    """
    generated_dir = Path(generated_dir)
    ground_truth_dir = Path(ground_truth_dir)
    
    pairs = []
    
    # 遍历生成目录
    for gen_file in generated_dir.rglob(f"*{generated_suffix}"):
        # 获取相对路径作为样本名
        rel_path = gen_file.relative_to(generated_dir)
        sample_name = str(rel_path.with_suffix(''))
        
        # 查找对应的真实mesh
        gt_file = ground_truth_dir / rel_path.parent / f"{rel_path.stem}{gt_suffix}"
        
        if not gt_file.exists():
            # 尝试不同的命名模式
            # 尝试直接在ground_truth根目录查找
            gt_file_alt = ground_truth_dir / f"{rel_path.stem}{gt_suffix}"
            if gt_file_alt.exists():
                gt_file = gt_file_alt
            else:
                print(f"⚠️  警告: 未找到真实mesh: {gt_file}")
                continue
        
        pairs.append((sample_name, str(gen_file), str(gt_file)))
    
    return pairs


def batch_evaluate(generated_dir: str,
                   ground_truth_dir: str,
                   output_json: str = None,
                   n_points: int = 10000,
                   use_icp: bool = True,
                   f_score_threshold: float = 0.02,
                   generated_suffix: str = ".obj",
                   gt_suffix: str = ".obj") -> Dict:
    """
    批量评估mesh
    
    Args:
        generated_dir: 生成mesh目录
        ground_truth_dir: 真实mesh目录
        output_json: 输出JSON文件路径（可选）
        n_points: 采样点数
        use_icp: 是否使用ICP对齐
        f_score_threshold: F-Score阈值
        generated_suffix: 生成mesh文件后缀
        gt_suffix: 真实mesh文件后缀
    
    Returns:
        评估结果字典
    """
    # 查找配对
    pairs = find_mesh_pairs(generated_dir, ground_truth_dir, generated_suffix, gt_suffix)
    
    if not pairs:
        print("❌ 错误: 未找到任何mesh配对")
        return None
    
    print(f"📊 找到 {len(pairs)} 对mesh待评估\n")
    
    # 批量评估
    all_results = {}
    failed = []
    
    for sample_name, gen_path, gt_path in tqdm(pairs, desc="评估进度"):
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
            'total_samples': len(pairs),
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
    print(f"  总样本数: {summary['total_samples']}")
    print(f"  成功评估: {summary['successful']}")
    print(f"  失败评估: {summary['failed']}")
    
    if summary['failed'] > 0:
        print(f"\n❌ 失败样本:")
        for name in summary['failed_samples']:
            print(f"  - {name}")
    
    print(f"\n📈 指标统计:")
    print(f"\n{'指标':<30} {'均值':<12} {'标准差':<12} {'中位数':<12} {'最小值':<12} {'最大值':<12}")
    print("-"*80)
    
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
    
    # 按CD排序
    sorted_by_cd = sorted(results.items(), key=lambda x: x[1]['chamfer_distance'])
    
    print(f"\n🏆 Chamfer Distance 最好的5个样本 (越小越好):")
    for i, (name, result) in enumerate(sorted_by_cd[:5], 1):
        print(f"{i}. {name}: CD={result['chamfer_distance']:.6f}")
    
    print(f"\n⚠️  Chamfer Distance 最差的5个样本:")
    for i, (name, result) in enumerate(sorted_by_cd[-5:][::-1], 1):
        print(f"{i}. {name}: CD={result['chamfer_distance']:.6f}")
    
    # 按F-Score排序
    sorted_by_fscore = sorted(results.items(), key=lambda x: x[1]['f_score'], reverse=True)
    
    print(f"\n🏆 F-Score 最好的5个样本 (越大越好):")
    for i, (name, result) in enumerate(sorted_by_fscore[:5], 1):
        print(f"{i}. {name}: F-Score={result['f_score']:.4f}")
    
    print(f"\n⚠️  F-Score 最差的5个样本:")
    for i, (name, result) in enumerate(sorted_by_fscore[-5:][::-1], 1):
        print(f"{i}. {name}: F-Score={result['f_score']:.4f}")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='批量评估3D mesh质量')
    parser.add_argument('--generated_dir', '-g', type=str, required=True,
                        help='生成mesh的目录')
    parser.add_argument('--ground_truth_dir', '-t', type=str, required=True,
                        help='真实mesh的目录')
    parser.add_argument('--output', '-o', type=str, default='evaluation_results.json',
                        help='输出JSON文件路径')
    parser.add_argument('--n_points', '-n', type=int, default=10000,
                        help='采样点数 (默认: 10000)')
    parser.add_argument('--no_icp', action='store_true',
                        help='禁用ICP对齐')
    parser.add_argument('--f_threshold', '-f', type=float, default=0.01,
                        help='F-Score距离阈值 (默认: 0.01)')
    parser.add_argument('--gen_suffix', type=str, default='.obj',
                        help='生成mesh文件后缀 (默认: .obj)')
    parser.add_argument('--gt_suffix', type=str, default='.obj',
                        help='真实mesh文件后缀 (默认: .obj)')
    
    args = parser.parse_args()
    
    print("🚀 开始批量评估...")
    print(f"生成mesh目录: {args.generated_dir}")
    print(f"真实mesh目录: {args.ground_truth_dir}")
    print(f"采样点数: {args.n_points}")
    print(f"使用ICP对齐: {not args.no_icp}")
    print(f"F-Score阈值: {args.f_threshold}\n")
    
    summary = batch_evaluate(
        generated_dir=args.generated_dir,
        ground_truth_dir=args.ground_truth_dir,
        output_json=args.output,
        n_points=args.n_points,
        use_icp=not args.no_icp,
        f_score_threshold=args.f_threshold,
        generated_suffix=args.gen_suffix,
        gt_suffix=args.gt_suffix
    )
    
    if summary:
        print_summary(summary)
        print("\n✅ 评估完成!")
    else:
        print("\n❌ 评估失败!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

