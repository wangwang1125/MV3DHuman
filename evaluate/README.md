# 3D Mesh评估工具

用于评估生成的3D mesh与真实mesh的质量，支持Chamfer Distance、Earth Mover's Distance和F-Score等指标。

## 特性

✅ **自动对齐归一化**: 自动处理生成mesh与原始mesh位置、大小不一致的问题
- 归一化：将点云缩放到单位球内
- 中心化：将点云中心移到原点
- ICP对齐：可选的刚性对齐优化

✅ **多种评估指标**:
- **Chamfer Distance (CD)**: 双向最近邻距离，越小越好
- **Earth Mover's Distance (EMD)**: Wasserstein距离，越小越好
- **F-Score**: 精度和召回率的调和平均，越大越好

✅ **批量评估**: 支持批量评估多个样本并生成统计报告

## 安装依赖

```bash
pip install numpy scipy trimesh tqdm
```

或使用requirements文件:

```bash
cd evaluate
pip install -r requirements.txt
```

## 使用方法

### 1. 单个mesh评估

```bash
python mesh_metrics.py <生成的mesh> <真实mesh> [采样点数] [是否使用ICP]
```

示例:
```bash
# 使用默认参数（10000点，启用ICP）
python mesh_metrics.py generated.obj ground_truth.obj

# 自定义采样点数
python mesh_metrics.py generated.obj ground_truth.obj 20000

# 禁用ICP对齐
python mesh_metrics.py generated.obj ground_truth.obj 10000 0
```

### 2. 批量评估

```bash
python batch_evaluate.py -g <生成mesh目录> -t <真实mesh目录> [选项]
```

参数说明:
- `-g, --generated_dir`: 生成mesh的目录（必需）
- `-t, --ground_truth_dir`: 真实mesh的目录（必需）
- `-o, --output`: 输出JSON文件路径（默认: evaluation_results.json）
- `-n, --n_points`: 采样点数（默认: 10000）
- `--no_icp`: 禁用ICP对齐
- `-f, --f_threshold`: F-Score距离阈值（默认: 0.01）
- `--gen_suffix`: 生成mesh文件后缀（默认: .obj）
- `--gt_suffix`: 真实mesh文件后缀（默认: .obj）

示例:
```bash
# 基本用法
python batch_evaluate.py \
    -g /path/to/generated_meshes \
    -t /path/to/ground_truth_meshes \
    -o results.json

# 高分辨率评估（更多采样点）
python batch_evaluate.py \
    -g /path/to/generated \
    -t /path/to/ground_truth \
    -n 50000 \
    -o high_res_results.json

# 评估PLY格式的mesh
python batch_evaluate.py \
    -g /path/to/generated \
    -t /path/to/ground_truth \
    --gen_suffix .ply \
    --gt_suffix .ply
```

### 3. 在Python代码中使用

```python
from mesh_metrics import evaluate_mesh_pair, print_evaluation_results

# 评估单对mesh
results = evaluate_mesh_pair(
    generated_mesh_path='generated.obj',
    ground_truth_mesh_path='ground_truth.obj',
    n_points=10000,
    use_icp=True,
    f_score_threshold=0.01
)

# 打印结果
print_evaluation_results(results, name="我的模型")

# 访问具体指标
print(f"CD: {results['chamfer_distance']}")
print(f"F-Score: {results['f_score']}")
```

## 输出示例

### 单个评估输出

```
============================================================
评估结果: sample_001
============================================================
Chamfer Distance (CD):         0.003245  (越小越好)
Earth Mover's Distance (EMD):  0.001876  (越小越好)
Precision:                     0.9234
Recall:                        0.8967
F-Score:                       0.9098  (越大越好)
============================================================
```

### 批量评估输出

批量评估会生成JSON文件，包含：
- 每个样本的详细指标
- 所有指标的统计信息（均值、标准差、中位数、最大最小值）
- 最好和最差的样本排名

JSON结构示例:
```json
{
  "total_samples": 100,
  "successful": 98,
  "failed": 2,
  "statistics": {
    "chamfer_distance": {
      "mean": 0.0045,
      "std": 0.0012,
      "median": 0.0043,
      "min": 0.0021,
      "max": 0.0098
    },
    ...
  },
  "individual_results": {
    "sample_001": {
      "chamfer_distance": 0.003245,
      "emd": 0.001876,
      "precision": 0.9234,
      "recall": 0.8967,
      "f_score": 0.9098
    },
    ...
  }
}
```

## 评估指标说明

### Chamfer Distance (CD)
- **定义**: 双向最近邻点距离的平均值
- **计算**: `CD = mean(min_dist(P1→P2)) + mean(min_dist(P2→P1))`
- **特点**: 对称距离，衡量整体形状匹配度
- **范围**: [0, +∞)，**越小越好**

### Earth Mover's Distance (EMD)
- **定义**: Wasserstein距离，将一个点云"搬运"到另一个所需的最小工作量
- **特点**: 对点云分布和密度敏感，能更好地评估整体结构
- **范围**: [0, +∞)，**越小越好**

### F-Score
- **定义**: 精度和召回率的调和平均
- **精度**: 生成点云中有多少点能在阈值内找到真实点
- **召回率**: 真实点云中有多少点能在阈值内找到生成点
- **计算**: `F = 2 × (Precision × Recall) / (Precision + Recall)`
- **范围**: [0, 1]，**越大越好**

## 关于位置和大小归一化

本工具自动处理生成mesh与真实mesh在位置、大小、姿态上的差异：

1. **归一化处理**:
   - 将每个点云中心化（移到原点）
   - 缩放到单位球内（最大距离为1）
   - 确保不同尺度的mesh可比较

2. **ICP对齐**（可选）:
   - 使用迭代最近点算法进行刚性对齐
   - 优化旋转和平移，找到最佳匹配姿态
   - 默认开启，可通过`--no_icp`禁用

3. **何时禁用ICP**:
   - 如果生成mesh的姿态已经对齐
   - 需要评估姿态估计的准确性
   - ICP收敛到局部最优导致错误对齐

## 性能建议

- **采样点数**: 默认10000点在速度和精度间平衡良好
  - 快速评估: 5000点
  - 标准评估: 10000点
  - 高精度评估: 20000-50000点
  
- **ICP对齐**: 会增加计算时间，但显著提高评估准确性

- **批量评估**: 使用进度条显示，支持中断后继续

## 常见问题

**Q: 为什么我的CD值很大？**
A: 可能原因：
1. 生成质量确实不好
2. 需要更多采样点
3. ICP对齐失败，尝试调整参数或禁用ICP

**Q: F-Score阈值如何选择？**
A: 
- 0.01: 适用于单位球归一化后的点云（推荐）
- 阈值越大，F-Score越高
- 根据具体应用场景调整

**Q: EMD计算很慢怎么办？**
A: 
- 减少采样点数
- EMD复杂度较高，对大规模点云可能较慢

## 许可证

本工具遵循项目主许可证。

