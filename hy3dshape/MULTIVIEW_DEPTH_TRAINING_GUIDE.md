# Multi-View Depth LoRA Training Guide - 方案2

## 🎯 概述

我们已经成功实现了**方案2**：支持多视图深度的完整训练系统。该系统能够同时利用前后左右四个视图的RGB图像和对应的深度图进行LoRA微调。

## 🏗️ 架构设计

### 核心组件

1. **MultiViewDepthControlNet** (`controlnet_multiview.py`)
   - 处理4个视图的深度图：前(0°), 右(90°), 后(180°), 左(270°)
   - 支持多种融合策略：attention, concat, average, weighted
   - 输出768维特征向量供DiT使用

2. **AlignedShapeLatentMultiViewDataset** (`dit_asl_multiview.py`)
   - 加载指定的多视图图像和深度图
   - 支持不同的深度融合策略
   - 返回格式：`image: (4, 3, H, W)`, `depth: (4, 1, H, W)` 或融合后的格式

3. **Enhanced Flow Matching Model** (`flow_matching_sit.py`)
   - 自动检测单视图/多视图模式
   - 智能选择对应的ControlNet类型
   - 支持混合精度训练

## 📁 文件结构

```
hy3dshape/
├── hy3dshape/
│   ├── data/
│   │   └── dit_asl_multiview.py           # 多视图数据加载器
│   └── models/
│       ├── controlnet_multiview.py        # 多视图ControlNet
│       └── diffusion/
│           └── flow_matching_sit.py       # 更新的扩散模型
└── configs/
    └── hunyuandit-multiview-depth-lora-flowmatching-dinol518-bf16-lr1e5-4096.yaml  # 多视图配置
```

## 🚀 使用方法

### 1. 数据准备

确保你的数据结构如下：
```
tools/mini_mv_depth_trainset/preprocessed/
└── [object_id]/
    └── render_cond/
        ├── 000.png          # 前视图 (0°)
        ├── 000_depth.exr    # 前视图深度
        ├── 006.png          # 右视图 (90°)
        ├── 006_depth.exr    # 右视图深度
        ├── 012.png          # 后视图 (180°)
        ├── 012_depth.exr    # 后视图深度
        ├── 018.png          # 左视图 (270°)
        └── 018_depth.exr    # 左视图深度
```

### 2. 开始训练

```bash
# 使用多视图深度LoRA训练
python main.py --config configs/hunyuandit-multiview-depth-lora-flowmatching-dinol518-bf16-lr1e5-4096.yaml
```

### 3. 配置选项

#### 深度融合策略选择：

```yaml
dataset:
  params:
    depth_fusion_strategy: "multiview"  # 推荐：保持所有视图供ControlNet处理
    # 其他选项：
    # "average"   - 平均融合多个深度图
    # "max"       - 取最大值融合
    # "weighted"  - 加权融合
```

#### MultiViewControlNet融合策略：

```yaml
model:
  params:
    control_net_config:
      fusion_strategy: "attention"  # 推荐：注意力融合
      # 其他选项：
      # "concat"    - 特征拼接
      # "average"   - 平均融合
      # "weighted"  - 学习权重融合
```

## 📊 数据格式说明

### 输入数据格式
- **图像**: `(batch_size, 4, 3, 518, 518)` - 4个视图的RGB图像
- **深度**: `(batch_size, 4, 1, 518, 518)` - 4个视图的深度图（multiview策略时）
- **表面点云**: `(batch_size, num_points, 6)` - 3D表面点和法向量

### 内部处理流程
1. **多视图图像** → `DinoImageEncoderMV` → 多视图视觉特征
2. **多视图深度** → `MultiViewDepthControlNet` → 融合深度特征
3. **特征融合** → `HunYuanDiT` → 3D生成

## 🎛️ 高级配置

### 视图选择自定义
```yaml
dataset:
  params:
    multiview_indices: [0, 6, 12, 18]  # 默认：前右后左
    # 可以自定义其他视图组合，例如：
    # [0, 8, 16, 4]  # 不同角度组合
```

### ControlNet架构调整
```yaml
model:
  params:
    num_views: 4                    # 视图数量
    control_in_channels: 1          # 每个深度图的通道数
    additional_cond_hidden_state: 768  # 深度特征维度
```

## 🔧 性能优化建议

### 1. 内存优化
- **batch_size**: 建议设为2（因为包含4个视图）
- **混合精度**: 使用bf16减少显存占用
- **梯度累积**: 如需要更大的有效batch size

### 2. 训练稳定性
- **梯度裁剪**: `gradient_clip_val: 1.0`
- **学习率调度**: Cosine annealing with warmup
- **LoRA rank**: 8（平衡性能和参数量）

### 3. 融合策略选择指南
- **Attention**: 最佳性能，但计算量较大
- **Weighted**: 平衡性能和效率
- **Average**: 最快速，适合快速实验
- **Concat**: 保留最多信息，但参数量大

## 📈 预期性能提升

相比单视图方案，多视图深度方案预期能带来：

1. **几何一致性提升**: 多角度深度信息提供更完整的几何约束
2. **细节恢复改进**: 不同视图的深度互补，减少遮挡问题
3. **训练稳定性增强**: 多视图约束降低训练不稳定性
4. **生成质量提升**: 更准确的3D结构理解

## 🐛 常见问题与解决方案

### 1. 深度图格式不匹配
```python
# 如果遇到深度图读取问题，检查：
- EXR文件完整性
- OpenEXR库安装状态
- 通道名称匹配（R/G/B/Y/Z/depth）
```

### 2. 视图索引错误
```yaml
# 确保视图索引在0-23范围内
multiview_indices: [0, 6, 12, 18]  # ✅ 正确
multiview_indices: [0, 6, 12, 24]  # ❌ 错误：24超出范围
```

### 3. 内存不足
```yaml
# 降低batch size和分辨率
batch_size: 1
image_size: 256  # 从518降低
```

## 🎉 总结

方案2成功实现了完整的多视图深度训练系统，具有：

- ✅ **完整的多视图深度支持**
- ✅ **灵活的融合策略选择**
- ✅ **自动的单/多视图模式切换**
- ✅ **高效的注意力融合机制**
- ✅ **完善的错误处理和回退机制**

该系统能够充分利用多视图深度信息，为3D生成任务提供更强的几何约束和更高的生成质量。
