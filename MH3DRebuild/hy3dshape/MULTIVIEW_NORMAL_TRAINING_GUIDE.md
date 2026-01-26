# 多视图法线图训练指南

本指南介绍如何使用多视图法线贴图进行微调训练。

## 概述

多视图法线图训练使用法线贴图（`_normal.png`）作为条件输入，通过 `DinoImageEncoderMV` 进行特征提取，生成与多视图彩色图相同数量的token，用于后续的注意力机制。

## 关键特性

1. **法线图格式**: `{view_index:03d}_normal.png` (例如: `000_normal.png`, `001_normal.png`)
2. **特征提取**: 使用 `DinoImageEncoderMV`，与RGB图相同的处理流程
3. **Token生成**: 生成与多视图RGB图同样多的token用于注意力机制
4. **无ControlNet**: 法线图不经过ControlNet处理，直接通过DINO编码器

## 配置说明

主要配置文件: `configs/hunyuandit-multiview-normal-lora-flowmatching-dinol518-bf16-lr1e5-4096.yaml`

### 关键参数

```yaml
dataset:
  params:
    # 启用法线图加载
    load_normal: true
    
    # 多视图索引（0和1表示前视图和后视图）
    multiview_indices: [0, 1]
    
    # 法线图融合策略
    normal_fusion_strategy: "multiview"

model:
  params:
    # 使用DinoImageEncoderMV处理法线图
    cond_stage_config:
      params:
        main_image_encoder:
          type: DinoImageEncoderMV 
          kwargs:
            view_num: 2  # 2个视图
    
    # 不使用额外的条件处理
    denoiser_cfg:
      params:
        with_decoupled_ca: false
        num_views: 2
```

## 数据准备

### 目录结构

```
preprocessed/
├── sample_001/
│   ├── render_cond/
│   │   ├── 000.png          # 前视图RGB
│   │   ├── 000_normal.png   # 前视图法线图
│   │   ├── 001.png          # 后视图RGB
│   │   ├── 001_normal.png   # 后视图法线图
│   │   └── ...
│   └── geo_data/
│       └── sample_001_surface.npz
├── sample_002/
│   └── ...
```

### 法线图要求

1. **格式**: PNG格式，RGB三通道
2. **值域**: 法线向量映射到RGB颜色空间（通常用于可视化）
3. **尺寸**: 与RGB图相同的分辨率
4. **命名**: `{view_index:03d}_normal.png`

### 法线图生成

可以使用Blender或其他渲染工具生成法线贴图：

```python
# 在渲染脚本中生成法线图
import bpy

# 配置法线输出节点
bpy.context.scene.view_layers[0].use_pass_normal = True

# 渲染并保存为PNG
bpy.ops.render.render()
bpy.data.images['Render Result'].save_render(
    filepath=output_path.replace('.png', '_normal.png')
)
```

## 训练流程

### 1. 准备训练数据

确保数据目录结构正确，包含法线图文件。

### 2. 运行训练

```bash
python main.py \
    --config configs/hunyuandit-multiview-normal-lora-flowmatching-dinol518-bf16-lr1e5-4096.yaml \
    --epochs 100 \
    --accumulate_grad_batches 4
```

### 3. 监控训练

训练过程中会输出：
- 训练损失
- 验证损失  
- LoRA checkpoint保存路径: `output_folder/dit/multiview_normal_lora_checkpoints/`

## 与深度图的区别

| 特性 | 深度图 | 法线图 |
|------|--------|--------|
| 文件格式 | `.exr` | `.png` |
| 通道数 | 单通道 | 三通道RGB |
| 特征提取 | ControlNet | DinoImageEncoderMV |
| Token生成 | 16个固定token | 动态token数量（与RGB相同） |
| 输入尺寸 | 1×H×W | 3×H×W |
| 预处理 | 归一化到[0,1] | RGB标准化 |

## 优势

1. **丰富语义**: 法线图提供表面朝向信息
2. **一致性**: 与RGB图使用相同的特征提取流程
3. **灵活性**: 不依赖ControlNet，更灵活的token生成
4. **融合**: 法线信息自然融入注意力机制

## 注意事项

1. 确保法线图值域正确（通常需要归一化）
2. 法线图与RGB图需要对齐
3. 如果使用自定义法线空间，需要相应调整预处理
4. Batch size可能需要根据GPU内存调整

## 故障排除

### 常见错误

1. **文件未找到**: 检查法线图文件是否存在且命名正确
2. **维度不匹配**: 确保法线图与RGB图尺寸一致
3. **内存不足**: 减小batch size或增加gradient accumulation

### 调试建议

```python
# 检查数据加载
from hy3dshape.data.dit_asl_multiview import AlignedShapeLatentMultiViewDataset

dataset = AlignedShapeLatentMultiViewDataset(
    data_list="path/to/data",
    load_normal=True,
    multiview_indices=[0, 1]
)

# 获取一个样本检查
sample = next(iter(dataset))
print(f"Normal shape: {sample['normal'].shape}")
print(f"Normal mask shape: {sample['normal_mask'].shape}")
```

## 扩展

可以结合多种条件进行训练：

```yaml
# 同时使用RGB、深度和法线图
dataset:
  params:
    load_depth: true
    load_normal: true
```

这样可以提供更丰富的几何和颜色信息。

