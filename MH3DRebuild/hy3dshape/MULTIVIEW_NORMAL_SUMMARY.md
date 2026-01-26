# 多视图法线图训练实施总结

## 概述

本文档总结了为支持多视图法线图训练所做的代码修改。法线图作为条件输入，通过 `DinoImageEncoderMV` 进行特征提取，生成与多视图RGB图相同数量的token用于注意力机制。

## 核心特点

### 与深度图的区别

- **深度图**: 使用 `MultiViewDepthControlNet` → 生成16个固定token
- **法线图**: 使用 `DinoImageEncoderMV` → 生成动态数量的token（与RGB图相同）

### 法线图的优势

1. **更丰富的语义信息**: 提供表面朝向信息
2. **一致性处理**: 与RGB图使用相同的特征提取流程
3. **灵活性**: 不依赖ControlNet架构
4. **自然融合**: 法线信息通过注意力机制自然融合

## 修改文件清单

### 1. 新增配置文件
**文件**: `configs/hunyuandit-multiview-normal-lora-flowmatching-dinol518-bf16-lr1e5-4096.yaml`

**关键配置**:
- `load_normal: true` - 启用法线图加载
- `normal_fusion_strategy: "multiview"` - 保持多视图格式
- 使用 `DinoImageEncoderMV` 编码器
- `with_decoupled_ca: false` - 不使用额外的条件处理

### 2. 修改数据加载器
**文件**: `hy3dshape/data/dit_asl_multiview.py`

**主要修改**:

#### 2.1 类参数添加
```python
def __init__(
    self,
    ...,
    load_normal: bool = False,  # 新增参数
    normal_fusion_strategy: str = "multiview"  # 新增参数
):
```

#### 2.2 decode方法
添加法线图路径生成逻辑：
```python
# Load normal maps if enabled
if self.load_normal:
    normal_img_paths = [
        os.path.join(item, f'render_cond/{i:03d}_normal.png') 
        for i in self.multiview_indices
    ]
    sample["normal"] = normal_img_paths
```

#### 2.3 transform方法
添加法线图加载和处理：
```python
# Load and process multi-view normal maps if enabled
# Normal maps are processed like RGB images using load_multiview_render
if self.load_normal and "normal" in sample:
    normal_input, normal_mask = self.load_multiview_render(sample['normal'])
    result_sample["normal"] = normal_input  # Shape: (num_views, C, H, W)
    result_sample["normal_mask"] = normal_mask  # Shape: (num_views, 1, H, W)
```

#### 2.4 LightningDataModule更新
- 添加 `load_normal` 参数
- 添加 `normal_fusion_strategy` 参数
- 更新 `train_dataloader` 和 `val_dataloader` 的参数传递

### 3. 新增训练脚本
**文件**: `train_normal_lora.sh`

用于快速启动多视图法线图训练。

### 4. 新增文档
- `MULTIVIEW_NORMAL_TRAINING_GUIDE.md` - 详细的训练指南
- `MULTIVIEW_NORMAL_SUMMARY.md` - 本文档

## 数据格式

### 文件命名规范
- RGB图: `{view_index:03d}.png` (例如: `000.png`, `001.png`)
- 法线图: `{view_index:03d}_normal.png` (例如: `000_normal.png`, `001_normal.png`)
- 深度图: `{view_index:03d}_depth.exr` (例如: `000_depth.exr`, `001_depth.exr`)

### 法线图要求
- **格式**: PNG, RGB三通道
- **尺寸**: 与RGB图相同
- **值域**: 法线向量映射到RGB颜色空间

## 使用流程

### 1. 准备数据
确保数据目录包含：
```
preprocessed/
└── sample_001/
    └── render_cond/
        ├── 000.png
        ├── 000_normal.png
        ├── 001.png
        ├── 001_normal.png
        └── ...
```

### 2. 训练命令
```bash
bash train_normal_lora.sh
```

或直接使用Python:
```bash
python main.py \
    --config configs/hunyuandit-multiview-normal-lora-flowmatching-dinol518-bf16-lr1e5-4096.yaml
```

## 技术细节

### 特征提取流程对比

#### 深度图路径
```
深度图 (.exr) 
  ↓
load_multiview_depth_maps()
  ↓
MultiViewDepthControlNet
  ↓
16个固定token
  ↓
decoupled_ca (decoupled cross-attention)
```

#### 法线图路径
```
法线图 (.png)
  ↓
load_multiview_render()  # 与RGB图相同的加载方式
  ↓
DinoImageEncoderMV
  ↓
动态token数量（= patches × num_views）
  ↓
融合到条件输入中
```

### 为什么法线图使用不同的处理方式？

1. **深度图**: 单通道数值，需要专门的特征提取 → ControlNet
2. **法线图**: RGB图像格式，可以直接使用视觉编码器 → DINO
3. **语义对齐**: 法线图与RGB图在视觉空间，更好的语义对齐

## 配置参数说明

### 关键参数
```yaml
dataset:
  params:
    load_normal: true              # 启用法线图
    multiview_indices: [0, 1]      # 2个视图（前、后）
    normal_fusion_strategy: "multiview"  # 保持多视图格式

model:
  params:
    cond_stage_config:
      params:
        main_image_encoder:
          type: DinoImageEncoderMV  # 使用多视图DINO编码器
          kwargs:
            view_num: 2            # 2个视图
    
    denoiser_cfg:
      params:
        with_decoupled_ca: false    # 不使用额外条件处理
        num_views: 2                # 2个视图
```

## 故障排除

### 常见问题

1. **法线图找不到**
   - 检查文件是否存在
   - 验证命名格式是否正确

2. **维度不匹配**
   - 确保法线图与RGB图尺寸一致
   - 检查通道数（应该是3通道）

3. **内存不足**
   - 减小batch_size
   - 增加gradient_accumulation

## 扩展可能

### 组合多种条件
可以同时使用RGB、深度和法线图：

```yaml
dataset:
  params:
    load_depth: true
    load_normal: true
```

这样可以获得最丰富的几何和颜色信息。

## 总结

通过本次修改，实现了：
1. ✅ 法线图加载支持
2. ✅ 使用DINO编码器进行特征提取
3. ✅ 生成与RGB图相同数量的token
4. ✅ 完整的多视图支持
5. ✅ 详细的文档和训练脚本

现在可以通过多视图法线图进行高质量的3D形状生成微调训练。

