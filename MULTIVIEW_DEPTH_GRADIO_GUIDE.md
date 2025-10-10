# 多视图深度条件 Gradio 前端使用指南

## 概述

本文档说明如何使用改造后的 Gradio 前端进行多视图+深度图的3D生成。改造后的前端支持三种模式：

1. **单视图RGB模式** - 标准的单图像到3D生成
2. **单视图深度模式** - 单图像+深度图到3D生成
3. **多视图深度模式** - 多视图图像+多视图深度图到3D生成（新增）

## 快速开始

### 1. 启动多视图深度模式

```bash
# 启动多视图深度条件模式
python gradio_app.py \
    --model_path tencent/Hunyuan3D-2.1 \
    --subfolder hunyuan3d-dit-v2-1 \
    --enable_multiview_depth \
    --depth_lora_path ./hy3dshape/output_folder/dit/multiview_depth_lora_checkpoints/step_1000 \
    --port 6008
```

**参数说明：**
- `--enable_multiview_depth`: 启用多视图深度模式（这会自动启用MV_MODE和DEPTH_MODE）
- `--depth_lora_path`: 指向训练好的多视图深度LoRA权重路径
- `--model_path` / `--subfolder`: 基础模型路径

### 2. 启动单视图深度模式（原有功能）

```bash
# 启动单视图深度条件模式
python gradio_app.py \
    --model_path tencent/Hunyuan3D-2.1 \
    --subfolder hunyuan3d-dit-v2-1 \
    --enable_depth \
    --depth_lora_path ./hy3dshape/output_folder/dit/depth_lora_checkpoints/step_1000 \
    --port 6008
```

### 3. 启动标准模式（无深度）

```bash
# 标准RGB模式
python gradio_app.py \
    --model_path tencent/Hunyuan3D-2.1 \
    --subfolder hunyuan3d-dit-v2-1 \
    --port 6008
```

## 多视图深度模式详细说明

### 数据准备

#### 视图顺序和命名约定

训练代码使用以下视图顺序（与训练数据集一致）：
- **Front** (0°) - 索引 0
- **Right** (90°) - 索引 1  
- **Back** (180°) - 索引 2
- **Left** (270°) - 索引 3

这对应训练配置文件中的 `multiview_indices: [0, 1, 2, 3]`

#### RGB 图像要求

- **格式**: PNG, JPG, 或任何PIL支持的格式
- **通道**: RGBA（带Alpha通道）或RGB
- **尺寸**: 建议512x512或518x518（会自动调整）
- **背景**: 支持透明背景（推荐）或白色背景
- **数量**: 至少提供一个视图，最多4个视图

#### 深度图要求

- **格式**: 16位PNG、TIFF或TIF文件（**重要**：必须使用File组件上传，因为Gradio的Image组件会压缩为8位）
- **尺寸**: 建议与RGB图像相同（518x518）
- **通道**: 单通道灰度图
- **值域**: 任意深度值（代码会自动归一化到[0,1]）
- **数量**: 至少提供一个视图的深度图，建议与RGB视图一一对应

### UI 操作步骤

1. **访问界面**
   - 打开浏览器访问 `http://localhost:6008`
   - 标题应显示 "Hunyuan-3D-2.1 (多视图深度条件模式)"

2. **上传 RGB 图像**
   - 切换到 "MultiView Prompt" 标签页
   - 在 "RGB 图像" 部分上传各个视图的图像：
     - Front: 前视图
     - Back: 后视图  
     - Left: 左视图
     - Right: 右视图
   - 至少上传一个视图

3. **上传深度图**
   - 在 "深度图 (16位PNG)" 部分上传对应视图的深度图：
     - Front Depth: 前视图深度图
     - Back Depth: 后视图深度图
     - Left Depth: 左视图深度图
     - Right Depth: 右视图深度图
   - 至少上传一个视图的深度图

4. **调整参数**（可选）
   - 在 "Advanced Options" 中调整：
     - Inference Steps: 推理步数（默认30，训练配置中使用50）
     - Guidance Scale: 引导强度（默认5.0）
     - Octree Resolution: 网格分辨率（默认256）
     - Remove Background: 是否移除背景（默认开启）

5. **生成3D模型**
   - 点击 "Gen Shape" 生成几何体
   - 或点击 "Gen Textured Shape" 生成带纹理的模型

### 数据流程

```
用户输入
  ├── RGB图像字典: {'front': PIL.Image, 'right': PIL.Image, 'back': PIL.Image, 'left': PIL.Image}
  └── 深度图字典: {'front': file, 'right': file, 'back': file, 'left': file}
       ↓
预处理阶段
  ├── RGB: 通过 MVImageProcessorV2 处理
  │   └── 输出: (num_views, 3, 518, 518)
  └── 深度: 通过 load_multiview_depths 处理
      └── 输出: (num_views, 1, 518, 518)
       ↓
模型推理
  ├── DinoImageEncoderMV: 编码多视图RGB特征
  ├── MultiViewControlNet: 处理多视图深度图
  └── HunYuanDiT: 生成3D潜在表示
       ↓
后处理
  └── ShapeVAE解码 → Marching Cubes → Trimesh输出
```

## 代码架构

### 关键函数

#### 1. `load_multiview_depths(depth_files_dict, target_size=518)`
加载多视图深度图并格式化为模型输入格式。

**输入:**
- `depth_files_dict`: 包含各视图深度图文件的字典
- `target_size`: 目标尺寸（默认518）

**输出:**
- `torch.Tensor`: 形状为 `(num_views, 1, H, W)` 的多视图深度图张量

**处理逻辑:**
1. 按顺序加载 front→right→back→left 的深度图
2. 对每个深度图：
   - 加载16位PNG数据
   - 归一化到[0, 1]范围
   - 调整大小到target_size
   - 转换为torch张量
3. 堆叠所有视图到batch维度

#### 2. `_gen_shape(...)`
核心生成函数，支持三种模式。

**新增参数:**
- `mv_depth_front/back/left/right`: 多视图深度图文件

**处理流程:**
1. 验证输入（根据模式）
2. 处理RGB图像（单视图或多视图）
3. 处理深度图（单视图或多视图）
4. 调用模型推理
5. 导出网格

#### 3. 模式控制变量

```python
MV_MODE = 'mv' in args.model_path or args.enable_multiview_depth
DEPTH_MODE = args.enable_depth or args.enable_multiview_depth
MULTIVIEW_DEPTH_MODE = args.enable_multiview_depth
```

- `MV_MODE`: 控制是否显示多视图UI
- `DEPTH_MODE`: 控制是否启用深度图处理
- `MULTIVIEW_DEPTH_MODE`: 控制是否使用多视图深度图

## 模型配置对应关系

### Gradio前端参数 ↔ 训练配置

| Gradio参数 | 训练配置路径 | 说明 |
|-----------|------------|------|
| 视图顺序: front/right/back/left | `dataset.params.multiview_indices: [0, 1, 2, 3]` | 四个90度间隔的视图 |
| 深度图格式: 16位PNG | `dataset.params.load_depth: true` | 训练时使用EXR，推理时支持PNG |
| 图像尺寸: 518x518 | `dataset.params.image_size: 518` | 与训练保持一致 |
| 深度融合策略: multiview | `dataset.params.depth_fusion_strategy: "multiview"` | 保持多视图格式供ControlNet处理 |

### 模型组件映射

| 前端调用 | 训练模型组件 |
|---------|------------|
| `MVImageProcessorV2` | `preprocessors.MVImageProcessorV2` |
| `DinoImageEncoderMV` | `cond_stage_config.main_image_encoder.type: DinoImageEncoderMV` |
| `MultiViewControlNet` | `control_net_config` + LoRA微调 |
| Pipeline | `Hunyuan3DDiTFlowMatchingPipeline` |

## 故障排查

### 常见问题

1. **深度图加载失败**
   - 确保使用16位PNG格式
   - 不要使用Gradio的Image组件（会压缩为8位）
   - 使用File组件上传深度图

2. **视图顺序错误**
   - 确保按照 front→right→back→left 的顺序上传
   - 检查配置文件中的 `multiview_indices`

3. **内存不足**
   - 减小 `octree_resolution`（256→196）
   - 减小 `num_chunks`
   - 启用 `--low_vram_mode`

4. **LoRA权重加载失败**
   - 检查 `--depth_lora_path` 路径是否正确
   - 确保路径指向包含 `adapter_config.json` 的目录（PEFT格式）
   - 或指向Lightning checkpoint `.ckpt` 文件

5. **多视图处理速度慢**
   - 多视图处理需要更多计算资源
   - 建议使用较少的推理步数（10-30步）
   - 考虑使用turbo模式（5步）

## 性能优化建议

### 推理速度优化

1. **减少推理步数**
   ```python
   # 快速预览: 5-10步
   # 标准质量: 30步（默认）
   # 高质量: 50步
   ```

2. **启用模型编译** (仅Linux/CUDA)
   ```bash
   python gradio_app.py --compile --enable_multiview_depth ...
   ```

3. **启用低显存模式**
   ```bash
   python gradio_app.py --low_vram_mode --enable_multiview_depth ...
   ```

### 质量优化

1. **提高输入质量**
   - 使用高分辨率RGB图像（512x512或更高）
   - 使用准确的深度图（从专业深度估计模型获得）
   - 确保视图间的一致性

2. **调整推理参数**
   - `guidance_scale`: 5.0-7.5（更高=更符合条件，但可能过拟合）
   - `octree_resolution`: 256-384（更高=更细节，但更慢）

3. **后处理**
   - 启用 "Simplify Mesh" 减少面数
   - 使用 "Transform" 导出不同格式

## 与训练代码的集成

### 数据格式一致性

前端处理后的数据格式与训练数据完全一致：

```python
# 训练数据格式
batch = {
    'image': torch.Tensor,  # (batch, num_views, 3, 518, 518)
    'depth': torch.Tensor,  # (batch, num_views, 1, 518, 518)
    'surface': torch.Tensor, # (batch, num_points, 6/7)
}

# Gradio前端输出（传给模型）
model_inputs = {
    'image': dict or PIL.Image,  # 会被pipeline.prepare_image处理
    'depth': torch.Tensor,       # (num_views, 1, 518, 518)
    'num_inference_steps': int,
    'guidance_scale': float,
    ...
}
```

### Pipeline处理流程

1. **图像预处理** (`prepare_image`)
   - 单视图: PIL.Image → (1, 3, 518, 518)
   - 多视图: dict → MVImageProcessorV2 → (num_views, 3, 518, 518)

2. **条件编码** (`encode_cond`)
   - RGB: DinoImageEncoderMV → 图像特征
   - 深度: MultiViewControlNet → 深度特征

3. **扩散采样**
   - Flow Matching with velocity prediction
   - 支持 Classifier-Free Guidance

4. **VAE解码**
   - Latent → Surface Points
   - Marching Cubes → Mesh

## 进阶使用

### 自定义视图配置

如果需要使用不同的视图配置（例如6个视图），需要修改：

1. **前端UI** (`gradio_app.py`)
   - 添加更多视图的上传组件
   - 更新 `load_multiview_depths` 函数

2. **数据处理**
   - 修改 `view_order` 列表
   - 更新视图索引映射

3. **模型配置**
   - 训练配置中的 `multiview_indices`
   - DinoImageEncoderMV 的 `view_num` 参数

### 批量推理

目前前端不支持批量推理，如需批量处理：

```python
# 使用Python脚本进行批量推理
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(...)

for sample in dataset:
    outputs = pipeline(
        image=sample['images'],
        depth=sample['depths'],
        num_inference_steps=30,
        ...
    )
    # 保存结果
```

## 参考资料

- 训练配置: `hy3dshape/configs/hunyuandit-multiview-depth-lora-flowmatching-dinol518-bf16-lr1e5-4096.yaml`
- 数据加载器: `hy3dshape/hy3dshape/data/dit_asl_multiview.py`
- 预处理器: `hy3dshape/hy3dshape/preprocessors.py`
- 训练指南: `hy3dshape/MULTIVIEW_DEPTH_TRAINING_GUIDE.md`

## 更新日志

### v1.0 (当前版本)
- ✅ 支持多视图RGB输入
- ✅ 支持多视图深度图输入
- ✅ 16位深度图处理
- ✅ 与训练代码数据格式对齐
- ✅ 三种模式切换（单视图/单视图深度/多视图深度）
- ✅ 自动LoRA权重加载

### 未来计划
- 🔄 批量推理支持
- 🔄 实时深度图预览
- 🔄 视图数量自定义
- 🔄 深度图可视化

