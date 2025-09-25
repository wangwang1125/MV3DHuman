# Hunyuan3D-2.1 深度图支持使用说明

本文档说明如何使用新增的深度图条件功能来生成3D模型。

## 🚀 快速开始

### 1. 启用深度图模式

使用 `--enable_depth` 参数启动Gradio应用：

```bash
python gradio_app.py --enable_depth
```

### 2. 可选：指定LoRA权重路径

如果你有训练好的深度LoRA权重，可以指定路径：

```bash
python gradio_app.py --enable_depth --depth_lora_path /path/to/your/depth_lora_checkpoints/step_1000
```

如果不指定路径，系统会自动在以下位置搜索最新的权重：
- `output_folder/dit/depth_lora_checkpoints/`
- `hy3dshape/output_folder/dit/depth_lora_checkpoints/`
- `./hy3dshape/output_folder/dit/depth_lora_checkpoints/`

## 📁 深度图要求

### 支持的格式
- **16位PNG** (推荐)
- **TIFF/TIF** 文件
- 单通道或多通道深度图（会使用第一个通道）

### 深度值要求
- 任意深度值范围（系统会自动归一化到 [0,1]）
- 支持16位无符号整数 (0-65535)
- 支持8位无符号整数 (0-255)
- 支持浮点数格式

### 尺寸要求
- 建议与RGB图像尺寸相同
- 系统会自动调整到518x518像素

## 🎯 使用步骤

1. **启动应用**
   ```bash
   python gradio_app.py --enable_depth
   ```

2. **上传RGB图像**
   - 在"RGB图像"区域上传你的输入图像
   
3. **上传深度图**
   - 在"深度图文件 (16位PNG)"区域上传对应的深度图
   - 系统会验证文件格式并显示状态
   
4. **调整参数**（可选）
   - 推理步数 (Inference Steps)
   - 引导尺度 (Guidance Scale)
   - 八叉树分辨率 (Octree Resolution)
   
5. **生成3D模型**
   - 点击"Gen Shape"生成基础白模
   - 点击"Gen Textured Shape"生成带纹理的模型

## ⚙️ 技术细节

### 深度图处理流程
1. **加载**: 支持16位PNG/TIFF格式
2. **预处理**: 过滤无效值（NaN、Inf、超大值）
3. **归一化**: 映射到 [0,1] 范围
4. **尺寸调整**: 调整到518x518像素
5. **张量转换**: 转换为 (1,1,H,W) 张量格式

### LoRA权重加载
- 自动搜索训练产生的LoRA检查点
- 按步数排序，加载最新的权重
- 如果加载失败，回退到基础模型

### 模型架构
- **ControlNet**: 提取深度图特征
- **LoRA**: 轻量级微调主DiT模型
- **深度条件**: 通过additional contexts注入到diffusion过程

## 🔧 故障排除

### 常见问题

1. **深度图文件无效**
   - 确保使用16位PNG或TIFF格式
   - 检查文件是否损坏
   - 尝试用其他图像查看器打开验证

2. **模型加载失败**
   - 检查LoRA权重路径是否正确
   - 确保权重文件完整
   - 查看终端输出的详细错误信息

3. **生成结果不理想**
   - 检查RGB图像和深度图是否匹配
   - 尝试调整引导尺度 (建议5.0-10.0)
   - 增加推理步数（建议30-50步）

### 调试信息

启动时会显示以下信息：
```
正在加载深度条件模型...
自动发现深度LoRA权重: /path/to/weights
深度图处理完成，形状: torch.Size([1, 1, 518, 518])
深度图已添加到模型输入
```

## 📋 命令行参数参考

| 参数 | 描述 | 默认值 |
|------|------|--------|
| `--enable_depth` | 启用深度图模式 | False |
| `--depth_lora_path` | 深度LoRA权重路径 | None (自动搜索) |
| `--model_path` | 基础模型路径 | 'tencent/Hunyuan3D-2.1' |
| `--device` | 运行设备 | 'cuda' |

## 📄 示例命令

```bash
# 基础深度模式
python gradio_app.py --enable_depth

# 指定LoRA权重
python gradio_app.py --enable_depth --depth_lora_path ./checkpoints/step_5000

# 完整配置
python gradio_app.py \
    --enable_depth \
    --depth_lora_path ./output_folder/dit/depth_lora_checkpoints/step_3000 \
    --device cuda \
    --port 8080
```

## 🎨 训练自己的深度LoRA

如果你想训练自己的深度条件模型，请参考：
- `hy3dshape/DEPTH_LORA_README.md` - 详细的训练指南
- `hy3dshape/configs/hunyuandit-depth-lora-flowmatching-dinol518-bf16-lr1e5-4096.yaml` - 训练配置
- `hy3dshape/train_depth_lora.sh` - 训练脚本

---

如有问题，请检查终端输出中的详细错误信息，或参考 `hy3dshape/` 目录下的相关文档。
