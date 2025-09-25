# Hunyuan3D 深度条件LoRA微调

本文档说明如何使用深度图作为条件对Hunyuan3D模型进行LoRA微调。

## 功能特性

- ✅ **深度条件输入**: 支持单通道深度图(.exr格式)作为额外条件
- ✅ **ControlNet风格架构**: 轻量级深度特征提取器，不影响主模型结构
- ✅ **LoRA微调**: 仅训练低秩适配器，保持原模型权重不变
- ✅ **彩色图特征分离**: 深度图仅在diffusion阶段起作用，不影响DINO特征提取

## 数据准备

### 数据集结构
```
tools/mini_depth_trainset/preprocessed/
├── object_001/
│   ├── geo_data/
│   │   └── object_001_surface.npz
│   └── render_cond/
│       ├── 000.png          # RGB图像
│       ├── 000_depth.exr    # 对应深度图
│       ├── 001.png
│       ├── 001_depth.exr
│       └── ...
└── object_002/
    └── ...
```

### 深度图要求
- **格式**: EXR文件，单通道或多通道（使用第一通道）
- **值域**: 原始深度值，代码会自动归一化到[0,1]
- **分辨率**: 与对应RGB图像相同
- **命名**: `{index:03d}_depth.exr`

## 训练配置

### 配置文件
使用 `configs/hunyuandit-depth-lora-flowmatching-dinol518-bf16-lr1e5-4096.yaml`

关键配置项：
```yaml
dataset:
  params:
    load_depth: true  # 启用深度图加载

model:
  params:
    # LoRA配置
    lora_config:
      rank: 8
      target_modules: ["to_q", "to_k", "to_v", "to_out.0"]
    
    # ControlNet配置  
    control_net_config:
      pretrained_model_name_or_path: "lllyasviel/sd-controlnet-depth"
      train_unet: false
      
    # 深度输入通道
    control_in_channels: 1
```

### 启动训练
```bash
cd hy3dshape
chmod +x train_depth_lora.sh
./train_depth_lora.sh
```

或手动运行：
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python main.py \
    --base configs/hunyuandit-depth-lora-flowmatching-dinol518-bf16-lr1e5-4096.yaml \
    --train \
    --name depth_lora_experiment \
    --logdir output_folder/dit/depth_lora_finetuning \
    --gpus 8 \
    --strategy ddp \
    --precision bf16-mixed
```

## 推理使用

### 基本推理
```python
import torch
from hy3dshape.models.diffusion.flow_matching_sit import Diffuser
from peft import PeftModel

# 加载模型
model = Diffuser(**config['model']['params'])

# 加载LoRA权重
lora_path = "output_folder/dit/depth_lora_checkpoints/step_1000"
model.model = PeftModel.from_pretrained(model.model, lora_path)
if model.controlnet is not None:
    model.controlnet = PeftModel.from_pretrained(model.controlnet, lora_path)

# 准备输入
batch = {
    'image': rgb_image,      # PIL Image
    'depth': depth_tensor,   # (1, 1, H, W) torch.FloatTensor
}

# 生成
with torch.no_grad():
    outputs = model.sample(batch, output_type='trimesh')
```

### 测试脚本
```bash
python test_depth_inference.py
```

## 架构说明

### 深度条件处理流程
1. **数据加载**: EXR深度图 → 归一化 → (B,1,H,W)张量
2. **特征提取**: 深度张量 → ControlNet → 深度特征向量
3. **条件注入**: 深度特征注入到diffusion contexts
4. **生成**: 主DiT模型使用RGB+深度条件生成3D

### ControlNet架构
```python
DepthControlNet(
  depth_encoder: Sequential(
    Conv2d(1, 64, 3, padding=1)
    ReLU()
    Conv2d(64, 128, 3, stride=2, padding=1)  
    ReLU()
    Conv2d(128, 256, 3, stride=2, padding=1)
    ReLU()
    AdaptiveAvgPool2d((1, 1))
    Flatten()
    Linear(256, 768)  # 匹配DiT hidden_dim
  )
)
```

### LoRA应用范围
- **主DiT模型**: 注意力层 `["to_q", "to_k", "to_v", "to_out.0"]`
- **ControlNet**: 同样的注意力层（如果存在）
- **rank**: 8（可调整为4/16视性能需求）

## 输出和检查点

### 训练输出
```
output_folder/dit/depth_lora_finetuning/
├── checkpoints/           # 完整模型检查点
├── depth_lora_checkpoints/# LoRA权重检查点
│   ├── step_1000/
│   ├── step_2000/
│   └── ...
└── logs/                  # 训练日志
```

### LoRA权重结构
```
step_1000/
├── adapter_config.json
├── adapter_model.bin     # LoRA权重
└── README.md
```

## 故障排除

### 常见问题

1. **深度图加载失败**
   - 检查EXR文件路径和格式
   - 确保OpenCV支持EXR格式：`pip install opencv-contrib-python`

2. **内存不足**
   - 减小batch_size
   - 降低LoRA rank
   - 使用梯度检查点

3. **训练不收敛**
   - 检查深度图质量和对齐
   - 调整学习率
   - 增加warmup步数

4. **推理结果差**
   - 确保深度图预处理一致
   - 检查LoRA权重加载
   - 验证条件注入是否正确

### 调试技巧
```python
# 检查深度图加载
print(f"Depth shape: {batch['depth'].shape}")
print(f"Depth range: [{batch['depth'].min():.3f}, {batch['depth'].max():.3f}]")

# 检查ControlNet输出
if model.controlnet is not None:
    depth_features = model.controlnet(batch['depth'])
    print(f"Depth features shape: {depth_features.shape}")
```

## 性能优化

### 训练加速
- 使用混合精度：`--precision bf16-mixed`
- 多GPU训练：`--strategy ddp`
- 数据并行加载：增加`num_workers`

### 推理优化
- 模型量化
- 批量推理
- 缓存深度特征

## 扩展方向

1. **多模态条件**: 法线图、位置图等
2. **更复杂ControlNet**: 多尺度特征、注意力机制
3. **动态LoRA rank**: 根据层重要性调整rank
4. **条件权重控制**: 可调节深度条件强度

---

更多详细信息请参考原始Hunyuan3D文档和代码注释。
