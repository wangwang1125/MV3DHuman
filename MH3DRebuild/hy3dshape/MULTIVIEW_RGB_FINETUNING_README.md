# 多视图RGB全量微调训练指南

## 概述

本配置将多视图RGB LoRA训练改为**全量微调**模式，所有DiT模型权重都可训练，支持从预训练权重加载和从checkpoint继续训练。

## 主要变化

### 与原LoRA配置的区别

| 特性 | LoRA配置 | 全量微调配置 |
|------|---------|-------------|
| **主DiT模型权重** | ❌ 冻结（仅LoRA可训练） | ✅ **全部可训练** |
| **可训练参数** | ~3K (仅LoRA) | ~3.3B (全部) |
| **内存占用** | 低 | 高 |
| **训练速度** | 快 | 慢 |
| **Checkpoint格式** | PEFT格式 | PyTorch Lightning完整checkpoint |

## 配置文件

### 配置文件位置
```
configs/hunyuandit-multiview-rgb-finetuning-flowmatching-dinol518-bf16-lr1e5-4096.yaml
```

### 关键配置说明

#### 1. 移除LoRA配置
```yaml
# LoRA Configuration - DISABLED for full fine-tuning
# lora_config:
#   rank: 8
#   target_modules: ["to_q", "to_k", "to_v", "to_out.0"]
```

#### 2. 从预训练权重加载
```yaml
denoiser_cfg:
  target: hy3dshape.models.denoisers.hunyuandit.HunYuanDiTPlain
  from_pretrained: tencent/Hunyuan3D-2.1  # 加载预训练权重
```

#### 3. Checkpoint恢复配置
```yaml
training:
  ckpt_path: ""  # 设置为checkpoint路径以继续训练
```

## 使用方法

### 1. 首次训练（从预训练权重开始）

```bash
cd hy3dshape
chmod +x train_multiview_rgb_finetuning.sh
./train_multiview_rgb_finetuning.sh
```

或手动运行：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python main.py \
    --base configs/hunyuandit-multiview-rgb-finetuning-flowmatching-dinol518-bf16-lr1e5-4096.yaml \
    --train \
    --name multiview_rgb_finetuning \
    --logdir output_folder/dit/multiview_rgb_finetuning \
    --gpus 8 \
    --strategy ddp \
    --precision bf16-mixed \
    --output_dir output_folder/dit/multiview_rgb_finetuning
```

### 2. 从Checkpoint继续训练

#### 方法1：通过命令行参数
```bash
python main.py \
    --base configs/hunyuandit-multiview-rgb-finetuning-flowmatching-dinol518-bf16-lr1e5-4096.yaml \
    --train \
    --name multiview_rgb_finetuning \
    --logdir output_folder/dit/multiview_rgb_finetuning \
    --gpus 8 \
    --strategy ddp \
    --precision bf16-mixed \
    --output_dir output_folder/dit/multiview_rgb_finetuning \
    --ckpt_path output_folder/dit/multiview_rgb_finetuning/ckpt/ckpt-00020000.ckpt
```

#### 方法2：修改配置文件
在配置文件中设置：
```yaml
training:
  ckpt_path: "output_folder/dit/multiview_rgb_finetuning/ckpt/ckpt-00020000.ckpt"
```

#### 方法3：使用训练脚本
修改 `train_multiview_rgb_finetuning.sh`：
```bash
export ckpt_path=output_folder/dit/multiview_rgb_finetuning/ckpt/ckpt-00020000.ckpt
```

## Checkpoint管理

### Checkpoint保存位置
```
output_folder/dit/multiview_rgb_finetuning/
├── ckpt/
│   ├── ckpt-00002000.ckpt
│   ├── ckpt-00004000.ckpt
│   ├── ckpt-00006000.ckpt
│   └── ...
└── log/
    └── tensorboard/
```

### Checkpoint保存频率
- 默认每 `every_n_train_steps: 2000` 步保存一次
- 可通过配置文件中的 `training.every_n_train_steps` 调整

### Checkpoint内容
PyTorch Lightning checkpoint包含：
- 模型权重（`state_dict`）
- 优化器状态
- 学习率调度器状态
- 训练步数（`global_step`）
- 其他训练状态

## 权重冻结情况

### ✅ 冻结的组件
- **VAE (ShapeVAE)**: 所有参数冻结
- **图像编码器 (DinoImageEncoderMV)**: 所有参数冻结

### ✅ 可训练的组件
- **主DiT模型 (HunYuanDiTPlain)**: **全部参数可训练** (~3.3B参数)

## 训练参数对比

| 配置 | 可训练参数 | 内存占用 | 训练速度 |
|------|-----------|---------|---------|
| **LoRA** | ~3K | 低 | 快 |
| **全量微调** | ~3.3B | 高 | 慢 |

## 注意事项

### 1. 显存要求
- 全量微调需要更多显存（约是LoRA的2-3倍）
- 建议使用至少24GB显存的GPU
- 如果显存不足，可以：
  - 减小 `batch_size`
  - 使用梯度累积（`--update_every`）
  - 启用CPU卸载（如果支持）

### 2. 训练时间
- 全量微调训练速度较慢
- 建议使用多GPU训练（DDP策略）

### 3. Checkpoint兼容性
- 全量微调的checkpoint与LoRA checkpoint格式不同
- 不能直接混用
- 如果需要从LoRA切换到全量微调，需要重新开始训练

### 4. 学习率
- 当前配置使用 `base_lr: 1e-5`
- 全量微调可能需要更小的学习率
- 可以根据训练情况调整

## 故障排除

### 问题1：CUDA Out of Memory
**解决方法：**
- 减小 `batch_size`（在配置文件中）
- 增加 `--update_every`（梯度累积）
- 使用更少的GPU或更小的模型

### 问题2：Checkpoint加载失败
**解决方法：**
- 检查checkpoint路径是否正确
- 确保checkpoint文件完整
- 检查配置文件是否与checkpoint匹配

### 问题3：训练不收敛
**解决方法：**
- 降低学习率
- 检查数据质量
- 增加warmup步数
- 检查梯度裁剪设置

## 性能优化建议

1. **使用混合精度训练**：已启用 `bf16-mixed`
2. **多GPU训练**：使用DDP策略
3. **梯度累积**：通过 `--update_every` 参数
4. **定期保存checkpoint**：避免训练中断导致进度丢失

## 总结

全量微调配置适合：
- ✅ 有大规模数据集
- ✅ 有充足的计算资源
- ✅ 需要最大化的模型性能
- ✅ 需要从预训练权重继续训练

LoRA配置适合：
- ✅ 快速适配新任务
- ✅ 资源有限
- ✅ 只需要小幅调整模型
