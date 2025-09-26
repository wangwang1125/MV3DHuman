#!/bin/bash

# Multi-view + Depth LoRA fine-tuning script for Hunyuan3D
# This script trains the model with multi-view images and corresponding depth maps

export CUDA_VISIBLE_DEVICES=0
export config=configs/hunyuandit-multiview-depth-lora-flowmatching-dinol518-bf16-lr1e5-4096.yaml
export output_dir=output_folder/dit/multiview_depth_lora_finetuning

echo "Starting Multi-view + Depth LoRA fine-tuning..."
echo "Config: $config"
echo "Output dir: $output_dir"
echo "Expected data format:"
echo "  - Multi-view RGB images: front, back, left, right"
echo "  - Corresponding depth maps for each view"
echo "  - Higher memory requirements due to multi-view processing"

# 检查数据集是否存在
if [ ! -d "tools/mini_mv_depth_trainset" ]; then
    echo "Warning: Multi-view depth training dataset not found!"
    echo "Expected path: tools/mini_mv_depth_trainset/preprocessed/"
    echo "Please prepare your dataset according to the format described in the README."
fi

python main.py \
    -c $config \
    -ng 1 \
    --output_dir $output_dir \
    --use_amp \
    --amp_type bf16 \
    --deepspeed \
    --strategy ddp

echo "Training completed. Check results in: $output_dir"
echo "LoRA weights saved to: output_folder/dit/multiview_depth_lora_checkpoints/"
