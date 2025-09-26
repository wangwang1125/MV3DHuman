#!/bin/bash

# Depth LoRA fine-tuning script for Hunyuan3D
# This script trains the model with depth conditioning via ControlNet and LoRA

export CUDA_VISIBLE_DEVICES=0
export config=configs/hunyuandit-depth-mv-lora-flowmatching-dinol518-bf16-lr1e5-4096.yaml
export output_dir=output_folder/dit/depth_lora_finetuning

echo "Starting Depth LoRA fine-tuning..."
echo "Config: $config"
echo "Output dir: $output_dir"

python main.py \
    -c $config \
    -ng 1 \
    --output_dir $output_dir \
    --use_amp \
    --amp_type bf16 \
    --deepspeed

echo "Training completed. Check results in: $output_dir"
