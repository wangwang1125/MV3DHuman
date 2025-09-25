#!/bin/bash

# Depth LoRA fine-tuning script for Hunyuan3D
# This script trains the model with depth conditioning via ControlNet and LoRA

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export config=configs/hunyuandit-depth-lora-flowmatching-dinol518-bf16-lr1e5-4096.yaml
export output_dir=output_folder/dit/depth_lora_finetuning

echo "Starting Depth LoRA fine-tuning..."
echo "Config: $config"
echo "Output dir: $output_dir"

python main.py \
    -c $config \
    -ng 8 \
    --output_dir $output_dir \
    --use_amp \
    --amp_type bf16 \
    --deepspeed

echo "Training completed. Check results in: $output_dir"
