#!/bin/bash

# Multi-View Normal Map LoRA Fine-tuning Script for Hunyuan3D
# This script trains the model with normal map conditioning via DinoImageEncoderMV and LoRA

export CUDA_VISIBLE_DEVICES=0
export config=configs/hunyuandit-multiview-normal-lora-flowmatching-dinol518-bf16-lr1e5-4096.yaml
export output_dir=output_folder/dit/normal_lora_finetuning

echo "Starting Multi-View Normal Map LoRA Fine-tuning..."
echo "Config: $config"
echo "Output dir: $output_dir"
echo ""

python main.py \
    -c $config \
    -ng 1 \
    --output_dir $output_dir \
    --use_amp \
    --amp_type bf16 \
    --deepspeed

echo ""
echo "Training completed. Check results in: $output_dir"
echo "LoRA checkpoints saved in: output_folder/dit/multiview_normal_lora_checkpoints/"

