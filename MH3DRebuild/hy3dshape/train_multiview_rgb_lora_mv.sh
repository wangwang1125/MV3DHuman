#!/bin/bash

# Multi-View RGB Image LoRA Fine-tuning Script for Hunyuan3D-2mv
# This script trains the model with RGB image conditioning from 4 views via DinoImageEncoderMV and LoRA
# Uses Hunyuan3D-2mv as pretrained model
# No normal maps are used, only RGB images from 4 viewpoints

export CUDA_VISIBLE_DEVICES=0
export config=configs/hunyuandit-multiview-rgb-lora-flowmatching-dinol518-bf16-lr1e5-4096-mv.yaml
export output_dir=output_folder/dit/multiview_rgb_lora_finetuning_mv

echo "Starting Multi-View RGB Image LoRA Fine-tuning (from Hunyuan3D-2mv)..."
echo "Config: $config"
echo "Output dir: $output_dir"
echo "Using 4 views: front, right, back, left (indices: 0, 1, 2, 3)"
echo "Pretrained model: tencent/Hunyuan3D-2mv (subfolder: hunyuan3d-dit-v2-mv)"
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
echo "LoRA checkpoints saved in: output_folder/dit/multiview_rgb_lora_checkpoints_mv/"
