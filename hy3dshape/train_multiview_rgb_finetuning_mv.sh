#!/bin/bash

# Multi-View RGB Full Fine-tuning Training Script (from Hunyuan3D-2mv)
# This script trains all DiT model weights (not LoRA) for multi-view RGB reconstruction
# Uses Hunyuan3D-2mv as pretrained model

export CUDA_VISIBLE_DEVICES=0
export num_gpu_per_node=1

export config=configs/hunyuandit-multiview-rgb-finetuning-flowmatching-dinol518-bf16-lr1e5-4096-mv.yaml
export output_dir=output_folder/dit/multiview_rgb_finetuning_mv

# To resume from checkpoint, uncomment and set the checkpoint path:
# export ckpt_path=output_folder/dit/multiview_rgb_finetuning_mv/ckpt/ckpt-00020000.ckpt

echo "Starting Multi-View RGB Full Fine-tuning (from Hunyuan3D-2mv)..."
echo "Config: $config"
echo "Output dir: $output_dir"
echo "GPUs: $num_gpu_per_node"
echo "Pretrained model: tencent/Hunyuan3D-2mv (subfolder: hunyuan3d-dit-v2-mv)"
echo ""

python main.py \
    -c ${config} \
    -ng ${num_gpu_per_node} \
    --output_dir ${output_dir} \
    --use_amp \
    --amp_type bf16 \
    ${ckpt_path:+--ckpt_path ${ckpt_path}} \
    --deepspeed

echo ""
echo "Training completed. Check results in: $output_dir"
echo "Checkpoints saved in: $output_dir/ckpt/"
