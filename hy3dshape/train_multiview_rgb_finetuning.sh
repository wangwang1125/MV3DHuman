#!/bin/bash

# Multi-View RGB Full Fine-tuning Training Script
# This script trains all DiT model weights (not LoRA) for multi-view RGB reconstruction

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export num_gpu_per_node=8

export node_num=1
export node_rank=0
export master_ip=0.0.0.0

export config=configs/hunyuandit-multiview-rgb-finetuning-flowmatching-dinol518-bf16-lr1e5-4096.yaml
export output_dir=output_folder/dit/multiview_rgb_finetuning

# To resume from checkpoint, uncomment and set the checkpoint path:
# export ckpt_path=output_folder/dit/multiview_rgb_finetuning/ckpt/ckpt-00020000.ckpt

python main.py \
    --base ${config} \
    --train \
    --name multiview_rgb_finetuning \
    --logdir ${output_dir} \
    --gpus ${num_gpu_per_node} \
    --strategy ddp \
    --precision bf16-mixed \
    --output_dir ${output_dir} \
    ${ckpt_path:+--ckpt_path ${ckpt_path}}
