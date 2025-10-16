#!/bin/bash

# Multi-View Depth LoRA Training with Token Merging
# This script trains the multi-view depth LoRA with Token Merging optimization

echo "Starting Multi-View Depth LoRA Training with Token Merging..."

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Training parameters
CONFIG_PATH="configs/hunyuandit-multiview-depth-lora-tome-flowmatching-dinol518-bf16-lr1e5-4096.yaml"
OUTPUT_DIR="output_folder/dit/multiview_depth_lora_tome"

echo "Config: $CONFIG_PATH"
echo "Output dir: $OUTPUT_DIR"

# Create directories
mkdir -p $OUTPUT_DIR

# Training command with Token Merging configuration
# Using the correct argument format for main.py
python main.py \
    -c $CONFIG_PATH \
    -ng 1 \
    --output_dir $OUTPUT_DIR \
    --use_amp \
    --amp_type bf16 \
    --deepspeed \
    --gradient_clip_val 1.0 \
    --every_n_train_steps 2000 \
    --val_check_interval 1500 \
    --limit_val_batches 16 \
    --log_every_n_steps 50

echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"