#!/bin/bash

# Multi-View Depth LoRA Training with Token Merging
# This script trains the multi-view depth LoRA with Token Merging optimization

echo "Starting Multi-View Depth LoRA Training with Token Merging..."

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Training parameters
CONFIG_PATH="configs/hunyuandit-multiview-depth-lora-tome-flowmatching-dinol518-bf16-lr1e5-4096.yaml"
OUTPUT_DIR="output_folder/dit/multiview_depth_lora_tome"
LOG_DIR="logs/multiview_depth_lora_tome"

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Training command with Token Merging configuration
python main.py \
    --config $CONFIG_PATH \
    --train \
    --work_dir $OUTPUT_DIR \
    --log_dir $LOG_DIR \
    --accelerator gpu \
    --devices 1 \
    --precision bf16-mixed \
    --gradient_clip_val 1.0 \
    --max_epochs 100 \
    --check_val_every_n_epoch 5 \
    --log_every_n_steps 50 \
    --val_check_interval 1500 \
    --limit_val_batches 16 \
    --every_n_train_steps 2000 \
    --limit_train_batches 1.0 \
    --num_sanity_val_steps 2 \
    --enable_progress_bar true \
    --enable_model_summary true \
    --detect_anomaly false \
    --benchmark true \
    --deterministic false \
    --sync_batchnorm false \
    --reload_dataloaders_every_n_epochs 0 \
    --default_root_dir $OUTPUT_DIR \
    --resume_from_checkpoint null \
    --fast_dev_run false \
    --overfit_batches 0.0 \
    --profiler null \
    --accumulate_grad_batches 1 \
    --track_grad_norm -1 \
    --auto_lr_find false \
    --auto_scale_batch_size null \
    --auto_select_gpus false \
    --replace_sampler_ddp true \
    --multiple_trainloader_mode min_size \
    --inference_mode true \
    --use_distributed_sampler true \
    --find_unused_parameters false \
    --strategy auto \
    --plugins null \
    --num_nodes 1 \
    --ipus null \
    --tpu_cores null \
    --enable_checkpointing true \
    --enable_model_summary true \
    --enable_progress_bar true \
    --logger true \
    --callbacks null \
    --max_time null \
    --min_time null \
    --max_steps null \
    --min_steps null \
    --limit_test_batches 1.0 \
    --limit_predict_batches 1.0 \
    --stochastic_weight_avg false \
    --stochastic_weight_avg_lrs null \
    --terminate_on_nan false \
    --reload_dataloaders_every_n_epochs 0 \
    --num_sanity_val_steps 2 \
    --check_val_every_n_epoch 1 \
    --log_every_n_steps 50 \
    --fast_dev_run false \
    --overfit_batches 0.0 \
    --accumulate_grad_batches 1 \
    --track_grad_norm -1 \
    --auto_lr_find false \
    --auto_scale_batch_size null \
    --auto_select_gpus false \
    --replace_sampler_ddp true \
    --multiple_trainloader_mode min_size \
    --inference_mode true \
    --use_distributed_sampler true \
    --find_unused_parameters false \
    --strategy auto \
    --plugins null \
    --num_nodes 1 \
    --ipus null \
    --tpu_cores null \
    --enable_checkpointing true \
    --enable_model_summary true \
    --enable_progress_bar true \
    --logger true \
    --callbacks null \
    --max_time null \
    --min_time null \
    --max_steps null \
    --min_steps null \
    --limit_test_batches 1.0 \
    --limit_predict_batches 1.0 \
    --stochastic_weight_avg false \
    --stochastic_weight_avg_lrs null \
    --terminate_on_nan false

echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "Logs saved to: $LOG_DIR"
