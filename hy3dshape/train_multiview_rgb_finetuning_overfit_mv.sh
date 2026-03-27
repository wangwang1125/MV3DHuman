#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export num_gpu_per_node=1

export config=configs/hunyuandit-multiview-rgb-finetuning-flowmatching-dinol518-bf16-overfit-mv.yaml
export output_dir=output_folder/dit/multiview_rgb_finetuning_mv_overfit

echo "Starting Multi-View RGB Full Fine-tuning Overfit Debug..."
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
    --deepspeed
