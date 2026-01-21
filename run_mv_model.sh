#!/bin/bash

# Hunyuan3D-2mv 模型启动脚本
# 使用方法: bash run_mv_model.sh

echo "=========================================="
echo "启动 Hunyuan3D-2mv 模型"
echo "=========================================="
echo ""
echo "模型信息:"
echo "  - 模型路径: tencent/Hunyuan3D-2mv"
echo "  - Subfolder: hunyuan3d-dit-v2-mv"
echo "  - Hugging Face: https://huggingface.co/tencent/Hunyuan3D-2mv"
echo ""
echo "首次运行将自动从 Hugging Face 下载模型（约4.93 GB）"
echo ""

# 设置默认参数
MODEL_PATH="tencent/Hunyuan3D-2mv"
PORT=${PORT:-6008}
HOST=${HOST:-"0.0.0.0"}
DEVICE=${DEVICE:-"cuda"}

# 解析命令行参数
LOW_VRAM=""
MULTI_GPU=""
GPU_IDS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --low_vram)
            LOW_VRAM="--low_vram_mode"
            shift
            ;;
        --multi_gpu)
            MULTI_GPU="--use_multi_gpu"
            shift
            ;;
        --gpu_ids)
            GPU_IDS="--gpu_ids $2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            echo "可用参数:"
            echo "  --low_vram        启用低显存模式"
            echo "  --multi_gpu       启用多GPU"
            echo "  --gpu_ids IDS     指定GPU ID（如: 0,1）"
            echo "  --port PORT       指定端口（默认: 6008）"
            echo "  --host HOST       指定主机（默认: 0.0.0.0）"
            echo "  --device DEVICE   指定设备（默认: cuda）"
            exit 1
            ;;
    esac
done

echo "启动参数:"
echo "  - 端口: $PORT"
echo "  - 主机: $HOST"
echo "  - 设备: $DEVICE"
if [ -n "$LOW_VRAM" ]; then
    echo "  - 低显存模式: 启用"
fi
if [ -n "$MULTI_GPU" ]; then
    echo "  - 多GPU模式: 启用"
    if [ -n "$GPU_IDS" ]; then
        echo "  - GPU IDs: $GPU_IDS"
    fi
fi
echo ""

# 构建命令
CMD="python gradio_app.py \
    --model_path $MODEL_PATH \
    --port $PORT \
    --host $HOST \
    --device $DEVICE \
    $LOW_VRAM \
    $MULTI_GPU \
    $GPU_IDS"

echo "执行命令:"
echo "$CMD"
echo ""
echo "=========================================="
echo ""

# 执行命令
eval $CMD
