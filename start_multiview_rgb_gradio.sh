#!/bin/bash
# 启动多视图RGB重建 Gradio 界面（仅四视图彩色图像，不使用法线图）

# 设置默认参数
MODEL_PATH="tencent/Hunyuan3D-2.1"
SUBFOLDER="hunyuan3d-dit-v2-1"
PORT=6008
HOST="0.0.0.0"
DEVICE="cuda"

# 多视图RGB全量微调checkpoint路径（根据实际情况修改）
# 优先级：全量微调checkpoint > LoRA checkpoint
RGB_CHECKPOINT_PATH=""

# 1. 首先检查是否已指定具体的checkpoint路径
if [ -n "$RGB_CHECKPOINT_PATH" ] && [ -f "$RGB_CHECKPOINT_PATH" ]; then
    echo "✅ 使用指定的checkpoint: $RGB_CHECKPOINT_PATH"
else
    # 2. 优先查找全量微调的checkpoint (.ckpt文件)
    echo "正在查找最新的checkpoint..."
    echo "1. 查找全量微调checkpoint (.ckpt)..."
    FINETUNING_DIRS=(
        "./hy3dshape/output_folder/dit/multiview_rgb_finetuning/ckpt"
        "./output_folder/dit/multiview_rgb_finetuning/ckpt"
        "hy3dshape/output_folder/dit/multiview_rgb_finetuning/ckpt"
        "output_folder/dit/multiview_rgb_finetuning/ckpt"
    )

    for dir in "${FINETUNING_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            # 查找最新的.ckpt文件
            latest_ckpt=$(ls -t "$dir"/*.ckpt 2>/dev/null | head -n 1)
            if [ -n "$latest_ckpt" ]; then
                RGB_CHECKPOINT_PATH="$latest_ckpt"
                echo "✅ 找到全量微调checkpoint: $RGB_CHECKPOINT_PATH"
                break
            fi
        fi
    done
fi

# 3. 如果没有找到全量微调checkpoint，查找LoRA格式（向后兼容）
if [ -z "$RGB_CHECKPOINT_PATH" ]; then
    echo "2. 查找LoRA格式checkpoint..."
    LORA_DIRS=(
        "./hy3dshape/output_folder/dit/multiview_rgb_lora_finetuning/ckpt"
        "./output_folder/dit/multiview_rgb_lora_finetuning/ckpt"
        "./hy3dshape/output_folder/dit/multiview_rgb_lora_checkpoints"
        "./output_folder/dit/multiview_rgb_lora_checkpoints"
    )
    
    for dir in "${LORA_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            # 先查找.ckpt文件
            latest_ckpt=$(ls -t "$dir"/*.ckpt 2>/dev/null | head -n 1)
            if [ -n "$latest_ckpt" ]; then
                RGB_CHECKPOINT_PATH="$latest_ckpt"
                echo "✅ 找到LoRA checkpoint: $RGB_CHECKPOINT_PATH"
                break
            fi
            # 再查找PEFT格式目录
            latest_ckpt=$(ls -d "$dir"/step_* 2>/dev/null | sort -V | tail -n 1)
            if [ -n "$latest_ckpt" ]; then
                RGB_CHECKPOINT_PATH="$latest_ckpt"
                echo "✅ 找到PEFT checkpoint: $RGB_CHECKPOINT_PATH"
                break
            fi
        fi
    done
fi

# 检查是否找到了checkpoint文件（而不是目录）
if [ -n "$RGB_CHECKPOINT_PATH" ] && [ ! -f "$RGB_CHECKPOINT_PATH" ] && [ -d "$RGB_CHECKPOINT_PATH" ]; then
    # 如果是目录，查找其中的.ckpt文件
    ckpt_files=$(ls -t "$RGB_CHECKPOINT_PATH"/*.ckpt 2>/dev/null | head -n 1)
    if [ -n "$ckpt_files" ]; then
        RGB_CHECKPOINT_PATH="$ckpt_files"
        echo "✅ 在目录中找到最新的checkpoint: $RGB_CHECKPOINT_PATH"
    else
        echo "⚠️  目录中没有找到.ckpt文件: $RGB_CHECKPOINT_PATH"
        RGB_CHECKPOINT_PATH=""
    fi
fi

if [ -z "$RGB_CHECKPOINT_PATH" ] || [ ! -f "$RGB_CHECKPOINT_PATH" ]; then
    echo "⚠️  警告: 未找到多视图RGB checkpoint，将使用基础模型"
    echo "请确保已完成训练并设置正确的checkpoint路径"
    CHECKPOINT_ARG=""
else
    CHECKPOINT_ARG="--rgb_lora_path $RGB_CHECKPOINT_PATH"
    echo "✅ 使用checkpoint: $RGB_CHECKPOINT_PATH"
fi

# 启动Gradio应用
echo ""
echo "================================================"
echo "启动多视图RGB重建 Gradio 界面"
echo "================================================"
echo "模型: $MODEL_PATH / $SUBFOLDER"
echo "端口: $PORT"
echo "地址: http://$HOST:$PORT"
echo "模式: 仅四视图RGB重建（不使用法线图）"
echo "================================================"
echo ""

python gradio_app.py \
    --model_path "$MODEL_PATH" \
    --subfolder "$SUBFOLDER" \
    --enable_multiview_rgb \
    $CHECKPOINT_ARG \
    --port $PORT \
    --host "$HOST" \
    --device "$DEVICE" \
    --num_views 4

