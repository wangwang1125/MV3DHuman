#!/bin/bash
# 启动多视图RGB重建 Gradio 界面（仅四视图彩色图像，不使用法线图）

# 设置默认参数
MODEL_PATH="tencent/Hunyuan3D-2.1"
SUBFOLDER="hunyuan3d-dit-v2-1"
PORT=6009
HOST="0.0.0.0"
DEVICE="cuda"

# 多视图RGB LoRA权重路径（根据实际情况修改）
# 优先级：Lightning checkpoint (.ckpt) > PEFT格式 (step_*)
RGB_LORA_PATH="./hy3dshape/output_folder/dit/multiview_rgb_lora_checkpoints"

# 1. 首先检查是否已指定具体的checkpoint路径
if [ -n "$RGB_LORA_PATH" ] && [ -f "$RGB_LORA_PATH" ]; then
    echo "✅ 使用指定的checkpoint: $RGB_LORA_PATH"
else
    # 2. 如果没有指定或文件不存在，查找Lightning checkpoint (.ckpt文件)
    LIGHTNING_DIRS=(
        "./hy3dshape/output_folder/dit/multiview_rgb_lora_finetuning/ckpt"
        "./output_folder/dit/multiview_rgb_lora_finetuning/ckpt"
        "./hy3dshape/output_folder/dit/multiview_rgb_lora_checkpoints/ckpt"
        "./output_folder/dit/multiview_rgb_lora_checkpoints/ckpt"
    )

    echo "正在查找最新的checkpoint..."
    echo "1. 查找Lightning checkpoint (.ckpt)..."
    for dir in "${LIGHTNING_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            # 查找最新的.ckpt文件
            latest_ckpt=$(ls -t "$dir"/*.ckpt 2>/dev/null | head -n 1)
            if [ -n "$latest_ckpt" ]; then
                RGB_LORA_PATH="$latest_ckpt"
                echo "✅ 找到Lightning checkpoint: $RGB_LORA_PATH"
                break
            fi
        fi
    done
fi

# 3. 如果没有找到Lightning checkpoint，查找PEFT格式
if [ -z "$RGB_LORA_PATH" ]; then
    echo "2. 查找PEFT格式 LoRA checkpoint (step_*)..."
    PEFT_DIRS=(
        "./hy3dshape/output_folder/dit/multiview_rgb_lora_checkpoints"
        "./output_folder/dit/multiview_rgb_lora_checkpoints"
    )
    
    for dir in "${PEFT_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            # 查找最新的step_*目录
            latest_ckpt=$(ls -d "$dir"/step_* 2>/dev/null | sort -V | tail -n 1)
            if [ -n "$latest_ckpt" ]; then
                RGB_LORA_PATH="$latest_ckpt"
                echo "✅ 找到PEFT checkpoint: $RGB_LORA_PATH"
                break
            fi
        fi
    done
fi

# 检查是否找到了checkpoint文件（而不是目录）
if [ -n "$RGB_LORA_PATH" ] && [ ! -f "$RGB_LORA_PATH" ] && [ -d "$RGB_LORA_PATH" ]; then
    # 如果是目录，查找其中的.ckpt文件
    ckpt_files=$(ls -t "$RGB_LORA_PATH"/*.ckpt 2>/dev/null | head -n 1)
    if [ -n "$ckpt_files" ]; then
        RGB_LORA_PATH="$ckpt_files"
        echo "✅ 在目录中找到最新的checkpoint: $RGB_LORA_PATH"
    else
        echo "⚠️  目录中没有找到.ckpt文件: $RGB_LORA_PATH"
        RGB_LORA_PATH=""
    fi
fi

if [ -z "$RGB_LORA_PATH" ] || [ ! -f "$RGB_LORA_PATH" ]; then
    echo "⚠️  警告: 未找到多视图RGB LoRA权重，将使用基础模型"
    echo "请确保已完成训练并设置正确的checkpoint路径"
    LORA_ARG=""
else
    LORA_ARG="--rgb_lora_path $RGB_LORA_PATH"
    echo "✅ 使用LoRA权重: $RGB_LORA_PATH"
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
    $LORA_ARG \
    --port $PORT \
    --host "$HOST" \
    --device "$DEVICE" \
    --num_views 4

