#!/bin/bash
# 启动多视图法线条件 Gradio 界面

# 设置默认参数
MODEL_PATH="tencent/Hunyuan3D-2.1"
SUBFOLDER="hunyuan3d-dit-v2-1"
PORT=6008
HOST="0.0.0.0"
DEVICE="cuda"

# 多视图法线LoRA权重路径（根据实际情况修改）
# 优先级：Lightning checkpoint (.ckpt) > PEFT格式 (step_*)
NORMAL_LORA_PATH="./hy3dshape/output_folder/dit/multiview_normal_lora_checkpoints"

# 1. 首先检查是否已指定具体的checkpoint路径
if [ -n "$NORMAL_LORA_PATH" ] && [ -f "$NORMAL_LORA_PATH" ]; then
    echo "✅ 使用指定的checkpoint: $NORMAL_LORA_PATH"
else
    # 2. 如果没有指定或文件不存在，查找Lightning checkpoint (.ckpt文件)
    LIGHTNING_DIRS=(
        "./hy3dshape/output_folder/dit/normal_lora_finetuning/ckpt"
        "./output_folder/dit/normal_lora_finetuning/ckpt"
        "./hy3dshape/output_folder/dit/multiview_normal_lora_checkpoints/ckpt"
        "./output_folder/dit/multiview_normal_lora_checkpoints/ckpt"
    )

    echo "正在查找最新的checkpoint..."
    echo "1. 查找Lightning checkpoint (.ckpt)..."
    for dir in "${LIGHTNING_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            # 查找最新的.ckpt文件
            latest_ckpt=$(ls -t "$dir"/*.ckpt 2>/dev/null | head -n 1)
            if [ -n "$latest_ckpt" ]; then
                NORMAL_LORA_PATH="$latest_ckpt"
                echo "✅ 找到Lightning checkpoint: $NORMAL_LORA_PATH"
                break
            fi
        fi
    done
fi

# 3. 如果没有找到Lightning checkpoint，查找PEFT格式
if [ -z "$NORMAL_LORA_PATH" ]; then
    echo "2. 查找PEFT格式 LoRA checkpoint (step_*)..."
    PEFT_DIRS=(
        "./hy3dshape/output_folder/dit/multiview_normal_lora_checkpoints"
        "./output_folder/dit/multiview_normal_lora_checkpoints"
    )
    
    for dir in "${PEFT_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            # 查找最新的step_*目录
            latest_ckpt=$(ls -d "$dir"/step_* 2>/dev/null | sort -V | tail -n 1)
            if [ -n "$latest_ckpt" ]; then
                NORMAL_LORA_PATH="$latest_ckpt"
                echo "✅ 找到PEFT checkpoint: $NORMAL_LORA_PATH"
                break
            fi
        fi
    done
fi

# 检查是否找到了checkpoint文件（而不是目录）
if [ -n "$NORMAL_LORA_PATH" ] && [ ! -f "$NORMAL_LORA_PATH" ] && [ -d "$NORMAL_LORA_PATH" ]; then
    # 如果是目录，查找其中的.ckpt文件
    ckpt_files=$(ls -t "$NORMAL_LORA_PATH"/*.ckpt 2>/dev/null | head -n 1)
    if [ -n "$ckpt_files" ]; then
        NORMAL_LORA_PATH="$ckpt_files"
        echo "✅ 在目录中找到最新的checkpoint: $NORMAL_LORA_PATH"
    else
        echo "⚠️  目录中没有找到.ckpt文件: $NORMAL_LORA_PATH"
        NORMAL_LORA_PATH=""
    fi
fi

if [ -z "$NORMAL_LORA_PATH" ] || [ ! -f "$NORMAL_LORA_PATH" ]; then
    echo "⚠️  警告: 未找到多视图法线LoRA权重，将使用基础模型"
    echo "请确保已完成训练并设置正确的checkpoint路径"
    LORA_ARG=""
else
    LORA_ARG="--normal_lora_path $NORMAL_LORA_PATH"
    echo "✅ 使用LoRA权重: $NORMAL_LORA_PATH"
fi

# 启动Gradio应用
echo ""
echo "================================================"
echo "启动多视图法线条件 Gradio 界面"
echo "================================================"
echo "模型: $MODEL_PATH / $SUBFOLDER"
echo "端口: $PORT"
echo "地址: http://$HOST:$PORT"
echo "================================================"
echo ""

python gradio_app.py \
    --model_path "$MODEL_PATH" \
    --subfolder "$SUBFOLDER" \
    --enable_multiview_normal \
    $LORA_ARG \
    --port $PORT \
    --host "$HOST" \
    --device "$DEVICE" \
    --num_views 2

