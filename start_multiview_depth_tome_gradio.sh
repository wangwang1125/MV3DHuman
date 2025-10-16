#!/bin/bash
# 启动多视图深度条件 Gradio 界面 (支持Token Merging)
# 使用训练好的Token Merging优化模型进行推理

# 设置默认参数
MODEL_PATH="tencent/Hunyuan3D-2.1"
SUBFOLDER="hunyuan3d-dit-v2-1"
PORT=6009  # 使用不同端口避免冲突
HOST="0.0.0.0"
DEVICE="cuda"

# Token Merging优化的多视图深度LoRA权重路径
# 优先级：Lightning checkpoint (.ckpt) > PEFT格式 (step_*)
TOME_DEPTH_LORA_PATH="./hy3dshape/output_folder/dit/multiview_depth_lora_tome_checkpoints/ckpt/"

# 1. 首先检查是否已指定具体的checkpoint路径
if [ -n "$TOME_DEPTH_LORA_PATH" ] && [ -f "$TOME_DEPTH_LORA_PATH" ]; then
    echo "✅ 使用指定的Token Merging checkpoint: $TOME_DEPTH_LORA_PATH"
else
    # 2. 如果没有指定或文件不存在，查找Lightning checkpoint (.ckpt文件)
    TOME_LIGHTNING_DIRS=(
        "./hy3dshape/output_folder/dit/multiview_depth_lora_tome_checkpoints/ckpt"
        "./output_folder/dit/multiview_depth_lora_tome_checkpoints/ckpt"
        "./hy3dshape/output_folder/dit/multiview_depth_lora_tome_checkpoints"
        "./output_folder/dit/multiview_depth_lora_tome_checkpoints"
    )

    echo "正在查找Token Merging优化的checkpoint..."
    echo "1. 查找Lightning checkpoint (.ckpt)..."
    for dir in "${TOME_LIGHTNING_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            # 查找最新的.ckpt文件
            latest_ckpt=$(ls -t "$dir"/*.ckpt 2>/dev/null | head -n 1)
            if [ -n "$latest_ckpt" ]; then
                TOME_DEPTH_LORA_PATH="$latest_ckpt"
                echo "✅ 找到Token Merging Lightning checkpoint: $TOME_DEPTH_LORA_PATH"
                break
            fi
        fi
    done
fi

# 3. 如果没有找到Token Merging checkpoint，查找PEFT格式
if [ -z "$TOME_DEPTH_LORA_PATH" ]; then
    echo "2. 查找Token Merging PEFT格式 LoRA checkpoint (step_*)..."
    TOME_PEFT_DIRS=(
        "./hy3dshape/output_folder/dit/multiview_depth_lora_tome_checkpoints"
        "./output_folder/dit/multiview_depth_lora_tome_checkpoints"
    )
    
    for dir in "${TOME_PEFT_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            # 查找最新的step_*目录
            latest_ckpt=$(ls -d "$dir"/step_* 2>/dev/null | sort -V | tail -n 1)
            if [ -n "$latest_ckpt" ]; then
                TOME_DEPTH_LORA_PATH="$latest_ckpt"
                echo "✅ 找到Token Merging PEFT checkpoint: $TOME_DEPTH_LORA_PATH"
                break
            fi
        fi
    done
fi

# 4. 如果仍然没有找到Token Merging checkpoint，回退到原始多视图深度checkpoint
if [ -z "$TOME_DEPTH_LORA_PATH" ]; then
    echo "⚠️  未找到Token Merging优化的checkpoint，回退到原始多视图深度LoRA权重..."
    
    # 查找原始的多视图深度checkpoint
    ORIGINAL_DIRS=(
        "./hy3dshape/output_folder/dit/multiview_depth_lora_checkpoints"
        "./output_folder/dit/multiview_depth_lora_checkpoints"
    )
    
    for dir in "${ORIGINAL_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            # 查找最新的step_*目录
            latest_ckpt=$(ls -d "$dir"/step_* 2>/dev/null | sort -V | tail -n 1)
            if [ -n "$latest_ckpt" ]; then
                TOME_DEPTH_LORA_PATH="$latest_ckpt"
                echo "✅ 使用原始多视图深度checkpoint: $TOME_DEPTH_LORA_PATH"
                break
            fi
        fi
    done
fi

if [ -z "$TOME_DEPTH_LORA_PATH" ]; then
    echo "⚠️  警告: 未找到多视图深度LoRA权重，将使用基础模型"
    echo "请确保已完成训练并设置正确的checkpoint路径"
    LORA_ARG=""
else
    LORA_ARG="--depth_lora_path $TOME_DEPTH_LORA_PATH"
    echo "✅ 使用LoRA权重: $TOME_DEPTH_LORA_PATH"
fi

# 启动Gradio应用
echo ""
echo "================================================"
echo "启动多视图深度条件 Gradio 界面 (Token Merging优化)"
echo "================================================"
echo "模型: $MODEL_PATH / $SUBFOLDER"
echo "端口: $PORT"
echo "地址: http://$HOST:$PORT"
echo "LoRA权重: $TOME_DEPTH_LORA_PATH"
echo "Token Merging: 已启用 (减少token数量，提高推理效率)"
echo "================================================"
echo ""

# 设置环境变量以启用Token Merging模式
export ENABLE_TOKEN_MERGING=true
export TOKEN_MERGE_RATIO=0.75
export TOKEN_MERGE_STRATEGY="attention"

python gradio_app.py \
    --model_path "$MODEL_PATH" \
    --subfolder "$SUBFOLDER" \
    --enable_multiview_depth \
    $LORA_ARG \
    --port $PORT \
    --host "$HOST" \
    --device "$DEVICE" \
    --low_vram_mode  # 启用低显存模式以优化Token Merging性能
