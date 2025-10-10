#!/bin/bash
# 启动多视图深度条件 Gradio 界面

# 设置默认参数
MODEL_PATH="tencent/Hunyuan3D-2.1"
SUBFOLDER="hunyuan3d-dit-v2-1"
PORT=6008
HOST="0.0.0.0"
DEVICE="cuda"

# 多视图深度LoRA权重路径（根据实际情况修改）
# 优先级：最新的checkpoint > 默认路径
DEPTH_LORA_PATH=""

# 自动查找最新的checkpoint
CHECKPOINT_DIRS=(
    "./hy3dshape/output_folder/dit/multiview_depth_lora_checkpoints"
    "./output_folder/dit/multiview_depth_lora_checkpoints"
)

echo "正在查找最新的LoRA checkpoint..."
for dir in "${CHECKPOINT_DIRS[@]}"; do
    if [ -d "$dir" ]; then
        # 查找最新的step_*目录
        latest_ckpt=$(ls -d "$dir"/step_* 2>/dev/null | sort -V | tail -n 1)
        if [ -n "$latest_ckpt" ]; then
            DEPTH_LORA_PATH="$latest_ckpt"
            echo "找到checkpoint: $DEPTH_LORA_PATH"
            break
        fi
    fi
done

if [ -z "$DEPTH_LORA_PATH" ]; then
    echo "⚠️  警告: 未找到多视图深度LoRA权重，将使用基础模型"
    echo "请确保已完成训练并设置正确的checkpoint路径"
    LORA_ARG=""
else
    LORA_ARG="--depth_lora_path $DEPTH_LORA_PATH"
    echo "✅ 使用LoRA权重: $DEPTH_LORA_PATH"
fi

# 启动Gradio应用
echo ""
echo "================================================"
echo "启动多视图深度条件 Gradio 界面"
echo "================================================"
echo "模型: $MODEL_PATH / $SUBFOLDER"
echo "端口: $PORT"
echo "地址: http://$HOST:$PORT"
echo "================================================"
echo ""

python gradio_app.py \
    --model_path "$MODEL_PATH" \
    --subfolder "$SUBFOLDER" \
    --enable_multiview_depth \
    $LORA_ARG \
    --port $PORT \
    --host "$HOST" \
    --device "$DEVICE"

