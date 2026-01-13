#!/bin/bash
# 启动多视图RGB重建 API 服务器

# 设置默认参数
MODEL_PATH="tencent/Hunyuan3D-2.1"
SUBFOLDER="hunyuan3d-dit-v2-1"
PORT=6008
HOST="0.0.0.0"
DEVICE="cuda"
CONCURRENCY=2  # 多视图模式建议降低并发数

# 批处理参数
BATCH_SIZE=2       # 批处理大小上限（同时处理的任务数），根据GPU内存调整
BATCH_TIMEOUT=1.0  # 批处理超时时间（秒），等待多久后即使未满批次也开始处理

# Worker 数量
NUM_WORKERS=2      # Worker 实例数量（默认2个，提高并发能力）

# 多视图RGB LoRA权重路径（根据实际情况修改）
# 优先级：Lightning checkpoint (.ckpt) > PEFT格式 (step_*)
RGB_LORA_PATH=""

# 1. 首先检查是否已指定具体的checkpoint路径
if [ -n "$RGB_LORA_PATH" ] && [ -f "$RGB_LORA_PATH" ]; then
    echo "✅ 使用指定的checkpoint: $RGB_LORA_PATH"
else
    # 2. 如果没有指定或文件不存在，查找Lightning checkpoint (.ckpt文件)
    LIGHTNING_DIRS=(
        "./hy3dshape/output_folder/dit/multiview_rgb_lora_finetuning/ckpt"
        "./output_folder/dit/multiview_rgb_lora_finetuning/ckpt"
        "hy3dshape/output_folder/dit/multiview_rgb_lora_finetuning/ckpt"
        "output_folder/dit/multiview_rgb_lora_finetuning/ckpt"
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
        "hy3dshape/output_folder/dit/multiview_rgb_lora_checkpoints"
        "output_folder/dit/multiview_rgb_lora_checkpoints"
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

# 启动API服务器
echo ""
echo "================================================"
echo "启动多视图RGB重建 API 服务器"
echo "================================================"
echo "模型: $MODEL_PATH / $SUBFOLDER"
echo "端口: $PORT"
echo "地址: http://$HOST:$PORT"
echo "并发数: $CONCURRENCY"
echo "Worker数量: $NUM_WORKERS (并行处理模型实例)"
echo "批处理大小: $BATCH_SIZE (同时处理的任务数)"
echo "批处理超时: ${BATCH_TIMEOUT}s (等待聚合时间)"
echo "模式: 4视图RGB重建（不使用法线图或深度图）"
echo "================================================"
echo ""
echo "API 文档: http://$HOST:$PORT/docs"
echo "API 端点:"
echo "  - POST /generate - 同步生成"
echo "  - POST /send - 异步生成"
echo "  - GET /status/{uid} - 查询状态"
echo "  - GET /health - 健康检查"
echo "  - GET /workers/status - Worker状态监控"
echo ""
echo "测试脚本: python test_multiview_api.py"
echo "================================================"
echo ""

python api_server.py \
    --model_path "$MODEL_PATH" \
    --subfolder "$SUBFOLDER" \
    --enable_multiview_rgb \
    $LORA_ARG \
    --num_views 4 \
    --port $PORT \
    --host "$HOST" \
    --device "$DEVICE" \
    --limit-model-concurrency $CONCURRENCY \
    --batch-size $BATCH_SIZE \
    --batch-timeout $BATCH_TIMEOUT \
    --num-workers $NUM_WORKERS
