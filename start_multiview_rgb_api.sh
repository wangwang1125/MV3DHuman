#!/bin/bash
# 启动多视图RGB重建 API 服务器（参考 start_multiview_rgb_gradio_mv.sh）
# 接口：接收四视图 RGB、可选法线图、可选深度图（深度不参与推理）；返回含抠图四视图

# 设置默认参数（与 gradio_mv 一致：Hunyuan3D-2mv）
MODEL_PATH="tencent/Hunyuan3D-2mv"
SUBFOLDER="hunyuan3d-dit-v2-mv"
PORT=6008
HOST="0.0.0.0"
DEVICE="cuda"
CONCURRENCY=2  # 多视图模式建议降低并发数

# 批处理参数
BATCH_SIZE=2       # 批处理大小上限（同时处理的任务数），根据GPU内存调整
BATCH_TIMEOUT=1.0  # 批处理超时时间（秒），等待多久后即使未满批次也开始处理

# Worker 数量
NUM_WORKERS=2      # Worker 实例数量（默认2个，提高并发能力）

# 多视图RGB全量微调checkpoint路径（根据实际情况修改）
# 优先级：全量微调checkpoint > LoRA checkpoint，与 gradio_mv 一致优先 mv 版本
RGB_CHECKPOINT_PATH=""

# 1. 首先检查是否已指定具体的checkpoint路径
if [ -n "$RGB_CHECKPOINT_PATH" ] && [ -f "$RGB_CHECKPOINT_PATH" ]; then
    echo "✅ 使用指定的checkpoint: $RGB_CHECKPOINT_PATH"
else
    # 2. 优先查找全量微调的checkpoint (.ckpt文件) - mv版本
    echo "正在查找最新的checkpoint..."
    echo "1. 查找全量微调checkpoint (.ckpt) - mv版本..."
    FINETUNING_DIRS=(
        "./hy3dshape/output_folder/dit/multiview_rgb_finetuning_mv/ckpt"
        "./output_folder/dit/multiview_rgb_finetuning_mv/ckpt"
        "hy3dshape/output_folder/dit/multiview_rgb_finetuning_mv/ckpt"
        "output_folder/dit/multiview_rgb_finetuning_mv/ckpt"
    )

    for dir in "${FINETUNING_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            latest_ckpt=$(ls -t "$dir"/*.ckpt 2>/dev/null | head -n 1)
            if [ -n "$latest_ckpt" ]; then
                RGB_CHECKPOINT_PATH="$latest_ckpt"
                echo "✅ 找到全量微调checkpoint (mv): $RGB_CHECKPOINT_PATH"
                break
            fi
        fi
    done
fi

# 3. 若无全量微调，查找LoRA格式 - mv版本（向后兼容）
if [ -z "$RGB_CHECKPOINT_PATH" ]; then
    echo "2. 查找LoRA格式checkpoint - mv版本..."
    LORA_DIRS=(
        "./hy3dshape/output_folder/dit/multiview_rgb_lora_finetuning_mv/ckpt"
        "./output_folder/dit/multiview_rgb_lora_finetuning_mv/ckpt"
        "./hy3dshape/output_folder/dit/multiview_rgb_lora_checkpoints_mv"
        "./output_folder/dit/multiview_rgb_lora_checkpoints_mv"
    )
    
    for dir in "${LORA_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            latest_ckpt=$(ls -t "$dir"/*.ckpt 2>/dev/null | head -n 1)
            if [ -n "$latest_ckpt" ]; then
                RGB_CHECKPOINT_PATH="$latest_ckpt"
                echo "✅ 找到LoRA checkpoint (mv): $RGB_CHECKPOINT_PATH"
                break
            fi
            latest_ckpt=$(ls -d "$dir"/step_* 2>/dev/null | sort -V | tail -n 1)
            if [ -n "$latest_ckpt" ]; then
                RGB_CHECKPOINT_PATH="$latest_ckpt"
                echo "✅ 找到PEFT checkpoint (mv): $RGB_CHECKPOINT_PATH"
                break
            fi
        fi
    done
fi

# 4. 若仍未找到，尝试原始版本路径（向后兼容）
if [ -z "$RGB_CHECKPOINT_PATH" ]; then
    echo "3. 查找原始版本checkpoint（向后兼容）..."
    ORIGINAL_FINETUNING_DIRS=(
        "./hy3dshape/output_folder/dit/multiview_rgb_finetuning/ckpt"
        "./output_folder/dit/multiview_rgb_finetuning/ckpt"
    )
    for dir in "${ORIGINAL_FINETUNING_DIRS[@]}"; do
        if [ -d "$dir" ]; then
            latest_ckpt=$(ls -t "$dir"/*.ckpt 2>/dev/null | head -n 1)
            if [ -n "$latest_ckpt" ]; then
                RGB_CHECKPOINT_PATH="$latest_ckpt"
                echo "✅ 找到原始版本checkpoint: $RGB_CHECKPOINT_PATH"
                break
            fi
        fi
    done
    if [ -z "$RGB_CHECKPOINT_PATH" ]; then
        ORIGINAL_LORA_DIRS=(
            "./hy3dshape/output_folder/dit/multiview_rgb_lora_checkpoints"
            "./output_folder/dit/multiview_rgb_lora_checkpoints"
        )
        for dir in "${ORIGINAL_LORA_DIRS[@]}"; do
            if [ -d "$dir" ]; then
                latest_ckpt=$(ls -d "$dir"/step_* 2>/dev/null | sort -V | tail -n 1)
                if [ -n "$latest_ckpt" ]; then
                    RGB_CHECKPOINT_PATH="$latest_ckpt"
                    echo "✅ 找到原始版本LoRA checkpoint: $RGB_CHECKPOINT_PATH"
                    break
                fi
            fi
        done
    fi
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
echo "模式: 4视图 RGB + 可选法线（参与推理）+ 可选深度（仅接收不推理）；返回含抠图四视图"
echo "================================================"
echo ""
echo "API 文档: http://$HOST:$PORT/docs"
echo "API 端点:"
echo "  - POST /generate - 同步生成"
echo "  - POST /send - 异步生成（请求体可含 image_*/normal_*/depth_* base64）"
echo "  - GET /status/{uid} - 查询状态（完成时含 model_base64 与 image_*_matted_base64）"
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
    $CHECKPOINT_ARG \
    --num_views 4 \
    --port $PORT \
    --host "$HOST" \
    --device "$DEVICE" \
    --limit-model-concurrency $CONCURRENCY \
    --batch-size $BATCH_SIZE \
    --batch-timeout $BATCH_TIMEOUT \
    --num-workers $NUM_WORKERS
