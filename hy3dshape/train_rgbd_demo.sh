#!/bin/bash

# RGBD微调训练启动脚本
# 基于Hunyuan3D-2.1架构的RGB+深度图微调训练

set -e

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Hunyuan3D RGBD微调训练启动脚本${NC}"
echo -e "${BLUE}========================================${NC}"

# 检查Python环境
echo -e "${YELLOW}检查Python环境...${NC}"
if ! command -v python &> /dev/null; then
    echo -e "${RED}错误: 未找到Python解释器${NC}"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}Python版本: $PYTHON_VERSION${NC}"

# 检查CUDA
echo -e "${YELLOW}检查CUDA环境...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}CUDA可用:${NC}"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
else
    echo -e "${YELLOW}警告: 未检测到CUDA，将使用CPU训练${NC}"
fi

# 设置环境变量
echo -e "${YELLOW}设置环境变量...${NC}"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3,4,5,6,7"}

# 默认配置
DEFAULT_CONFIG="configs/hunyuandit-rgbd-depth-guided-fusion.yaml"
CONFIG_FILE=${1:-$DEFAULT_CONFIG}
RESUME_PATH=${2:-""}
BATCH_SIZE=${3:-""}
LEARNING_RATE=${4:-""}
TRAINING_STEPS=${5:-""}

# 检查配置文件
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}错误: 配置文件不存在: $CONFIG_FILE${NC}"
    echo -e "${YELLOW}请确保配置文件路径正确，或使用默认配置:${NC}"
    echo -e "${BLUE}  $DEFAULT_CONFIG${NC}"
    exit 1
fi

echo -e "${GREEN}使用配置文件: $CONFIG_FILE${NC}"

# 检查必要的目录
echo -e "${YELLOW}检查训练目录...${NC}"
mkdir -p checkpoints
mkdir -p logs

# 检查数据目录
if [ ! -d "tools/mini_depth_trainset/preprocessed" ]; then
    echo -e "${YELLOW}警告: 训练数据目录不存在: tools/mini_depth_trainset/preprocessed${NC}"
    echo -e "${YELLOW}请确保已准备好RGBD训练数据${NC}"
fi

# 构建训练命令
TRAIN_CMD="python train_rgbd_finetuning.py --config $CONFIG_FILE"

if [ -n "$RESUME_PATH" ]; then
    if [ -f "$RESUME_PATH" ]; then
        TRAIN_CMD="$TRAIN_CMD --resume $RESUME_PATH"
        echo -e "${GREEN}将从检查点恢复训练: $RESUME_PATH${NC}"
    else
        echo -e "${RED}错误: 检查点文件不存在: $RESUME_PATH${NC}"
        exit 1
    fi
fi

if [ -n "$BATCH_SIZE" ]; then
    TRAIN_CMD="$TRAIN_CMD --batch_size $BATCH_SIZE"
    echo -e "${GREEN}批次大小: $BATCH_SIZE${NC}"
fi

if [ -n "$LEARNING_RATE" ]; then
    TRAIN_CMD="$TRAIN_CMD --lr $LEARNING_RATE"
    echo -e "${GREEN}学习率: $LEARNING_RATE${NC}"
fi

if [ -n "$TRAINING_STEPS" ]; then
    TRAIN_CMD="$TRAIN_CMD --steps $TRAINING_STEPS"
    echo -e "${GREEN}训练步数: $TRAINING_STEPS${NC}"
fi

# 显示训练信息
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}训练配置信息:${NC}"
echo -e "${GREEN}  配置文件: $CONFIG_FILE${NC}"
echo -e "${GREEN}  CUDA设备: $CUDA_VISIBLE_DEVICES${NC}"
echo -e "${GREEN}  Python路径: $PYTHONPATH${NC}"
if [ -n "$RESUME_PATH" ]; then
    echo -e "${GREEN}  恢复路径: $RESUME_PATH${NC}"
fi
echo -e "${BLUE}========================================${NC}"

# 显示使用说明
echo -e "${YELLOW}使用说明:${NC}"
echo -e "${YELLOW}  基本用法: ./train_rgbd_demo.sh${NC}"
echo -e "${YELLOW}  指定配置: ./train_rgbd_demo.sh configs/your_config.yaml${NC}"
echo -e "${YELLOW}  恢复训练: ./train_rgbd_demo.sh configs/your_config.yaml checkpoints/last.ckpt${NC}"
echo -e "${YELLOW}  自定义参数: ./train_rgbd_demo.sh config.yaml resume.ckpt batch_size lr steps${NC}"
echo -e "${BLUE}========================================${NC}"

# 确认开始训练
echo -e "${YELLOW}即将开始RGBD微调训练...${NC}"
echo -e "${YELLOW}按Enter继续，或Ctrl+C取消${NC}"
read -r

echo -e "${GREEN}开始训练...${NC}"
echo -e "${BLUE}执行命令: $TRAIN_CMD${NC}"
echo -e "${BLUE}========================================${NC}"

# 执行训练
eval "$TRAIN_CMD"

# 训练完成
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}RGBD微调训练完成!${NC}"
echo -e "${GREEN}检查点保存在: checkpoints/目录${NC}"
echo -e "${GREEN}日志保存在: logs/目录${NC}"
echo -e "${BLUE}========================================${NC}"

# 显示后续步骤
echo -e "${YELLOW}后续步骤:${NC}"
echo -e "${YELLOW}1. 检查训练日志和损失曲线${NC}"
echo -e "${YELLOW}2. 使用最佳检查点进行推理测试${NC}"
echo -e "${YELLOW}3. 评估模型在RGBD数据上的性能${NC}"
echo -e "${YELLOW}4. 根据需要调整超参数并重新训练${NC}"