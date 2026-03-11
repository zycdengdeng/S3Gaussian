#!/bin/bash
# ============================================================
# S3Gaussian 路侧单帧重建 训练脚本
#
# 用法:
#   bash scripts/roadside/run_roadside.sh <GPU_ID> <DATA_DIR> [MODEL_DIR]
#
# 示例:
#   # 使用 GPU 0, 数据在 ./data/roadside/scene_001
#   bash scripts/roadside/run_roadside.sh 0 ./data/roadside/scene_001
#
#   # 指定输出目录
#   bash scripts/roadside/run_roadside.sh 0 ./data/roadside/scene_001 ./work_dirs/roadside/scene_001
# ============================================================
set -e

GPU_ID=${1:-0}
DATA_DIR=${2:?"Usage: $0 <GPU_ID> <DATA_DIR> [MODEL_DIR]"}
MODEL_DIR=${3:-""}

# Auto-generate model path if not specified
if [ -z "$MODEL_DIR" ]; then
    DATE=$(date '+%m%d')
    SCENE_NAME=$(basename "$DATA_DIR")
    MODEL_DIR="./work_dirs/${DATE}/roadside/${SCENE_NAME}"
fi

echo "=========================================="
echo " S3Gaussian Roadside Training"
echo " GPU: ${GPU_ID}"
echo " Data: ${DATA_DIR}"
echo " Output: ${MODEL_DIR}"
echo "=========================================="

# Verify data directory exists and has frame_info.json
if [ ! -f "${DATA_DIR}/frame_info.json" ]; then
    echo "ERROR: ${DATA_DIR}/frame_info.json not found!"
    echo "Please run prepare_roadside_data.py first to convert your data."
    exit 1
fi

# Create output directory
mkdir -p "$MODEL_DIR"

# Run training
CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    -s "$DATA_DIR" \
    --model_path "$MODEL_DIR" \
    --expname "roadside" \
    --configs "arguments/roadside.py"

echo "=========================================="
echo " Training complete!"
echo " Results: ${MODEL_DIR}"
echo "=========================================="
