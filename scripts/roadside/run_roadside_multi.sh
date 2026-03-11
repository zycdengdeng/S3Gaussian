#!/bin/bash
# ============================================================
# S3Gaussian 路侧多帧重建 训练脚本
#
# 用法:
#   bash scripts/roadside/run_roadside_multi.sh <GPU_ID> <DATA_DIR> [MODEL_DIR]
#
# 示例:
#   bash scripts/roadside/run_roadside_multi.sh 0 ./data/roadside/scene_053_multi
# ============================================================
set -e

GPU_ID=${1:-0}
DATA_DIR=${2:?"Usage: $0 <GPU_ID> <DATA_DIR> [MODEL_DIR]"}
MODEL_DIR=${3:-""}

if [ -z "$MODEL_DIR" ]; then
    DATE=$(date '+%m%d')
    SCENE_NAME=$(basename "$DATA_DIR")
    MODEL_DIR="./work_dirs/${DATE}/roadside_multi/${SCENE_NAME}"
fi

echo "=========================================="
echo " S3Gaussian Roadside Multi-Frame Training"
echo " GPU: ${GPU_ID}"
echo " Data: ${DATA_DIR}"
echo " Output: ${MODEL_DIR}"
echo "=========================================="

if [ ! -f "${DATA_DIR}/frame_info.json" ]; then
    echo "ERROR: ${DATA_DIR}/frame_info.json not found!"
    echo "Run prepare_roadside_multiframe.py first."
    exit 1
fi

mkdir -p "$MODEL_DIR"

CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
    -s "$DATA_DIR" \
    --model_path "$MODEL_DIR" \
    --expname "roadside_multi" \
    --configs "arguments/roadside_multi.py"

echo "=========================================="
echo " Training complete!"
echo " Results: ${MODEL_DIR}"
echo "=========================================="
