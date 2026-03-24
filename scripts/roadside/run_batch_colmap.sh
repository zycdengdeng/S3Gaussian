#!/bin/bash
# ============================================================
# S3Gaussian 批量训练脚本 — COLMAP 静态场景
#
# 用法:
#   bash scripts/roadside/run_batch_colmap.sh <GPU_ID> <DATA_ROOT> [OUTPUT_ROOT]
#
# 示例:
#   # 训练 /mnt/zyc_wzh/SparseGS/data/car_road/ 下所有场景
#   bash scripts/roadside/run_batch_colmap.sh 0 /mnt/zyc_wzh/SparseGS/data/car_road
#
#   # 指定输出目录
#   bash scripts/roadside/run_batch_colmap.sh 0 /mnt/zyc_wzh/SparseGS/data/car_road ./work_dirs/car_road
#
# 数据要求:
#   每个子目录下需包含:
#     images/          — 原始图片
#     sparse/0/        — COLMAP 稀疏重建 (cameras.bin, images.bin, points3D.bin)
# ============================================================
set -e

GPU_ID=${1:-0}
DATA_ROOT=${2:?"Usage: $0 <GPU_ID> <DATA_ROOT> [OUTPUT_ROOT]"}
OUTPUT_ROOT=${3:-""}

# Auto-generate output root if not specified
if [ -z "$OUTPUT_ROOT" ]; then
    DATE=$(date '+%m%d')
    OUTPUT_ROOT="./work_dirs/${DATE}/colmap_static"
fi

CONFIG="arguments/colmap_static.py"

echo "=========================================="
echo " S3Gaussian Batch COLMAP Training"
echo " GPU: ${GPU_ID}"
echo " Data root: ${DATA_ROOT}"
echo " Output root: ${OUTPUT_ROOT}"
echo " Config: ${CONFIG}"
echo "=========================================="

# Collect all scene directories that have sparse/0/
scenes=()
for scene_dir in "${DATA_ROOT}"/*/; do
    if [ -d "${scene_dir}sparse/0" ]; then
        scenes+=("$scene_dir")
    fi
done

total=${#scenes[@]}
echo "Found ${total} scenes with COLMAP data"
echo ""

if [ "$total" -eq 0 ]; then
    echo "ERROR: No scenes found with sparse/0/ under ${DATA_ROOT}"
    exit 1
fi

# Train each scene
count=0
failed=0
for scene_dir in "${scenes[@]}"; do
    scene_name=$(basename "$scene_dir")
    model_path="${OUTPUT_ROOT}/${scene_name}"
    count=$((count + 1))

    echo "=========================================="
    echo " [${count}/${total}] Training: ${scene_name}"
    echo " Data: ${scene_dir}"
    echo " Output: ${model_path}"
    echo "=========================================="

    # Skip if already trained (checkpoint exists)
    if [ -f "${model_path}/chkpnt_fine_30000.pth" ]; then
        echo "SKIP: ${scene_name} already has chkpnt_fine_30000.pth"
        continue
    fi

    mkdir -p "$model_path"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
        -s "$scene_dir" \
        --model_path "$model_path" \
        --expname "colmap_static" \
        --configs "$CONFIG" \
    || {
        echo "FAILED: ${scene_name}"
        failed=$((failed + 1))
        continue
    }

    echo "DONE: ${scene_name}"
    echo ""
done

echo "=========================================="
echo " Batch training complete!"
echo " Total: ${total}, Failed: ${failed}"
echo " Results: ${OUTPUT_ROOT}"
echo "=========================================="
