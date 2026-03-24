#!/bin/bash
# ============================================================
# S3Gaussian 批量训练脚本 — COLMAP → Roadside 格式转换 + 训练
#
# 用法:
#   bash scripts/roadside/run_batch_colmap.sh <GPU_ID> <DATA_ROOT> [OUTPUT_ROOT]
#
# 示例:
#   bash scripts/roadside/run_batch_colmap.sh 3 /mnt/zyc_wzh/SparseGS/data/car_road
#   bash scripts/roadside/run_batch_colmap.sh 3 /mnt/zyc_wzh/SparseGS/data/car_road ./work_dirs/car_road
#
# 流程:
#   1. 批量将 COLMAP 数据转换为 roadside 格式 (frame_info.json 等)
#   2. 使用 roadside.py 配置逐个训练
# ============================================================
set -e

# Ensure we run from the project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

GPU_ID=${1:-0}
DATA_ROOT=${2:?"Usage: $0 <GPU_ID> <DATA_ROOT> [OUTPUT_ROOT]"}
OUTPUT_ROOT=${3:-""}

# Auto-generate output root if not specified
if [ -z "$OUTPUT_ROOT" ]; then
    DATE=$(date '+%m%d')
    OUTPUT_ROOT="./work_dirs/${DATE}/roadside_colmap"
fi

CONFIG="arguments/roadside.py"

echo "=========================================="
echo " S3Gaussian Batch Training (COLMAP → Roadside)"
echo " GPU: ${GPU_ID}"
echo " Data root: ${DATA_ROOT}"
echo " Output root: ${OUTPUT_ROOT}"
echo " Config: ${CONFIG}"
echo "=========================================="

# Step 1: Batch convert COLMAP to roadside format
echo ""
echo "Step 1: Converting COLMAP data to roadside format..."
python scripts/roadside/convert_colmap_to_roadside.py --data_root "$DATA_ROOT"

# Step 2: Collect all converted scenes (have frame_info.json)
scenes=()
for scene_dir in "${DATA_ROOT}"/*/; do
    if [ -f "${scene_dir}frame_info.json" ]; then
        scenes+=("$scene_dir")
    fi
done

total=${#scenes[@]}
echo ""
echo "Found ${total} scenes ready for training"
echo ""

if [ "$total" -eq 0 ]; then
    echo "ERROR: No converted scenes found under ${DATA_ROOT}"
    exit 1
fi

# Step 3: Train each scene
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

    # Skip if already trained
    if [ -f "${model_path}/chkpnt_fine_30000.pth" ]; then
        echo "SKIP: ${scene_name} already has chkpnt_fine_30000.pth"
        continue
    fi

    mkdir -p "$model_path"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
        -s "$scene_dir" \
        --model_path "$model_path" \
        --expname "roadside" \
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
