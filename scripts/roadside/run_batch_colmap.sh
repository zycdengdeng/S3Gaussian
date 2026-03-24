#!/bin/bash
# ============================================================
# S3Gaussian 批量训练脚本 — COLMAP → Roadside 格式转换 + 训练
#
# 不修改源数据，转换结果输出到独立目录（图片用 symlink）
#
# 用法:
#   bash scripts/roadside/run_batch_colmap.sh <GPU_ID> <DATA_ROOT> [WORK_DIR]
#
# 示例:
#   bash scripts/roadside/run_batch_colmap.sh 3 /mnt/zyc_wzh/SparseGS/data/car_road
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

GPU_ID=${1:-0}
DATA_ROOT=${2:?"Usage: $0 <GPU_ID> <DATA_ROOT> [WORK_DIR]"}
WORK_DIR=${3:-""}

if [ -z "$WORK_DIR" ]; then
    DATE=$(date '+%m%d')
    WORK_DIR="./work_dirs/${DATE}/roadside_colmap"
fi

# Converted data goes here (symlinks to source images, no copy)
CONVERTED_ROOT="${WORK_DIR}/data"
CONFIG="arguments/roadside.py"

echo "=========================================="
echo " S3Gaussian Batch Training (COLMAP → Roadside)"
echo " GPU: ${GPU_ID}"
echo " Source data: ${DATA_ROOT}"
echo " Converted data: ${CONVERTED_ROOT}"
echo " Model output: ${WORK_DIR}"
echo " Config: ${CONFIG}"
echo "=========================================="

# Step 1: Convert COLMAP to roadside format (to separate dir, source untouched)
echo ""
echo "Step 1: Converting COLMAP data to roadside format..."
python scripts/roadside/convert_colmap_to_roadside.py \
    --data_root "$DATA_ROOT" \
    --output_root "$CONVERTED_ROOT"

# Step 2: Collect converted scenes
scenes=()
for scene_dir in "${CONVERTED_ROOT}"/*/; do
    if [ -f "${scene_dir}frame_info.json" ]; then
        scenes+=("$scene_dir")
    fi
done

total=${#scenes[@]}
echo ""
echo "Found ${total} scenes ready for training"

if [ "$total" -eq 0 ]; then
    echo "ERROR: No converted scenes found"
    exit 1
fi

# Step 3: Train each scene
count=0
failed=0
for scene_dir in "${scenes[@]}"; do
    scene_name=$(basename "$scene_dir")
    model_path="${WORK_DIR}/models/${scene_name}"
    count=$((count + 1))

    echo "=========================================="
    echo " [${count}/${total}] Training: ${scene_name}"
    echo " Data: ${scene_dir}"
    echo " Output: ${model_path}"
    echo "=========================================="

    if [ -f "${model_path}/chkpnt_fine_30000.pth" ]; then
        echo "SKIP: already trained"
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
done

echo "=========================================="
echo " Batch training complete!"
echo " Total: ${total}, Failed: ${failed}"
echo " Results: ${WORK_DIR}/models/"
echo "=========================================="
