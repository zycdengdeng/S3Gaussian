#!/bin/bash
# ============================================================
# S3Gaussian 批量训练脚本 — COLMAP → Roadside 格式转换 + 并行训练
#
# 不修改源数据，转换结果输出到独立目录（图片用 symlink）
# 每张GPU独占一个训练任务，多GPU并行
#
# 用法:
#   bash scripts/roadside/run_batch_colmap.sh <DATA_ROOT> [WORK_DIR] [GPUS]
#
# 参数:
#   DATA_ROOT  - 源数据目录 (必须)
#   WORK_DIR   - 输出目录 (默认: /mnt/zyc_wzh/S3Gaussian/work_dirs/roadside_colmap)
#   GPUS       - 使用的GPU列表，逗号分隔 (默认: 0,1,2)
#
# 示例:
#   bash scripts/roadside/run_batch_colmap.sh /mnt/zyc_wzh/SparseGS/data/car_road
#   bash scripts/roadside/run_batch_colmap.sh /mnt/zyc_wzh/SparseGS/data/car_road "" "0,1,2"
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT=${1:?"Usage: $0 <DATA_ROOT> [WORK_DIR] [GPUS]"}
WORK_DIR=${2:-"/mnt/zyc_wzh/S3Gaussian/work_dirs/roadside_colmap"}
GPU_LIST=${3:-"0,1,2"}

# Parse GPU list
IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}

# Converted data goes here (symlinks to source images, no copy)
CONVERTED_ROOT="${WORK_DIR}/data"
MODEL_ROOT="${WORK_DIR}/models"
LOG_DIR="${WORK_DIR}/logs"
CONFIG="arguments/roadside.py"

mkdir -p "$LOG_DIR"

echo "=========================================="
echo " S3Gaussian Batch Training (COLMAP → Roadside)"
echo " GPUs: ${GPU_LIST} (${NUM_GPUS} parallel)"
echo " Source data: ${DATA_ROOT}"
echo " Converted data: ${CONVERTED_ROOT}"
echo " Model output: ${MODEL_ROOT}"
echo " Logs: ${LOG_DIR}"
echo " Config: ${CONFIG}"
echo "=========================================="

# Step 1: Convert COLMAP to roadside format (to separate dir, source untouched)
echo ""
echo "Step 1: Converting COLMAP data to roadside format..."
python scripts/roadside/convert_colmap_to_roadside.py \
    --data_root "$DATA_ROOT" \
    --output_root "$CONVERTED_ROOT"

# Step 2: Collect scenes that need training
scenes=()
for scene_dir in "${CONVERTED_ROOT}"/*/; do
    if [ -f "${scene_dir}frame_info.json" ]; then
        scene_name=$(basename "$scene_dir")
        model_path="${MODEL_ROOT}/${scene_name}"
        # Skip already-trained scenes
        if [ -f "${model_path}/chkpnt_fine_30000.pth" ]; then
            echo "  SKIP (trained): ${scene_name}"
            continue
        fi
        scenes+=("$scene_dir")
    fi
done

total=${#scenes[@]}
echo ""
echo "Found ${total} scenes to train (${NUM_GPUS} GPUs parallel)"

if [ "$total" -eq 0 ]; then
    echo "All scenes already trained or no scenes found."
    exit 0
fi

# Step 3: Split scenes across GPUs, each GPU runs its share sequentially
# GPU 0 gets scenes 0, 3, 6, ...
# GPU 1 gets scenes 1, 4, 7, ...
# GPU 2 gets scenes 2, 5, 8, ...

run_gpu_batch() {
    local gpu_id="$1"
    shift
    local gpu_scenes=("$@")
    local gpu_total=${#gpu_scenes[@]}
    local gpu_failed=0
    local gpu_done=0

    for scene_dir in "${gpu_scenes[@]}"; do
        scene_name=$(basename "$scene_dir")
        model_path="${MODEL_ROOT}/${scene_name}"
        log_file="${LOG_DIR}/${scene_name}.log"
        gpu_done=$((gpu_done + 1))

        # Double-check skip (in case another GPU finished it)
        if [ -f "${model_path}/chkpnt_fine_30000.pth" ]; then
            echo "[GPU ${gpu_id}] SKIP (${gpu_done}/${gpu_total}): ${scene_name}"
            continue
        fi

        mkdir -p "$model_path"
        echo "[GPU ${gpu_id}] START (${gpu_done}/${gpu_total}): ${scene_name}"

        if CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
            -s "$scene_dir" \
            --model_path "$model_path" \
            --expname "roadside" \
            --configs "$CONFIG" \
            > "$log_file" 2>&1; then
            echo "[GPU ${gpu_id}] DONE (${gpu_done}/${gpu_total}): ${scene_name}"
        else
            echo "[GPU ${gpu_id}] FAILED (${gpu_done}/${gpu_total}): ${scene_name} (see ${log_file})"
            gpu_failed=$((gpu_failed + 1))
        fi
    done

    return $gpu_failed
}

# Distribute scenes to GPUs round-robin
declare -a gpu_scene_lists
for i in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_scene_lists[$i]=""
done

for i in "${!scenes[@]}"; do
    gpu_idx=$((i % NUM_GPUS))
    gpu_scene_lists[$gpu_idx]+="${scenes[$i]}"$'\n'
done

# Launch one background process per GPU
echo ""
echo "Launching ${NUM_GPUS} GPU workers..."
echo ""

pids=()
for gpu_idx in $(seq 0 $((NUM_GPUS - 1))); do
    gpu_id=${GPUS[$gpu_idx]}
    # Parse newline-separated scene list into array
    IFS=$'\n' read -ra gpu_scenes <<< "${gpu_scene_lists[$gpu_idx]}"
    # Remove empty entries
    clean_scenes=()
    for s in "${gpu_scenes[@]}"; do
        [ -n "$s" ] && clean_scenes+=("$s")
    done

    if [ ${#clean_scenes[@]} -eq 0 ]; then
        continue
    fi

    echo "[GPU ${gpu_id}] Assigned ${#clean_scenes[@]} scenes"
    run_gpu_batch "$gpu_id" "${clean_scenes[@]}" &
    pids+=($!)
done

# Wait for all GPU workers
failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=$((failed + $?))
done

echo ""
echo "=========================================="
echo " Batch training complete!"
echo " Total: ${total}, Failed: ${failed}"
echo " Results: ${MODEL_ROOT}/"
echo " Logs: ${LOG_DIR}/"
echo "=========================================="

# List any failures
if [ $failed -gt 0 ]; then
    echo ""
    echo "Failed scenes (check logs):"
    for log in "${LOG_DIR}"/*.log; do
        scene_name=$(basename "$log" .log)
        model_path="${MODEL_ROOT}/${scene_name}"
        if [ ! -f "${model_path}/chkpnt_fine_30000.pth" ]; then
            echo "  - ${scene_name}: ${log}"
        fi
    done
fi
