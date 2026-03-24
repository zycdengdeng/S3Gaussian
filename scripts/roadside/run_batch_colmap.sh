#!/bin/bash
# ============================================================
# S3Gaussian 批量训练脚本 — COLMAP → Roadside 格式转换 + 并行训练
#
# 不修改源数据，转换结果输出到独立目录（图片用 symlink）
# 支持多GPU并行，每GPU可跑多个任务
#
# 用法:
#   bash scripts/roadside/run_batch_colmap.sh <DATA_ROOT> [WORK_DIR] [GPUS] [PER_GPU]
#
# 参数:
#   DATA_ROOT  - 源数据目录 (必须)
#   WORK_DIR   - 输出目录 (默认: /mnt/zyc_wzh/S3Gaussian/work_dirs/roadside_colmap)
#   GPUS       - 使用的GPU列表，逗号分隔 (默认: 0,1,2,3,4,5)
#   PER_GPU    - 每GPU并行数 (默认: 3)
#
# 示例:
#   bash scripts/roadside/run_batch_colmap.sh /mnt/zyc_wzh/SparseGS/data/car_road
#   bash scripts/roadside/run_batch_colmap.sh /mnt/zyc_wzh/SparseGS/data/car_road "" "0,1,2,3,4,5" 3
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

DATA_ROOT=${1:?"Usage: $0 <DATA_ROOT> [WORK_DIR] [GPUS] [PER_GPU]"}
WORK_DIR=${2:-"/mnt/zyc_wzh/S3Gaussian/work_dirs/roadside_colmap"}
GPU_LIST=${3:-"0,1,2,3,4,5"}
PER_GPU=${4:-3}

# Parse GPU list
IFS=',' read -ra GPUS <<< "$GPU_LIST"
NUM_GPUS=${#GPUS[@]}
TOTAL_SLOTS=$((NUM_GPUS * PER_GPU))

# Converted data goes here (symlinks to source images, no copy)
CONVERTED_ROOT="${WORK_DIR}/data"
MODEL_ROOT="${WORK_DIR}/models"
LOG_DIR="${WORK_DIR}/logs"
CONFIG="arguments/roadside.py"

mkdir -p "$LOG_DIR"

echo "=========================================="
echo " S3Gaussian Batch Training (COLMAP → Roadside)"
echo " GPUs: ${GPU_LIST} (${NUM_GPUS} GPUs × ${PER_GPU}/GPU = ${TOTAL_SLOTS} parallel)"
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
echo "Found ${total} scenes to train (${TOTAL_SLOTS} parallel slots)"

if [ "$total" -eq 0 ]; then
    echo "All scenes already trained or no scenes found."
    exit 0
fi

# Step 3: Launch parallel training
# Use a job-slot approach: maintain up to TOTAL_SLOTS concurrent jobs
# Each job gets assigned to a GPU in round-robin fashion

train_one() {
    local scene_dir="$1"
    local gpu_id="$2"
    local scene_name
    scene_name=$(basename "$scene_dir")
    local model_path="${MODEL_ROOT}/${scene_name}"
    local log_file="${LOG_DIR}/${scene_name}.log"

    mkdir -p "$model_path"

    echo "[GPU ${gpu_id}] START: ${scene_name}"

    CUDA_VISIBLE_DEVICES=${gpu_id} python train.py \
        -s "$scene_dir" \
        --model_path "$model_path" \
        --expname "roadside" \
        --configs "$CONFIG" \
        > "$log_file" 2>&1

    local status=$?
    if [ $status -eq 0 ]; then
        echo "[GPU ${gpu_id}] DONE: ${scene_name}"
    else
        echo "[GPU ${gpu_id}] FAILED: ${scene_name} (see ${log_file})"
    fi
    return $status
}

# Track PIDs and their scene names for final summary
declare -A pid_scene
declare -A pid_gpu
active_pids=()
slot_idx=0
failed=0
done_count=0

# Function to wait for a free slot
wait_for_slot() {
    while [ ${#active_pids[@]} -ge $TOTAL_SLOTS ]; do
        # Wait for any one child to finish
        local new_active=()
        local found_done=false
        for pid in "${active_pids[@]}"; do
            if ! kill -0 "$pid" 2>/dev/null; then
                # Process finished
                wait "$pid" || {
                    echo "  FAILED: ${pid_scene[$pid]}"
                    failed=$((failed + 1))
                }
                done_count=$((done_count + 1))
                unset pid_scene[$pid]
                unset pid_gpu[$pid]
                found_done=true
            else
                new_active+=("$pid")
            fi
        done
        active_pids=("${new_active[@]}")
        if ! $found_done; then
            sleep 2
        fi
    done
}

echo ""
echo "Launching ${total} training jobs across ${TOTAL_SLOTS} slots..."
echo ""

for scene_dir in "${scenes[@]}"; do
    wait_for_slot

    # Assign GPU round-robin
    gpu_idx=$((slot_idx % NUM_GPUS))
    gpu_id=${GPUS[$gpu_idx]}
    slot_idx=$((slot_idx + 1))

    train_one "$scene_dir" "$gpu_id" &
    pid=$!
    active_pids+=("$pid")
    pid_scene[$pid]=$(basename "$scene_dir")
    pid_gpu[$pid]=$gpu_id
done

# Wait for all remaining jobs
echo ""
echo "Waiting for remaining ${#active_pids[@]} jobs to finish..."
for pid in "${active_pids[@]}"; do
    wait "$pid" || {
        echo "  FAILED: ${pid_scene[$pid]}"
        failed=$((failed + 1))
    }
    done_count=$((done_count + 1))
done

echo ""
echo "=========================================="
echo " Batch training complete!"
echo " Total: ${total}, Succeeded: $((total - failed)), Failed: ${failed}"
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
