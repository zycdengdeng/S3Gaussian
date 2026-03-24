#!/bin/bash
# ============================================================
# S3Gaussian 批量训练脚本 — COLMAP → Roadside 格式转换 + 并行训练
#
# 不修改源数据，转换结果输出到独立目录（图片用 symlink）
# 在同一张GPU上同时跑N个训练任务
#
# 用法:
#   bash scripts/roadside/run_batch_colmap.sh <GPU_ID> <DATA_ROOT> [WORK_DIR] [PARALLEL]
#
# 参数:
#   GPU_ID     - 使用的GPU编号 (必须)
#   DATA_ROOT  - 源数据目录 (必须)
#   WORK_DIR   - 输出目录 (默认: /mnt/zyc_wzh/S3Gaussian/work_dirs/roadside_colmap)
#   PARALLEL   - 同时跑几个任务 (默认: 3)
#
# 示例:
#   bash scripts/roadside/run_batch_colmap.sh 3 /mnt/zyc_wzh/SparseGS/data/car_road
#   bash scripts/roadside/run_batch_colmap.sh 3 /mnt/zyc_wzh/SparseGS/data/car_road "" 3
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

GPU_ID=${1:?"Usage: $0 <GPU_ID> <DATA_ROOT> [WORK_DIR] [PARALLEL]"}
DATA_ROOT=${2:?"Usage: $0 <GPU_ID> <DATA_ROOT> [WORK_DIR] [PARALLEL]"}
WORK_DIR=${3:-"/mnt/zyc_wzh/S3Gaussian/work_dirs/roadside_colmap"}
PARALLEL=${4:-3}

CONVERTED_ROOT="${WORK_DIR}/data"
MODEL_ROOT="${WORK_DIR}/models"
LOG_DIR="${WORK_DIR}/logs"
CONFIG="arguments/roadside.py"

mkdir -p "$LOG_DIR"

echo "=========================================="
echo " S3Gaussian Batch Training (COLMAP → Roadside)"
echo " GPU: ${GPU_ID}, Parallel: ${PARALLEL}"
echo " Source data: ${DATA_ROOT}"
echo " Converted data: ${CONVERTED_ROOT}"
echo " Model output: ${MODEL_ROOT}"
echo " Logs: ${LOG_DIR}"
echo " Config: ${CONFIG}"
echo "=========================================="

# Step 1: Convert COLMAP to roadside format
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
        if [ -f "${model_path}/chkpnt_fine_30000.pth" ]; then
            echo "  SKIP (trained): ${scene_name}"
            continue
        fi
        scenes+=("$scene_dir")
    fi
done

total=${#scenes[@]}
echo ""
echo "Found ${total} scenes to train (${PARALLEL} parallel on GPU ${GPU_ID})"

if [ "$total" -eq 0 ]; then
    echo "All scenes already trained or no scenes found."
    exit 0
fi

# Step 3: Split scenes into PARALLEL workers, each runs its share sequentially
run_worker() {
    local worker_id="$1"
    shift
    local worker_scenes=("$@")
    local wtotal=${#worker_scenes[@]}
    local wdone=0
    local wfailed=0

    for scene_dir in "${worker_scenes[@]}"; do
        scene_name=$(basename "$scene_dir")
        model_path="${MODEL_ROOT}/${scene_name}"
        log_file="${LOG_DIR}/${scene_name}.log"
        wdone=$((wdone + 1))

        if [ -f "${model_path}/chkpnt_fine_30000.pth" ]; then
            echo "[W${worker_id}] SKIP (${wdone}/${wtotal}): ${scene_name}"
            continue
        fi

        mkdir -p "$model_path"
        echo "[W${worker_id}] START (${wdone}/${wtotal}): ${scene_name}"

        if CUDA_VISIBLE_DEVICES=${GPU_ID} python train.py \
            -s "$scene_dir" \
            --model_path "$model_path" \
            --expname "roadside" \
            --configs "$CONFIG" \
            > "$log_file" 2>&1; then
            echo "[W${worker_id}] DONE (${wdone}/${wtotal}): ${scene_name}"
        else
            echo "[W${worker_id}] FAILED (${wdone}/${wtotal}): ${scene_name} (see ${log_file})"
            wfailed=$((wfailed + 1))
        fi
    done

    return $wfailed
}

# Distribute scenes round-robin across workers
declare -a worker_lists
for i in $(seq 0 $((PARALLEL - 1))); do
    worker_lists[$i]=""
done

for i in "${!scenes[@]}"; do
    w=$((i % PARALLEL))
    worker_lists[$w]+="${scenes[$i]}"$'\n'
done

# Launch workers
echo ""
echo "Launching ${PARALLEL} workers on GPU ${GPU_ID}..."
echo ""

pids=()
for w in $(seq 0 $((PARALLEL - 1))); do
    IFS=$'\n' read -ra wscenes <<< "${worker_lists[$w]}"
    clean=()
    for s in "${wscenes[@]}"; do
        [ -n "$s" ] && clean+=("$s")
    done
    [ ${#clean[@]} -eq 0 ] && continue

    echo "[W${w}] Assigned ${#clean[@]} scenes"
    run_worker "$w" "${clean[@]}" &
    pids+=($!)
done

# Wait for all workers
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
