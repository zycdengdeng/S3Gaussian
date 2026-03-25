#!/bin/bash
# ============================================================
# S3Gaussian 批量渲染脚本 — 渲染到车端视角 (1280x720)
#
# 投影链: World --(world2lidar)--> VirtualLiDAR --(virtualLidarToCam)--> Camera
#
# 用法:
#   bash scripts/roadside/run_batch_render.sh <GPU_ID> <CALIB_JSON> <TRANSFORM_ROOT> [WORK_DIR] [PARALLEL]
#
# 参数:
#   GPU_ID          - 使用的GPU编号 (必须)
#   CALIB_JSON      - calib.json路径 (必须)
#   TRANSFORM_ROOT  - transform_json根目录 (必须)
#   WORK_DIR        - 工作目录 (默认: /mnt/zyc_wzh/S3Gaussian/work_dirs/roadside_colmap)
#   PARALLEL        - 同时跑几个任务 (默认: 6)
#
# 示例:
#   bash scripts/roadside/run_batch_render.sh 3 \
#     /mnt/car_road_data_TianJin/support_info/calib.json \
#     /mnt/car_road_data_TianJin/support_info/transform_json
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

GPU_ID=${1:?"Usage: $0 <GPU_ID> <CALIB_JSON> <TRANSFORM_ROOT> [WORK_DIR] [PARALLEL]"}
CALIB_JSON=${2:?"Usage: $0 <GPU_ID> <CALIB_JSON> <TRANSFORM_ROOT> [WORK_DIR] [PARALLEL]"}
TRANSFORM_ROOT=${3:?"Usage: $0 <GPU_ID> <CALIB_JSON> <TRANSFORM_ROOT> [WORK_DIR] [PARALLEL]"}
WORK_DIR=${4:-"/mnt/zyc_wzh/S3Gaussian/work_dirs/roadside_colmap"}
PARALLEL=${5:-6}

MODEL_ROOT="${WORK_DIR}/models"
OUTPUT_ROOT="${WORK_DIR}/renders"
LOG_DIR="${WORK_DIR}/logs_render"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

echo "=========================================="
echo " S3Gaussian Batch Vehicle Rendering (1280x720)"
echo " GPU: ${GPU_ID}, Parallel: ${PARALLEL}"
echo " Calib: ${CALIB_JSON}"
echo " Transforms: ${TRANSFORM_ROOT}"
echo " Models: ${MODEL_ROOT}"
echo " Output: ${OUTPUT_ROOT}"
echo "=========================================="

# Collect scenes that have trained checkpoints
scenes=()
for model_dir in "${MODEL_ROOT}"/*/; do
    scene_name=$(basename "$model_dir")
    if [ -f "${model_dir}chkpnt_fine_30000.pth" ]; then
        # Skip already rendered scenes
        render_dir="${OUTPUT_ROOT}/${scene_name}"
        if [ -d "$render_dir" ] && [ "$(find "$render_dir" -name '*.png' 2>/dev/null | head -1)" ]; then
            echo "  SKIP (rendered): ${scene_name}"
            continue
        fi
        scenes+=("$scene_name")
    fi
done

total=${#scenes[@]}
echo ""
echo "Found ${total} scenes to render (${PARALLEL} parallel on GPU ${GPU_ID})"

if [ "$total" -eq 0 ]; then
    echo "All scenes already rendered or no trained models found."
    exit 0
fi

# Worker function
run_worker() {
    local worker_id="$1"
    shift
    local worker_scenes=("$@")
    local wtotal=${#worker_scenes[@]}
    local wdone=0
    local wfailed=0

    for scene_name in "${worker_scenes[@]}"; do
        log_file="${LOG_DIR}/${scene_name}.log"
        wdone=$((wdone + 1))

        echo "[W${worker_id}] START (${wdone}/${wtotal}): ${scene_name}"

        if CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/roadside/render_roadside_batch.py \
            --model_root "$MODEL_ROOT" \
            --calib_json "$CALIB_JSON" \
            --transform_root "$TRANSFORM_ROOT" \
            --output_root "$OUTPUT_ROOT" \
            --scene_name "$scene_name" \
            > "$log_file" 2>&1; then
            echo "[W${worker_id}] DONE (${wdone}/${wtotal}): ${scene_name}"
        else
            echo "[W${worker_id}] FAILED (${wdone}/${wtotal}): ${scene_name} (see ${log_file})"
            wfailed=$((wfailed + 1))
        fi
    done

    return $wfailed
}

# Round-robin distribute
declare -a worker_lists
for i in $(seq 0 $((PARALLEL - 1))); do
    worker_lists[$i]=""
done

for i in "${!scenes[@]}"; do
    w=$((i % PARALLEL))
    worker_lists[$w]+="${scenes[$i]}"$'\n'
done

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

failed=0
for pid in "${pids[@]}"; do
    wait "$pid" || failed=$((failed + $?))
done

echo ""
echo "=========================================="
echo " Batch rendering complete!"
echo " Total: ${total}, Failed: ${failed}"
echo " Output: ${OUTPUT_ROOT}/"
echo " Logs: ${LOG_DIR}/"
echo "=========================================="

if [ $failed -gt 0 ]; then
    echo ""
    echo "Failed scenes (check logs):"
    for scene_name in "${scenes[@]}"; do
        render_dir="${OUTPUT_ROOT}/${scene_name}"
        if [ ! -d "$render_dir" ] || [ ! "$(find "$render_dir" -name '*.png' 2>/dev/null | head -1)" ]; then
            echo "  - ${scene_name}: ${LOG_DIR}/${scene_name}.log"
        fi
    done
fi
