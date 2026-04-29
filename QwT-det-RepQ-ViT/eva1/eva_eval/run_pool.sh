#!/usr/bin/env bash
# Pool-based runner for 7 EVA eval tasks:
#   1) fp${N}                     FP baseline (via quant_eval_shard --no-quant)
#   2) v01_w6a6_rp                int6 + reparam
#   3) v02_w7a7_rp                int7 + reparam
#   4) v03_w8a8_rp                int8 + reparam
#   5) v04_w6a6_rp_qwt            int6 + reparam + QwT
#   6) v05_w7a7_rp_qwt            int7 + reparam + QwT
#   7) v06_w8a8_rp_qwt            int8 + reparam + QwT
#
# Two scheduling knobs:
#   - GPUs:           positional args  (e.g. ./run_pool.sh 0 1 2 3 4 5 6 7)
#   - SHARDS_PER_TASK env             (default 1)
#
# How they combine:
#   #GPUs / SHARDS_PER_TASK  =  number of tasks running concurrently.
#   Each task uses SHARDS_PER_TASK GPUs (data-parallel sharded), the rest of
#   the tasks queue up. When a task's group of GPUs frees up, the next pending
#   task takes that group.
#
# Examples:
#   ./run_pool.sh 0 1 2 3 4 5 6 7              # 8 tasks parallel × 1 GPU
#   SHARDS_PER_TASK=2 ./run_pool.sh 0 1 2 3 4 5 6 7   # 4 tasks parallel × 2 GPUs
#   SHARDS_PER_TASK=8 ./run_pool.sh 0 1 2 3 4 5 6 7   # 1 task at a time × 8 GPUs
#   ./run_pool.sh 0                            # 1 task × 1 GPU (serial)
#
# Per-task log:    results/logs/pool_<tag>.log
# Per-shard log:   results/<tag>/logs/shard<i>.log   (multi-shard tasks)
# Status log:      results/logs/pool_status_<run_key>.log
# Resume:          tasks with results/<tag>/metrics.json are skipped.
#                  shard.py uses --resume to skip already-evaluated images.
set -uo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <gpu_id> [<gpu_id> ...]" >&2
  exit 1
fi
GPUS=("$@")

cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2
module load cuda/12.8.1 gcc/11.2.0 2>/dev/null || true

# Local data + ckpt paths (the upstream defaults point to a /scratch path that
# is not mounted on this host).
export DETECTRON2_DATASETS="${DETECTRON2_DATASETS:-/home/azrsadmin/vit_sc/data}"
export EVA_CKPT="${EVA_CKPT:-/home/azrsadmin/vit_sc/data/pretrained/eva_coco_det.pth}"
if [ ! -f "$EVA_CKPT" ]; then
  echo "[pool] EVA_CKPT not found: $EVA_CKPT" >&2
  exit 1
fi
if [ ! -d "$DETECTRON2_DATASETS/coco" ]; then
  echo "[pool] DETECTRON2_DATASETS/coco missing: $DETECTRON2_DATASETS/coco" >&2
  exit 1
fi

mkdir -p results/logs

# Number of COCO val images to score per task (full val = 5000).
N_EVAL="${N_EVAL:-5000}"
export N_EVAL

# Image size override (square_pad + ResizeShortestEdge). 1280 = cfg default.
EVA_SIZE="${EVA_SIZE:-1280}"
export EVA_SIZE
if [ $((EVA_SIZE % 256)) -ne 0 ]; then
  echo "[pool] EVA_SIZE=$EVA_SIZE must be a multiple of 256" >&2
  exit 1
fi
SZ_SUFFIX=""
if [ "$EVA_SIZE" != "1280" ]; then
  SZ_SUFFIX="_sz${EVA_SIZE}"
fi

# Shard-per-task knob: how many GPUs each task uses (data-parallel shards).
SHARDS_PER_TASK="${SHARDS_PER_TASK:-1}"
if [ "$SHARDS_PER_TASK" -lt 1 ]; then
  echo "[pool] SHARDS_PER_TASK=$SHARDS_PER_TASK must be >= 1" >&2
  exit 1
fi
if [ ${#GPUS[@]} -lt "$SHARDS_PER_TASK" ]; then
  echo "[pool] need at least SHARDS_PER_TASK=$SHARDS_PER_TASK GPUs, got ${#GPUS[@]}" >&2
  exit 1
fi
N_GPU_GROUPS=$((${#GPUS[@]} / SHARDS_PER_TASK))
LEFTOVER=$((${#GPUS[@]} % SHARDS_PER_TASK))

# Build GPU groups: each group is a comma-separated list of physical GPU IDs.
declare -a GPU_GROUPS
for ((g=0; g<N_GPU_GROUPS; g++)); do
  start=$((g * SHARDS_PER_TASK))
  group=""
  for ((i=0; i<SHARDS_PER_TASK; i++)); do
    [ -n "$group" ] && group="${group},"
    group="${group}${GPUS[$((start + i))]}"
  done
  GPU_GROUPS+=("$group")
done

# Per-run status log.
RUN_KEY="sz${EVA_SIZE}_n${N_EVAL}_spt${SHARDS_PER_TASK}"
STATUS="results/logs/pool_status_${RUN_KEY}.log"
{
  echo "=== pool started $(date) ==="
  echo "  gpus=${GPUS[*]}  groups=${GPU_GROUPS[*]}"
  echo "  shards_per_task=${SHARDS_PER_TASK}  concurrent_tasks=${N_GPU_GROUPS}"
  if [ "$LEFTOVER" -gt 0 ]; then
    echo "  WARNING: ${LEFTOVER} GPU(s) will be idle (not divisible by SHARDS_PER_TASK)"
  fi
  echo "  n_eval=${N_EVAL}  eva_size=${EVA_SIZE}  run_key=${RUN_KEY}"
} > "$STATUS"
cat "$STATUS"

# Task table. Note: FP now also goes through quant_eval_shard.py via --no-quant,
# so it can shard across multiple GPUs identically to the quant tasks.
TASKS=(
  "fp${N_EVAL}${SZ_SUFFIX}|--no-quant"
  "v01_w6a6_rp_n${N_EVAL}${SZ_SUFFIX}|--w-bits 6 --a-bits 6 --reparam"
  "v02_w7a7_rp_n${N_EVAL}${SZ_SUFFIX}|--w-bits 7 --a-bits 7 --reparam"
  "v03_w8a8_rp_n${N_EVAL}${SZ_SUFFIX}|--w-bits 8 --a-bits 8 --reparam"
  "v04_w6a6_rp_qwt_n${N_EVAL}${SZ_SUFFIX}|--w-bits 6 --a-bits 6 --reparam --qwt --qwt-n-samples 128"
  "v05_w7a7_rp_qwt_n${N_EVAL}${SZ_SUFFIX}|--w-bits 7 --a-bits 7 --reparam --qwt --qwt-n-samples 128"
  "v06_w8a8_rp_qwt_n${N_EVAL}${SZ_SUFFIX}|--w-bits 8 --a-bits 8 --reparam --qwt --qwt-n-samples 128"
)


# Run a single task on a GPU group. Launches SHARDS_PER_TASK shard processes
# in parallel (one per GPU in the group), waits for all of them, then runs
# merge. Caller already redirected stdout/stderr to the per-task log file.
run_task() {
  local group_gpus="$1"   # e.g. "0,1,2,3"
  local tag="$2"
  local extra="$3"
  local log="results/logs/pool_${tag}.log"
  local t0=$SECONDS

  IFS=',' read -ra gpus <<< "$group_gpus"
  local n_shards=${#gpus[@]}

  (
    set -e
    mkdir -p "results/${tag}/logs"
    pids=()
    for sid in $(seq 0 $((n_shards - 1))); do
      local gpu="${gpus[$sid]}"
      CUDA_VISIBLE_DEVICES="$gpu" python -u quant_eval_shard.py \
          --shard_id "$sid" --num_shards "$n_shards" \
          --n_eval "$N_EVAL" --tag "$tag" --resume $extra \
          > "results/${tag}/logs/shard${sid}.log" 2>&1 &
      pids+=($!)
    done
    # Wait for all shards; if any fails, propagate non-zero.
    fail=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then fail=1; fi
    done
    if [ $fail -ne 0 ]; then
      echo "[run_task] one or more shards failed; tails:"
      tail -n 15 "results/${tag}/logs/shard"*.log
      exit 1
    fi
    # Merge once all shards finished.
    CUDA_VISIBLE_DEVICES="${gpus[0]}" python -u quant_eval_merge.py \
        --tag "$tag" --rm-shards
  ) > "$log" 2>&1
  local rc=$?
  local dt=$((SECONDS - t0))
  {
    if [ $rc -ne 0 ]; then
      echo "[pool] !!! $tag FAILED on GPUs $group_gpus (${dt}s) — see $log"
      tail -20 "$log"
    else
      grep -E "\[bbox\] AP|\[segm\] AP|ΔAP" "$log" | head -4 || true
      echo "[pool] <<< $tag done on GPUs $group_gpus in ${dt}s"
    fi
  } | tee -a "$STATUS"
  return $rc
}

# Optional task subset filter. Comma-separated list of task-id prefixes
# matched against the leading "fp" / "vNN" of each tag.
TASKS_INCLUDE="${TASKS_INCLUDE:-}"
declare -A INCLUDE_SET
if [ -n "$TASKS_INCLUDE" ]; then
  IFS=',' read -ra _inc <<< "$TASKS_INCLUDE"
  for k in "${_inc[@]}"; do INCLUDE_SET["$k"]=1; done
fi

# Filter out tasks whose metrics.json already exists, or that aren't in
# TASKS_INCLUDE (when set).
PENDING=()
for entry in "${TASKS[@]}"; do
  IFS='|' read -r tag extra <<< "$entry"
  if [[ "$tag" =~ ^(fp|v[0-9]+) ]]; then
    task_id="${BASH_REMATCH[1]}"
  else
    task_id="$tag"
  fi
  if [ -n "$TASKS_INCLUDE" ] && [ -z "${INCLUDE_SET[$task_id]:-}" ]; then
    echo "[pool] skip $tag (not in TASKS_INCLUDE=$TASKS_INCLUDE)" | tee -a "$STATUS"
    continue
  fi
  if [ -f "results/${tag}/metrics.json" ]; then
    echo "[pool] skip $tag (metrics.json present)" | tee -a "$STATUS"
    grep -E "\"AP\":" "results/${tag}/metrics.json" | head -2 | tee -a "$STATUS"
  else
    PENDING+=("$entry")
  fi
done

if [ ${#PENDING[@]} -eq 0 ]; then
  echo "=== pool finished $(date) — nothing to do ===" | tee -a "$STATUS"
  exit 0
fi

# Dispatcher: groups (not individual GPUs) are the schedulable unit.
declare -A PID_GPU_GROUP PID_TAG
FREE_GPU_GROUPS=("${GPU_GROUPS[@]}")
overall_rc=0

while [ ${#PENDING[@]} -gt 0 ] || [ ${#PID_GPU_GROUP[@]} -gt 0 ]; do
  while [ ${#FREE_GPU_GROUPS[@]} -gt 0 ] && [ ${#PENDING[@]} -gt 0 ]; do
    group="${FREE_GPU_GROUPS[0]}"
    FREE_GPU_GROUPS=("${FREE_GPU_GROUPS[@]:1}")
    entry="${PENDING[0]}"
    PENDING=("${PENDING[@]:1}")
    IFS='|' read -r tag extra <<< "$entry"
    run_task "$group" "$tag" "$extra" &
    pid=$!
    PID_GPU_GROUP[$pid]=$group
    PID_TAG[$pid]=$tag
    echo "[pool] >>> launched $tag on GPUs $group (pid $pid)" | tee -a "$STATUS"
  done

  done_pid=0
  wait -n -p done_pid
  rc=$?
  if [ -z "${done_pid:-}" ] || [ "${done_pid}" = "0" ]; then
    for pid in "${!PID_GPU_GROUP[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then done_pid=$pid; break; fi
    done
  fi
  group="${PID_GPU_GROUP[$done_pid]:-}"
  tag="${PID_TAG[$done_pid]:-}"
  if [ -n "$group" ]; then
    unset 'PID_GPU_GROUP[$done_pid]' 'PID_TAG[$done_pid]'
    FREE_GPU_GROUPS+=("$group")
    [ $rc -ne 0 ] && overall_rc=1
  fi
done

echo "=== pool finished $(date) rc=$overall_rc ===" | tee -a "$STATUS"
exit $overall_rc
