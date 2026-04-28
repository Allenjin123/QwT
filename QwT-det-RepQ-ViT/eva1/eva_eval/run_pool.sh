#!/usr/bin/env bash
# Pool-based runner for 7 EVA eval tasks:
#   1) fp100                      FP baseline
#   2) v01_w6a6_rp                int6 + reparam
#   3) v02_w7a7_rp                int7 + reparam
#   4) v03_w8a8_rp                int8 + reparam
#   5) v04_w6a6_rp_qwt            int6 + reparam + QwT
#   6) v05_w7a7_rp_qwt            int7 + reparam + QwT
#   7) v06_w8a8_rp_qwt            int8 + reparam + QwT
#
# One GPU per task. Pass the physical GPU IDs to use as positional args; the
# pool size equals the number of GPUs you list.
#
# Usage:
#   ./run_pool.sh 3 4 5            # uses GPUs 3,4,5 -> 3-way concurrent
#   ./run_pool.sh 0                # serial on GPU 0
#
# Per-task log: results/logs/pool_<tag>.log
# Status log:   results/logs/pool_status.log
# Resume:       tasks with results/<tag>/metrics.json are skipped.
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
# Must be a multiple of 256. Setting non-default also enables interp_type="beit"
# inside the eval scripts (mirrors cascade_mask_rcnn_vitdet_eva_1536.py).
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

# Per-run status log (namespaced so concurrent pools at different sizes don't
# stomp each other).
RUN_KEY="sz${EVA_SIZE}_n${N_EVAL}"
STATUS="results/logs/pool_status_${RUN_KEY}.log"
echo "=== pool started $(date) gpus=${GPUS[*]} run_key=${RUN_KEY} ===" > "$STATUS"

# tag | kind(fp|quant) | extra args to quant_eval_shard.py
TASKS=(
  "fp${N_EVAL}${SZ_SUFFIX}|fp|"
  "v01_w6a6_rp_n${N_EVAL}${SZ_SUFFIX}|quant|--w-bits 6 --a-bits 6 --reparam"
  "v02_w7a7_rp_n${N_EVAL}${SZ_SUFFIX}|quant|--w-bits 7 --a-bits 7 --reparam"
  "v03_w8a8_rp_n${N_EVAL}${SZ_SUFFIX}|quant|--w-bits 8 --a-bits 8 --reparam"
  "v04_w6a6_rp_qwt_n${N_EVAL}${SZ_SUFFIX}|quant|--w-bits 6 --a-bits 6 --reparam --qwt --qwt-n-samples 128"
  "v05_w7a7_rp_qwt_n${N_EVAL}${SZ_SUFFIX}|quant|--w-bits 7 --a-bits 7 --reparam --qwt --qwt-n-samples 128"
  "v06_w8a8_rp_qwt_n${N_EVAL}${SZ_SUFFIX}|quant|--w-bits 8 --a-bits 8 --reparam --qwt --qwt-n-samples 128"
)

run_task() {
  local gpu="$1"; local tag="$2"; local kind="$3"; local extra="$4"
  local log="results/logs/pool_${tag}.log"
  local t0=$SECONDS
  (
    set -e
    if [ "$kind" = "fp" ]; then
      CUDA_VISIBLE_DEVICES="$gpu" python -u fp_eval100.py
    else
      rm -f "results/${tag}"/pred_shard*.pth 2>/dev/null || true
      mkdir -p "results/${tag}/logs"
      CUDA_VISIBLE_DEVICES="$gpu" python -u quant_eval_shard.py \
          --shard-idx 0 --n-shards 1 --n-eval "$N_EVAL" --tag "$tag" $extra
      CUDA_VISIBLE_DEVICES="$gpu" python -u quant_eval_merge.py --tag "$tag"
    fi
  ) > "$log" 2>&1
  local rc=$?
  local dt=$((SECONDS - t0))
  {
    if [ $rc -ne 0 ]; then
      echo "[pool] !!! $tag FAILED on GPU $gpu (${dt}s) — see $log"
      tail -20 "$log"
    else
      grep -E "\[bbox\] AP|\[segm\] AP|ΔAP" "$log" | head -4 || true
      echo "[pool] <<< $tag done on GPU $gpu in ${dt}s"
    fi
  } | tee -a "$STATUS"
  return $rc
}

# Optional task subset filter. Comma-separated list of task-id prefixes
# matched against the leading "fp" / "vNN" of each tag.
#   TASKS_INCLUDE=fp,v01,v02       -> only those four
#   (unset / empty)                -> all 7 (default)
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
  IFS='|' read -r tag kind extra <<< "$entry"
  # Derive a stable task id from the tag for filter matching:
  #   fp5000_sz1024            -> fp
  #   v01_w6a6_rp_n5000_sz1024 -> v01
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

declare -A PID_GPU PID_TAG
FREE_GPUS=("${GPUS[@]}")
overall_rc=0

while [ ${#PENDING[@]} -gt 0 ] || [ ${#PID_GPU[@]} -gt 0 ]; do
  # Fill all free GPUs from the pending queue.
  while [ ${#FREE_GPUS[@]} -gt 0 ] && [ ${#PENDING[@]} -gt 0 ]; do
    gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")
    entry="${PENDING[0]}"
    PENDING=("${PENDING[@]:1}")
    IFS='|' read -r tag kind extra <<< "$entry"
    run_task "$gpu" "$tag" "$kind" "$extra" &
    pid=$!
    PID_GPU[$pid]=$gpu
    PID_TAG[$pid]=$tag
    echo "[pool] >>> launched $tag on GPU $gpu (pid $pid)" | tee -a "$STATUS"
  done

  # Block until any one child exits; -p captures its PID so we know which GPU to free.
  done_pid=0
  wait -n -p done_pid
  rc=$?
  if [ -z "${done_pid:-}" ] || [ "${done_pid}" = "0" ]; then
    # Defensive: scan map for anything no longer alive.
    for pid in "${!PID_GPU[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then done_pid=$pid; break; fi
    done
  fi
  gpu="${PID_GPU[$done_pid]:-}"
  tag="${PID_TAG[$done_pid]:-}"
  if [ -n "$gpu" ]; then
    unset 'PID_GPU[$done_pid]' 'PID_TAG[$done_pid]'
    FREE_GPUS+=("$gpu")
    [ $rc -ne 0 ] && overall_rc=1
  fi
done

echo "=== pool finished $(date) rc=$overall_rc ===" | tee -a "$STATUS"
exit $overall_rc
