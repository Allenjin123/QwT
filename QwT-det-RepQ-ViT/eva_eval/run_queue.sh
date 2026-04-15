#!/usr/bin/env bash
# Sequential queue of 7 runs: FP baseline + 6 quant configs (all with reparam).
# Each entry: (version_tag, bits, extra-flags-to-shard).
# Runs use 4 GPUs data-parallel.
set -e
cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2
module load cuda/12.8.1 gcc/11.2.0 2>/dev/null || true

mkdir -p results/logs
STATUS=results/logs/queue_status.log
echo "=== queue started $(date) ===" > "$STATUS"

run() {
  local tag="$1"; local w="$2"; local a="$3"; local extra="$4"
  echo "[queue] >>> $tag  (W=$w A=$a extra='$extra')" | tee -a "$STATUS"
  local t0=$SECONDS
  EXTRA="$extra" ./run_multi.sh $w $a $tag 2>&1 | tail -60 > "results/logs/queue_${tag}.log" || {
    echo "[queue] !!! $tag FAILED" | tee -a "$STATUS"
    return 1
  }
  local dt=$((SECONDS - t0))
  # Print summary line from the run's own log
  grep -E "\[bbox\] AP|\[segm\] AP|ΔAP" "results/logs/queue_${tag}.log" | tee -a "$STATUS"
  echo "[queue] <<< $tag done in ${dt}s" | tee -a "$STATUS"
}

# --- FP baseline (no quant, no reparam, no QwT) ---
if [ ! -f results/fp100/metrics.json ]; then
  echo "[queue] >>> v00_fp100" | tee -a "$STATUS"
  t0=$SECONDS
  CUDA_VISIBLE_DEVICES=0 python fp_eval100.py > results/logs/queue_v00_fp100.log 2>&1
  dt=$((SECONDS - t0))
  grep -E "\[bbox\] AP|\[segm\] AP" results/logs/queue_v00_fp100.log | tee -a "$STATUS"
  echo "[queue] <<< v00_fp100 done in ${dt}s" | tee -a "$STATUS"
fi

# --- 6 quant configs (all with reparam) ---
run v01_w8a8_rp     8 8 "--reparam"
run v02_w6a6_rp     6 6 "--reparam"
run v03_w4a4_rp     4 4 "--reparam"
run v04_w8a8_rp_qwt 8 8 "--reparam --qwt --qwt-n-samples 32"
run v05_w6a6_rp_qwt 6 6 "--reparam --qwt --qwt-n-samples 32"
run v06_w4a4_rp_qwt 4 4 "--reparam --qwt --qwt-n-samples 32"

echo "=== queue finished $(date) ===" | tee -a "$STATUS"
