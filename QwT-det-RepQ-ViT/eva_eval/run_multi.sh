#!/usr/bin/env bash
# Launch 4-way data-parallel quant eval, then merge.
# Usage: ./run_multi.sh <W_BITS> <A_BITS> [tag]
set -e
W=${1:-8}
A=${2:-8}
TAG=${3:-w${W}a${A}}
EXTRA="${EXTRA:-}"            # e.g. EXTRA=--reparam ./run_multi.sh 4 4 w4a4_rp
N_GPU=4
N_EVAL=100

cd "$(dirname "$0")"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2
module load cuda/12.8.1 gcc/11.2.0 2>/dev/null || true

rm -rf results/${TAG}/pred_shard*.pth 2>/dev/null || true
mkdir -p results/${TAG}/logs

echo "[launcher] W${W}/A${A} across ${N_GPU} GPUs, tag=${TAG}"
pids=()
for i in $(seq 0 $((N_GPU-1))); do
  CUDA_VISIBLE_DEVICES=$i python quant_eval_shard.py \
      --w-bits $W --a-bits $A \
      --shard-idx $i --n-shards $N_GPU \
      --n-eval $N_EVAL --tag $TAG $EXTRA \
      > results/${TAG}/logs/shard${i}.log 2>&1 &
  pids+=($!)
  echo "[launcher]   GPU$i -> pid ${pids[-1]}"
done

echo "[launcher] waiting for ${N_GPU} shards..."
fail=0
for pid in "${pids[@]}"; do
  if ! wait $pid; then fail=1; fi
done
if [ $fail -ne 0 ]; then
  echo "[launcher] at least one shard failed; see results/${TAG}/logs/"
  tail -n 10 results/${TAG}/logs/*.log
  exit 1
fi

echo "[launcher] all shards done; merging..."
python quant_eval_merge.py --tag $TAG
