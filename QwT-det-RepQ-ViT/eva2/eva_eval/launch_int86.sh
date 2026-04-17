#!/bin/bash
# Launch int8 on GPU 0 and int6 on GPU 1 in parallel
set -e
WORKDIR=/home/yjrcs/SC_V/QwT/QwT-det-RepQ-ViT/eva2/eva_eval
LOGDIR=$WORKDIR/logs
mkdir -p $LOGDIR

source /home/yjrcs/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2

cd $WORKDIR

CUDA_VISIBLE_DEVICES=0 nohup python $WORKDIR/quant_eval_single.py 8 rp_qwt > $LOGDIR/w8a8_rp_qwt.log 2>&1 &
PID8=$!
echo "int8 PID: $PID8 (GPU 0)"

sleep 2

CUDA_VISIBLE_DEVICES=1 nohup python $WORKDIR/quant_eval_single.py 6 rp_qwt > $LOGDIR/w6a6_rp_qwt.log 2>&1 &
PID6=$!
echo "int6 PID: $PID6 (GPU 1)"

sleep 4
echo "---"
ps -p $PID8,$PID6 -o pid,cmd --no-headers || echo "(some processes already exited — check logs)"
