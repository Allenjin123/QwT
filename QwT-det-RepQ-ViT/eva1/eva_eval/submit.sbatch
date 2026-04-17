#!/bin/bash
#SBATCH --job-name=eva_qwt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/sbatch_%j.out
#SBATCH --error=results/logs/sbatch_%j.err
#SBATCH --partition=gpu-rtx6000
#SBATCH --account=nbleier_owned1

set -e
cd "$(dirname "$(readlink -f "$0")")"
mkdir -p results/logs

echo "[sbatch] job $SLURM_JOB_ID  node=$SLURMD_NODENAME  gpus=$SLURM_GPUS  $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

source ~/miniconda3/etc/profile.d/conda.sh
conda activate qwt_d2
module load cuda/12.8.1 gcc/11.2.0 2>/dev/null || true

./run_queue.sh
echo "[sbatch] queue done $(date)"
