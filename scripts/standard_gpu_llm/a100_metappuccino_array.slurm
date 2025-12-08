#!/bin/bash
#SBATCH --job-name=metappuccino_a100
#SBATCH --output=metappuccino_a100_%A_%a.out
#SBATCH --error=metappuccino_a100_%A_%a.err
#SBATCH -A tfz@a100
#SBATCH --time=20:00:00
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --hint=nomultithread
#SBATCH --array=0-3

module purge
module load arch/a100

source .venv/bin/activate

export METAPP_SHARD_ID=${SLURM_ARRAY_TASK_ID}
export METAPP_NUM_SHARDS=4
export METAPP_MAX_SECONDS=$((18*3600))

echo "[SLURM] Job ID: ${SLURM_JOB_ID}, array task: ${SLURM_ARRAY_TASK_ID}"
echo "[SLURM] Shard ${METAPP_SHARD_ID}/${METAPP_NUM_SHARDS}, MAX_SECONDS=${METAPP_MAX_SECONDS}"

python metappuccino_llm.py
