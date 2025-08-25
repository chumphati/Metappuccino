#!/bin/bash

#PBS -N install_requirements
#PBS -l walltime=01:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:ngpus=1

#SBATCH --job-name=install_requirements
#SBATCH --partition=alphafold
#SBATCH --time=01:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

METAPPUCCINO=${1:-$METAPPUCCINO}
RES=${2:-RES}
ENV_REQUIREMENT=${3:-$ENV_REQUIREMENT}
PATH_CUDA=${4:-$PATH_CUDA}
NODE_WORK_PATH=${5:-$NODE_WORK_PATH}

LOG_DIR=$RES/logs
TMP_DIR=$RES/tmp

#SCRATCH_DIR="/scratchlocal/$USER/${PBS_JOBID:-$SLURM_JOB_ID}"
if [[ -n "${PBS_JOBID:-}" ]]; then
  SCRATCH_DIR="$NODE_WORK_PATH/${PBS_JOBID}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRATCH_DIR="$NODE_WORK_PATH/${SLURM_JOB_ID}"
else
  SCRATCH_DIR="$(mktemp -d -p "${TMP_DIR}" "install_requirements")"
fi

mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

exec >"$LOG_DIR/install_requirements.out" 2>"$LOG_DIR/install_requirements.err"

cleanup() {
  echo "End $(date)"
  rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

source $ENV_REQUIREMENT/bin/activate

echo "Start $(date)"

echo "Installing llama-cpp-python with CUDA support." \
  >> "$SCRATCH_DIR/llm_log_install.txt"
export CMAKE_ARGS="-DGGML_CUDA=on \
  -DCUDA_PATH=$PATH_CUDA \
  -DCUDAToolkit_ROOT=$PATH_CUDA \
  -DCUDAToolkit_INCLUDE_DIR=$PATH_CUDA/include \
  -DCUDAToolkit_LIBRARY_DIR=$PATH_CUDA/lib64"
export CUDACXX=$PATH_CUDA/bin/nvcc
python3 -m pip install llama-cpp-python \
  --upgrade --force-reinstall --no-cache-dir
