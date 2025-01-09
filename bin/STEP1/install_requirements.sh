#!/bin/bash
#PBS -N install_requirements
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=12:mem=64gb

METAMAP=$1
ENV_REQUIREMENT=$2
PATH_CUDA=$3

LOG_DIR=$METAMAP/results/logs
SCRATCH_DIR=/scratchlocal/$USER/$PBS_JOBID
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

exec > "$LOG_DIR/install_requirements.out" 2> "$LOG_DIR/install_requirements.err"

#clean and copy in case of fail
cleanup() {
    echo "End date: $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

#activate requirements venv
source $ENV_REQUIREMENT/bin/activate

echo "Begin date: $(date)"

#install in venv
python3 -m pip install -r "$METAMAP/requirements.txt"

#install de llama-cpp-python with CUDA support
echo "Installing llama-cpp-python with CUDA support." >> "/scratchlocal/$USER/$PBS_JOBID/llm_log_SB.txt"
export CMAKE\_ARGS="-DGGML\_CUDA=on -DCUDA\_PATH=$PATH_CUDA -DCUDAToolkit\_ROOT=$PATH_CUDA -DCUDAToolkit\_INCLUDE\_DIR=$PATH_CUDA/include -DCUDAToolkit\_LIBRARY\_DIR=$PATH_CUDA/lib64"
export CUDACXX=$PATH_CUDA/bin/nvcc
python3 -m pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir