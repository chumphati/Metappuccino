#!/bin/bash
#PBS -N install_requirements
#PBS -l walltime=12:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:host=node51:ncpus=30:ngpus=1:mem=80gb

METAMAP=${1:-$METAMAP}
ENV_REQUIREMENT=${2:-$ENV_REQUIREMENT}
PATH_CUDA=${3:-$PATH_CUDA}

#METAMAP='/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap'

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

#python3 -m pip uninstall -y llama_cpp_python llama_cpp_python_cuda llama_cpp_python_cuda_tensorcores
#git clone https://github.com/JamePeng/llama-cpp-python.git
#cd llama-cpp-python/vendor
#git clone https://github.com/ggerganov/llama.cpp
#cd ..
#set CMAKE_ARGS = "-DGGML_CUDA=on"
#set FORCE_CMAKE=1
#python3 -m pip install . --no-cache-dir --verbose