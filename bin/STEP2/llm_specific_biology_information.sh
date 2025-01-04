#!/bin/bash
#PBS -N llm_specific_biology_information
#PBS -o /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/llm_specific_biology_information.out
#PBS -e /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/llm_specific_biology_information.err
#PBS -l walltime=1000:00:00
#PBS -l select=1:host=node51:ncpus=30:ngpus=2:mem=80gb

SCRATCH_DIR=/scratchlocal/$USER/$PBS_JOBID
OUTPUT_DIR=/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results
LOG_DIR=$OUTPUT_DIR/logs
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

# necessary files
cp /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/models/Llama-3.1-Nemotron-70B-Instruct-HF-Q4_K_M.gguf $SCRATCH_DIR/
cp /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/LLM_METADATA_READY/test.txt $SCRATCH_DIR/
cp /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/RAW_FINAL_INFO.txt $SCRATCH_DIR/
cp /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/scripts/fill_missing_metadata/get_biology_information_LLM.py $SCRATCH_DIR/

# activate requirements venv
source /store/EQUIPES/SSFA/MEMBERS/fiona.hak/llama_env/bin/activate

# install de llama-cpp-python with CUDA support
#echo "Installing llama-cpp-python with CUDA support." >> "/scratchlocal/$USER/$PBS_JOBID/llm_log_SB.txt"
#export CMAKE\_ARGS="-DGGML\_CUDA=on -DCUDA\_PATH=/usr/local/cuda -DCUDAToolkit\_ROOT=/usr/local/cuda -DCUDAToolkit\_INCLUDE\_DIR=/usr/local/cuda/include -DCUDAToolkit\_LIBRARY\_DIR=/usr/local/cuda/lib64"
#export CUDACXX=/usr/local/cuda/bin/nvcc
#python3 -m pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

#python3 -m pip install llama-cpp-python \
#  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu125

echo "Begin date: $(date)"

python3 -u $SCRATCH_DIR/get_biology_information_LLM.py --base_path $SCRATCH_DIR

# copy results
cp -r $SCRATCH_DIR/INFO_BIO_LLM $OUTPUT_DIR/
cp $SCRATCH_DIR/llm_log_SB.txt $LOG_DIR/
echo "End date: $(date)"
rm -rf "$SCRATCH_DIR"
