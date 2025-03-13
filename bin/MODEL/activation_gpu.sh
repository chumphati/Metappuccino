#!/bin/bash
#PBS -N activation_gpu
#PBS -l walltime=100:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:host=node50:ncpus=80:ngpus=2:mem=250gb

METAMAP="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap"
ENV="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv"
LOG_DIR="$METAMAP/results/logs"
RESULT_DIR="$METAMAP/results/PRUNING_MODEL"

SCRATCH_DIR="/scratchlocal/$USER/$PBS_JOBID"
mkdir -p $SCRATCH_DIR
mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR

exec > "$LOG_DIR/activation_gpu.out" 2> "$LOG_DIR/activation_gpu.err"
echo "Début du job : $(date)"

#clean and copy in case of fail
cleanup() {
    cp $SCRATCH_DIR/activations_gpu.pkl $RESULT_DIR/ 2>/dev/null || echo "Activations non trouvées."
    rm -rf "$SCRATCH_DIR"
    echo "Fin du job : $(date)"
}
trap cleanup EXIT

SCRATCH_DIR=/scratchlocal/$USER/$PBS_JOBID
mkdir -p $SCRATCH_DIR
mkdir -p $METAMAP/results/PRUNING_MODEL

source $ENV/bin/activate

#necessary files
cp $METAMAP/scripts/model_processing/activation_gpu.py $SCRATCH_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_CACHE=$METAMAP/results/PRUNING_MODEL/hf_cache
mkdir -p $TRANSFORMERS_CACHE
cd $SCRATCH_DIR

python activation_gpu.py --base_path $SCRATCH_DIR

