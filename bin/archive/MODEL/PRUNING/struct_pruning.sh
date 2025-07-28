#!/bin/bash
#PBS -N struct_pruning
#PBS -l walltime=100:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:host=node41:ncpus=30:mem=600gb

METAMAP="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap"
ENV="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv"
LOG_DIR="$METAMAP/results/logs"
RESULT_DIR="$METAMAP/results/PRUNING_MODEL"

SCRATCH_DIR="/scratchlocal/$USER/$PBS_JOBID"
mkdir -p $SCRATCH_DIR
mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR

exec > "$LOG_DIR/struct_pruning.out" 2> "$LOG_DIR/struct_pruning.err"
echo "Begin job : $(date)"

#clean and copy in case of fail
cleanup() {
    cp $SCRATCH_DIR/activations_struct.pkl $RESULT_DIR/ 2>/dev/null || echo "Activations non trouvées."
    rm -rf "$SCRATCH_DIR"
    echo "End job : $(date)"
}
trap cleanup EXIT

SCRATCH_DIR=/scratchlocal/$USER/$PBS_JOBID
mkdir -p $SCRATCH_DIR
mkdir -p $METAMAP/results/PRUNING_MODEL

source $ENV/bin/activate

#necessary files
cp $METAMAP/scripts/model_processing/structural_pruning.py $SCRATCH_DIR

export TRANSFORMERS_CACHE=/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/store/first_pruned_results/PRUNING_MODEL/hf_cache
cd $SCRATCH_DIR

python structural_pruning.py --base_path $SCRATCH_DIR

