#!/bin/bash
#PBS -N pruning_ml
#PBS -l walltime=100:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:ncpus=19:mem=700gb


METAMAP="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap"
ENV="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv"
LOG_DIR="$METAMAP/results/logs"
RESULT_DIR="$METAMAP/results/PRUNING_MODEL"

SCRATCH_DIR="/scratchlocal/$USER/$PBS_JOBID"
mkdir -p $SCRATCH_DIR
mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR

exec > "$LOG_DIR/pruning_ml.out" 2> "$LOG_DIR/pruning_ml.err"
echo "Début du job : $(date)"

#clean and copy in case of fail
cleanup() {
    cp -r $SCRATCH_DIR/llama-3-pruned $RESULT_DIR/ 2>/dev/null || echo "Modèle pruné non trouvé."
    cp $SCRATCH_DIR/activations.pkl $RESULT_DIR/ 2>/dev/null || echo "Activations non trouvées."
    rm -rf "$SCRATCH_DIR"
    echo "Fin du job : $(date)"
}
trap cleanup EXIT

SCRATCH_DIR=/scratchlocal/$USER/$PBS_JOBID
mkdir -p $SCRATCH_DIR
mkdir -p $METAMAP/results/PRUNING_MODEL

source $ENV/bin/activate

#necessary files
cp $METAMAP/scripts/model_processing/pruning_model.py $SCRATCH_DIR

export TRANSFORMERS_CACHE=$METAMAP/results/PRUNING_MODEL/hf_cache
mkdir -p $TRANSFORMERS_CACHE
cd $SCRATCH_DIR

python pruning_model.py --base_path $SCRATCH_DIR

