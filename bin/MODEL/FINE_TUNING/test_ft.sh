#!/bin/bash
#PBS -N test_ft.sh
#PBS -l walltime=1000:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:host=node51:ncpus=30:ngpus=1:mem=150gb

METAMAP="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap"
ENV="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv"
LOG_DIR="$METAMAP/results/FINE_TUNING_TEST"
RESULT_DIR="$METAMAP/results/FINE_TUNING_TEST"
SCRATCH_DIR=/scratchlocal/$USER/$PBS_JOBID

mkdir -p $SCRATCH_DIR
mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR

exec > "$LOG_DIR/test_ft.sh.out" 2> "$LOG_DIR/test_ft.sh.err"
echo "Begin job : $(date)"

cleanup() {
    rm -rf "$SCRATCH_DIR"
    echo "End job : $(date)"
}
trap cleanup EXIT

source $ENV/bin/activate

cp -r /store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/Mistral-7B-Instruct-v0.3 $SCRATCH_DIR
cp $RESULT_DIR/test_finetune_data.csv $SCRATCH_DIR
cp $METAMAP/scripts/model_processing/fine_tuning/test_ft.py $SCRATCH_DIR

cd $SCRATCH_DIR

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python test_ft.py --base_path $SCRATCH_DIR