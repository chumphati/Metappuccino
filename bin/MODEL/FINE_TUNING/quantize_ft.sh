#!/bin/bash
#PBS -N quantize_ft
#PBS -l walltime=100:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:ncpus=18:mem=600gb

METAMAP="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap"
ENV="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv"
LOG_DIR="$METAMAP/results/logs"
RESULT_DIR="$METAMAP/results/FINE_TUNING"

SCRATCH_DIR="/scratchlocal/$USER/$PBS_JOBID"
mkdir -p $SCRATCH_DIR
mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR

exec > "$LOG_DIR/quantize_ft.out" 2> "$LOG_DIR/quantize_ft.err"
echo "Début du job : $(date)"

#clean and copy in case of fail
cleanup() {
    cp $SCRATCH_DIR/activations_unstruct.pkl $RESULT_DIR/ 2>/dev/null || echo "Activations non trouvées."
    rm -rf "$SCRATCH_DIR"
    echo "Fin du job : $(date)"
}
trap cleanup EXIT

SCRATCH_DIR=/scratchlocal/$USER/$PBS_JOBID
mkdir -p $SCRATCH_DIR
mkdir -p $METAMAP/results/FINE_TUNING

source $ENV/bin/activate

#necessary files
cp $METAMAP/scripts/model_processing/fine_tuning/quantize_finetuned.py $SCRATCH_DIR

cd $SCRATCH_DIR

python quantize_finetuned.py --base_path $SCRATCH_DIR

