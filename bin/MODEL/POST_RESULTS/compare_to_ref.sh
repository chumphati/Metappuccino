#!/bin/bash
#PBS -N compare_to_ref
#PBS -l walltime=100:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:ncpus=20:mem=600gb

METAMAP="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap"
ENV="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv"
LOG_DIR="$METAMAP/results/logs"

SCRATCH_DIR="/scratchlocal/$USER/$PBS_JOBID"
mkdir -p $SCRATCH_DIR
mkdir -p $LOG_DIR

exec > "$LOG_DIR/compare_to_ref.out" 2> "$LOG_DIR/compare_to_ref.err"
echo "Début du job : $(date)"

#clean and copy in case of fail
cleanup() {
    rm -rf "$SCRATCH_DIR"
    echo "Fin du job : $(date)"
}
trap cleanup EXIT

SCRATCH_DIR=/scratchlocal/$USER/$PBS_JOBID
mkdir -p $SCRATCH_DIR
source $ENV/bin/activate

#necessary files
cp $METAMAP/scripts/model_processing/fine_tuning/compare_ft_out.py $SCRATCH_DIR
cd $SCRATCH_DIR

python compare_ft_out.py

