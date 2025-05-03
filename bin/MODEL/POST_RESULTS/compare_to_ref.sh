#!/bin/bash
#SBATCH --job-name=compare_to_ref
#SBATCH --partition=common
#SBATCH --time=10:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=600G

METAMAP="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap"
ENV="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv"
LOG_DIR="$METAMAP/results/logs"

SCRATCH_DIR="/scratchlocal/$USER/$SLURM_JOB_ID"
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

