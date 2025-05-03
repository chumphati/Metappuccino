#!/bin/bash
#SBATCH --job-name=quantize_ft
#SBATCH --partition=alphafold
#SBATCH --time=10:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --nodelist=node49
#SBATCH --cpus-per-task=18
#SBATCH --mem=150G

METAMAP="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap"
ENV="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv"
LOG_DIR="$METAMAP/results/logs"

SCRATCH_DIR="/scratchlocal/$USER/$SLURM_JOB_ID"
mkdir -p $SCRATCH_DIR
mkdir -p $LOG_DIR

exec > "$LOG_DIR/quantize_ft.out" 2> "$LOG_DIR/quantize_ft.err"
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
cp $METAMAP/scripts/model_processing/fine_tuning/process_final_model/quantize_finetuned.py $SCRATCH_DIR

cd $SCRATCH_DIR

python quantize_finetuned.py --base_path $SCRATCH_DIR

