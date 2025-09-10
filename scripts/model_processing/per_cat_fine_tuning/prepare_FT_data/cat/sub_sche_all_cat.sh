#!/bin/bash

#PBS -N get_data_train
#PBS -l walltime=500:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:host=node51:ncpus=30:mem=60gb

#SBATCH --job-name=get_data_train
#SBATCH --partition=alphafold
#SBATCH --time=100:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --nodelist=node49
#SBATCH --cpus-per-task=30
#SBATCH --mem=60G

ENV="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv"
LOG_DIR="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/DATA_PER_CAT/unified_v2/logs"
RESULT_DIR="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/DATA_PER_CAT/unified_v2"
SCRATCH_DIR="/scratchlocal/$USER/${PBS_JOBID:-$SLURM_JOB_ID}"

mkdir -p $SCRATCH_DIR
mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR

exec > "$LOG_DIR/get_data_train.out" 2> "$LOG_DIR/get_data_train.err"
echo "Begin job : $(date)"

cleanup() {
    rm -rf "$SCRATCH_DIR"
    echo "End job : $(date)"
}
trap cleanup EXIT

source $ENV/bin/activate
cp /store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/scripts/model_processing/per_cat_fine_tuning/prepare_FT_data/cat/all_sra_cat_get_real_data.sh $SCRATCH_DIR
cd $SCRATCH_DIR

bash all_sra_cat_get_real_data.sh