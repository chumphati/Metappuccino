#!/bin/bash

#PBS -N ft_percat_eval
#PBS -l walltime=500:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:host=node51:ncpus=30:ngpus=1:mem=80gb

#SBATCH --job-name=ft_percat_eval
#SBATCH --partition=alphafold
#SBATCH --time=100:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --nodelist=node49
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:1
#SBATCH --mem=80G

METAPPUCCINO="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino"
ENV="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv"
LOG_DIR="$METAPPUCCINO/results/logs"
RESULT_DIR="$METAPPUCCINO/results/FINE_TUNING"
SCRATCH_DIR="/scratchlocal/$USER/${PBS_JOBID:-$SLURM_JOB_ID}"

mkdir -p $SCRATCH_DIR
mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR

exec > "$LOG_DIR/ft_percat_eval.out" 2> "$LOG_DIR/ft_percat_eval.err"
echo "Begin job : $(date)"

cleanup() {
    cp -r $SCRATCH_DIR/cat_cell_line $RESULT_DIR 2>/dev/null || echo "Model could not be saved."
    rm -rf "$SCRATCH_DIR"
    echo "End job : $(date)"
}
trap cleanup EXIT

source $ENV/bin/activate

cp -r /store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/Mistral-7B-Instruct-v0.3 $SCRATCH_DIR
cp $RESULT_DIR/finetune_data_train.csv $SCRATCH_DIR
cp $RESULT_DIR/finetune_data_val.csv $SCRATCH_DIR
cp $RESULT_DIR/finetune_data_test.csv $SCRATCH_DIR
cp $METAPPUCCINO/scripts/model_processing/per_cat_fine_tuning/cell_line_FT.py $SCRATCH_DIR

cd $SCRATCH_DIR

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python cell_line_FT.py --base_path $SCRATCH_DIR