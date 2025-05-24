#!/bin/bash
#SBATCH --job-name=ft_complete_cv_eval
#SBATCH --partition=alphafold
#SBATCH --time=1000:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --nodelist=node49
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:2
#SBATCH --mem=150G


METAMAP="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap"
ENV="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv"
LOG_DIR="$METAMAP/results/logs"
RESULT_DIR="$METAMAP/results/FINE_TUNING"
#SCRATCH_DIR=/scratchlocal/$USER/$PBS_JOBID
SCRATCH_DIR="/scratchlocal/$USER/$SLURM_JOB_ID"

mkdir -p $SCRATCH_DIR
mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR

exec > "$LOG_DIR/ft_complete_cv_eval.out" 2> "$LOG_DIR/ft_complete_cv_eval.err"
echo "Begin job : $(date)"

cleanup() {
    cp -r $SCRATCH_DIR/mistral7B_fine_tuned $RESULT_DIR 2>/dev/null || echo "Model could not be saved."
    cp -r $SCRATCH_DIR/mistral7B_full_finetuned $RESULT_DIR 2>/dev/null || echo "Model could not be saved."
    cp -r $SCRATCH_DIR/mistral7B_train_final $RESULT_DIR 2>/dev/null || echo "Model could not be saved."
    rm -rf "$SCRATCH_DIR"
    echo "End job : $(date)"
}
trap cleanup EXIT

source $ENV/bin/activate

cp -r /store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/Mistral-7B-Instruct-v0.3 $SCRATCH_DIR
cp $RESULT_DIR/finetune_data.csv $SCRATCH_DIR
cp $METAMAP/scripts/model_processing/fine_tuning/ft_complete_cv_eval.py $SCRATCH_DIR

cd $SCRATCH_DIR

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ft_complete_cv_eval.py --base_path $SCRATCH_DIR