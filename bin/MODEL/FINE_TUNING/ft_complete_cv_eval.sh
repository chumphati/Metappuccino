#!/bin/bash
#PBS -N ft_complete_cv_eval
#PBS -l walltime=100:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:host=node50:ncpus=30:ngpus=2:mem=80gb


METAMAP="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap"
ENV="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv"
LOG_DIR="$METAMAP/results/logs"
RESULT_DIR="$METAMAP/results/FINE_TUNING"
SCRATCH_DIR=/scratchlocal/$USER/$PBS_JOBID

mkdir -p $SCRATCH_DIR
mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR

exec > "$LOG_DIR/ft_complete_cv_eval.out" 2> "$LOG_DIR/ft_complete_cv_eval.err"
echo "Begin job : $(date)"

cleanup() {
    cp -r $SCRATCH_DIR/mistral7B_fine_tuned $RESULT_DIR 2>/dev/null || echo "Model saved."
    cp -r $SCRATCH_DIR/mistral7B_full_finetuned $RESULT_DIR 2>/dev/null || echo "Model saved."
    cp -r $SCRATCH_DIR/mistral7B_train $RESULT_DIR 2>/dev/null || echo "Model saved."
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