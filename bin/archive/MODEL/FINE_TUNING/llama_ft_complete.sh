#!/bin/bash
#PBS -N llama_ft_complete
#PBS -l walltime=1000:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:host=node51:ncpus=30:ngpus=2:mem=150gb

METAMAP="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap"
ENV="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv"
LOG_DIR="$METAMAP/results/logs"
RESULT_DIR="$METAMAP/results/FINE_TUNING_LLAMA"
SCRATCH_DIR=/scratchlocal/$USER/$PBS_JOBID

mkdir -p $SCRATCH_DIR
mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR

exec > "$LOG_DIR/llama_ft_complete.out" 2> "$LOG_DIR/llama_ft_complete.err"
echo "Begin job : $(date)"

cleanup() {
    cp -r $SCRATCH_DIR/llama8B_fine_tuned $RESULT_DIR 2>/dev/null || echo "Model could not be saved."
    cp -r $SCRATCH_DIR/llama8B_full_finetuned $RESULT_DIR 2>/dev/null || echo "Model could not be saved."
    cp -r $SCRATCH_DIR/llama8B_train $RESULT_DIR 2>/dev/null || echo "Model could not be saved."
    rm -rf "$SCRATCH_DIR"
    echo "End job : $(date)"
}
trap cleanup EXIT

source $ENV/bin/activate

cp -r /store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/Llama-3.1-8B-Instruct $SCRATCH_DIR
cp $RESULT_DIR/finetune_data.csv $SCRATCH_DIR
cp $METAMAP/scripts/model_processing/fine_tuning/llama_ft_complete.py $SCRATCH_DIR

cd $SCRATCH_DIR

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python llama_ft_complete.py --base_path $SCRATCH_DIR