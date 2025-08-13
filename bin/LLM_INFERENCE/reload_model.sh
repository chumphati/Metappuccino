#!/bin/bash

#PBS -N reload_context_llm
#PBS -l walltime=500:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:host=node51:ncpus=30:ngpus=1:mem=80gb

#SBATCH --job-name=reload_context_llm
#SBATCH --partition=alphafold
#SBATCH --time=500:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --nodelist=node49
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:1
#SBATCH --mem=80G

METAPPUCCINO=${1:-$METAPPUCCINO}
RES=${2:-$RES}
ENV_REQUIREMENT=${3:-$ENV_REQUIREMENT}
MODEL=${4:-$MODEL}
ITERATION_LIMIT=${5:-$ITERATION_LIMIT}

RESULTS_DIR=$RES
TMP_DIR=$RESULTS_DIR/tmp
LOG_DIR=$RESULTS_DIR/logs
exec > "$LOG_DIR/reload_context_llm.out" 2> "$LOG_DIR/reload_context_llm.err"

SCRATCH_DIR="/scratchlocal/$USER/${PBS_JOBID:-$SLURM_JOB_ID}"
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

cleanup() {
    cp -r "$SCRATCH_DIR/METADATA_LLM_INFERENCE/" "$RES/COMPLETED_INFERENCE/" \
      || echo "METADATA_LLM_INFERENCE not founded"
    cp "$SCRATCH_DIR/skipped_runs.txt"            "$TMP_DIR/" 2>/dev/null || echo "No skipped_runs"
    cp "$SCRATCH_DIR/reload_model_bio_info.txt"   "$TMP_DIR/" 2>/dev/null || echo "No reload_model"
    cp "$SCRATCH_DIR/llm_log_reload.txt"              "$LOG_DIR/" 2>/dev/null || echo "No log"
    cp "$SCRATCH_DIR/STEP3_2.flag"                "$TMP_DIR/" 2>/dev/null || echo "No flag"
    echo "End $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT


if [ ! -f "$TMP_DIR/reload_model_bio_info.txt" ] ; then
    echo "✔ No file $TMP_DIR/reload_model_bio_info.txt to analyse."
    touch "$SCRATCH_DIR/STEP3_2.flag"
    exit 0
fi

cp $MODEL $SCRATCH_DIR/
cp $TMP_DIR/reload_model_bio_info.txt $SCRATCH_DIR/
cp $TMP_DIR/database_metadata_curated.csv $SCRATCH_DIR/
cp $METAPPUCCINO/scripts/fill_missing_metadata/LLM_metadata_inference.py $SCRATCH_DIR/
source $ENV_REQUIREMENT/bin/activate

echo "Begin date: $(date)"

iteration_limit=ITERATION_LIMIT
for (( i=0; i<$iteration_limit; i++ )); do

    if [ -f "$TMP_DIR/reload_model_bio_info.txt" ]; then
      python3 -u $SCRATCH_DIR/LLM_metadata_inference.py --base_path $SCRATCH_DIR \
          --input_metadata_path $SCRATCH_DIR/reload_model_bio_info.txt \
          --error_file_path $SCRATCH_DIR/reload_model_bio_info_bis.txt \
          --log_file_path $SCRATCH_DIR/llm_log_reload.txt \
          --flag_file $SCRATCH_DIR/STEP3_2.flag \
          --initial_n_ctx 3500 \
          --model "$SCRATCH_DIR/$(basename "$MODEL")"
    fi

    if [ ! -f "$SCRATCH_DIR/reload_model_bio_info.txt" ] ; then
        echo "All inferences completed.."
        touch "$SCRATCH_DIR/STEP3_2.flag"
        break
    fi

    mv $SCRATCH_DIR/reload_model_bio_info_bis.txt $SCRATCH_DIR/reload_model_bio_info.txt
done

