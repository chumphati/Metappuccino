#!/bin/bash

#PBS -N llm_inference_pbs
#PBS -l walltime=500:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:host=node51:ncpus=30:ngpus=2:mem=80gb

#SBATCH --job-name=llm_inference_slurm
#SBATCH --partition=alphafold
#SBATCH --time=500:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --nodelist=node49
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:2
#SBATCH --mem=80G

METAPPUCCINO=${1:-$METAPPUCCINO}
RES=${2:-$RES}
ENV_REQUIREMENT=${3:-$ENV_REQUIREMENT}
MODEL=${4:-$MODEL}

RESULTS_DIR=$METAPPUCCINO/$RES
TMP_DIR=$RESULTS_DIR/tmp
LOG_DIR=$RESULTS_DIR/logs

exec > "$LOG_DIR/llm_inference.out" \
     2> "$LOG_DIR/llm_inference.err"

SCRATCH_DIR="/scratchlocal/$USER/${PBS_JOBID:-$SLURM_JOB_ID}"
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

cleanup() {
    cp -r "$SCRATCH_DIR/METADATA_LLM_INFERENCE" "$RESULTS_DIR/COMPLETED_INFERENCE/" \
      2>/dev/null || echo "METADATA_LLM_INFERENCE not founded"
    cp "$SCRATCH_DIR/skipped_runs.txt"            "$TMP_DIR/" 2>/dev/null || echo "No skipped_runs"
    cp "$SCRATCH_DIR/reload_model_bio_info.txt"   "$TMP_DIR/" 2>/dev/null || echo "No reload_model"
    cp "$SCRATCH_DIR/llm_log_SB.txt"              "$LOG_DIR/" 2>/dev/null || echo "No log"
    cp "$SCRATCH_DIR/STEP3_1.flag"                "$TMP_DIR/" 2>/dev/null || echo "No flag"
    echo "End $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

cp "$MODEL"                                 $SCRATCH_DIR/
cp "$METAPPUCCINO/$RES/ORIGINAL_METADATA/metadata_sra_summarized.txt"          $SCRATCH_DIR/
cp "$TMP_DIR/database_metadata_curated.csv"      $SCRATCH_DIR/
cp "$METAPPUCCINO/scripts/fill_missing_metadata/LLM_metadata_inference.py"  $SCRATCH_DIR/

source $ENV_REQUIREMENT/bin/activate

echo "Start $(date)"

python3 -u LLM_metadata_inference.py \
  --base_path $SCRATCH_DIR \
  --input_metadata_path metadata_sra_summarized.txt \
  --error_file_path reload_model_bio_info.txt \
  --log_file_path llm_log_SB.txt \
  --flag_file STEP3_1.flag \
  --initial_n_ctx 3000 \
  --model "$(basename "$MODEL")"
