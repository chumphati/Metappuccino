#!/bin/bash

#PBS -N LLM2_pbs
#PBS -l walltime=500:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:host=node51:ncpus=30:ngpus=1:mem=80gb

#SBATCH --job-name=LLM2_slurm
#SBATCH --partition=alphafold
#SBATCH --time=500:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --nodelist=node49
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:1
#SBATCH --mem=80G

#METAPPUCCINO=${1:-$METAPPUCCINO}
#RES=${2:-$RES}
#ENV_REQUIREMENT=${3:-$ENV_REQUIREMENT}
#MODEL=${4:-$MODEL}

METAPPUCCINO="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino"
RES="results_rpl25a"
ENV_REQUIREMENT="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv"
MODEL="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/gguf/Mistral-7B-Instruct-v0.3-original.gguf"


RESULTS_DIR=$METAPPUCCINO/$RES/LLM2
TMP_DIR="$RESULTS_DIR/"
LOG_DIR="$RESULTS_DIR/"

exec > "$LOG_DIR/LLM2.out" \
     2> "$LOG_DIR/LLM2.err"

SCRATCH_DIR="/scratchlocal/$USER/${PBS_JOBID:-$SLURM_JOB_ID}"
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

cleanup() {
    cp -r "$SCRATCH_DIR/METADATA_LLM_INFERENCE" "$RESULTS_DIR/" \
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
cp "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_rpl25a/ORIGINAL_METADATA/metadata_sra_summarized2.txt"          $SCRATCH_DIR/
cp "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_rpl25a/tmp/database_metadata_curated.csv"      $SCRATCH_DIR/
cp "$METAPPUCCINO/scripts/fill_missing_metadata/LLM_metadata_inference.py"  $SCRATCH_DIR/

source $ENV_REQUIREMENT/bin/activate

echo "Start $(date)"

python3 -u LLM_metadata_inference.py \
  --base_path $SCRATCH_DIR \
  --input_metadata_path metadata_sra_summarized2.txt \
  --error_file_path reload_model_bio_info.txt \
  --log_file_path llm_log_SB.txt \
  --flag_file STEP3_1.flag \
  --initial_n_ctx 4000 \
  --model "$(basename "$MODEL")"
