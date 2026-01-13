#!/bin/bash

#PBS -N normalize_final
#PBS -l walltime=100:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1

#SBATCH --job-name=normalize_final
#SBATCH --time=100:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1

set -euo pipefail

METAPPUCCINO=${1:-$METAPPUCCINO}
RES=${2:-$RES}
ENV_REQUIREMENT=${3:-$ENV_REQUIREMENT}
VERBOSE=${4:-${VERBOSE:-FALSE}}
NODE_WORK_PATH=${5:-${NODE_WORK_PATH:-}}
WITHOUT_CELLOSAURUS=${6:-${WITHOUT_CELLOSAURUS:-}}

LOG_DIR="$RES/logs"
TMP_DIR="$RES/tmp"
mkdir -p "$LOG_DIR" "$TMP_DIR"

source "$ENV_REQUIREMENT/bin/activate" || true

if [[ -n "${PBS_JOBID:-}" ]]; then
  SCRATCH_DIR="$NODE_WORK_PATH/${PBS_JOBID}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRATCH_DIR="$NODE_WORK_PATH/${SLURM_JOB_ID}"
else
  SCRATCH_DIR="$(mktemp -d -p "$TMP_DIR" normalize_final.XXXXX)"
fi

mkdir -p "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

exec > "$LOG_DIR/normalize_final.out" 2> "$LOG_DIR/normalize_final.err"

cleanup() {
    cp "$SCRATCH_DIR/completed_metadata.csv" "$RES/COMPLETED_INFERENCE/" 2>/dev/null || echo "csv file not found, skipping."
    cp "$SCRATCH_DIR/completed_metadata.xlsx" "$RES/COMPLETED_INFERENCE/" 2>/dev/null || echo "xlsx file not found, skipping."
    cp "$SCRATCH_DIR/completed_metadata.parquet" "$RES/COMPLETED_INFERENCE/" 2>/dev/null || echo "parquet file not found, skipping."
    cp "$SCRATCH_DIR/completed_metadata.json" "$RES/COMPLETED_INFERENCE/" 2>/dev/null || echo "json file not found, skipping."
    cp "$SCRATCH_DIR/completed_metadata.tsv" "$RES/COMPLETED_INFERENCE/" 2>/dev/null || echo "tsv file not found, skipping."
    cp "$SCRATCH_DIR/completed_metadata.feather" "$RES/COMPLETED_INFERENCE/" 2>/dev/null || echo "feather file not found, skipping."
    cp "$SCRATCH_DIR/nll_inference.csv" "$RES/COMPLETED_INFERENCE/MODEL_CONFIDENCE" 2>/dev/null || echo "nll file not found, skipping."
    cp "$SCRATCH_DIR/ppl_inference.csv" "$RES/COMPLETED_INFERENCE/MODEL_CONFIDENCE" 2>/dev/null || echo "ppl file not found, skipping."
    cp "$SCRATCH_DIR/STEP4_1.flag" "$TMP_DIR/" 2>/dev/null || echo "Flag not found, skipping."
    echo "End date: $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

cp "$TMP_DIR/database_metadata_curated.csv" "$SCRATCH_DIR/"
cp -r "$RES/COMPLETED_INFERENCE/METADATA_LLM_INFERENCE" "$SCRATCH_DIR/"
cp "$METAPPUCCINO/data/CELLOSAURUS_CLEAN.csv" "$SCRATCH_DIR/"
cp "$METAPPUCCINO/data/DOT_TABLE_CLEAN.csv" "$SCRATCH_DIR/"
cp "$METAPPUCCINO/data/UBERON_TABLE_CLEAN.csv" "$SCRATCH_DIR/"
cp "$METAPPUCCINO/scripts/normalize_graph/norm_complete.py" "$SCRATCH_DIR/"
cp "$RES/ORIGINAL_METADATA/metadata_sra_summarized.txt" "$SCRATCH_DIR/"

echo "Start $(date)"

PY_VERBOSE=()
VERBOSE_UP=$(printf '%s' "${VERBOSE:-}" | tr '[:lower:]' '[:upper:]')
if [[ "$VERBOSE_UP" = "TRUE" ]]; then
  PY_VERBOSE+=(--verbose)
fi

PY_WITHOUT_CELLOSAURUS=()
WITHOUT_CELLOSAURUS_UP=$(printf '%s' "${WITHOUT_CELLOSAURUS:-}" | tr '[:lower:]' '[:upper:]')
if [[ "$WITHOUT_CELLOSAURUS_UP" = "TRUE" ]]; then
  PY_WITHOUT_CELLOSAURUS+=(--without_cellosaurus)
fi

python3 -u norm_complete.py --base_path "$SCRATCH_DIR" "${PY_VERBOSE[@]}" "${PY_WITHOUT_CELLOSAURUS[@]}"
