#!/bin/bash

#PBS -N extract_preprocess
#PBS -l walltime=12:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1

#SBATCH --job-name=extract_preprocess
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

METAPPUCCINO=${1:-$METAPPUCCINO}
RES=${2:-$RES}
ENV_REQUIREMENT=${3:-$ENV_REQUIREMENT}
LOGAN_PATH=${4:-$LOGAN_PATH}
VERBOSE=${5:-${VERBOSE:-FALSE}}
NODE_WORK_PATH=${6:-$NODE_WORK_PATH}

LOG_DIR=$RES/logs
TMP_DIR=$RES/tmp
mkdir -p "$LOG_DIR" "$TMP_DIR"

source "$ENV_REQUIREMENT/bin/activate" || true

if [[ -n "${PBS_JOBID:-}" ]]; then
  SCRATCH_DIR="$NODE_WORK_PATH/${PBS_JOBID}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRATCH_DIR="$NODE_WORK_PATH/${SLURM_JOB_ID}"
else
  SCRATCH_DIR="$(mktemp -d "$TMP_DIR/extract_preprocess")"
fi

mkdir -p "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

exec > "$LOG_DIR/extract_preprocess.out" 2> "$LOG_DIR/extract_preprocess.err"

cleanup() {
    cp "$SCRATCH_DIR/database_metadata_curated.csv" "$TMP_DIR/" 2>/dev/null || echo "database_metadata_curated file not found, skipping."
    cp "$SCRATCH_DIR/ambiguous_cell_lines.csv" "$TMP_DIR/" 2>/dev/null || echo "ambiguous_cell_lines file not found, skipping."
    cp "$SCRATCH_DIR/cleaned_metadata_sra.txt" "$TMP_DIR/" 2>/dev/null || echo "logan_comp file not found, skipping."
    cp "$SCRATCH_DIR/STEP2_1.flag" "$TMP_DIR/" 2>/dev/null || echo "Flag not found, skipping."
    echo "End date: $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

cp "$METAPPUCCINO/data/CELLOSAURUS_CLEAN.csv" "$SCRATCH_DIR/"
cp "$METAPPUCCINO/data/CELLOSAURUS_PRECUT.csv" "$SCRATCH_DIR/"
cp "$METAPPUCCINO/data/DOT_TABLE_CLEAN.csv" "$SCRATCH_DIR/"
cp "$METAPPUCCINO/data/UBERON_TABLE_CLEAN.csv" "$SCRATCH_DIR/"
cp -r "$RES/ORIGINAL_METADATA/metadata" "$SCRATCH_DIR/"
cp "$RES/ORIGINAL_METADATA/metadata_sra.txt" "$SCRATCH_DIR/"
cp "$TMP_DIR/cleaned_metadata_sra.txt" "$SCRATCH_DIR/"
cp "$METAPPUCCINO/scripts/get_clean_metadata/fetch_existing_cat.py" "$SCRATCH_DIR/"
cp "$METAPPUCCINO/scripts/get_clean_metadata/logan_add_search.py" "$SCRATCH_DIR/"

echo "Start $(date)"

PY_VERBOSE=()
VERBOSE_UP=$(printf '%s' "${VERBOSE:-}" | tr '[:lower:]' '[:upper:]')
if [[ "$VERBOSE_UP" = "TRUE" ]]; then
  PY_VERBOSE+=(--verbose)
fi

python3 -u fetch_existing_cat.py --base_path "$SCRATCH_DIR" "${PY_VERBOSE[@]}"

if [[ -n "$LOGAN_PATH" ]]; then
    python3 -u logan_add_search.py --base_path "$SCRATCH_DIR" --input_logan_path "$LOGAN_PATH" "${PY_VERBOSE[@]}"
    mv "$SCRATCH_DIR/metadata_sra_with_logan.txt" "$SCRATCH_DIR/cleaned_metadata_sra.txt"
fi
