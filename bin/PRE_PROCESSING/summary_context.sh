#!/bin/bash

#PBS -N summary_context
#PBS -l walltime=5000:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1

#SBATCH --job-name=summary_context
#SBATCH --time=5000:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1

set -euo pipefail

METAPPUCCINO=${1:-$METAPPUCCINO}
RES=${2:-$RES}
ENV_REQUIREMENT=${3:-$ENV_REQUIREMENT}
VERBOSE=${4:-${VERBOSE:-FALSE}}
NODE_WORK_PATH=${5:-${NODE_WORK_PATH:-}}

LOG_DIR=$RES/logs
TMP_DIR=$RES/tmp
mkdir -p "$LOG_DIR" "$TMP_DIR"

source "$ENV_REQUIREMENT/bin/activate" || true

if [[ -n "${PBS_JOBID:-}" ]]; then
  SCRATCH_DIR="$NODE_WORK_PATH/${PBS_JOBID}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRATCH_DIR="$NODE_WORK_PATH/${SLURM_JOB_ID}"
else
  SCRATCH_DIR="$(mktemp -d -p "$TMP_DIR" summary_context.XXXXX)"
fi

mkdir -p "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

exec > "$LOG_DIR/summary_context.out" 2> "$LOG_DIR/summary_context.err"

cleanup() {
    cp "$SCRATCH_DIR/metadata_sra_summarized.txt" "$RES/ORIGINAL_METADATA/" 2>/dev/null || echo "Output file not found, skipping."
    cp "$SCRATCH_DIR/STEP2_2.flag" "$TMP_DIR/" 2>/dev/null || echo "Flag not found, skipping."
    echo "End date: $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

cp "$TMP_DIR/cleaned_metadata_sra.txt" "$SCRATCH_DIR/"
cp "$METAPPUCCINO/scripts/get_clean_metadata/summarize_inputs.py" "$SCRATCH_DIR/"

echo "Start $(date)"

PY_VERBOSE=()
VERBOSE_UP=$(printf '%s' "${VERBOSE:-}" | tr '[:lower:]' '[:upper:]')
if [[ "$VERBOSE_UP" = "TRUE" ]]; then
  PY_VERBOSE+=(--verbose)
fi

python3 -u summarize_inputs.py --base_path "$SCRATCH_DIR" "${PY_VERBOSE[@]}"
