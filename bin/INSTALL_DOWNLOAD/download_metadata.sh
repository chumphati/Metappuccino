#!/bin/bash

#PBS -N download_metadata
#PBS -l walltime=12:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1

#SBATCH --job-name=download_metadata
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

METAPPUCCINO=${1:-${METAPPUCCINO:-}}
RES=${2:-${RES:-}}
ENV_REQUIREMENT=${3:-${ENV_REQUIREMENT:-}}
VERBOSE=${4:-${VERBOSE:-FALSE}}
NODE_WORK_PATH=${5:-${NODE_WORK_PATH:-}}
RUNS_INPUTS=${6:-${RUNS_INPUTS:-}}

if [[ -z "${RES:-}" ]]; then
  echo "RES (results dir) is required" >&2
  exit 1
fi

LOG_DIR="$RES/logs"
TMP_DIR="$RES/tmp"
mkdir -p "$LOG_DIR" "$TMP_DIR"

if [[ -n "${ENV_REQUIREMENT:-}" && -d "$ENV_REQUIREMENT" ]]; then
  source "$ENV_REQUIREMENT/bin/activate" || true
fi

if [[ -n "${PBS_JOBID:-}" ]]; then
  SCRATCH_DIR="$NODE_WORK_PATH/${PBS_JOBID}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRATCH_DIR="$NODE_WORK_PATH/${SLURM_JOB_ID}"
else
  SCRATCH_DIR="$(mktemp -d "$TMP_DIR/download_metadata")"
fi

mkdir -p "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

exec >"$LOG_DIR/download_metadata.out" 2>"$LOG_DIR/download_metadata.err"

cleanup() {
  cp -r "$SCRATCH_DIR/metadata" "$RES/ORIGINAL_METADATA/" 2>/dev/null || echo "No metadata"
  cp "$SCRATCH_DIR/metadata_sra.txt" "$RES/ORIGINAL_METADATA/" 2>/dev/null || echo "No metadata_sra.txt"
  cp "$SCRATCH_DIR/STEP1_1.flag" "$TMP_DIR/" 2>/dev/null || echo "Flag not found, skipping."
  echo "End $(date)"
  rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

if [[ -z "${METAPPUCCINO:-}" ]]; then
  echo "METAPPUCCINO path is required" >&2
  exit 1
fi
if [[ -z "${RUNS_INPUTS:-}" || ! -f "$RUNS_INPUTS" ]]; then
  echo "RUNS_INPUTS file is missing: $RUNS_INPUTS" >&2
  exit 1
fi

cp "$METAPPUCCINO/scripts/get_clean_metadata/get_metadata_ncbi_ena.py" "$SCRATCH_DIR/"
cp "$RUNS_INPUTS" "$SCRATCH_DIR/runs.txt"

echo "Start $(date)"

VERBOSE_UP=$(printf '%s' "${VERBOSE:-}" | tr '[:lower:]' '[:upper:]')
PY_VERBOSE=""
if [[ "$VERBOSE_UP" = "TRUE" ]]; then
  PY_VERBOSE="--verbose"
fi

python3 -u get_metadata_ncbi_ena.py --base_path "$SCRATCH_DIR" $PY_VERBOSE
