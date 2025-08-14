#!/bin/bash

#PBS -N summary_context
#PBS -l walltime=12:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:ncpus=10:mem=16gb

#SBATCH --job-name=summary_context
#SBATCH --partition=common
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G

METAPPUCCINO=${1:-$METAPPUCCINO}
RES=${2:-$RES}
ENV_REQUIREMENT=${3:-$ENV_REQUIREMENT}
VERBOSE=${4:-${VERBOSE:-FALSE}}

LOG_DIR=$RES/logs
TMP_DIR=$RES/tmp
SCRATCH_DIR="/scratchlocal/$USER/${PBS_JOBID:-$SLURM_JOB_ID}"

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

cp "$TMP_DIR/cleaned_metadata_sra.txt" $SCRATCH_DIR/
cp "$METAPPUCCINO/scripts/get_clean_metadata/summarize_inputs.py" $SCRATCH_DIR/

source $ENV_REQUIREMENT/bin/activate

echo "Start $(date)"

PY_VERBOSE=()
if [[ "${VERBOSE^^}" == "TRUE" ]]; then
  PY_VERBOSE+=(--verbose)
fi

python3 -u summarize_inputs.py --base_path "$SCRATCH_DIR" "${PY_VERBOSE[@]}"
