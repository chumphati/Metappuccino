#!/bin/bash

#PBS -N visualisation
#PBS -l walltime=12:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1

#SBATCH --job-name=visualisation
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1

METAPPUCCINO=${1:-$METAPPUCCINO}
RES=${2:-$RES}
ENV_REQUIREMENT=${3:-$ENV_REQUIREMENT}
VERBOSE=${4:-${VERBOSE:-FALSE}}
NODE_WORK_PATH=${5:-$NODE_WORK_PATH}

LOG_DIR=$RES/logs
TMP_DIR=$RES/tmp

#SCRATCH_DIR="/scratchlocal/$USER/${PBS_JOBID:-$SLURM_JOB_ID}"
if [[ -n "${PBS_JOBID:-}" ]]; then
  SCRATCH_DIR="$NODE_WORK_PATH/${PBS_JOBID}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRATCH_DIR="$NODE_WORK_PATH/${SLURM_JOB_ID}"
else
  SCRATCH_DIR="$(mktemp -d -p "${TMP_DIR}" "visualisation")"
fi

mkdir -p "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

exec > "$LOG_DIR/visualisation.out" 2> "$LOG_DIR/visualisation.err"

cleanup() {
    cp -r "$SCRATCH_DIR/VISUALISATION" "$RES/COMPLETED_INFERENCE/" 2>/dev/null || echo "Output file not found, skipping."
    cp "$SCRATCH_DIR/STEP4_2.flag" "$TMP_DIR/" 2>/dev/null || echo "Flag not found, skipping."
    echo "End date: $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

cp "$RES/COMPLETED_INFERENCE/completed_metadata.csv" $SCRATCH_DIR/
cp "$METAPPUCCINO/scripts/normalize_graph/vizualisation_data.py" $SCRATCH_DIR/

source $ENV_REQUIREMENT/bin/activate

echo "Start $(date)"

PY_VERBOSE=()
if [[ "${VERBOSE^^}" == "TRUE" ]]; then
  PY_VERBOSE+=(--verbose)
fi

python3 -u vizualisation_data.py --base_path "$SCRATCH_DIR" --input "$SCRATCH_DIR/completed_metadata.csv" --outdir "$SCRATCH_DIR/VISUALISATION" "${PY_VERBOSE[@]}"
