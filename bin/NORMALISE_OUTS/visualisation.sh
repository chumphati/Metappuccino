#!/bin/bash

#PBS -N visualisation_pbs
#PBS -l walltime=12:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:ncpus=10:mem=16gb

#SBATCH --job-name=visualisation_slurm
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

LOG_DIR=$METAPPUCCINO/$RES/logs
TMP_DIR=$METAPPUCCINO/$RES/tmp
SCRATCH_DIR="/scratchlocal/$USER/${PBS_JOBID:-$SLURM_JOB_ID}"

mkdir -p "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

exec > "$LOG_DIR/visualisation.out" 2> "$LOG_DIR/visualisation.err"

cleanup() {
    cp -r "$SCRATCH_DIR/VISUALISATION" "$METAPPUCCINO/$RES/COMPLETED_INFERENCE/" 2>/dev/null || echo "Output file not found, skipping."
    cp "$SCRATCH_DIR/STEP4_2.flag" "$TMP_DIR/" 2>/dev/null || echo "Flag not found, skipping."
    echo "End date: $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

cp "$METAPPUCCINO/$RES/COMPLETED_INFERENCE/completed_metadata.csv" $SCRATCH_DIR/
cp "$METAPPUCCINO/scripts/normalize_graph/vizualisation_data.py" $SCRATCH_DIR/

source $ENV_REQUIREMENT/bin/activate

echo "Start $(date)"

python3 -u vizualisation_data.py --base_path "$SCRATCH_DIR" --input "$SCRATCH_DIR/completed_metadata.csv" --outdir "$SCRATCH_DIR/VISUALISATION"
