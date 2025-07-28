#!/bin/bash

#PBS -N extract_preprocess_pbs
#PBS -l walltime=12:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:ncpus=10:mem=16gb

#SBATCH --job-name=extract_preprocess_slurm
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

exec > "$LOG_DIR/extract_preprocess.out" 2> "$LOG_DIR/extract_preprocess.err"

cleanup() {
    cp "$SCRATCH_DIR/database_metadata_curated.csv" "$TMP_DIR/" 2>/dev/null || echo "Output file not found, skipping."
    cp "$SCRATCH_DIR/STEP2_1.flag" "$TMP_DIR/" 2>/dev/null || echo "Flag not found, skipping."
    echo "End date: $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

cp "$METAPPUCCINO/data/CELLOSAURUS_CLEAN.csv" $SCRATCH_DIR/
cp "$METAPPUCCINO/data/DOT_TABLE_CLEAN.csv" $SCRATCH_DIR/
cp "$METAPPUCCINO/data/UBERON_TABLE_CLEAN.csv" $SCRATCH_DIR/
cp -r "$TMP_DIR/metadata" $SCRATCH_DIR/
cp "$TMP_DIR/metadata_sra.txt" $SCRATCH_DIR/
cp "$METAPPUCCINO/scripts/get_clean_metadata/fetch_existing_cat.py" $SCRATCH_DIR/

source $ENV_REQUIREMENT/bin/activate

echo "Start $(date)"

python3 -u fetch_existing_cat.py --base_path "$SCRATCH_DIR"
