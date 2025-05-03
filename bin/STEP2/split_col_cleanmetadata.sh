#!/bin/bash
#SBATCH --job-name=split_col_cleanmetadata
#SBATCH --partition=common
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G

OUTPUT_DIR=${1:-$OUTPUT_DIR}
ENV_REQUIREMENT=${2:-$ENV_REQUIREMENT}

LOG_DIR=$OUTPUT_DIR/results/logs
SCRATCH_DIR="/scratchlocal/$USER/$SLURM_JOB_ID"
TMP_DIR=$OUTPUT_DIR/results/tmp
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

exec > "$LOG_DIR/split_col_cleanmetadata.out" 2> "$LOG_DIR/split_col_cleanmetadata.err"

#clean and copy in case of fail
cleanup() {
    cp $SCRATCH_DIR/tmp/* $TMP_DIR/ 2>/dev/null || echo "Output directory not found, skipping."
    cp $SCRATCH_DIR/STEP2_2.flag $TMP_DIR/ 2>/dev/null || echo "Flag not found, skipping."
    echo "End date: $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

#necessary files
cp "$OUTPUT_DIR/results/METADATA/cleaned_metadata_sra.txt" $SCRATCH_DIR/
cp "$OUTPUT_DIR/scripts/fill_missing_metadata/split_col_cleanmetadata.py" $SCRATCH_DIR/

#activate requirements venv
source $ENV_REQUIREMENT/bin/activate

echo "Begin date: $(date)"

python3 -u $SCRATCH_DIR/split_col_cleanmetadata.py --base_path $SCRATCH_DIR
