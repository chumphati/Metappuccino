#!/bin/bash
#PBS -N split_col_cleanmetadata
#PBS -l walltime=12:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:ncpus=10:mem=16gb

OUTPUT_DIR=${1:-$OUTPUT_DIR}
ENV_REQUIREMENT=${2:-$ENV_REQUIREMENT}

LOG_DIR=$OUTPUT_DIR/results_mistral7B_original/logs
#SCRATCH_DIR="/scratchlocal/$USER/$SLURM_JOB_ID"
SCRATCH_DIR=/scratchlocal/$USER/$PBS_JOBID
TMP_DIR=$OUTPUT_DIR/results_mistral7B_original/tmp
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
cp "$OUTPUT_DIR/results_mistral7B_original/METADATA/cleaned_metadata_sra.txt" $SCRATCH_DIR/
cp "$OUTPUT_DIR/scripts/fill_missing_metadata/split_col_cleanmetadata.py" $SCRATCH_DIR/

#activate requirements venv
source $ENV_REQUIREMENT/bin/activate

echo "Begin date: $(date)"

python3 -u $SCRATCH_DIR/split_col_cleanmetadata.py --base_path $SCRATCH_DIR
