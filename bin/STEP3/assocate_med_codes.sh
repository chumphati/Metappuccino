#!/bin/bash
#PBS -N assocate_med_codes
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=10:mem=16gb

METAMAP=$1
ENV_REQUIREMENT=$2
LOG_DIR=$METAMAP/results/logs
TMP_DIR=$METAMAP/results/tmp

SCRATCH_DIR=/scratchlocal/$USER/$PBS_JOBID
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

exec > "$LOG_DIR/assocate_med_codes.out" 2> "$LOG_DIR/assocate_med_codes.err"

#clean and copy in case of fail
cleanup() {
    cp $SCRATCH_DIR/RAW_FINAL_INFO.txt $TMP_DIR/ 2>/dev/null || echo "Metrics file not found, skipping."
    cp $SCRATCH_DIR/STEP3_1.flag $TMP_DIR/ 2>/dev/null || echo "Flag not found, skipping."
    echo "End date: $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

#necessary files
cp -r "$METAMAP/results/SPECIFIC_RUN_ANALYSIS/INFO_BIO_LLM/" $SCRATCH_DIR/
cp "$METAMAP/data/UBERON_TABLE_CLEAN.csv" $SCRATCH_DIR/
cp "$TMP_DIR/RAW_FINAL_INFO.txt" $SCRATCH_DIR/
cp "$METAMAP/scripts/associate_code/associate_uberon_reftable.py" $SCRATCH_DIR/

#activate requirements venv
source $ENV_REQUIREMENT/bin/activate

echo "Begin date: $(date)"

python3 -u $SCRATCH_DIR/associate_uberon_reftable.py --base_path $SCRATCH_DIR
