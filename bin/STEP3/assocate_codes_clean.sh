#!/bin/bash
#PBS -N assocate_codes_clean
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=10:mem=16gb

METAMAP=$1
ENV_REQUIREMENT=$2
LOG_DIR=$METAMAP/results/logs
TMP_DIR=$METAMAP/results/tmp
FINAL_DIR=$METAMAP/results/SPECIFIC_RUN_ANALYSIS

SCRATCH_DIR=/scratchlocal/$USER/$PBS_JOBID
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

exec > "$LOG_DIR/assocate_codes_clean.out" 2> "$LOG_DIR/assocate_codes_clean.err"

#clean and copy in case of fail
cleanup() {
    cp $SCRATCH_DIR/raw_final_info.txt $TMP_DIR/ 2>/dev/null || echo "Raw LLM file not found, skipping."
    cp $SCRATCH_DIR/final_llm_sample_analysis.csv $FINAL_DIR/ 2>/dev/null || echo "Final LLM file not found, skipping."
    cp $SCRATCH_DIR/STEP3.flag $TMP_DIR/ 2>/dev/null || echo "Flag not found, skipping."
    echo "End date: $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

#necessary files
cp -r "$METAMAP/results/SPECIFIC_RUN_ANALYSIS/INFO_BIO_LLM/" $SCRATCH_DIR/
cp "$METAMAP/data/UBERON_TABLE_CLEAN.csv" $SCRATCH_DIR/
cp "$METAMAP/data/DOT_TABLE_CLEAN.csv" $SCRATCH_DIR/
cp "$TMP_DIR/raw_final_info.txt" $SCRATCH_DIR/
cp "$METAMAP/scripts/associate_code/associate_uberon_reftable.py" $SCRATCH_DIR/
cp "$METAMAP/scripts/associate_code/associate_dot_reftable.py" $SCRATCH_DIR/
cp "$METAMAP/scripts/associate_code/associate_stable_info.py" $SCRATCH_DIR/
cp "$METAMAP/scripts/associate_code/clean_raw_info.py" $SCRATCH_DIR/

#activate requirements venv
source $ENV_REQUIREMENT/bin/activate

echo "Begin date: $(date)"

python3 -u $SCRATCH_DIR/associate_uberon_reftable.py --base_path $SCRATCH_DIR
python3 -u $SCRATCH_DIR/associate_dot_reftable.py --base_path $SCRATCH_DIR
python3 -u $SCRATCH_DIR/associate_stable_info.py --base_path $SCRATCH_DIR
python3 -u $SCRATCH_DIR/clean_raw_info.py --base_path $SCRATCH_DIR
