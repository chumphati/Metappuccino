#!/bin/bash
#PBS -N associate_codes_clean
#PBS -l walltime=12:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:ncpus=10:mem=16gb

#METAMAP='/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap'
#ENV_REQUIREMENT='/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv'
METAMAP=${1:-$METAMAP}
RES=${2:-$RES}
ENV_REQUIREMENT=${3:-$ENV_REQUIREMENT}
LOG_DIR=$METAMAP/$RES/logs
TMP_DIR=$METAMAP/$RES/tmp
FINAL_DIR=$METAMAP/$RES/SPECIFIC_RUN_ANALYSIS

#SCRATCH_DIR="/scratchlocal/$USER/$SLURM_JOB_ID"
SCRATCH_DIR=/scratchlocal/$USER/$PBS_JOBID
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

exec > "$LOG_DIR/associate_codes_clean.out" 2> "$LOG_DIR/associate_codes_clean.err"

#clean and copy in case of fail
cleanup() {
    cp $SCRATCH_DIR/raw_final_info.txt $TMP_DIR/ 2>/dev/null || echo "Raw LLM file not found, skipping."
    cp $SCRATCH_DIR/*_high_entropy.txt $TMP_DIR/high_entropy 2>/dev/null || echo "High entropy files not found, skipping."
    cp $SCRATCH_DIR/tmp_final_llm_sample_analysis.csv $FINAL_DIR/final_llm_sample_analysis.csv 2>/dev/null || echo "Final LLM file not found, skipping."
    cp $SCRATCH_DIR/STEP3.flag $TMP_DIR/ 2>/dev/null || echo "Flag not found, skipping."
    echo "End date: $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

#necessary files
cp -r "$METAMAP/$RES/SPECIFIC_RUN_ANALYSIS/INFO_BIO_LLM/" $SCRATCH_DIR/
cp "$METAMAP/data/UBERON_TABLE_CLEAN.csv" $SCRATCH_DIR/
cp "$METAMAP/data/DOT_TABLE_CLEAN.csv" $SCRATCH_DIR/
cp "$METAMAP/data/CELLOSAURUS_CLEAN.csv" $SCRATCH_DIR/
cp "$TMP_DIR/raw_final_info.txt" $SCRATCH_DIR/
cp "$TMP_DIR/initial_raw_metadata.txt" $SCRATCH_DIR/
cp "$METAMAP/scripts/associate_code/associate_uberon_reftable.py" $SCRATCH_DIR/
cp "$METAMAP/scripts/associate_code/associate_dot_reftable.py" $SCRATCH_DIR/
cp "$METAMAP/scripts/associate_code/associate_stable_info.py" $SCRATCH_DIR/
cp "$METAMAP/scripts/associate_code/clean_raw_info.py" $SCRATCH_DIR/
cp "$METAMAP/scripts/associate_code/unif_cell_line.py" $SCRATCH_DIR/

#activate requirements venv
source $ENV_REQUIREMENT/bin/activate

echo "Begin date: $(date)"

sed -i -e 's/[*"\x27]//g' $SCRATCH_DIR/INFO_BIO_LLM/*

python3 -u $SCRATCH_DIR/associate_uberon_reftable.py --base_path $SCRATCH_DIR
python3 -u $SCRATCH_DIR/associate_dot_reftable.py --base_path $SCRATCH_DIR
python3 -u $SCRATCH_DIR/associate_stable_info.py --base_path $SCRATCH_DIR
python3 -u $SCRATCH_DIR/clean_raw_info.py --base_path $SCRATCH_DIR
python3 -u $SCRATCH_DIR/unif_cell_line.py --base_path $SCRATCH_DIR
