#!/bin/bash
#PBS -N associate_codes_clean
#PBS -l walltime=12:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:ncpus=10:mem=16gb

#METAMAP='/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap'
#ENV_REQUIREMENT='/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv'
METAMAP=${1:-$METAMAP}
ENV_REQUIREMENT=${2:-$ENV_REQUIREMENT}
LOG_DIR=$METAMAP/results/logs
TMP_DIR=$METAMAP/results/tmp
FINAL_DIR=$METAMAP/results/SPECIFIC_RUN_ANALYSIS

SCRATCH_DIR=/scratchlocal/$USER/$PBS_JOBID
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

exec > "$LOG_DIR/associate_codes_clean.out" 2> "$LOG_DIR/associate_codes_clean.err"

#clean and copy in case of fail
cleanup() {
    cp $SCRATCH_DIR/raw_final_info.txt $TMP_DIR/ 2>/dev/null || echo "Raw LLM file not found, skipping."
    cp $SCRATCH_DIR/uberon_high_entropy.txt $SCRATCH_DIR/dot_high_entropy.txt $SCRATCH_DIR/tissue_high_entropy.txt $SCRATCH_DIR/cellline_high_entropy.txt $SCRATCH_DIR/celltype_high_entropy.txt $TMP_DIR/high_entropy 2>/dev/null || echo "High entropy files not found, skipping."
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
cp "$TMP_DIR/initial_raw_metadata.txt" $SCRATCH_DIR/
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
