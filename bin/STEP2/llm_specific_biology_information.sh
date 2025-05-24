#!/bin/bash
#PBS -N llm_specific_biology_information
#PBS -l walltime=500:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:host=node51:ncpus=30:ngpus=1:mem=80gb

METAMAP=${1:-$METAMAP}
RES=${2:-$RES}
ENV_REQUIREMENT=${3:-$ENV_REQUIREMENT}
MODEL=${4:-$ENV_REQUIREMENT}
#METAMAP='/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap'
#ENV_REQUIREMENT='/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv'

RESULTS_DIR=$METAMAP/$RES
TMP_DIR=$RESULTS_DIR/tmp
LOG_DIR=$RESULTS_DIR/logs

exec > "$LOG_DIR/llm_specific_biology_information.out" 2> "$LOG_DIR/llm_specific_biology_information.err"

SCRATCH_DIR=/scratchlocal/$USER/$PBS_JOBID
#SCRATCH_DIR="/scratchlocal/$USER/$SLURM_JOB_ID"
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

#clean and copy in case of fail
cleanup() {
    cp -r $SCRATCH_DIR/INFO_BIO_LLM $RESULTS_DIR/SPECIFIC_RUN_ANALYSIS/ 2>/dev/null || echo "INFO_BIO_LLM not found, skipping."
    cp $SCRATCH_DIR/reload_model_bio_info.txt $TMP_DIR/ 2>/dev/null || echo "No additional file to process."
    cp $SCRATCH_DIR/context_model_bio_info.txt $TMP_DIR/ 2>/dev/null || echo "No additional context to increase."
    cp $SCRATCH_DIR/llm_log_SB.txt $LOG_DIR/ 2>/dev/null || echo "Log file not found, skipping."
    cp $SCRATCH_DIR/STEP2_3.flag $TMP_DIR/ 2>/dev/null || echo "Flag not found, skipping."
    echo "End date: $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

#necessary files
cp $MODEL $SCRATCH_DIR/
cp $TMP_DIR/sample_info.txt $SCRATCH_DIR/
cp $TMP_DIR/initial_raw_metadata.txt $SCRATCH_DIR/
cp $METAMAP/scripts/fill_missing_metadata/get_biology_information_LLM.py $SCRATCH_DIR/

#activate requirements venv
source $ENV_REQUIREMENT/bin/activate

echo "Begin date: $(date)"

python3 -u $SCRATCH_DIR/get_biology_information_LLM.py --base_path $SCRATCH_DIR --input_metadata_path $TMP_DIR/sample_info.txt --context_file_path $SCRATCH_DIR/context_model_bio_info.txt --error_file_path $SCRATCH_DIR/reload_model_bio_info.txt --log_file_path $SCRATCH_DIR/llm_log_SB.txt --flag_file $SCRATCH_DIR/STEP2_3.flag --initial_n_ctx 2500 --model "$SCRATCH_DIR/$(basename "$MODEL")"

