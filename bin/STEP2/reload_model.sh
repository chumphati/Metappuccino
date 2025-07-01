#!/bin/bash
#PBS -N reload_context_llm
#PBS -l walltime=500:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:host=node51:ncpus=30:ngpus=1:mem=80gb

METAPPUCCINO=${1:-$METAPPUCCINO}
RES=${2:-$RES}
ENV_REQUIREMENT=${3:-$ENV_REQUIREMENT}
MODEL=${4:-$MODEL}

RESULTS_DIR=$METAPPUCCINO/$RES
TMP_DIR=$RESULTS_DIR/tmp
LOG_DIR=$RESULTS_DIR/logs
exec > "$LOG_DIR/reload_context_llm.out" 2> "$LOG_DIR/reload_context_llm.err"

#SCRATCH_DIR="/scratchlocal/$USER/$SLURM_JOB_ID"
SCRATCH_DIR=/scratchlocal/$USER/$PBS_JOBID
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

cleanup() {
    cp -r $SCRATCH_DIR/INFO_BIO_LLM $RESULTS_DIR/SPECIFIC_RUN_ANALYSIS/ 2>/dev/null || echo "INFO_BIO_LLM not found, skipping."
    cp $SCRATCH_DIR/llm_log_reload.txt $LOG_DIR/ 2>/dev/null || echo "Log file not found, skipping."
    cp $SCRATCH_DIR/STEP2_4.flag $TMP_DIR/ 2>/dev/null || echo "Flag not found, skipping."
    echo "End date: $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

if [ ! -f "$TMP_DIR/reload_model_bio_info.txt" ] && [ ! -f "$TMP_DIR/context_model_bio_info.txt" ]; then
    echo "✔ No file $TMP_DIR/reload_model_bio_info.txt to analyse."
    echo "✔ No file $TMP_DIR/context_model_bio_info.txt to analyse."
    touch "$SCRATCH_DIR/STEP2_4.flag"
    exit 0
fi

cp $MODEL $SCRATCH_DIR/
cp $TMP_DIR/reload_model_bio_info.txt $SCRATCH_DIR/
cp $TMP_DIR/context_model_bio_info.txt $SCRATCH_DIR/
cp $TMP_DIR/initial_raw_metadata.txt $SCRATCH_DIR/
cp $METAPPUCCINO/scripts/fill_missing_metadata/get_biology_information_LLM.py $SCRATCH_DIR/
source $ENV_REQUIREMENT/bin/activate

echo "Begin date: $(date)"

iteration_limit=2
for (( i=0; i<$iteration_limit; i++ )); do

    if [ -f "$TMP_DIR/reload_model_bio_info.txt" ]; then
      python3 -u $SCRATCH_DIR/get_biology_information_LLM.py --base_path $SCRATCH_DIR \
          --input_metadata_path $SCRATCH_DIR/reload_model_bio_info.txt \
          --context_file_path $SCRATCH_DIR/context_model_bio_info.txt \
          --error_file_path $SCRATCH_DIR/reload_model_bio_info_bis.txt \
          --log_file_path $SCRATCH_DIR/llm_log_reload.txt \
          --flag_file $SCRATCH_DIR/STEP2_4.flag \
          --initial_n_ctx 2500 \
          --model "$SCRATCH_DIR/$(basename "$MODEL")"
    fi

    if [ -f "$TMP_DIR/context_model_bio_info.txt" ]; then
      python3 -u $SCRATCH_DIR/get_biology_information_LLM.py --base_path $SCRATCH_DIR \
          --input_metadata_path $SCRATCH_DIR/context_model_bio_info.txt \
          --error_file_path $SCRATCH_DIR/context_model_bio_info_bis.txt \
          --context_file_path $SCRATCH_DIR/context_model_bio_info.txt \
          --log_file_path $SCRATCH_DIR/llm_log_reload.txt \
          --flag_file $SCRATCH_DIR/STEP2_4.flag \
          --initial_n_ctx $(( 2500 + 10000 * i )) \
          --model "$SCRATCH_DIR/$(basename "$MODEL")"
    fi

    if [ ! -f "$SCRATCH_DIR/reload_model_bio_info.txt" ] && [ ! -f "$SCRATCH_DIR/context_model_bio_info.txt" ]; then
        echo "All inferences completed.."
        touch "$SCRATCH_DIR/STEP2_4.flag"
        break
    fi

    mv $SCRATCH_DIR/reload_model_bio_info_bis.txt $SCRATCH_DIR/reload_model_bio_info.txt
done

