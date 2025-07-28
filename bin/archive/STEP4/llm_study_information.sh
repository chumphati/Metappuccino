#!/bin/bash
#SBATCH --job-name=llm_study_information
#SBATCH --partition=alphafold
#SBATCH --time=1000:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --nodelist=node49
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:1
#SBATCH --mem=100G

METAPPUCCINO=${1:-$METAPPUCCINO}
RES=${2:-$RES}
ENV_REQUIREMENT=${3:-$ENV_REQUIREMENT}
MODEL=${4:-$MODEL}

#METAPPUCCINO="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino"

RESULTS_DIR=$METAPPUCCINO/$RES
TMP_DIR=$RESULTS_DIR/tmp
LOG_DIR=$RESULTS_DIR/logs

exec > "$LOG_DIR/llm_study_information.out" 2> "$LOG_DIR/llm_study_information.err"

SCRATCH_DIR="/scratchlocal/$USER/$SLURM_JOB_ID"
#SCRATCH_DIR=/scratchlocal/$USER/$PBS_JOBID
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

#clean and copy in case of fail
cleanup() {
    cp -r $SCRATCH_DIR/INFO_STUDY_LLM $RESULTS_DIR/SPECIFIC_RUN_ANALYSIS/ 2>/dev/null || echo "INFO_STUDY_LLM not found, skipping."
    cp $SCRATCH_DIR/reload_model_study_info.txt $TMP_DIR/ 2>/dev/null || echo "No need of more context."
    cp $SCRATCH_DIR/llm_log_study.txt $LOG_DIR/ 2>/dev/null || echo "Log file not found, skipping."
    cp $SCRATCH_DIR/STEP4_1.flag $TMP_DIR/ 2>/dev/null || echo "Flag not found, skipping."
    echo "End date: $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

#necessary files
cp -r $METAPPUCCINO/$RES/SPECIFIC_RUN_ANALYSIS/INFO_BIO_LLM $SCRATCH_DIR/
cp $MODEL $SCRATCH_DIR/
cp $TMP_DIR/study_info.txt $SCRATCH_DIR/
cp $RESULTS_DIR/SPECIFIC_RUN_ANALYSIS/final_llm_sample_analysis.csv $SCRATCH_DIR/
cp $METAPPUCCINO/scripts/fill_missing_metadata/similar_study_accession_process.py $SCRATCH_DIR/

echo "Begin date: $(date)"

python3 -u $SCRATCH_DIR/similar_study_accession_process.py --base_path $SCRATCH_DIR --model "$SCRATCH_DIR/$(basename "$MODEL")"
