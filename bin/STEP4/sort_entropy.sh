#!/bin/bash
#SBATCH --job-name=process_study_llm
#SBATCH --partition=common
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G

METAMAP=${1:-$METAMAP}
ENV_REQUIREMENT=${2:-$ENV_REQUIREMENT}
LOG_DIR=$METAMAP/results/logs
TMP_DIR=$METAMAP/results/tmp
FINAL_DIR=$METAMAP/results/SPECIFIC_RUN_ANALYSIS

SCRATCH_DIR="/scratchlocal/$USER/$SLURM_JOB_ID"
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

exec > "$LOG_DIR/process_study_llm.out" 2> "$LOG_DIR/process_study_llm.err"

#clean and copy in case of fail
cleanup() {
    cp $SCRATCH_DIR/best_inferences_per_run.csv $TMP_DIR/ 2>/dev/null || echo "Study inference not found, skipping."
    cp $SCRATCH_DIR/tmp_final_llm_sample_analysis.csv $FINAL_DIR/final_llm_sample_analysis.csv 2>/dev/null || echo "Final LLM file not found, skipping."
    cp $SCRATCH_DIR/STEP4_2.flag $TMP_DIR/ 2>/dev/null || echo "Flag not found, skipping."
    echo "End date: $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

#necessary files
cp -r "$METAMAP/results/SPECIFIC_RUN_ANALYSIS/INFO_STUDY_LLM/" $SCRATCH_DIR/
cp -r "$TMP_DIR/high_entropy/" $SCRATCH_DIR/
cp "$METAMAP/data/UBERON_TABLE_CLEAN.csv" $SCRATCH_DIR/
cp "$METAMAP/data/DOT_TABLE_CLEAN.csv" $SCRATCH_DIR/
cp "$METAMAP/results/SPECIFIC_RUN_ANALYSIS/final_llm_sample_analysis.csv" $SCRATCH_DIR/
cp "$TMP_DIR/raw_final_info.txt" $SCRATCH_DIR/
cp "$TMP_DIR/study_info.txt" $SCRATCH_DIR/
cp "$METAMAP/results/METADATA/cleaned_metadata_sra.txt" $SCRATCH_DIR/

cp "$METAMAP/scripts/associate_code/study_llm_process.py" $SCRATCH_DIR/
cp "$METAMAP/scripts/associate_code/fill_study_blanks.py" $SCRATCH_DIR/
cp "$METAMAP/scripts/reference_clean/clean_final_tab.py" $SCRATCH_DIR/

#activate requirements venv
source $ENV_REQUIREMENT/bin/activate

echo "Begin date: $(date)"

python3 -u $SCRATCH_DIR/study_llm_process.py --base_path $SCRATCH_DIR && \
python3 -u $SCRATCH_DIR/fill_study_blanks.py --base_path $SCRATCH_DIR && \
python3 -u $SCRATCH_DIR/clean_final_tab.py --base_path $SCRATCH_DIR
