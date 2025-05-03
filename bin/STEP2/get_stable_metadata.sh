#!/bin/bash
#SBATCH --job-name=get_stable_metadata
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

SCRATCH_DIR="/scratchlocal/$USER/$SLURM_JOB_ID"
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

exec > "$LOG_DIR/get_stable_metadata.out" 2> "$LOG_DIR/get_stable_metadata.err"

#clean and copy in case of fail
cleanup() {
    cp $SCRATCH_DIR/raw_final_info.txt $TMP_DIR/ 2>/dev/null || echo "Output file not found, skipping."
    cp $SCRATCH_DIR/initial_raw_metadata.txt $TMP_DIR/ 2>/dev/null || echo "Output file not found, skipping."
    cp $SCRATCH_DIR/STEP2_1.flag $TMP_DIR/ 2>/dev/null || echo "Flag not found, skipping."
    echo "End date: $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

#necessary files
cp "$METAMAP/results/METADATA/cleaned_metadata_sra.txt" $SCRATCH_DIR/
cp "$METAMAP/scripts/fill_missing_metadata/get_initial_raw_metadata.py" $SCRATCH_DIR/
cp "$METAMAP/scripts/fill_missing_metadata/get_stable_metadata.py" $SCRATCH_DIR/

#activate requirements venv
source $ENV_REQUIREMENT/bin/activate

echo "Begin date: $(date)"

python3 -u $SCRATCH_DIR/get_initial_raw_metadata.py --base_path $SCRATCH_DIR
python3 -u $SCRATCH_DIR/get_stable_metadata.py --base_path $SCRATCH_DIR
