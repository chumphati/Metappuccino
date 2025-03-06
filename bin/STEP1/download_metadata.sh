#!/bin/bash
#PBS -N download_metadata
#PBS -l walltime=12:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:ncpus=10:mem=16gb

METAMAP=${1:-$METAMAP}
ENV_REQUIREMENT=${2:-$ENV_REQUIREMENT}

#METAMAP="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap"
#ENV_REQUIREMENT="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv"

LOG_DIR=$METAMAP/results/logs
TMP_DIR=$METAMAP/results/tmp

SCRATCH_DIR=/scratchlocal/$USER/$PBS_JOBID
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

exec > "$LOG_DIR/download_metadata.out" 2> "$LOG_DIR/download_metadata.err"

#clean and copy in case of fail
cleanup() {
    cp $SCRATCH_DIR/runs.tsv $METAMAP/results/METADATA 2>/dev/null || echo "run file not found, skipping."
    cp -r $SCRATCH_DIR/metadata $TMP_DIR/ 2>/dev/null || echo "Metadata extraction directory not found, skipping."
    cp $SCRATCH_DIR/metadata_sra.txt $TMP_DIR/ 2>/dev/null || echo "Final extraction file not found, skipping."
    cp $SCRATCH_DIR/STEP1_1.flag $TMP_DIR/ 2>/dev/null || echo "Flag not found, skipping."
    echo "End date: $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

#necessary files
#cp "$METAMAP/data/raw/mela-select.tsv" $SCRATCH_DIR/
cp "$METAMAP/data/raw/annotated_totalRNA.csv" $SCRATCH_DIR/
cp "$METAMAP/scripts/get_clean_metadata/get_metadata_ncbi_ena.py" $SCRATCH_DIR/

#cp "$METAMAP/results/METADATA/runs.tsv" $SCRATCH_DIR/
#cp -r "$METAMAP/results/tmp/metadata" $SCRATCH_DIR/

#activate requirements venv
source $ENV_REQUIREMENT/bin/activate

echo "Begin date: $(date)"

python3 -u $SCRATCH_DIR/get_metadata_ncbi_ena.py --base_path $SCRATCH_DIR
