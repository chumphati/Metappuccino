#!/bin/bash

#PBS -N download_metadata_pbs
#PBS -l walltime=12:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:ncpus=10:mem=16gb

#SBATCH --job-name=download_metadata_slurm
#SBATCH --partition=common
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --mem=16G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

METAPPUCCINO=${1:-$METAPPUCCINO}
RES=${2:-$RES}
ENV_REQUIREMENT=${3:-$ENV_REQUIREMENT}

LOG_DIR=$METAPPUCCINO/$RES/logs
TMP_DIR=$METAPPUCCINO/$RES/tmp
SCRATCH_DIR="/scratchlocal/$USER/${PBS_JOBID:-$SLURM_JOB_ID}"

mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR
exec >"$LOG_DIR/download_metadata.out" 2>"$LOG_DIR/download_metadata.err"

cleanup() {
  cp -r "$SCRATCH_DIR/metadata" "$TMP_DIR/" 2>/dev/null || echo "No metadata"
  cp "$SCRATCH_DIR/metadata_sra.txt" "$TMP_DIR/" 2>/dev/null || echo "No metadata_sra.txt"
  cp "$SCRATCH_DIR/STEP1_1.flag" "$TMP_DIR/" 2>/dev/null || echo "Flag not found, skipping."
  echo "End $(date)"
  rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

cp "$METAPPUCCINO/scripts/get_clean_metadata/get_metadata_ncbi_ena.py" $SCRATCH_DIR/
cp "$METAPPUCCINO/$RES/runs.txt" $SCRATCH_DIR/

source $ENV_REQUIREMENT/bin/activate
echo "Start $(date)"
python3 -u get_metadata_ncbi_ena.py --base_path $SCRATCH_DIR
