#!/bin/bash

#PBS -N init
#PBS -l walltime=12:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1

#SBATCH --job-name=init
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

METAPPUCCINO=${1:-$METAPPUCCINO}
RES=${2:-$RES}
NODE_WORK_PATH=${3:-$NODE_WORK_PATH}
ENV_REQUIREMENT=${4:-$ENV_REQUIREMENT}

source $ENV_REQUIREMENT/bin/activate

LOG_DIR=$RES/logs
TMP_DIR=$RES/tmp

#create dir
mkdir -p $RES/logs
mkdir -p $RES/tmp
mkdir -p $RES/ORIGINAL_METADATA
mkdir -p $RES/COMPLETED_INFERENCE/VISUALISATION

if [[ -n "${PBS_JOBID:-}" ]]; then
  SCRATCH_DIR="$NODE_WORK_PATH/${PBS_JOBID}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRATCH_DIR="$NODE_WORK_PATH/${SLURM_JOB_ID}"
else
  SCRATCH_DIR="$TMP_DIR"
fi

mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR
exec >"$LOG_DIR/init.out" 2>"$LOG_DIR/init.err"

cleanup() {
  cp "$SCRATCH_DIR/STEP1_0.flag" "$TMP_DIR/" 2>/dev/null || echo "Flag not found, skipping."
  echo "End $(date)"
  rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

print_metappuccino_logo() {
  echo "=========================================="
  echo "  __  __      _        __  __            "
  echo " |  \/  |    | |      |  \/  |           "
  echo " | \  / | ___| |_ __ _| \  / | __ _ _ __ "
  echo " | |\/| |/ _ \ __/ _ | |\/| |/ _ | '_ \\"
  echo " | |  | |  __/ || (_| | |  | | (_| | |_) |"
  echo " |_|  |_|\___|\__\__,_|_|  |_|\__,_| .__/ "
  echo "                                   | |    "
  echo "                                   |_|    "
  echo
  echo "       Metadata Reconstruction using LLMs"
  echo "       Version 1.0.0                      "
  echo "       License Apache License 2.0         "
  echo "=========================================="
  echo " "
}

print_metappuccino_logo

echo "Beginning of Metappuccino analysis"
echo "Beginning date: $(date)"
echo "Please wait while your data ara analyzed..."

touch "$SCRATCH_DIR/STEP1_0.flag"