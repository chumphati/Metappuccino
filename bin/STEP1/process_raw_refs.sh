#!/bin/bash
#SBATCH --job-name=ref_process
#SBATCH --partition=common
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=46G

METAMAP="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap"
ENV_REQUIREMENT="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv"

SCRATCH_DIR="/scratchlocal/$USER/$SLURM_JOB_ID"
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

exec > "$METAMAP/results/logs/ref.out" 2> "$METAMAP/results/logs/ref.err"

#clean and copy in case of fail
cleanup() {
    echo "End date: $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

#activate requirements venv
source $ENV_REQUIREMENT/bin/activate

echo "Begin date: $(date)"

bash /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/scripts/reference_clean/clean_cellosaurus.sh
