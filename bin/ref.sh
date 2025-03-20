#!/bin/bash
#PBS -N ref
#PBS -l walltime=12:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:ncpus=30:mem=46gb

METAMAP="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap"
ENV_REQUIREMENT="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv"

SCRATCH_DIR=/scratchlocal/$USER/$PBS_JOBID
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
