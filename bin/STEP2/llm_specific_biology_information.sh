#!/bin/bash
#PBS -N llm_specific_biology_information
#PBS -o /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/llm_specific_biology_information.out
#PBS -e /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/llm_specific_biology_information.err
#PBS -l walltime=1000:00:00
#PBS -l select=1:host=node51:ncpus=30:ngpus=2:mem=80gb

SCRATCH_DIR=/scratchlocal/$USER/$PBS_JOBID
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR
source /store/EQUIPES/SSFA/MEMBERS/fiona.hak/llama_env/bin/activate
echo "Begin date: $(date)"

python3 -u /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/scripts/fill_missing_metadata/get_biology_information_LLM.py

echo "End date: $(date)"
rm -rf "$SCRATCH_DIR"
