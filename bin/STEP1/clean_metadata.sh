#!/bin/bash
#PBS -N clean_metadata
#PBS -o /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/clean_metadata.out
#PBS -e /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/clean_metadata.err
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=8:mem=16gb

echo "Begin date: $(date)"
bash /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/scripts/get_clean_metadata/clean_metadata_ncbi.sh
echo "End date: $(date)"
