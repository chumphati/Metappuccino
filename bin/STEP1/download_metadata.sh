#!/bin/bash
#PBS -N download_metadata
#PBS -o /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/download_metadata.out
#PBS -e /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/download_metadata.err
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=20:mem=64gb

echo "Begin date: $(date)"
python3 /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/scripts/get_clean_metadata/get_metadata_ncbi_ena.py
echo "End date: $(date)"
