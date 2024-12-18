#!/bin/bash
#PBS -N get_stable_metadata
#PBS -o /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/get_stable_metadata.out
#PBS -e /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/get_stable_metadata.err
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=10:mem=16gb

echo "Begin date: $(date)"
python3 /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/scripts/fill_missing_metadata/get_stable_metadata.py
echo "End date: $(date)"
