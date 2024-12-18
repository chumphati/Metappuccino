#!/bin/bash
#PBS -N install_requirements
#PBS -o /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/install_requirements.out
#PBS -e /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/install_requirements.err
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=10:mem=64gb

echo "Begin date: $(date)"
python3 -m pip install -r /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/requirements.txt
echo "End date: $(date)"
