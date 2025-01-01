#!/bin/bash
#PBS -N install_requirements
#PBS -o /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/install_requirements.out
#PBS -e /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/install_requirements.err
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=12:mem=64gb

echo "Begin date: $(date)"

python3 -m pip install -r /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/requirements.txt

#cd /store/EQUIPES/SSFA/MEMBERS/fiona.hak/llama-cpp-python
#git submodule update --init --recursive
#export LLAMA_CPP_CMAKE_ARGS="-DCMAKE_CXX_FLAGS='-O3 -mavx2 -mfma'"
#pip install --force-reinstall --no-binary /store/EQUIPES/SSFA/MEMBERS/fiona.hak/llama-cpp-python .

echo "End date: $(date)"
