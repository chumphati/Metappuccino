#!/bin/bash
#PBS -N MetaMap
#PBS -o /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/MetaMap.out
#PBS -e /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/MetaMap.err
#PBS -l walltime=9999:00:00
#PBS -l select=1:ncpus=10:mem=16gb

METAMAP_DIR="/Users/fionahak/Documents/phd/phd_code/MetaMap"
ENV_REQUIREMENT="/Users/fionahak/Documents/phd/phd_code/.venv"

print_metamap_logo() {
  echo # ==========================================
  echo #  __  __      _        __  __
  echo # |  \/  |    | |      |  \/  |
  echo # | \  / | ___| |_ __ _| \  / | __ _ _ __
  echo # | |\/| |/ _ \ __/ _` | |\/| |/ _` | '_ \
  echo # | |  | |  __/ || (_| | |  | | (_| | |_) |
  echo # |_|  |_|\___|\__\__,_|_|  |_|\__,_| .__/
  echo #                                   | |
  echo #                                   |_|
  echo #
  echo #       Metadata Reconstruction using LLMs
  echo #       Version 1.0.0
  echo #       License Apache License 2.0
  echo # ==========================================
}

print_metamap_logo

echo "✨ Beginning of MetaMap analysis✨"
echo "Beginning date: $(date)"
echo "🔄 Please wait while we analyze your data..."

#call script that manage the analysis
python3 "$METAMAP_DIR/bin/MetaMap.py" \
    --metamap_dir "$METAMAP_DIR" \
    --env_requirement "$ENV_REQUIREMENT" \
    --getmetadata --fillmetadata \
    >> "$METAMAP_DIR/results/logs/MetaMap.out" 2>> "$METAMAP_DIR/results/logs/MetaMap.err"

echo "End date: $(date)"