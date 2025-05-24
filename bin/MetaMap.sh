#!/bin/bash
#PBS -N MetaMap
#PBS -l walltime=10000:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:ncpus=1:mem=8gb

METAMAP_DIR="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap"
RES="results_llama70b_original"
ENV_REQUIREMENT="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv"
MODEL="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/gguf/Llama-3.1-70B-Instruct-original.gguf"

exec > "$METAMAP_DIR/$RES/logs/MetaMap.out" 2> "$METAMAP_DIR/$RES/logs/MetaMap.err"

print_metamap_logo() {
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

print_metamap_logo

echo "✨Beginning of MetaMap analysis✨"
echo "Beginning date: $(date)"
echo "🔄 Please wait while we analyze your data..."

#create dir
mkdir -p $METAMAP_DIR/$RES/logs
mkdir -p $METAMAP_DIR/$RES/tmp/high_entropy
mkdir -p $METAMAP_DIR/$RES/METADATA
mkdir -p $METAMAP_DIR/$RES/SPECIFIC_RUN_ANALYSIS

#call script that manage the analysis
source "$ENV_REQUIREMENT/bin/activate"
python3 "$METAMAP_DIR/bin/MetaMap.py" \
    --metamap_dir "$METAMAP_DIR" \
    --res_dir "$RES" \
    --env_requirement "$ENV_REQUIREMENT" \
    --model "$MODEL" \
    --getmetadata --fillmetadata --associateinformation --completestudy \
    >> "$METAMAP_DIR/$RES/logs/MetaMap.out" 2>> "$METAMAP_DIR/$RES/logs/MetaMap.err"
deactivate

echo "End date: $(date)"
echo "✨End of MetaMap analysis! All results are stored in '$METAMAP_DIR/$RES'✨"
