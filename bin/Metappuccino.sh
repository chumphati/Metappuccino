#!/bin/bash

#PBS -N Metappuccino
#PBS -l walltime=10000:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:ncpus=1:mem=8gb

#SBATCH --job-name=Metappuccino
#SBATCH --partition=common
#SBATCH --time=10000:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

METAPPUCCINO_DIR="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino"
RES="results"
ENV_REQUIREMENT="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv"
MODEL="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/4bits_quantified/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf"

exec > "$METAPPUCCINO_DIR/$RES/logs/Metappuccino.out" 2> "$METAPPUCCINO_DIR/$RES/logs/Metappuccino.err"

print_metappuccino_logo() {
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

print_metappuccino_logo

echo "Beginning of Metappuccino analysis"
echo "Beginning date: $(date)"
echo "Please wait while your data ara analyzed..."

#create dir
mkdir -p $METAPPUCCINO_DIR/$RES/logs
mkdir -p $METAPPUCCINO_DIR/$RES/tmp
mkdir -p $METAPPUCCINO_DIR/$RES/ORIGINAL_METADATA
mkdir -p $METAPPUCCINO_DIR/$RES/COMPLETED_INFERENCE/VISUALISATION

#call script that manage the analysis
source "$ENV_REQUIREMENT/bin/activate"
python3 "$METAPPUCCINO_DIR/bin/Metappuccino.py" \
    --metappuccino_dir "$METAPPUCCINO_DIR" \
    --res_dir "$RES" \
    --env_requirement "$ENV_REQUIREMENT" \
    --model "$MODEL" \
    --getmetadata --fillmetadata --associateinformation --visualisation \
    --iteration_limit 1 \
    >> "$METAPPUCCINO_DIR/$RES/logs/Metappuccino.out" 2>> "$METAPPUCCINO_DIR/$RES/logs/Metappuccino.err"
deactivate

echo "End date: $(date)"
echo "End of Metappuccino analysis! All results are stored in '$METAPPUCCINO_DIR/$RES'"
