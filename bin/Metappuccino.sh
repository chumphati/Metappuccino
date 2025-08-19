#!/bin/bash

#PBS -N Metappuccino
#PBS -l walltime=10000:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:ncpus=1:mem=8gb

#SBATCH --job-name=Metappuccino
#SBATCH --time=10000:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G

METAPPUCCINO_DIR="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino"
RES="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_Llama-3.1-8B-Instruct-original"
ENV_REQUIREMENT="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv"
MODEL="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/gguf/Llama-3.1-8B-Instruct-original.gguf"
#MODEL="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/test5_firstadapters/mistral7B_fine_tuned"

exec > "$RES/logs/Metappuccino.out" 2> "$RES/logs/Metappuccino.err"

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
mkdir -p $RES/logs
mkdir -p $RES/tmp
mkdir -p $RES/ORIGINAL_METADATA
mkdir -p $RES/COMPLETED_INFERENCE/VISUALISATION

#call script that manage the analysis
source "$ENV_REQUIREMENT/bin/activate"
python3 "$METAPPUCCINO_DIR/bin/Metappuccino.py" \
    --metappuccino_dir "$METAPPUCCINO_DIR" \
    --res_dir "$RES" \
    --tmp_keep \
    --env_requirement "$ENV_REQUIREMENT" \
    --working_dir "/scratchlocal/$USER" \
    --model "$MODEL" \
    --getmetadata --fillmetadata --associateinformation --visualisation \
    --iteration_limit 3 --gpus 1 --per_gpu_jobs --verbose \
    --node "node51" --partition "alphafold" \
    >> "$RES/logs/Metappuccino.out" 2>> "$RES/logs/Metappuccino.err"
deactivate

echo "End date: $(date)"
echo "End of Metappuccino analysis! All results are stored in '$RES'"
