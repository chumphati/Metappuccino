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

########################################################################################################################
#PARAMETERS

##REQUIRED
SAMPLE_INPUT="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_tests/run.txt"
METAPPUCCINO_DIR="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino"
RES_DIR="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_tests"
ENV_REQUIREMENT="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv"
WORKING_DIR="/scratchlocal/$USER"
MODEL_PATH="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/4bits_quantified/Mistral7B-Instruct-ft-Q4_K_M.gguf"
NODE_NAME="node51"
PARTITION="alphafold"

##OPTIONAL
ITERATION_LIMIT=3
NGPUS=1
NCPUS=20
MEM="50gb"
LOGAN_PATH=""
CUDA_PATH="/usr/local/cuda"

##STEPS
VERBOSE=TRUE
TMP_KEEP=TRUE
INSTALL_REQUIREMENTS=FALSE
GET_METADATA=TRUE
FILL_METADATA=TRUE
GGUF=TRUE
ASSOCIATE_INFORMATION=TRUE
VISUALISATION=TRUE
LOCAL=FALSE

########################################################################################################################

exec > "$RES_DIR/logs/Metappuccino.out" 2> "$RES_DIR/logs/Metappuccino.err"
source "$ENV_REQUIREMENT/bin/activate"

ARGS=()
[[ "$VERBOSE" == "TRUE" ]] && ARGS+=("--verbose")
[[ "$TMP_KEEP" == "TRUE" ]] && ARGS+=("--tmp_keep")
[[ "$INSTALL_REQUIREMENTS" == "TRUE" ]] && ARGS+=("--requirements")
[[ "$GET_METADATA" == "TRUE" ]] && ARGS+=("--getmetadata")
[[ "$FILL_METADATA" == "TRUE" ]] && ARGS+=("--fillmetadata")
[[ "$ASSOCIATE_INFORMATION" == "TRUE" ]] && ARGS+=("--associateinformation")
[[ "$VISUALISATION" == "TRUE" ]] && ARGS+=("--visualisation")
[[ "$LOCAL" == "TRUE" ]] && ARGS+=("--local")
[[ "$GGUF" == "TRUE" ]] && ARGS+=("--gguf")

python3 "$METAPPUCCINO_DIR/bin/Metappuccino.py" \
    --metappuccino_dir "$METAPPUCCINO_DIR" \
    --res_dir "$RES_DIR" \
    --sample_input "$SAMPLE_INPUT" \
    --env_requirement "$ENV_REQUIREMENT" \
    --working_dir "$WORKING_DIR" \
    --model "$MODEL_PATH" \
    --node "$NODE_NAME" --partition "$PARTITION" \
    --iteration_limit "$ITERATION_LIMIT" --gpus "$NGPUS" --cpus "$NCPUS" --mem "$MEM" --per_gpu_jobs \
    --logan_path "$LOGAN_PATH" --cuda "$CUDA_PATH" \
    "${ARGS[@]}" \
    >> "$RES_DIR/Metappuccino.out" 2>> "$RES_DIR/Metappuccino.err"

deactivate

mv "$RES_DIR/Metappuccino.out" "$RES_DIR/Metappuccino.err" "$RES_DIR/logs"