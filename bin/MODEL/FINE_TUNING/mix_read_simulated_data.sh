#!/bin/bash
#SBATCH --job-name=mix_sim_data
#SBATCH --partition=common
#SBATCH --time=10:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=16G

METAMAP="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap"
ENV="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/venv"
LOG_DIR="$METAMAP/results/logs"
mkdir -p $SCRATCH_DIR
cd $SCRATCH_DIR

exec > "$LOG_DIR/mix_sim_data.out" 2> "$LOG_DIR/mix_sim_data.err"
echo "Begin job : $(date)"

cleanup() {
    rm -rf "$SCRATCH_DIR"
    echo "End job : $(date)"
}
trap cleanup EXIT

source $ENV/bin/activate

#need clean metadata downloaded
python -u $METAMAP/scripts/model_processing/fine_tuning/generate_fake_data.py
python -u $METAMAP/scripts/model_processing/fine_tuning/prepare_fake_datasets.py
python -u $METAMAP/scripts/model_processing/fine_tuning/get_real_specific_cat.py

cat /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/real_finetune_data.csv \
    /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/simulated_finetune_data.csv \
    > /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/finetune_data.csv

python -u $METAMAP/scripts/model_processing/fine_tuning/clean_real_data.py
