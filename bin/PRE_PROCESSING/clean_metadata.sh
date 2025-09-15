#!/bin/bash

#PBS -N clean_metadata
#PBS -l walltime=12:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1

#SBATCH --job-name=clean_metadata
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

METAPPUCCINO=${1:-$METAPPUCCINO}
RES=${2:-$RES}
NODE_WORK_PATH=${3:-${NODE_WORK_PATH:-}}
ENV_REQUIREMENT=${4:-$ENV_REQUIREMENT}

LOG_DIR=$RES/logs
TMP_DIR=$RES/tmp
mkdir -p "$LOG_DIR" "$TMP_DIR"

source "$ENV_REQUIREMENT/bin/activate" || true

if [[ -n "${PBS_JOBID:-}" ]]; then
  SCRATCH_DIR="$NODE_WORK_PATH/${PBS_JOBID}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRATCH_DIR="$NODE_WORK_PATH/${SLURM_JOB_ID}"
else
  SCRATCH_DIR="$(mktemp -d -p "$TMP_DIR" clean_metadata.XXXXX)"
fi

mkdir -p "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

exec >"$LOG_DIR/clean_metadata.out" 2>"$LOG_DIR/clean_metadata.err"

cleanup() {
    cp "$SCRATCH_DIR/cleaned_metadata_sra.txt" "$TMP_DIR" 2>/dev/null || echo "Cleaned metadata file not found, skipping."
    cp "$SCRATCH_DIR/STEP2_0.flag" "$TMP_DIR/" 2>/dev/null || echo "Flag not found, skipping."
    echo "End date: $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

echo "Begin date: $(date)"

cp "$RES/ORIGINAL_METADATA/metadata_sra.txt" "$SCRATCH_DIR/"

awk -F'\t' '{
  for (i=NF-1; i<=NF; i++) {
    gsub(/\b[A-Z_,-]+\b/, " ", $i)
    gsub(/[[:punct:]]+/, " ", $i)
    gsub(/TAG/, " ", $i)
    gsub(/VALUE/, " ", $i)
    gsub(/SAMPLE ATTRIBUTE/, " ", $i)
    gsub(/SCIENTIFIC NAME/, " ", $i)
    gsub(/SAMPLE NAME/, " ", $i)
    gsub(/EXTERNAL ID/, " ", $i)
    gsub(/PRIMARY ID/, " ", $i)
    gsub(/External Id/, " ", $i)
    gsub(/DESCRIPTION/, " ", $i)
    gsub(/STUDY/, " ", $i)
    gsub(/LINK/, " ", $i)
    gsub(/URL/, " ", $i)
    gsub(/ATTRIBUTE/, " ", $i)
    gsub(/[0-9]{4} [0-9]{2} [0-9]{2}/, " ", $i)
    gsub(/ENA FIRST PUBLIC/, " ", $i)
    gsub(/ENA LAST UPDATE/, " ", $i)
    gsub(/ENA last update/, " ", $i)
    gsub(/ENA first public/, " ", $i)
    gsub(/ArrayExpress/, " ", $i)
    gsub(/LABEL/, " ", $i)
    gsub(/XREF/, " ", $i)
    gsub(/SAMPLE/, " ", $i)
    gsub(/ +/, " ", $i)
  }
  print $0
}' OFS='\t' "$SCRATCH_DIR/metadata_sra.txt" > "$SCRATCH_DIR/cleaned_metadata_sra.txt"

awk -F'\t' '{
  for (i=1; i<=NF; i++) {
    if ($i == "" || $i ~ /^ *$/) {
      $i = ""
    }
  }
  print $0
}' OFS='\t' "$SCRATCH_DIR/cleaned_metadata_sra.txt" > "$SCRATCH_DIR/cleaned_metadata_sra.tmp" && \
mv "$SCRATCH_DIR/cleaned_metadata_sra.tmp" "$SCRATCH_DIR/cleaned_metadata_sra.txt"

touch "$SCRATCH_DIR/STEP2_0.flag"
