#!/bin/bash

#PBS -N llm_inference
#PBS -l walltime=500:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:ncpus=30:mem=80gb

#SBATCH --job-name=llm_inference
#SBATCH --time=500:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1
#SBATCH --cpus-per-task=30
#SBATCH --mem=80G

METAPPUCCINO=${1:-$METAPPUCCINO}
RES=${2:-$RES}
ENV_REQUIREMENT=${3:-$ENV_REQUIREMENT}
MODEL=${4:-$MODEL}
VERBOSE=${5:-${VERBOSE:-FALSE}}
N_GPUS=${6:-${N_GPUS:-1}}

RESULTS_DIR=$RES
TMP_DIR=$RESULTS_DIR/tmp
LOG_DIR=$RESULTS_DIR/logs

exec > "$LOG_DIR/llm_inference.out" \
     2> "$LOG_DIR/llm_inference.err"

SCRATCH_DIR="/scratchlocal/$USER/${PBS_JOBID:-$SLURM_JOB_ID}"
mkdir -p "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

cleanup() {
    cp -r "$SCRATCH_DIR/METADATA_LLM_INFERENCE" "$RESULTS_DIR/COMPLETED_INFERENCE/" \
      2>/dev/null || echo "METADATA_LLM_INFERENCE not founded"
    if [[ -n "$PBS_JOBID" ]]; then jid=".$PBS_JOBID"; elif [[ -n "$SLURM_JOB_ID" ]]; then jid=".$SLURM_JOB_ID"; else jid=""; fi
    [[ -f "$SCRATCH_DIR/llm_log_SB.txt" ]] && cp "$SCRATCH_DIR/llm_log_SB.txt" "$LOG_DIR/llm_log_SB${jid}.txt" 2>/dev/null || echo "No log"
    [[ -f "$SCRATCH_DIR/reload_model_bio_info.txt" ]] && cp "$SCRATCH_DIR/reload_model_bio_info.txt" "$TMP_DIR/reload_model_bio_info${jid}.txt" 2>/dev/null || echo "No reload_model"
    [[ -f "$SCRATCH_DIR/skipped_runs.txt" ]] && cp "$SCRATCH_DIR/skipped_runs.txt" "$TMP_DIR/skipped_runs${jid}.txt" 2>/dev/null || echo "No skipped_runs"
    [[ -f "$SCRATCH_DIR/STEP3_1.flag" ]] && cp "$SCRATCH_DIR/STEP3_1.flag" "$TMP_DIR/" 2>/dev/null || echo "No flag"
    echo "End $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

cp "$MODEL"                                                 "$SCRATCH_DIR/"
cp "$RES/ORIGINAL_METADATA/metadata_sra_summarized.txt"     "$SCRATCH_DIR/"
cp "$TMP_DIR/database_metadata_curated.csv"                 "$SCRATCH_DIR/"
cp "$METAPPUCCINO/scripts/fill_missing_metadata/LLM_metadata_inference.py"  "$SCRATCH_DIR/"

source "$ENV_REQUIREMENT/bin/activate"

echo "Start $(date)"

PY_VERBOSE=()
if [[ "${VERBOSE^^}" == "TRUE" ]]; then
  PY_VERBOSE+=(--verbose)
fi

SHARD_TOTAL=${SHARD_TOTAL:-0}
SHARD_ID=${SHARD_ID:-0}

if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
  IFS=',' read -r -a ALL_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
  mapfile -t ALL_GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || echo 0)
fi
TOTAL_AVAIL=${#ALL_GPU_IDS[@]}
if [[ "$TOTAL_AVAIL" -eq 0 ]]; then
  echo "No GPU visible; falling back to CPU (set N_GPUS=0 or request GPUs)." >&2
  N_GPUS=0
fi
if [[ "$N_GPUS" -gt "$TOTAL_AVAIL" ]]; then
  echo "Requested N_GPUS=$N_GPUS but only $TOTAL_AVAIL visible; reducing to $TOTAL_AVAIL."
  N_GPUS=$TOTAL_AVAIL
fi

cat > "$SCRATCH_DIR/split_llm_inputs.py" << 'PYCODE'
import sys, os, json, argparse, re
import pandas as pd

def infer_sep(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        head = f.readline()
    if '\t' in head:
        return '\t'
    elif ';' in head and ',' not in head:
        return ';'
    else:
        return ','

def read_table(path):
    try:
        df = pd.read_csv(path, sep=None, engine='python', dtype=str, on_bad_lines='skip')
        return df
    except Exception:
        return None

def split_by_runs(db_csv, meta_path, out_dir, chunks):
    df_db = read_table(db_csv)
    if df_db is None or 'run_accession' not in df_db.columns:
        print("ERROR: cannot read database_metadata_curated.csv or missing 'run_accession' column.", file=sys.stderr)
        sys.exit(2)
    df_db = df_db.fillna('')
    runs = df_db['run_accession'].tolist()
    partitions = [runs[i::chunks] for i in range(chunks)]
    run2chunk = {r:i for i, part in enumerate(partitions) for r in part}

    meta_df = read_table(meta_path)
    meta_mode = None
    meta_sep = None
    meta_lines = None

    if meta_df is not None and 'run_accession' in meta_df.columns:
        meta_mode = 'table'
        meta_df = meta_df.fillna('')
        meta_sep = infer_sep(meta_path)
    else:
        # non-tabulaire → on ne split pas (plus sûr), on duplique
        meta_mode = 'copy_all'
        with open(meta_path, 'r', encoding='utf-8', errors='ignore') as f:
            meta_lines = f.read()

    os.makedirs(out_dir, exist_ok=True)
    db_sep = infer_sep(db_csv)

    for idx, subset in enumerate(partitions):
        cdir = os.path.join(out_dir, f"CHUNK_{idx}")
        os.makedirs(cdir, exist_ok=True)
        # CSV filtré par runs
        df_db_chunk = df_db[df_db['run_accession'].isin(subset)]
        df_db_chunk.to_csv(os.path.join(cdir, "database_metadata_curated.csv"), index=False, sep=db_sep)

        # Méta : split si table, sinon dupliquer
        if meta_mode == 'table':
            meta_chunk = meta_df[meta_df['run_accession'].isin(subset)]
            meta_chunk.to_csv(os.path.join(cdir, "metadata_sra_summarized.txt"), index=False, sep=meta_sep)
        else:
            with open(os.path.join(cdir, "metadata_sra_summarized.txt"), 'w', encoding='utf-8') as out:
                out.write(meta_lines)

        with open(os.path.join(cdir, "_chunk_info.txt"), "w", encoding="utf-8") as info:
            info.write(f"chunk_id={idx}\nnum_runs={len(subset)}\n")
PYCODE

if [[ "$SHARD_TOTAL" -ge 2 ]]; then
  N_GPUS=1
  python3 "$SCRATCH_DIR/split_llm_inputs.py" \
    --db   "$SCRATCH_DIR/database_metadata_curated.csv" \
    --meta "$SCRATCH_DIR/metadata_sra_summarized.txt" \
    --out  "$SCRATCH_DIR" \
    --chunks "$SHARD_TOTAL"

  CHUNK_DIR="$SCRATCH_DIR/CHUNK_${SHARD_ID}"
  mkdir -p "$CHUNK_DIR/METADATA_LLM_INFERENCE"
  cp -f "$SCRATCH_DIR/$(basename "$MODEL")" "$CHUNK_DIR/" 2>/dev/null

  gpu_id="${ALL_GPU_IDS[0]}"
  cd "$CHUNK_DIR"
  CUDA_VISIBLE_DEVICES="$gpu_id" python3 -u "$SCRATCH_DIR/LLM_metadata_inference.py" \
      --base_path "$CHUNK_DIR" \
      --input_metadata_path metadata_sra_summarized.txt \
      --error_file_path "reload_model_bio_info.${SHARD_ID}.txt" \
      --log_file_path "llm_log_SB.${SHARD_ID}.txt" \
      --flag_file "STEP3_1.flag.${SHARD_ID}" \
      --initial_n_ctx 3500 \
      --model "$(basename "$MODEL")" "${PY_VERBOSE[@]}"

  mkdir -p "$SCRATCH_DIR/METADATA_LLM_INFERENCE"
  [[ -d "$CHUNK_DIR/METADATA_LLM_INFERENCE" ]] && rsync -a --ignore-existing "$CHUNK_DIR/METADATA_LLM_INFERENCE/" "$SCRATCH_DIR/METADATA_LLM_INFERENCE/" 2>/dev/null || true
  : > "$SCRATCH_DIR/llm_log_SB.txt"
  : > "$SCRATCH_DIR/reload_model_bio_info.txt"
  [[ -f "$CHUNK_DIR/llm_log_SB.${SHARD_ID}.txt" ]] && cat "$CHUNK_DIR/llm_log_SB.${SHARD_ID}.txt" >> "$SCRATCH_DIR/llm_log_SB.txt"
  [[ -f "$CHUNK_DIR/reload_model_bio_info.${SHARD_ID}.txt" ]] && cat "$CHUNK_DIR/reload_model_bio_info.${SHARD_ID}.txt" >> "$SCRATCH_DIR/reload_model_bio_info.txt"
  touch "$SCRATCH_DIR/STEP3_1.flag"
  exit 0
fi

if [[ "$N_GPUS" -le 1 ]]; then
  mkdir -p "$SCRATCH_DIR/METADATA_LLM_INFERENCE"
  cd "$SCRATCH_DIR"
  python3 -u LLM_metadata_inference.py \
    --base_path "$SCRATCH_DIR" \
    --input_metadata_path metadata_sra_summarized.txt \
    --error_file_path reload_model_bio_info.txt \
    --log_file_path llm_log_SB.txt \
    --flag_file STEP3_1.flag \
    --initial_n_ctx 3500 \
    --model "$(basename "$MODEL")" "${PY_VERBOSE[@]}"

else
  python3 "$SCRATCH_DIR/split_llm_inputs.py" \
    --db   "$SCRATCH_DIR/database_metadata_curated.csv" \
    --meta "$SCRATCH_DIR/metadata_sra_summarized.txt" \
    --out  "$SCRATCH_DIR" \
    --chunks "$N_GPUS"

  pids=()
  for ((i=0; i< N_GPUS; i++)); do
    CHUNK_DIR="$SCRATCH_DIR/CHUNK_${i}"
    mkdir -p "$CHUNK_DIR/METADATA_LLM_INFERENCE"
    cp -f "$SCRATCH_DIR/$(basename "$MODEL")" "$CHUNK_DIR/" 2>/dev/null
    ERR_FILE="reload_model_bio_info.${i}.txt"
    LOG_FILE="llm_log_SB.${i}.txt"
    FLAG_FILE="STEP3_1.flag.${i}"
    gpu_id="${ALL_GPU_IDS[$i]}"

    (
      cd "$CHUNK_DIR"
      CUDA_VISIBLE_DEVICES="$gpu_id" \
      python3 -u "$SCRATCH_DIR/LLM_metadata_inference.py" \
        --base_path "$CHUNK_DIR" \
        --input_metadata_path metadata_sra_summarized.txt \
        --error_file_path "$ERR_FILE" \
        --log_file_path "$LOG_FILE" \
        --flag_file "$FLAG_FILE" \
        --initial_n_ctx 3500 \
        --model "$(basename "$MODEL")" "${PY_VERBOSE[@]}"
    ) &
    pids+=($!)
  done

  fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  if [[ "$fail" -ne 0 ]]; then
    echo "One or more GPU workers failed." >&2
    exit 1
  fi

  mkdir -p "$SCRATCH_DIR/METADATA_LLM_INFERENCE"
  if command -v rsync >/dev/null 2>&1; then
    for d in "$SCRATCH_DIR"/CHUNK_*; do
      if [[ -d "$d/METADATA_LLM_INFERENCE" ]]; then
        rsync -a --ignore-existing "$d/METADATA_LLM_INFERENCE/" "$SCRATCH_DIR/METADATA_LLM_INFERENCE/"
      fi
    done
  else
    for d in "$SCRATCH_DIR"/CHUNK_*; do
      if [[ -d "$d/METADATA_LLM_INFERENCE" ]]; then
        find "$d/METADATA_LLM_INFERENCE" -type f -print0 | while IFS= read -r -d '' f; do
          base="$(basename "$f")"
          dest="$SCRATCH_DIR/METADATA_LLM_INFERENCE/$base"
          if [[ ! -e "$dest" ]]; then
            mkdir -p "$SCRATCH_DIR/METADATA_LLM_INFERENCE"
            cp "$f" "$dest"
          fi
        done
      fi
    done
  fi

  : > "$SCRATCH_DIR/llm_log_SB.txt"
  : > "$SCRATCH_DIR/reload_model_bio_info.txt"
  for ((i=0; i< N_GPUS; i++)); do
    CHUNK_DIR="$SCRATCH_DIR/CHUNK_${i}"
    [[ -f "$CHUNK_DIR/llm_log_SB.${i}.txt" ]] && cat "$CHUNK_DIR/llm_log_SB.${i}.txt" >> "$SCRATCH_DIR/llm_log_SB.txt"
    [[ -f "$CHUNK_DIR/reload_model_bio_info.${i}.txt" ]] && cat "$CHUNK_DIR/reload_model_bio_info.${i}.txt" >> "$SCRATCH_DIR/reload_model_bio_info.txt"
  done
  touch "$SCRATCH_DIR/STEP3_1.flag"
fi
