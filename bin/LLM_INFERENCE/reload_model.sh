#!/bin/bash
#PBS -N reload_context_llm
#PBS -l walltime=500:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1:ncpus=30:mem=80gb

#SBATCH --job-name=reload_context_llm
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
ITERATION_LIMIT=${5:-$ITERATION_LIMIT}
VERBOSE=${6:-${VERBOSE:-FALSE}}
N_GPUS=${7:-${N_GPUS:-1}}

RESULTS_DIR=$RES
TMP_DIR=$RESULTS_DIR/tmp
LOG_DIR=$RESULTS_DIR/logs
exec > "$LOG_DIR/reload_context_llm.out" 2> "$LOG_DIR/reload_context_llm.err"

SCRATCH_DIR="/scratchlocal/$USER/${PBS_JOBID:-$SLURM_JOB_ID}"
mkdir -p "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

cleanup() {
    if command -v rsync >/dev/null 2>&1; then
        mkdir -p "$RES/COMPLETED_INFERENCE/METADATA_LLM_INFERENCE"
        rsync -a --ignore-existing "$SCRATCH_DIR/METADATA_LLM_INFERENCE/" "$RES/COMPLETED_INFERENCE/METADATA_LLM_INFERENCE/" 2>/dev/null || true
    else
        mkdir -p "$RES/COMPLETED_INFERENCE/METADATA_LLM_INFERENCE"
        find "$SCRATCH_DIR/METADATA_LLM_INFERENCE" -type f -print0 2>/dev/null | while IFS= read -r -d '' f; do
            base="$(basename "$f")"; dest="$RES/COMPLETED_INFERENCE/METADATA_LLM_INFERENCE/$base"; [[ -e "$dest" ]] || cp "$f" "$dest"
        done
    fi
    if [[ -n "$PBS_JOBID" ]]; then jid=".$PBS_JOBID"; elif [[ -n "$SLURM_JOB_ID" ]]; then jid=".$SLURM_JOB_ID"; else jid=""; fi
    [[ -f "$SCRATCH_DIR/llm_log_reload.txt" ]] && cp "$SCRATCH_DIR/llm_log_reload.txt" "$LOG_DIR/llm_log_reload${jid}.txt" 2>/dev/null || true
    [[ -s "$SCRATCH_DIR/reload_model_bio_info.txt" ]] && cp "$SCRATCH_DIR/reload_model_bio_info.txt" "$TMP_DIR/reload_model_bio_info.txt" 2>/dev/null || true
    [[ -f "$SCRATCH_DIR/skipped_runs.txt" ]] && cp "$SCRATCH_DIR/skipped_runs.txt" "$LOG_DIR/skipped_runs_reload${jid}.txt" 2>/dev/null || true
    [[ -f "$SCRATCH_DIR/STEP3_2.flag" ]] && cp "$SCRATCH_DIR/STEP3_2.flag" "$TMP_DIR/" 2>/dev/null || true
    echo "End $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

if [ ! -f "$TMP_DIR/reload_model_bio_info.txt" ] ; then
    echo "✔ No file $TMP_DIR/reload_model_bio_info.txt to analyse."
    touch "$SCRATCH_DIR/STEP3_2.flag"
    exit 0
fi

cp "$MODEL" "$SCRATCH_DIR/" || { echo "FATAL: cannot copy MODEL"; exit 4; }
cp "$TMP_DIR/reload_model_bio_info.txt" "$SCRATCH_DIR/" || { echo "FATAL: cannot copy reload_model_bio_info.txt"; exit 5; }
cp "$TMP_DIR/database_metadata_curated.csv" "$SCRATCH_DIR/" || { echo "FATAL: cannot copy database_metadata_curated.csv"; exit 6; }
cp "$METAPPUCCINO/scripts/fill_missing_metadata/LLM_metadata_inference.py" "$SCRATCH_DIR/" || { echo "FATAL: cannot copy LLM_metadata_inference.py"; exit 7; }
source "$ENV_REQUIREMENT/bin/activate"
echo "Begin date: $(date)"

PY_VERBOSE=()
if [[ "${VERBOSE^^}" == "TRUE" ]]; then PY_VERBOSE+=(--verbose); fi

SHARD_TOTAL=${SHARD_TOTAL:-0}
SHARD_ID=${SHARD_ID:-0}

if [[ -n "$CUDA_VISIBLE_DEVICES" ]]; then
  IFS=',' read -r -a ALL_GPU_IDS <<< "$CUDA_VISIBLE_DEVICES"
else
  mapfile -t ALL_GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || echo 0)
fi
TOTAL_AVAIL=${#ALL_GPU_IDS[@]}
if [[ "$TOTAL_AVAIL" -eq 0 ]]; then N_GPUS=0; fi
if [[ "$N_GPUS" -gt "$TOTAL_AVAIL" ]]; then N_GPUS=$TOTAL_AVAIL; fi

iteration_limit=${ITERATION_LIMIT:-1}
for (( i=0; i<iteration_limit; i++ )); do
    if [ ! -s "$SCRATCH_DIR/reload_model_bio_info.txt" ]; then
        echo "All inferences completed.."
        touch "$SCRATCH_DIR/STEP3_2.flag"
        break
    fi

    if [[ "$SHARD_TOTAL" -ge 2 ]]; then
      N_GPUS=1
      cat > "$SCRATCH_DIR/split_reload.py" << 'PYCODE'
import argparse, os
p=argparse.ArgumentParser()
p.add_argument("--input",required=True)
p.add_argument("--chunks",type=int,required=True)
p.add_argument("--out",required=True)
a=p.parse_args()
with open(a.input,encoding="utf-8",errors="ignore") as f:
    lines=[ln.rstrip("\n") for ln in f if ln.strip()]
hdr=lines[0] if lines else ""
data=lines[1:] if len(lines)>1 else []
parts=[data[i::a.chunks] for i in range(a.chunks)]
for i,seg in enumerate(parts):
    d=os.path.join(a.out,f"CHUNK_{i}")
    os.makedirs(d,exist_ok=True)
    with open(os.path.join(d,"reload_model_bio_info.txt"),"w",encoding="utf-8") as o:
        if hdr: o.write(hdr+"\n")
        if seg: o.write("\n".join(seg)+"\n")
    with open(os.path.join(d,"_chunk_info.txt"),"w",encoding="utf-8") as info:
        info.write(f"chunk_id={i}\nnum_lines={len(seg)}\n")
PYCODE
      python3 "$SCRATCH_DIR/split_reload.py" --input "$SCRATCH_DIR/reload_model_bio_info.txt" --chunks "$SHARD_TOTAL" --out "$SCRATCH_DIR" || { echo "FATAL: splitter failed"; exit 8; }
      CHUNK_DIR="$SCRATCH_DIR/CHUNK_${SHARD_ID}"
      META_CHUNK="$CHUNK_DIR/reload_model_bio_info.txt"
      if [[ ! -e "$CHUNK_DIR" ]]; then echo "FATAL: missing CHUNK dir $CHUNK_DIR" >&2; exit 8; fi
      if [[ ! -s "$META_CHUNK" ]]; then
        : > "$CHUNK_DIR/reload_model_bio_info_bis.${SHARD_ID}.txt"
      else
        if [[ $(wc -l < "$META_CHUNK") -le 1 ]]; then
          : > "$CHUNK_DIR/reload_model_bio_info_bis.${SHARD_ID}.txt"
        else
          cp -f "$SCRATCH_DIR/database_metadata_curated.csv" "$CHUNK_DIR/" 2>/dev/null
          cp -f "$SCRATCH_DIR/$(basename "$MODEL")" "$CHUNK_DIR/" 2>/dev/null
          gpu_id="${ALL_GPU_IDS[0]}"
          CUDA_VISIBLE_DEVICES="$gpu_id" python3 -u "$SCRATCH_DIR/LLM_metadata_inference.py" \
              --base_path "$CHUNK_DIR" \
              --input_metadata_path "$META_CHUNK" \
              --error_file_path "$CHUNK_DIR/reload_model_bio_info_bis.${SHARD_ID}.txt" \
              --log_file_path "$CHUNK_DIR/llm_log_reload.${SHARD_ID}.txt" \
              --flag_file "$CHUNK_DIR/STEP3_2.flag.${SHARD_ID}" \
              --initial_n_ctx 3500 \
              --model "$(basename "$MODEL")" "${PY_VERBOSE[@]}"
        fi
      fi
      mkdir -p "$SCRATCH_DIR/METADATA_LLM_INFERENCE"
      [[ -d "$CHUNK_DIR/METADATA_LLM_INFERENCE" ]] && rsync -a --ignore-existing "$CHUNK_DIR/METADATA_LLM_INFERENCE/" "$SCRATCH_DIR/METADATA_LLM_INFERENCE/" 2>/dev/null || true
      : > "$SCRATCH_DIR/llm_log_reload.txt"
      rm -f "$SCRATCH_DIR/reload_model_bio_info_bis.txt"
      f="$CHUNK_DIR/reload_model_bio_info_bis.${SHARD_ID}.txt"
      [[ -f "$CHUNK_DIR/llm_log_reload.${SHARD_ID}.txt" ]] && cat "$CHUNK_DIR/llm_log_reload.${SHARD_ID}.txt" >> "$SCRATCH_DIR/llm_log_reload.txt"
      if [[ -s "$f" ]]; then
        head -n 1 "$f" >> "$SCRATCH_DIR/reload_model_bio_info_bis.txt"
        tail -n +2 "$f" >> "$SCRATCH_DIR/reload_model_bio_info_bis.txt"
      fi
      rm -rf "$SCRATCH_DIR"/CHUNK_*
    else
      if [[ "$N_GPUS" -le 1 ]]; then
        python3 -u "$SCRATCH_DIR/LLM_metadata_inference.py" --base_path "$SCRATCH_DIR" --input_metadata_path "$SCRATCH_DIR/reload_model_bio_info.txt" --error_file_path "$SCRATCH_DIR/reload_model_bio_info_bis.txt" --log_file_path "$SCRATCH_DIR/llm_log_reload.txt" --flag_file "$SCRATCH_DIR/STEP3_2.flag" --initial_n_ctx 3500 --model "$SCRATCH_DIR/$(basename "$MODEL")" "${PY_VERBOSE[@]}"
      else
        cat > "$SCRATCH_DIR/split_reload.py" << 'PYCODE'
import argparse, os
p=argparse.ArgumentParser()
p.add_argument("--input",required=True)
p.add_argument("--chunks",type=int,required=True)
p.add_argument("--out",required=True)
a=p.parse_args()
with open(a.input,encoding="utf-8",errors="ignore") as f:
    lines=[ln.rstrip("\n") for ln in f if ln.strip()]
hdr=lines[0] if lines else ""
data=lines[1:] if len(lines)>1 else []
parts=[data[i::a.chunks] for i in range(a.chunks)]
for i,seg in enumerate(parts):
    d=os.path.join(a.out,f"CHUNK_{i}")
    os.makedirs(d,exist_ok=True)
    with open(os.path.join(d,"reload_model_bio_info.txt"),"w",encoding="utf-8") as o:
        if hdr: o.write(hdr+"\n")
        if seg: o.write("\n".join(seg)+"\n")
    with open(os.path.join(d,"_chunk_info.txt"),"w",encoding="utf-8") as info:
        info.write(f"chunk_id={i}\nnum_lines={len(seg)}\n")
PYCODE
        python3 "$SCRATCH_DIR/split_reload.py" --input "$SCRATCH_DIR/reload_model_bio_info.txt" --chunks "$N_GPUS" --out "$SCRATCH_DIR" || { echo "FATAL: splitter failed"; exit 9; }
        pids=()
        for ((g=0; g<N_GPUS; g++)); do
            CHUNK_DIR="$SCRATCH_DIR/CHUNK_${g}"
            META_CHUNK="$CHUNK_DIR/reload_model_bio_info.txt"
            if [[ ! -e "$CHUNK_DIR" ]]; then echo "FATAL: missing CHUNK dir $CHUNK_DIR" >&2; exit 9; fi
            if [[ ! -s "$META_CHUNK" || $(wc -l < "$META_CHUNK") -le 1 ]]; then
              : > "$CHUNK_DIR/reload_model_bio_info_bis.${g}.txt"
              continue
            fi
            cp -f "$SCRATCH_DIR/database_metadata_curated.csv" "$CHUNK_DIR/" 2>/dev/null
            cp -f "$SCRATCH_DIR/$(basename "$MODEL")" "$CHUNK_DIR/" 2>/dev/null
            gpu_id="${ALL_GPU_IDS[$g]}"
            ( CUDA_VISIBLE_DEVICES="$gpu_id" \
              python3 -u "$SCRATCH_DIR/LLM_metadata_inference.py" \
                --base_path "$CHUNK_DIR" \
                --input_metadata_path "$META_CHUNK" \
                --error_file_path "$CHUNK_DIR/reload_model_bio_info_bis.${g}.txt" \
                --log_file_path "$CHUNK_DIR/llm_log_reload.${g}.txt" \
                --flag_file "$CHUNK_DIR/STEP3_2.flag.${g}" \
                --initial_n_ctx 3500 \
                --model "$(basename "$MODEL")" "${PY_VERBOSE[@]}" ) &
            pids+=($!)
        done
        fail=0
        for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
        if [[ "$fail" -ne 0 ]]; then echo "One or more GPU workers failed." >&2; exit 1; fi
        mkdir -p "$SCRATCH_DIR/METADATA_LLM_INFERENCE"
        if command -v rsync >/dev/null 2>&1; then
            for d in "$SCRATCH_DIR"/CHUNK_*; do
                [[ -d "$d/METADATA_LLM_INFERENCE" ]] && rsync -a --ignore-existing "$d/METADATA_LLM_INFERENCE/" "$SCRATCH_DIR/METADATA_LLM_INFERENCE/" || true
            done
        else
            for d in "$SCRATCH_DIR"/CHUNK_*; do
                if [[ -d "$d/METADATA_LLM_INFERENCE" ]]; then
                    find "$d/METADATA_LLM_INFERENCE" -type f -print0 | while IFS= read -r -d '' f; do
                        base="$(basename "$f")"; dest="$SCRATCH_DIR/METADATA_LLM_INFERENCE/$base"; [[ -e "$dest" ]] || cp "$f" "$dest"
                    done
                fi
            done
        fi
        : > "$SCRATCH_DIR/llm_log_reload.txt"
        rm -f "$SCRATCH_DIR/reload_model_bio_info_bis.txt"
        header_done=0
        for ((g=0; g<N_GPUS; g++)); do
            CHUNK_DIR="$SCRATCH_DIR/CHUNK_${g}"
            LOG_CHUNK="$CHUNK_DIR/llm_log_reload.${g}.txt"
            NEXT_CHUNK="$CHUNK_DIR/reload_model_bio_info_bis.${g}.txt"
            [[ -f "$LOG_CHUNK" ]] && cat "$LOG_CHUNK" >> "$SCRATCH_DIR/llm_log_reload.txt"
            if [[ -s "$NEXT_CHUNK" ]]; then
              if [[ $header_done -eq 0 ]]; then
                head -n 1 "$NEXT_CHUNK" >> "$SCRATCH_DIR/reload_model_bio_info_bis.txt"
                tail -n +2 "$NEXT_CHUNK" >> "$SCRATCH_DIR/reload_model_bio_info_bis.txt"
                header_done=1
              else
                tail -n +2 "$NEXT_CHUNK" >> "$SCRATCH_DIR/reload_model_bio_info_bis.txt"
              fi
            fi
        done
        rm -rf "$SCRATCH_DIR"/CHUNK_*
      fi
    fi

    if [ ! -s "$SCRATCH_DIR/reload_model_bio_info_bis.txt" ] ; then
        echo "All inferences completed.."
        touch "$SCRATCH_DIR/STEP3_2.flag"
        break
    fi
    mv "$SCRATCH_DIR/reload_model_bio_info_bis.txt" "$SCRATCH_DIR/reload_model_bio_info.txt"
done
