#!/bin/bash
#PBS -N llm_inference
#PBS -l walltime=500:00:00
#PBS -o /dev/null
#PBS -e /dev/null
#PBS -l select=1

#SBATCH --job-name=llm_inference
#SBATCH --time=500:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --nodes=1

METAPPUCCINO=${1:-$METAPPUCCINO}
RES=${2:-$RES}
ENV_REQUIREMENT=${3:-$ENV_REQUIREMENT}
MODEL=${4:-$MODEL}
VERBOSE=${5:-${VERBOSE:-FALSE}}
N_GPUS=${6:-${N_GPUS:-1}}
NODE_WORK_PATH=${7:-$NODE_WORK_PATH}

RESULTS_DIR=$RES
TMP_DIR=$RESULTS_DIR/tmp
LOG_DIR=$RESULTS_DIR/logs

exec > "$LOG_DIR/llm_inference.out" 2> "$LOG_DIR/llm_inference.err"

#SCRATCH_DIR="/scratchlocal/$USER/${PBS_JOBID:-$SLURM_JOB_ID}"
if [[ -n "${PBS_JOBID:-}" ]]; then
  SCRATCH_DIR="$NODE_WORK_PATH/${PBS_JOBID}"
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  SCRATCH_DIR="$NODE_WORK_PATH/${SLURM_JOB_ID}"
else
  SCRATCH_DIR="$(mktemp -d -p "${TMP_DIR}" "llm_inference")"
fi

mkdir -p "$SCRATCH_DIR"
cd "$SCRATCH_DIR"

cleanup() {
    cp -r "$SCRATCH_DIR/METADATA_LLM_INFERENCE" "$RESULTS_DIR/COMPLETED_INFERENCE/" 2>/dev/null || true
    if [[ -n "$PBS_JOBID" ]]; then jid=".$PBS_JOBID"; elif [[ -n "$SLURM_JOB_ID" ]]; then jid=".$SLURM_JOB_ID"; else jid=""; fi
    [[ -f "$SCRATCH_DIR/llm_log_SB.txt" ]] && cp "$SCRATCH_DIR/llm_log_SB.txt" "$LOG_DIR/llm_log_SB${jid}.txt" 2>/dev/null || true
    [[ -s "$SCRATCH_DIR/reload_model_bio_info.txt" ]] && cp "$SCRATCH_DIR/reload_model_bio_info.txt" "$TMP_DIR/reload_model_bio_info.txt" 2>/dev/null || true
    [[ -f "$SCRATCH_DIR/skipped_runs.txt" ]] && cp "$SCRATCH_DIR/skipped_runs.txt" "$LOG_DIR/skipped_runs${jid}.txt" 2>/dev/null || true
    [[ -f "$SCRATCH_DIR/STEP3_1.flag" ]] && cp "$SCRATCH_DIR/STEP3_1.flag" "$TMP_DIR/" 2>/dev/null || true
    echo "End $(date)"
    rm -rf "$SCRATCH_DIR"
}
trap cleanup EXIT

META_SRC="$RES/ORIGINAL_METADATA/metadata_sra_summarized.txt"
DB_SRC="$TMP_DIR/database_metadata_curated.csv"
AMBI_CL="$TMP_DIR/ambiguous_cell_lines.csv"
if [[ ! -s "$META_SRC" ]]; then
  echo "FATAL: missing $META_SRC" >&2
  ls -l "$RES/ORIGINAL_METADATA" >&2 || true
  exit 2
fi
if [[ ! -s "$DB_SRC" ]]; then
  echo "FATAL: missing $DB_SRC" >&2
  ls -l "$TMP_DIR" >&2 || true
  exit 3
fi

#cp "$MODEL" "$SCRATCH_DIR/" || { echo "FATAL: cannot copy MODEL"; exit 4; }
MODEL_BASENAME="$(basename "$MODEL")"
ln -sf "$MODEL" "$SCRATCH_DIR/$MODEL_BASENAME" || cp -n "$MODEL" "$SCRATCH_DIR/"
cp "$META_SRC" "$SCRATCH_DIR/" || { echo "FATAL: cannot copy metadata_sra_summarized.txt"; exit 5; }
cp "$DB_SRC" "$SCRATCH_DIR/" || { echo "FATAL: cannot copy database_metadata_curated.csv"; exit 6; }
cp "$METAPPUCCINO/scripts/fill_missing_metadata/LLM_metadata_inference.py" "$SCRATCH_DIR/" || { echo "FATAL: cannot copy LLM_metadata_inference.py"; exit 7; }
cp "$AMBI_CL" "$SCRATCH_DIR/" || { echo "FATAL: cannot copy ambiguous_cell_lines.csv"; exit 8; }

META_ABS="$SCRATCH_DIR/metadata_sra_summarized.txt"
DB_ABS="$SCRATCH_DIR/database_metadata_curated.csv"
MODEL_BASENAME="$(basename "$MODEL")"

source "$ENV_REQUIREMENT/bin/activate"
echo "Start $(date)"

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

cat > "$SCRATCH_DIR/split_llm_inputs.py" << 'PYCODE'
import sys, os, json, csv, argparse
def sniff_delim(path):
    with open(path,'r',encoding='utf-8',errors='ignore',newline='') as f:
        sample=f.read(4096); f.seek(0)
        try: dialect=csv.Sniffer().sniff(sample,delimiters=[',','\t',';','|'])
        except Exception:
            class D: delimiter=','
            dialect=D()
    return dialect
def read_rows_with_runs(path):
    dialect=sniff_delim(path)
    with open(path,'r',encoding='utf-8',errors='ignore',newline='') as f:
        reader=csv.reader(f,dialect)
        try: headers=next(reader)
        except StopIteration: return [],[],dialect
        headers_norm=[h.strip().lower() for h in headers]
        if 'run_accession' not in headers_norm: return None,None,dialect
        idx=headers_norm.index('run_accession')
        rows=[headers]+[row for row in reader]
        runs=[row[idx] for row in rows[1:]]
        return rows,runs,dialect
def write_rows(path, rows, delimiter):
    with open(path,'w',encoding='utf-8',newline='') as f:
        w=csv.writer(f,delimiter=delimiter,quotechar='"',quoting=csv.QUOTE_MINIMAL,doublequote=True,escapechar='\\',lineterminator='\n')
        for r in rows: w.writerow(r)
def split_db(db_csv, chunks):
    rows,runs,dialect=read_rows_with_runs(db_csv)
    if rows is None: print("ERROR: 'run_accession' not found in DB header.",file=sys.stderr); sys.exit(2)
    parts=[runs[i::chunks] for i in range(chunks)]
    idx_run=[h.strip().lower() for h in rows[0]].index('run_accession')
    per_chunk=[]
    for i,subset in enumerate(parts):
        keep=set(subset)
        out=[rows[0]]+[r for r in rows[1:] if r[idx_run] in keep]
        per_chunk.append((i,out,getattr(dialect,'delimiter',',')))
    return per_chunk
def meta_mode(meta_path):
    rows,runs,dialect=read_rows_with_runs(meta_path)
    if rows is not None: return 'table',(rows,runs,getattr(dialect,'delimiter',','))
    try:
        with open(meta_path,'r',encoding='utf-8',errors='ignore') as f:
            for line in f:
                line=line.strip()
                if not line: continue
                obj=json.loads(line)
                if 'run_accession' not in obj: raise ValueError
        return 'jsonl',None
    except Exception:
        return 'copy_all',None
def split_meta(meta_path,chunks,parts_runs):
    mode,payload=meta_mode(meta_path); out=[]
    if mode=='table':
        rows,runs,delim=payload; hdr=rows[0]; idx_run=[h.strip().lower() for h in hdr].index('run_accession')
        for i,subset in enumerate(parts_runs):
            keep=set(subset)
            chunk_rows=[hdr]+[r for r in rows[1:] if r[idx_run] in keep]
            out.append(('table',i,chunk_rows,delim))
    else:
        with open(meta_path,'r',encoding='utf-8',errors='ignore') as f: blob=f.read()
        for i in range(chunks): out.append(('blob',i,blob,None))
    return out
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--db",required=True); ap.add_argument("--meta",required=True); ap.add_argument("--out",required=True); ap.add_argument("--chunks",type=int,required=True)
    a=ap.parse_args()
    per_chunk=split_db(a.db,a.chunks)
    parts_runs=[];
    for i,rows,delim in per_chunk:
        idx_run=[h.strip().lower() for h in rows[0]].index('run_accession')
        parts_runs.append([r[idx_run] for r in rows[1:]])
    meta_splits=split_meta(a.meta,a.chunks,parts_runs)
    os.makedirs(a.out,exist_ok=True)
    for i,rows,delim in per_chunk:
        cdir=os.path.join(a.out,f"CHUNK_{i}"); os.makedirs(cdir,exist_ok=True)
        write_rows(os.path.join(cdir,"database_metadata_curated.csv"),rows,delim)
        with open(os.path.join(cdir,"_chunk_info.txt"),"w",encoding="utf-8") as info: info.write(f"chunk_id={i}\nnum_runs={len(rows)-1}\n")
    for kind,i,payload,delim in meta_splits:
        cdir=os.path.join(a.out,f"CHUNK_{i}")
        if kind=='table': write_rows(os.path.join(cdir,"metadata_sra_summarized.txt"),payload,delim)
        else:
            with open(os.path.join(cdir,"metadata_sra_summarized.txt"),"w",encoding="utf-8") as f: f.write(payload)
    for i,rows,_ in per_chunk: print(f"[split] CHUNK_{i}: {len(rows)-1} runs")
if __name__=="__main__": main()
PYCODE

if [[ "$SHARD_TOTAL" -ge 2 ]]; then
  N_GPUS=1
  python3 "$SCRATCH_DIR/split_llm_inputs.py" --db "$DB_ABS" --meta "$META_ABS" --out "$SCRATCH_DIR" --chunks "$SHARD_TOTAL" || { echo "FATAL: splitter failed" >&2; exit 8; }
  CHUNK_DIR="$SCRATCH_DIR/CHUNK_${SHARD_ID}"
  META_CHUNK="$CHUNK_DIR/metadata_sra_summarized.txt"
  DB_CHUNK="$CHUNK_DIR/database_metadata_curated.csv"
  if [[ ! -e "$CHUNK_DIR" ]]; then echo "FATAL: missing CHUNK dir $CHUNK_DIR" >&2; exit 8; fi
  if [[ ! -s "$META_CHUNK" || ! -s "$DB_CHUNK" ]]; then echo "FATAL: missing chunk inputs: $META_CHUNK or $DB_CHUNK" >&2; ls -l "$CHUNK_DIR" >&2 || true; exit 8; fi
  mkdir -p "$CHUNK_DIR/METADATA_LLM_INFERENCE"
  cp -f "$SCRATCH_DIR/$MODEL_BASENAME" "$CHUNK_DIR/" 2>/dev/null
  python3 -u "$SCRATCH_DIR/LLM_metadata_inference.py" --base_path "$CHUNK_DIR" --input_metadata_path "$META_CHUNK" --error_file_path "$CHUNK_DIR/reload_model_bio_info.${SHARD_ID}.txt" --log_file_path "$CHUNK_DIR/llm_log_SB.${SHARD_ID}.txt" --flag_file "$CHUNK_DIR/STEP3_1.flag.${SHARD_ID}" --initial_n_ctx 3500 --model "$MODEL_BASENAME" "${PY_VERBOSE[@]}"
  mkdir -p "$SCRATCH_DIR/METADATA_LLM_INFERENCE"
  [[ -d "$CHUNK_DIR/METADATA_LLM_INFERENCE" ]] && rsync -a --ignore-existing "$CHUNK_DIR/METADATA_LLM_INFERENCE/" "$SCRATCH_DIR/METADATA_LLM_INFERENCE/" 2>/dev/null || true
  : > "$SCRATCH_DIR/llm_log_SB.txt"
  rm -f "$SCRATCH_DIR/reload_model_bio_info.txt"
  header_done=0
  f="$CHUNK_DIR/reload_model_bio_info.${SHARD_ID}.txt"
  [[ -f "$CHUNK_DIR/llm_log_SB.${SHARD_ID}.txt" ]] && cat "$CHUNK_DIR/llm_log_SB.${SHARD_ID}.txt" >> "$SCRATCH_DIR/llm_log_SB.txt"
  if [[ -s "$f" ]]; then
    head -n 1 "$f" >> "$SCRATCH_DIR/reload_model_bio_info.txt"
    tail -n +2 "$f" >> "$SCRATCH_DIR/reload_model_bio_info.txt"
  fi
  touch "$SCRATCH_DIR/STEP3_1.flag"
  exit 0
fi

if [[ "$N_GPUS" -le 1 ]]; then
  mkdir -p "$SCRATCH_DIR/METADATA_LLM_INFERENCE"
  python3 -u "$SCRATCH_DIR/LLM_metadata_inference.py" --base_path "$SCRATCH_DIR" --input_metadata_path "$META_ABS" --error_file_path "$SCRATCH_DIR/reload_model_bio_info.txt" --log_file_path "$SCRATCH_DIR/llm_log_SB.txt" --flag_file "$SCRATCH_DIR/STEP3_1.flag" --initial_n_ctx 3500 --model "$MODEL_BASENAME" "${PY_VERBOSE[@]}"
else
  python3 "$SCRATCH_DIR/split_llm_inputs.py" --db "$DB_ABS" --meta "$META_ABS" --out "$SCRATCH_DIR" --chunks "$N_GPUS" || { echo "FATAL: splitter failed" >&2; exit 9; }
  pids=()
  for ((i=0; i< N_GPUS; i++)); do
    CHUNK_DIR="$SCRATCH_DIR/CHUNK_${i}"
    META_CHUNK="$CHUNK_DIR/metadata_sra_summarized.txt"
    DB_CHUNK="$CHUNK_DIR/database_metadata_curated.csv"
    if [[ ! -e "$CHUNK_DIR" ]]; then echo "FATAL: missing CHUNK dir $CHUNK_DIR" >&2; exit 9; fi
    if [[ ! -s "$META_CHUNK" || ! -s "$DB_CHUNK" ]]; then echo "FATAL: missing chunk inputs for CHUNK_${i}" >&2; ls -l "$CHUNK_DIR" >&2 || true; exit 9; fi
    mkdir -p "$CHUNK_DIR/METADATA_LLM_INFERENCE"
    cp -f "$SCRATCH_DIR/$MODEL_BASENAME" "$CHUNK_DIR/" 2>/dev/null
    LOG_FILE="$CHUNK_DIR/llm_log_SB.${i}.txt"
    ERR_FILE="$CHUNK_DIR/reload_model_bio_info.${i}.txt"
    FLAG_FILE="$CHUNK_DIR/STEP3_1.flag.${i}"
    gpu_id="${ALL_GPU_IDS[$i]}"
    ( CUDA_VISIBLE_DEVICES="$gpu_id" python3 -u "$SCRATCH_DIR/LLM_metadata_inference.py" --base_path "$CHUNK_DIR" --input_metadata_path "$META_CHUNK" --error_file_path "$ERR_FILE" --log_file_path "$LOG_FILE" --flag_file "$FLAG_FILE" --initial_n_ctx 3500 --model "$MODEL_BASENAME" "${PY_VERBOSE[@]}" ) &
    pids+=($!)
  done
  fail=0
  for pid in "${pids[@]}"; do wait "$pid" || fail=1; done
  if [[ "$fail" -ne 0 ]]; then echo "One or more GPU workers failed." >&2; exit 10; fi
  mkdir -p "$SCRATCH_DIR/METADATA_LLM_INFERENCE"
  if command -v rsync >/dev/null 2>&1; then
    for d in "$SCRATCH_DIR"/CHUNK_*; do
      [[ -d "$d/METADATA_LLM_INFERENCE" ]] && rsync -a --ignore-existing "$d/METADATA_LLM_INFERENCE/" "$SCRATCH_DIR/METADATA_LLM_INFERENCE/" || true
    done
  else
    for d in "$SCRATCH_DIR"/CHUNK_*; do
      if [[ -d "$d/METADATA_LLM_INFERENCE" ]]; then
        find "$d/METADATA_LLM_INFERENCE" -type f -print0 | while IFS= read -r -d '' f; do
          base="$(basename "$f")"; dest="$SCRATCH_DIR/METADATA_LLM_INFERENCE/$base"; [[ ! -e "$dest" ]] && cp "$f" "$dest"
        done
      fi
    done
  fi
  : > "$SCRATCH_DIR/llm_log_SB.txt"
  rm -f "$SCRATCH_DIR/reload_model_bio_info.txt"
  header_done=0
  for ((i=0; i< N_GPUS; i++)); do
    CHUNK_DIR="$SCRATCH_DIR/CHUNK_${i}"
    LOG_CHUNK="$CHUNK_DIR/llm_log_SB.${i}.txt"
    ERR_CHUNK="$CHUNK_DIR/reload_model_bio_info.${i}.txt"
    [[ -f "$LOG_CHUNK" ]] && cat "$LOG_CHUNK" >> "$SCRATCH_DIR/llm_log_SB.txt"
    if [[ -s "$ERR_CHUNK" ]]; then
      if [[ $header_done -eq 0 ]]; then
        head -n 1 "$ERR_CHUNK" >> "$SCRATCH_DIR/reload_model_bio_info.txt"
        tail -n +2 "$ERR_CHUNK" >> "$SCRATCH_DIR/reload_model_bio_info.txt"
        header_done=1
      else
        tail -n +2 "$ERR_CHUNK" >> "$SCRATCH_DIR/reload_model_bio_info.txt"
      fi
    fi
  done
  touch "$SCRATCH_DIR/STEP3_1.flag"
fi
