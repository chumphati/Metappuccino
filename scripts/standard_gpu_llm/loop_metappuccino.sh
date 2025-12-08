#!/bin/bash

set -e

SCRIPT=metappuccino_a100_array.slurm
CSV=path_to_out_preprocessing_metappuccino/metadata_sra_summarized.csv
OUTDIR=out_path/METADATA_LLM_INFERENCE

while true; do
    total=$(tail -n +2 "$CSV" | wc -l)
    done_json=$(ls "$OUTDIR"/*.json 2>/dev/null | wc -l || echo 0)
    echo "[LOOP] Total CSV line: $total, JSON files: $done_json"

    if [ "$done_json" -ge "$total" ]; then
        echo "[LOOP] Stop: all done."
        break
    fi

    jid=$(sbatch "$SCRIPT" | awk '{print $4}')
    echo "[LOOP] Array submitted with job ID $jid, processing..."

    while squeue -j "$jid" 2>/dev/null | grep -q "$jid"; do
        sleep 300
    done

    echo "[LOOP] Array $jid done, relaunch if necessary."
done
