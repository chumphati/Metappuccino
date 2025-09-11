#!/bin/bash
set -euo pipefail

BASE_DIR="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/DATA_PER_CAT/cell_line"
LOG_DIR="${BASE_DIR}/logs"
RUNS_ALL_TSV="${BASE_DIR}/runs_with_cell_line.tsv"
RUNS_SEL_TSV="${BASE_DIR}/runs_with_cell_line_selected.tsv"
RUNS_RAND_TSV="${BASE_DIR}/runs_with_cell_line_randomized.tsv"
RUNS_RAND_SORTED_TSV="${BASE_DIR}/runs_with_cell_line_randomized_sorted.tsv"
RUNS_IDS_TXT="${BASE_DIR}/run_accessions_1000.txt"
END_FLAG="${LOG_DIR}/CELL_LINE_XMLS_DONE.flag"

TAX="tax_eq(9606)%20AND%20"
STRAT="library_strategy%3D%22RNA-seq%22%20AND%20"
DATES="first_public%3E%3D2012-01-01%20AND%20first_public%3C2025-01-01%20AND%20"
PLAT="instrument_platform%3D%22ILLUMINA%22%20AND%20"
COUNTS="read_count%3E%3D10000000"
K=2000

MAX_PER_STUDY=5        #at most 5 runs from the same study
MAX_PER_SAMPLE=1       #at most 1 run from the same sample
MAX_PER_CELLLINE=20    #at most 20 runs for the same exact cell_line string

/usr/bin/mkdir -p "${BASE_DIR}/xml"
/usr/bin/mkdir -p "${LOG_DIR}"

echo "[1/3] Search runs"
/usr/bin/curl -s -X POST -H "Content-Type: application/x-www-form-urlencoded" \
  -d "result=read_run&query=${TAX}${STRAT}${DATES}${PLAT}${COUNTS}&format=tsv&fields=run_accession,study_accession,sample_accession,cell_line&limit=100000" \
  "https://www.ebi.ac.uk/ena/portal/api/search" > "${RUNS_ALL_TSV}"

echo "[2/3] Filter not empty"
/usr/bin/awk -F'\t' 'NR>1 { sub(/\r$/, "", $0); if ($4 != "") print $1 "\t" $2 "\t" $3 "\t" $4 }' "${RUNS_ALL_TSV}" > "${RUNS_SEL_TSV}"

/usr/bin/awk -F'\t' 'BEGIN{srand()} {printf "%.12f\t%s\t%s\t%s\t%s\n", rand(), $1, $2, $3, $4}' \
  "${RUNS_SEL_TSV}" > "${RUNS_RAND_TSV}"
/usr/bin/sort -g -k1,1 "${RUNS_RAND_TSV}" -o "${RUNS_RAND_SORTED_TSV}"

/usr/bin/awk -F'\t' -v K="$K" -v MPS="$MAX_PER_STUDY" -v MPA="$MAX_PER_SAMPLE" -v MPCL="$MAX_PER_CELLLINE" '
  BEGIN { sel=0 }
  {
    run=$2; study=$3; sample=$4; cell=$5
    if (sel >= K) next
    if (seen[run]) next
    if (study_c[study] >= MPS) next
    if (sample_c[sample] >= MPA) next
    if (cell_c[cell] >= MPCL) next
    seen[run]=1
    study_c[study]++
    sample_c[sample]++
    cell_c[cell]++
    print run
    sel++
  }
  END { }' "${RUNS_RAND_SORTED_TSV}" > "${RUNS_IDS_TXT}"

SEL_COUNT=$(/usr/bin/wc -l < "${RUNS_IDS_TXT}" | /usr/bin/tr -d ' ')
echo "  -> ${SEL_COUNT} runs."

echo "[3/3] Download XML SRA (efetch NCBI) to ${BASE_DIR} ..."
N_DONE=0
while IFS=$'\n' read -r RUN_ACCESSION; do
  OUT_XML="${BASE_DIR}/xml/${RUN_ACCESSION}_metadata.xml"
  if [ ! -s "${OUT_XML}" ]; then
    /usr/bin/curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=sra&id=${RUN_ACCESSION}&retmode=text" -o "${OUT_XML}"
    /usr/bin/sleep 0.34
  fi
  N_DONE=$((N_DONE + 1))
  if (( N_DONE % 100 == 0 )); then
    echo "  -> ${N_DONE} / ${SEL_COUNT} XML downloaded..."
  fi
done < "${RUNS_IDS_TXT}"

/usr/bin/touch "${END_FLAG}"
