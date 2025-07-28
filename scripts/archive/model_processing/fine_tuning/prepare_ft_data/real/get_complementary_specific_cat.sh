#!/bin/bash

base_query="tax_eq(9606) AND library_strategy=%22RNA-seq%22 AND first_public>=2012-01-01 AND first_public<2025-01-01 AND instrument_platform=%22ILLUMINA%22 AND read_count>=10000000"
ENA_URL="https://www.ebi.ac.uk/ena/portal/api/search"
OUTPUT_TSV="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/ena_results.tsv"
LIMIT=10000
NB_PAR_TYPE=5
FILTERED_TMP=$(mktemp)
COMBOS_TMP=$(mktemp)
OUTPUT_CSV="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/ena_varied_tissue_only.csv"

echo "→ Téléchargement des métadonnées ENA..."
curl -G "$ENA_URL" \
  --data-urlencode "result=read_run" \
  --data-urlencode "query=${base_query}" \
  --data-urlencode "fields=run_accession,tissue_type" \
  --data-urlencode "limit=${LIMIT}" \
  --data-urlencode "format=tsv" \
  -o "$OUTPUT_TSV"

if [ ! -s "$OUTPUT_TSV" ]; then
  echo "Erreur : le fichier $OUTPUT_TSV est vide ou introuvable."
  exit 1
fi
echo "→ Métadonnées ENA téléchargées dans '$OUTPUT_TSV'."

header_line=$(head -n1 "$OUTPUT_TSV")
IFS=$'\t' read -r -a header_fields <<< "$header_line"
col_tissue=""
for idx in "${!header_fields[@]}"; do
  if [[ "${header_fields[$idx]}" == "tissue_type" ]]; then
    col_tissue=$((idx + 1))
    break
  fi
done

if [[ -z "$col_tissue" ]]; then
  echo "Erreur : colonne 'tissue_type' non trouvée."
  rm -f "$FILTERED_TMP" "$COMBOS_TMP"
  exit 1
fi
echo "→ Colonne 'tissue_type' détectée en position $col_tissue."

echo "→ Filtrage : garder seulement les runs avec tissue_type renseigné..."
awk -F"\t" -v t="$col_tissue" 'NR>1 {
    if ($t != "" && $t !~ /^[[:space:]]*$/) {
        print $1"\t"$t
    }
}' "$OUTPUT_TSV" > "$FILTERED_TMP"

if [ ! -s "$FILTERED_TMP" ]; then
  echo "Erreur : aucun run avec tissue_type renseigné."
  rm -f "$FILTERED_TMP" "$COMBOS_TMP"
  exit 1
fi
echo "→ Nombre de lignes après filtrage : $(wc -l < "$FILTERED_TMP")"

cut -f2 "$FILTERED_TMP" | sort | uniq > "$COMBOS_TMP"
echo "→ Nombre de tissue_type distincts : $(wc -l < "$COMBOS_TMP")"

echo "→ Échantillonnage par tissue_type (max $NB_PAR_TYPE runs par type)..."
echo "run_accession,tissue_type" > "$OUTPUT_CSV"

while read -r tissue; do
    runs=$(awk -F"\t" -v T="$tissue" '$2 == T {print $1}' "$FILTERED_TMP")
    if [[ -n "$runs" ]]; then
        echo "$runs" | shuf -n "$NB_PAR_TYPE" | while read -r run_id; do
            echo "${run_id},\"${tissue}\"" >> "$OUTPUT_CSV"
        done
    fi
done < "$COMBOS_TMP"

rm -f "$FILTERED_TMP" "$COMBOS_TMP"

echo "→ Fichier CSV généré : '$OUTPUT_CSV' ($(wc -l < "$OUTPUT_CSV") lignes)."
