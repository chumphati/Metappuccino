#!/bin/bash
base_query="tax_eq(9606) AND library_strategy=%22RNA-seq%22 AND first_public>=2012-01-01 AND first_public<2025-01-01 AND instrument_platform=%22ILLUMINA%22 AND read_count>=10000000"

curl -G "https://www.ebi.ac.uk/ena/portal/api/search" \
  --data-urlencode "result=read_run" \
  --data-urlencode "query=${base_query}" \
  --data-urlencode "fields=run_accession,cell_type,host_body_site,tissue_type,cell_line,disease,host_phenotype,library_selection,library_source" \
  --data-urlencode "limit=10" \
  --data-urlencode "format=tsv" \
  -o ena_results.tsv

if [ ! -s ena_results.tsv ]; then
  echo "Error: ena_results.tsv is empty."
  exit 1
fi

declare -A cats
cats["Cell type"]="cell_type"
cats["UBERON term"]="host_body_site"
cats["Tissue type"]="tissue_type"
cats["Cell line"]="cell_line"
cats["DOT term"]="disease"
cats["Phenotype"]="host_phenotype"
cats["Library selection fixed"]="library_selection"
cats["Library source"]="library_source"

tmp_union=$(mktemp)
for cat in "${!cats[@]}"; do
  field=${cats[$cat]}
  col=$(head -n 1 ena_results.tsv | awk -F"\t" -v f="$field" '{for(i=1;i<=NF;i++) if($i==f){print i; exit}}')
  [ -z "$col" ] && continue
  awk -F"\t" -v c="$col" 'NR>1 { if($c != "" && $c !~ /^[[:space:]]*$/) print $1 }' ena_results.tsv | shuf -n 10 >> "$tmp_union"
done
sort "$tmp_union" | uniq > union_runs.txt
rm "$tmp_union"

output_file="ena_random_runs.csv"
echo "Run Accession,Info" > "$output_file"
header_line=$(head -n 1 ena_results.tsv)
IFS=$'\t' read -r -a header_fields <<< "$header_line"

while read -r run; do
  row=$(awk -F"\t" -v r="$run" 'NR==1 {next} $1==r {print; exit}' ena_results.tsv)
  [ -z "$row" ] && continue
  IFS=$'\t' read -r -a fields_values <<< "$row"
  info=""
  for cat in "${!cats[@]}"; do
    key=${cats[$cat]}
    idx=0
    for i in "${!header_fields[@]}"; do
      if [ "${header_fields[$i]}" == "$key" ]; then
        idx=$i
        break
      fi
    done
    value="${fields_values[$idx]}"
    if [ -n "$value" ]; then
      [ -z "$info" ] && info="$cat: $value" || info="$info; $cat: $value"
    fi
  done
  echo "$run,\"$info\"" >> "$output_file"
done < union_runs.txt

rm union_runs.txt
echo "Output saved to $output_file"
