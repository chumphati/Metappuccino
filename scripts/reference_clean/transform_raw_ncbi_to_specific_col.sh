#!/bin/bash

input_file="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/store/good_cleaned_metadata_sra.txt"
output_file="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/store/mela-select_extraction.tsv"

header="run_accession\tlibrary_selection_fixed\tlibrary_source_fixed\tcell_type_fix\tcell_line_fix\ttissue_type_fix\tdisease\ttreatment\ttreatment_time\tresponse\tphenotype\tage\tsex"
echo -e "$header" > "$output_file"

awk -F"\t" 'BEGIN {OFS="\t"}
NR==1 {
  for (i=1; i<=NF; i++) {
    col[$i] = i
    colname[i] = $i
  }
  next
}
{
  disease_host_status = ""
  if ($col["disease"] != "NA" && $col["disease"] != "") disease_host_status = $col["disease"]
  if ($col["host_status"] != "NA" && $col["host_status"] != "") disease_host_status = (disease_host_status ? disease_host_status "," : "") $col["host_status"]

  if (disease_host_status == "" || disease_host_status == " ") disease_host_status = "NA"
  if (sex_merged == "" || sex_merged == " ") sex_merged = "NA"
  if (host_phenotype == "" || host_phenotype == " ") host_phenotype = "NA"

  print $col["run_accession"],
        $col["library_selection"],
        $col["library_source"],
        $col["cell_type"],
        $col["cell_line"],
        $col["tissue_type"],
        disease_host_status,
        "NA", "NA", "NA",
        $col["host_phenotype"],
        $col["age"],
        $col["sex"]
} ' "$input_file" >> "$output_file"
