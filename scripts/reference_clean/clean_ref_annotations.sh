#!/bin/bash

#GET INTERESTING LINE FROM REF MANUALLY ANNOTATED TABLE

input_file="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/data/annotated_totalRNA.csv"
output_file="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/data/processed_ref_metadata.csv"

awk -F';' '
BEGIN {
    OFS=",";
}
NR == 1 {
    for (i = 1; i <= NF; i++) {
        if ($i == "run_accession") run_accession_idx = i;
        if ($i == "tissue_lib") tissue_lib_idx = i;
        if ($i == "cell_line") cell_line_idx = i;
        if ($i == "cell_type") cell_type_idx = i;
        if ($i == "tissue_type") tissue_type_idx = i;
        if ($i == "Gtex") Gtex_idx = i;
    }
    print "run_accession,tissue_lib,cell_line,cell_type,tissue_type,Gtex";
    next;
}
{
    print $run_accession_idx, $tissue_lib_idx, $cell_line_idx, $cell_type_idx, $tissue_type_idx, $Gtex_idx;
}
' "$input_file" > "$output_file"