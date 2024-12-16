#!/bin/bash

#GET STABLE INFORMATION FROM METADATA EXTRACTION
#TARGET: run_accession, base_count, tissue_type, cell_line, cell_type, library_strategy, instrument_platform

INPUT_FILE="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/CLEAN_METADATA_SRA.txt"
OUTPUT_FILE="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/RAW_FINAL_INFO.csv"

#header
/usr/bin/echo "Run accession number,Number of base pairs,Tissue type,Cell line,Cell type,UBERON organ and code,Disease Ontology Term,Library strategy,Instrument platform,Donor information" > "$OUTPUT_FILE"

#extract columns raw
/usr/bin/awk -F'\t' '
BEGIN {
    OFS=",";
}
NR > 1 {
    print $1, $27, $20, $17, $18, "NA", "NA", $12, $16, "NA";
}' "$INPUT_FILE" >> "$OUTPUT_FILE"
