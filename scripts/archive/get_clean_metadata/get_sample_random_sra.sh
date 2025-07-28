#!/bin/bash

PATH="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_metappuccino"
TAX="tax_eq(9606)%20AND%20"
STRAT="library_strategy%3D%22RNA-seq%22%20AND%20"
DATES="first_public%3E%3D2012-01-01%20AND%20first_public%3C2025-01-01%20AND%20"
PLAT="instrument_platform%3D%22ILLUMINA%22%20AND%20"
COUNTS="read_count%3E%3D10000000"
FIELDS="study_accession,first_public,study_title,project_name,run_accession,sample_accession,sample_title,sample_description,library_name,library_selection,library_source,library_strategy,library_construction_protocol,library_layout,rna_integrity_num,instrument_platform,rt_prep_protocol,cell_line,cell_type,tissue_lib,tissue_type,host_phenotype,isolate,age,host_body_site,sampling_site,base_count,description"
OUTPUT_FILE="${PATH}/METADATA_SRA.txt"
HEADER_WRITTEN=false
END_STEP1_1="${PATH}/logs/STEP1_1.flag"

/usr/bin/curl -X POST -H "Content-Type: application/x-www-form-urlencoded" \
     -d "result=read_run&query=${TAX}${STRAT}${DATES}${COUNTS}&format=tsv&fields=run_accession&limit=0" \
     "https://www.ebi.ac.uk/ena/portal/api/search" > ${PATH}/runs_all.tsv

/usr/bin/head -n1 ${PATH}/runs_all.tsv > ${PATH}/runs_sample.tsv
/usr/bin/tail -n +2 ${PATH}/runs_all.tsv | /usr/bin/shuf -n 5000 >> ${PATH}/runs_sample.tsv
