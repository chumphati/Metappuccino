#!/bin/bash

PATH="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_metappuccino"
TAX="tax_eq(9606)%20AND%20"  # restrict to tax (human=9606)
STRAT="library_strategy%3D%22RNA-seq%22%20AND%20"  # Strategy= RNA-Seq, WXS, WGS
DATES="first_public%3E%3D2012-01-01%20AND%20first_public%3C2025-01-01%20AND%20"
PLAT="instrument_platform%3D%22ILLUMINA%22%20AND%20"  # "*"= %2A sinon "ILLUMINA"
COUNTS="read_count%3E%3D10000000"
FIELDS="study_accession,first_public,study_title,project_name,run_accession,sample_accession,sample_title,sample_description,library_name,library_selection,library_source,library_strategy,library_construction_protocol,library_layout,rna_integrity_num,instrument_platform,rt_prep_protocol,cell_line,cell_type,tissue_lib,tissue_type,host_phenotype,isolate,age,host_body_site,sampling_site,base_count,description"
OUTPUT_FILE="${PATH}/METADATA_SRA.txt"
HEADER_WRITTEN=false
END_STEP1_1="${PATH}/logs/STEP1_1.flag"

##RECUP RUN ACCESSION EN FONCTION DES PARAMÈTRES ÉTABLIS
/usr/bin/curl -X POST -H "Content-Type: application/x-www-form-urlencoded" \
     -d "result=read_run&query=${TAX}${STRAT}${DATES}${COUNTS}&format=tsv&fields=run_accession&limit=5000" \
     "https://www.ebi.ac.uk/ena/portal/api/search" > ${PATH}/runs.tsv

#from annotated total rna
#/usr/bin/awk -F';' 'NR==1 || $5 != "" { print $5 }' "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/raw/annotated_totalRNA.csv" > "${PATH}/runs.tsv"

##RECUP METADATA
#créer dossier out pour les metadatas
if [ -d "${PATH}/metadata/" ]; then
    echo "${PATH}/metadata/ already downloaded."
else
  /usr/bin/mkdir -p "${PATH}/metadata/"
  #récupérer metadata ncbi sra par le numéro d'accession
  /usr/bin/tail -n +2 "${PATH}/runs.tsv" | while IFS=$'\t' read -r RUN_ACCESSION; do
      OUTPUT_FILE="${PATH}/metadata/${RUN_ACCESSION}_metadata.xml"
      if [ ! -f "$OUTPUT_FILE" ]; then
          echo "Download metadonnées pour $RUN_ACCESSION"
          /usr/bin/curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=sra&id=${RUN_ACCESSION}&retmode=text" \
               -o "$OUTPUT_FILE"
      fi
  done
fi

##PARSER ET RÉCUP INFOS METADATA
#obtenir les metadonnées depuis le xml
extract_metadata_section () {
  section=$1
  /usr/bin/xmllint --xpath "//$section//*" "$METADATA_OUT" #| \
#  /usr/bin/sed 's/<\([^>]*\)>/\1 /g' | \
#  /usr/bin/sed 's/^\([^ ]*\)/\1 : /' | \
#  /usr/bin/sed 's/\/[A-Z_][A-Z_]*//g'
}

for xml_file in "${PATH}/metadata/"*_metadata.xml; do
    METADATA_OUT="$xml_file"
#    echo "$METADATA_OUT"
    run_accession=$(/usr/bin/basename "$xml_file" "_metadata.xml")

    #chercher dans sample et study
    sample_metadata=$(extract_metadata_section "SAMPLE" | /usr/bin/tr -d '\n')
    study_metadata=$(extract_metadata_section "STUDY" | /usr/bin/tr -d '\n')
#    echo "Sample Metadata: $sample_metadata"
#    echo "Study Metadata: $study_metadata"

    #chercher dans sample et study
    ena_metadata=$(/usr/bin/curl -s -X POST -H "Content-Type: application/x-www-form-urlencoded" \
        -d "result=read_run&query=run_accession%3D${run_accession}&format=tsv&fields=${FIELDS}&limit=1" \
        "https://www.ebi.ac.uk/ena/portal/api/search")

    #première ligne
    if [ "$HEADER_WRITTEN" = false ]; then
        first_line=$(echo "$ena_metadata" | /usr/bin/head -n 1)
        echo -e "${first_line}\tsample_metadata_ncbi\tstudy_metadata_ncbi" >> "$OUTPUT_FILE"
        HEADER_WRITTEN=true
    else
        #sinon
        echo -e "$(echo "$ena_metadata" | /usr/bin/tail -n +2)\t${sample_metadata}\t${study_metadata}" >> "$OUTPUT_FILE"
    fi
done

/usr/bin/touch "$END_STEP1_1"