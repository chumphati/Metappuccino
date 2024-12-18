#!/bin/bash

/usr/bin/awk -F'\t' '{
  for (i=NF-1; i<=NF; i++) {
    gsub(/\b[A-Z_,-]+\b/, "", $i)
    gsub(/[[:punct:]]+/, " ", $i)
    gsub(/TAG/, "", $i)
    gsub(/VALUE/, "", $i)
    gsub(/SAMPLE ATTRIBUTE/, "", $i)
    gsub(/SCIENTIFIC NAME/, "", $i)
    gsub(/SAMPLE NAME/, "", $i)
    gsub(/EXTERNAL ID/, "", $i)
    gsub(/PRIMARY ID/, "", $i)
    gsub(/External Id/, "", $i)
    gsub(/DESCRIPTION/, "", $i)
    gsub(/STUDY/, "", $i)
    gsub(/LINK/, "", $i)
    gsub(/URL/, "", $i)
    gsub(/ATTRIBUTE/, "", $i)
    gsub(/[0-9]{4} [0-9]{2} [0-9]{2}/, "", $i)
    gsub(/ENA FIRST PUBLIC/, "", $i)
    gsub(/ENA LAST UPDATE/, "", $i)
    gsub("  ", " ", $i)
    gsub(/ENA last update/, "", $i)
    gsub(/ENA first public/, "", $i)
    gsub(/ArrayExpress/, "", $i)
    gsub(/LABEL/, "", $i)
    gsub(/XREF/, "", $i)
    gsub(/SAMPLE/, "", $i)
  }
  print $0
}' OFS='\t' /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/METADATA_SRA.txt > /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/CLEAN_METADATA_SRA.txt

/usr/bin/awk -F'\t' '{
  for (i=1; i<=NF; i++) {
    if ($i == "" || $i ~ /^ *$/) {
      $i = "NA"
    }
  }
  print $0
}' OFS='\t' /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/CLEAN_METADATA_SRA.txt > /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/CLEAN_METADATA_SRA.tmp && \
/usr/bin/mv /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/CLEAN_METADATA_SRA.tmp /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/CLEAN_METADATA_SRA.txt

/usr/bin/touch "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/STEP1_2.flag"
