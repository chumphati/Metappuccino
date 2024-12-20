#!/bin/bash

#GET PROPORTION EMPTY LINE PER COLUMN

#output llm process
fichier_final_out_llm="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/results/final_out_llm/cleaned_table/llama_3_3_70B.csv"
output_file="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/results/final_out_llm/stats/llama_3_3_70B.csv"

#créate prop
en_tete=$(head -n 1 "$fichier_final_out_llm")
nb_colonnes=$(echo "$en_tete" | awk -F',' '{print NF}')

for ((col=1; col<=nb_colonnes; col++)); do
    total_lignes=$(tail -n +2 "$fichier_final_out_llm" | wc -l)

    #coutn empty or "INC" or "Not specified"
    lignes_vides=$(awk -F',' -v col="$col" 'NR > 1 && ($col == "" || $col ~ /INC/ || $col ~ /Not specified/) {count++} END {print count+0}' "$fichier_final_out_llm")

    if [[ $total_lignes -gt 0 ]]; then
        proportion=$(echo "scale=2; $lignes_vides/$total_lignes*100" | bc)
    else
        proportion=0
    fi

    colonne=$(echo "$en_tete" | cut -d',' -f"$col")
    echo "$colonne,$proportion%" >> $output_file
done
