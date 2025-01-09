#!/bin/bash

# Déclaration des chemins des fichiers d'entrée et de sortie
input_file_1="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/results/final_out_llm/cleaned_table/llama_3_3_70B.csv"
input_file_2="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/data/processed_ref_metadata.csv"
uberon_reference="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/data/UBERON_TABLE_CLEAN.csv"
output_stats="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/results/final_out_llm/stats/llama_3_3_70B.csv"
detailed_output="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/results/final_out_llm/detailed_uberon_results/llama_3_3_70B_detailed_uberon_results.txt"
good_matches_file="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/results/final_out_llm/mismatch_uberon/good_uberon_llama_3_3_70B.txt"
bad_matches_file="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/results/final_out_llm/mismatch_uberon/bad_uberon_llama_3_3_70B.txt"

# Création des dossiers si inexistants
mkdir -p "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/results/final_out_llm/stats"
mkdir -p "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/results/final_out_llm/detailed_uberon_results"
mkdir -p "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/results/final_out_llm/mismatch_uberon"

# Initialisation des comptages
correct_uberon_count=0
incorrect_uberon_count=0
no_code_count=0
match_count=0
skip_count=0
empty_line_count=0
total_line_count=0

# Initialisation des fichiers de sortie
> "$detailed_output"
echo "Accession,Reference UBERON Code,Reference Synonyms,Predicted Code,Result" >> "$detailed_output"
> "$good_matches_file"
echo "Predicted Code // Reference Code" >> "$good_matches_file"
> "$bad_matches_file"
echo "Predicted Code // Reference Code" >> "$bad_matches_file"

declare -A uberon_map

# Chargement du dictionnaire de référence UBERON
# Ce fichier contient les codes UBERON et leurs synonymes correspondants
while IFS=',' read -r code name synonyms; do
    cleaned_code=$(echo "$code" | tr '[:upper:]' '[:lower:]' | sed -E 's/[-_]/:/g' | xargs)
    cleaned_synonyms=$(echo "$name;$synonyms" | tr '[:upper:]' '[:lower:]' | sed -E 's/[-_]/:/g' | xargs)
    uberon_map["$cleaned_code"]="$cleaned_synonyms"
done < "$uberon_reference"

# Traitement des fichiers d'entrée
# Ce bloc analyse les codes prédits dans le fichier d'entrée et les compare à la référence
while IFS=',' read -r accession tissue cell_type uberon_info disease_term; do
    total_line_count=$((total_line_count + 1))

    # Comptage des lignes vides
    if [[ -z "$accession" || -z "$uberon_info" ]]; then
        empty_line_count=$((empty_line_count + 1))
        continue
    fi

    # Extraction des informations UBERON
    if [[ "$uberon_info" =~ (UBERON:[0-9]+) ]]; then
        extracted_code=${BASH_REMATCH[1]}
        extracted_text=$(echo "$uberon_info" | sed "s/$extracted_code//g" | xargs)

        # Validation avec la référence
        if [[ -n ${uberon_map[$extracted_code]} ]]; then
            synonyms=${uberon_map[$extracted_code]}
            match_found=false

            for synonym in $synonyms; do
                if [[ "$extracted_text" == *"$synonym"* ]]; then
                    match_found=true
                    break
                fi
            done

            if [[ $match_found == true ]]; then
                correct_uberon_count=$((correct_uberon_count + 1))
                result="Correct"
                echo "$accession,$extracted_code,${uberon_map[$extracted_code]},$extracted_text,$result" >> "$detailed_output"
            else
                incorrect_uberon_count=$((incorrect_uberon_count + 1))
                result="Incorrect"
                echo "$accession,$extracted_code,${uberon_map[$extracted_code]},$extracted_text,$result" >> "$detailed_output"
            fi
        else
            incorrect_uberon_count=$((incorrect_uberon_count + 1))
            result="Incorrect"
            echo "$accession,$extracted_code,N/A,$extracted_text,$result" >> "$detailed_output"
        fi
    else
        no_code_count=$((no_code_count + 1))
        echo "$accession,N/A,N/A,$uberon_info,No Code Found" >> "$detailed_output"
    fi

done < <(tail -n +2 "$input_file_1")

# Calcul de la proportion de lignes vides
if [[ $total_line_count -gt 0 ]]; then
    empty_line_proportion=$(echo "scale=2; $empty_line_count/$total_line_count*100" | bc)
else
    empty_line_proportion=0
fi

# Écriture des statistiques finales
# Ce fichier résume les statistiques sur les correspondances UBERON
> "$output_stats"
echo "Correct UBERON codes,$correct_uberon_count" >> "$output_stats"
echo "Incorrect UBERON codes,$incorrect_uberon_count" >> "$output_stats"
echo "Lines with no UBERON code,$no_code_count" >> "$output_stats"
echo "Total lines,$total_line_count" >> "$output_stats"
echo "Empty lines,$empty_line_count" >> "$output_stats"
echo "Proportion of empty lines,$empty_line_proportion%" >> "$output_stats"
