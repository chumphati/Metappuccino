#!/bin/bash

#CREATE CLEAN UBERON COMP TABLE

input_file="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/data/REF_UBERON.txt"
output_file="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/results/final_out_llm/UBERON_TABLE_CLEAN.csv"

echo "code_uberon,name,synonym" > "$output_file"

current_id=""
current_name=""
current_synonyms=""

while IFS= read -r line || [ -n "$line" ]; do
    if [[ $line =~ ^id:\ (UBERON:[0-9]+) ]]; then
        if [[ -n $current_id ]]; then
            echo "$current_id,$current_name,\"$current_synonyms\"" >> "$output_file"
        fi
        current_id="${BASH_REMATCH[1]}"
        current_name=""
        current_synonyms=""
    elif [[ $line =~ ^name:\ (.+) ]]; then
        current_name="${BASH_REMATCH[1]}"
    elif [[ $line =~ ^synonym:\ \"(.+)\" ]]; then
        #synonym
        if [[ -n $current_synonyms ]]; then
            current_synonyms="$current_synonyms;${BASH_REMATCH[1]}"
        else
            current_synonyms="${BASH_REMATCH[1]}"
        fi
    fi
done < "$input_file"

if [[ -n $current_id ]]; then
    echo "$current_id,$current_name,\"$current_synonyms\"" >> "$output_file"
fi
