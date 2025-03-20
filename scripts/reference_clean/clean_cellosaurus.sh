#!/bin/bash

# CREATE CLEAN CELLOSAURUS

input_file="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/data/raw/cellosaurus.txt"
output_file="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/data/CELLOSAURUS_CLEAN.csv"

echo "id_cell,name,synonym,disease" > "$output_file"

current_id=""
current_name=""
current_synonyms=""
current_disease=""

while IFS= read -r line || [ -n "$line" ]; do
    if [[ $line =~ ^id:\ (CVCL_[A-Z0-9]+) ]]; then
        if [[ -n $current_id ]]; then
            echo "$current_id,$current_name,\"$current_synonyms\",\"$current_disease\"" >> "$output_file"
        fi
        current_id="${BASH_REMATCH[1]}"
        current_name=""
        current_synonyms=""
        current_disease=""
    elif [[ $line =~ ^name:\ *(.*) ]]; then
        current_name="$(echo "${BASH_REMATCH[1]}" | sed 's/#//g')"  # Supprime tous les #
    elif [[ $line =~ ^synonym:\ \"(.+)\" ]]; then
        if [[ -n $current_synonyms ]]; then
            current_synonyms="$current_synonyms;${BASH_REMATCH[1]}"
        else
            current_synonyms="${BASH_REMATCH[1]}"
        fi
    elif [[ $line =~ ^xref:\ NCIt:[A-Za-z0-9]+\ !\ (.+) ]]; then
        if [[ -n $current_disease ]]; then
            current_disease="$current_disease;${BASH_REMATCH[1]}"
        else
            current_disease="${BASH_REMATCH[1]}"
        fi
    fi
done < "$input_file"

if [[ -n $current_id ]]; then
    echo "$current_id,$current_name,\"$current_synonyms\",\"$current_disease\"" >> "$output_file"
fi
