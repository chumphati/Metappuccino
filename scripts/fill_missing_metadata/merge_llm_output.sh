#!/bin/bash

#CREATE CLEAN TABLE OUTPUT FROM LLM OUT LONG PROMPT

#in/output
input_dir="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/results/LLM/llama_70B"
output_file="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/results/final_out_llm/cleaned_table/llama_70B.csv"
if [[ -f "$output_file" ]]; then
    rm "$output_file"
fi

#10 columns
echo "Run accession number,Number of base pairs,Tissue type,Cell line,Cell type,UBERON organ and code,Disease Ontology Term,Library strategy,Instrument platform,Donor information" > "$output_file"
#possible columns to transcript
columns=("Run accession number" "Number of base pairs" "Tissue type" "Cell line" "Cell type" "UBERON organ and code" "Disease Ontology Term" "Library strategy" "Instrument platform" "Donor information")

#for all the files in the out directory
for file in "$input_dir"/*; do
    #if line empty
    if [[ ! -s "$file" ]]; then
        continue
    fi
    #line to process ignoring prompt
    content=$(sed -n '35,48p' "$file")
    #initialisation
    file_data=()
    for column in "${columns[@]}"; do
        file_data+=("INC")
    done

    #extract info for each line
    while IFS= read -r line; do
        if [[ -z "$line" ]]; then
            continue
        fi

        key=$(echo "$line" | sed 's/:.*//' | xargs) #columns
        value=$(echo "$line" | sed 's/^[^:]*: //' | sed 's/\[end of text\]//g' | xargs) #
        #delete coma
        value=$(echo "$value" | tr ',' ' ')

        #is key part of columns we want
        for i in "${!columns[@]}"; do
            if [[ "${columns[$i]}" == "$key" ]]; then
                if [[ -z "$value" || "$value" == "Not specified" || "$value" == "Not specified." || "$value" == "NA" ]]; then
                    value="INC"
                fi
                file_data[$i]="$value"
                break
            fi
        done
    done <<< "$content"

    #to output
    line=$(IFS=,; echo "${file_data[*]}")
    echo "$line" >> "$output_file"
done
