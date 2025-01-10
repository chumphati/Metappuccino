#!/bin/bash

##########################################################################################
#INPUTS

input_final_out_llm="/Users/fionahak/Documents/phd/phd_code/MetaMap/results/INFO_BIO_LLM"
out_cleaned_table="/Users/fionahak/Documents/phd/phd_code/MetaMap/results/SPECIFIC_RUN_ANALYSIS/llm_generation_specific_info.csv"

if [[ -f "$out_cleaned_table" ]]; then
    rm "$out_cleaned_table"
fi

mkdir -p "/Users/fionahak/Documents/phd/phd_code/MetaMap/results/SPECIFIC_RUN_ANALYSIS"

##########################################################################################
#MAIN

#to extract
columns=("Run accession number" \
         "Number of base pairs" \
         "Tissue type" \
         "Cell line" \
         "Cell type" \
         "UBERON organ and code" \
         "Disease Ontology Term" \
         "Library strategy" \
         "Instrument platform" \
         "Donor information")

# Write the header to the output file
echo "${columns[*]}" | tr ' ' ',' > "$out_cleaned_table"

# Process each file in the input directory
for file in "$input_final_out_llm"/*; do
    # Skip empty files
    if [[ ! -s "$file" ]]; then
        continue
    fi

    # Initialize an array to store the extracted data for this file
    file_data=()
    for column in "${columns[@]}"; do
        file_data+=("INC") # Default value is "INC"
    done

    # Extract information from the file
    while IFS= read -r line; do
        if [[ -z "$line" ]]; then
            continue
        fi

        key=$(echo "$line" | sed 's/:.*//' | xargs) # Extract key
        value=$(echo "$line" | sed 's/^[^:]*: //' | sed 's/\[end of text\]//g' | xargs) # Extract value
        value=$(echo "$value" | tr ',' ' ') # Remove commas from the value

        # Check if the key matches one of the columns
        for i in "${!columns[@]}"; do
            if [[ "${columns[$i]}" == "$key" ]]; then
                if [[ -z "$value" || "$value" == "Not specified" || "$value" == "Not specified." || "$value" == "NA" ]]; then
                    value="INC"
                fi
                file_data[$i]="$value"
                break
            fi
        done
    done < "$file"

    # Write the extracted data to the output file
    line=$(IFS=,; echo "${file_data[*]}")
    echo "$line" >> "$out_cleaned_table"
done
