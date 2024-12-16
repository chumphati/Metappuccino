#!/bin/bash

#COUNT SAME UBERON NUMBER BETWEEN TWO FILES AND VERIFY CODE

file1="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/results/final_out_llm/cleaned_table/biollama_8B.csv"
file2="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/data/processed_ref_metadata.csv"
output="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/results/final_out_llm/mismatch_uberon/bad_uberon_biollama_8B.txt"
good_uberon="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/results/final_out_llm/mismatch_uberon/good_uberon_biollama_8B.txt"
output_stats="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/results/final_out_llm/stats/biollama_8B.csv"
detailed_output="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/results/final_out_llm/detailed_uberon_results/biollama_8B_detailed_uberon_results.txt"
uberon_ref="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/data/UBERON_TABLE_CLEAN.csv"

#counts
match_count=0
skip_count=0
correct_uberon_count=0
incorrect_uberon_count=0
no_code_count=0

> "$output"
echo "LLM // REFERENCE" >> "$output"
> "$detailed_output"
echo "Accession,Reference UBERON Code,Reference Names and Synonyms,Predicted Code,Result" >> "$detailed_output"

#find common accession number to compare the good lines
common_ids=$(comm -12 <(cut -d',' -f1 "$file1" | sort) <(cut -d',' -f1 "$file2" | sort))
#Same uberon nb+Unknown entries from LLM+output file

declare -A uberon_map
while IFS=',' read -r code name synonyms; do
    code_cleaned=$(echo "$code" | tr '[:upper:]' '[:lower:]' | xargs)
    synonyms_cleaned=$(echo "$name;$synonyms" | tr '[:upper:]' '[:lower:]' | sed 's/;/ /g' | xargs -0)
    uberon_map["$code_cleaned"]="$synonyms_cleaned"
done < "$uberon_ref"

#on common line
for id in $common_ids; do
    line1=$(grep "^$id," "$file1")
    line2=$(grep "^$id," "$file2")

    #clean extract columns
    col1=$(echo "$line1" | cut -d',' -f6 | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9: ]//g' | xargs -0)
    col2=$(echo "$line2" | cut -d',' -f6 | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9: ]//g' | xargs -0)

    #if llm "inc" ou "not specified"
    if [[ "$col1" == "inc" || "$col1" == "not specified" ]]; then
        skip_count=$((skip_count + 1))
        continue
    fi

    #extraction uberon
    if [[ "$col1" =~ (uberon:[0-9]+) ]]; then
        uberon_code=${BASH_REMATCH[1]}
        col1_text=$(echo "$col1" | sed "s/$uberon_code//g" | xargs -0)

        if [[ -n ${uberon_map["$uberon_code"]} ]]; then
            synonyms=${uberon_map["$uberon_code"]}
            match_found=false

            for synonym in $synonyms; do
                if [[ "$col1_text" == *"$synonym"* ]]; then
                    match_found=true
                    break
                fi
            done

            if [[ $match_found == true ]]; then
                correct_uberon_count=$((correct_uberon_count + 1))
                result="Correct"
            else
                incorrect_uberon_count=$((incorrect_uberon_count + 1))
                result="Incorrect"
            fi
        else
            incorrect_uberon_count=$((incorrect_uberon_count + 1))
            result="Incorrect"
        fi
        echo "$id,$uberon_code,${uberon_map[$uberon_code]},$col1_text,$result" >> "$detailed_output"
    else
        no_code_count=$((no_code_count + 1))
        echo "$id,N/A,N/A,$col1,No Code Found" >> "$detailed_output"
    fi

    #comparison words
    words_col1=($col1)
    words_col2=($col2)

    found_match=false
    for word1 in "${words_col1[@]}"; do
        for word2 in "${words_col2[@]}"; do
            if [[ "$word1" == "$word2" ]]; then
                found_match=true
                break
            fi
        done
        [[ $found_match == true ]] && break
    done

    if [[ $found_match == true ]]; then
        match_count=$((match_count + 1))
        echo "$col1 // $col2" >> "$good_uberon"
    else
        echo "$col1 // $col2" >> "$output"
    fi
done

#write output
echo "Same uberon nb,$match_count" >> "$output_stats"
echo "Unknown entries from LLM,$skip_count" >> "$output_stats"
echo "Correct UBERON codes,$correct_uberon_count" >> "$output_stats"
echo "Incorrect UBERON codes,$incorrect_uberon_count" >> "$output_stats"
echo "Lines with no code,$no_code_count" >> "$output_stats"
