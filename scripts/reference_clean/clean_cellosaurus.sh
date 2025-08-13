#!/bin/bash

input_file="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/raw/cellosaurus.txt"
output_file="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/CELLOSAURUS_HUMANS_CLEAN.csv"

echo "id_cell,name,synonym,disease,age,sex,ethnicity,localization,biopsy_type,biopsy_site,uberon_code,cell_type,ct_code" > "$output_file"

block=""
while IFS= read -r line || [ -n "$line" ]; do
    if [[ $line == "//" ]]; then
        id_cell="" name="" synonyms="" disease="" age="" sex="" ethnicity="" localization="" biopsy_type="" biopsy_site="" uberon_code="" cell_type="" ct_code=""
        population_val=""
        is_human=0
        while IFS= read -r block_line || [ -n "$block_line" ]; do
            if [[ $block_line =~ ^OX[[:space:]]+NCBI_TaxID=([0-9]+) ]]; then
                if [[ "${BASH_REMATCH[1]}" == "9606" ]]; then
                    is_human=1
                else
                    is_human=0
                fi
            fi
            [[ -z $block_line ]] && continue
            if [[ $block_line =~ ^AC[[:space:]]+(CVCL_[A-Z0-9]+) ]]; then
                id_cell="${BASH_REMATCH[1]}"
            elif [[ $block_line =~ ^ID[[:space:]]+(.+) ]]; then
                name="${BASH_REMATCH[1]//\#/}"
            elif [[ $block_line =~ ^SY[[:space:]]+(.+) ]]; then
                raw="${BASH_REMATCH[1]//,/;}"
                IFS=';' read -ra parts <<< "$raw"
                cleaned=""
                for p in "${parts[@]}"; do
                    p="${p#"${p%%[![:space:]]*}"}"
                    p="${p%"${p##*[![:space:]]}"}"
                    cleaned+="${cleaned:+;}$p"
                done
                synonyms="$cleaned"
            elif [[ -z $disease && $block_line =~ ^DI[[:space:]]+.*\;[[:space:]]*(.+) ]]; then
                disease="${BASH_REMATCH[1]}"
            elif [[ $block_line =~ ^AG[[:space:]]+(.*) ]]; then
                [[ "${BASH_REMATCH[1]}" != "Age unspecified" ]] && age="${BASH_REMATCH[1]}"
            elif [[ $block_line =~ ^SX[[:space:]]+([^[:punct:]]+) ]]; then
                [[ "${BASH_REMATCH[1]}" != "Sex unspecified" ]] && sex="${BASH_REMATCH[1]}"
            elif [[ $block_line =~ ^CC[[:space:]]+Population:[[:space:]]*([^[:punct:]]+) ]]; then
                population_val="${BASH_REMATCH[1]}"
            elif [[ $block_line =~ ^CC[[:space:]]+Genome[[:space:]]ancestry:[[:space:]]*(.+) ]]; then
                anc_line="${BASH_REMATCH[1]}"
                declare -A cont=( [African]=0 [American]=0 [Asian]=0 [European]=0 )
                IFS=';' read -ra entries <<< "$anc_line"
                for e in "${entries[@]}"; do
                    key="${e%%=*}" val="${e#*=}" val="${val%\%}"
                    case "$key" in
                        African) cont[African]=$(echo "${cont[African]} + $val" | bc) ;;
                        Native\ American) cont[American]=$(echo "${cont[American]} + $val" | bc) ;;
                        East\ Asian,*) cont[Asian]=$(echo "${cont[Asian]} + $val" | bc) ;;
                        South\ Asian) cont[Asian]=$(echo "${cont[Asian]} + $val" | bc) ;;
                        European,*) cont[European]=$(echo "${cont[European]} + $val" | bc) ;;
                    esac
                done
                for c in African American Asian European; do
                    if (( $(echo "${cont[$c]} >= 80" | bc -l) )); then
                        ethnicity="$c"
                        break
                    fi
                done
            elif [[ $block_line =~ ^CC[[:space:]]+From:[[:space:]]*(.*) ]]; then
                localization="${BASH_REMATCH[1]}"
            elif [[ $block_line =~ ^CC[[:space:]]+Derived[[:space:]]from[[:space:]]site:[[:space:]]*(.*) ]]; then
                raw_site="${BASH_REMATCH[1]%.}"
                IFS=';' read -ra site_parts <<< "$raw_site"
                for i in "${!site_parts[@]}"; do
                    site_parts[$i]="${site_parts[$i]#"${site_parts[$i]%%[![:space:]]*}"}"
                    site_parts[$i]="${site_parts[$i]%"${site_parts[$i]##*[![:space:]]}"}"
                done
                disposition="${site_parts[0]}" organ="${site_parts[1]}"
                if [[ $disposition == "Metastatic" ]]; then
                    biopsy_type="metastasis"
                elif [[ $disposition == "In situ" ]]; then
                    if [[ "${organ,,}" =~ blood ]]; then
                        biopsy_type="blood"
                    else
                        biopsy_type="primary"
                    fi
                fi
                biopsy_site="$organ"
                [[ $raw_site =~ UBERON=([^[:space:];\.]+) ]] && uberon_code="${BASH_REMATCH[1]}"
            elif [[ $block_line =~ ^CC[[:space:]]+Cell[[:space:]]type:[[:space:]]*([[:alnum:][:space:]-]+) ]]; then
                cell_type="${BASH_REMATCH[1]}"
                [[ $block_line =~ CL=([A-Za-z0-9:_-]+) ]] && ct_code="${BASH_REMATCH[1]}"
            fi
        done <<< "$block"
        if [[ -z $ethnicity && -n $population_val ]]; then
            ethnicity="$population_val"
        fi
        if [[ $is_human -eq 1 ]]; then
            echo "$id_cell,$name,\"$synonyms\",\"$disease\",\"$age\",\"$sex\",\"$ethnicity\",\"$localization\",\"$biopsy_type\",\"$biopsy_site\",\"$uberon_code\",\"$cell_type\",\"$ct_code\"" >> "$output_file"
        fi
        block=""
    else
        block+="$line"$'\n'
    fi
done < "$input_file"
