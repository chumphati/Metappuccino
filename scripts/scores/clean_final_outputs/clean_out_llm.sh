#!/bin/bash

#uniform output llm
path_to_raw_llm_output="/Users/fionahak/Documents/phd/phd_code/MetaMap/results/INFO_BIO_LLM"
for file in "$path_to_raw_llm_output"/*; do
    if [ -f "$file" ]; then
        sed -i '' '/./!d' "$file"
    fi
done
