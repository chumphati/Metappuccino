import os
import re

dir_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/clean_sra_ena_records/results/final_out_llm/cleaned_table"
keywords = ["age", "diagnosis", "gender", "location", "date", "host", "bmi", "weight", "years", "kg", "race"]
pattern = re.compile(r"|".join(keywords), re.IGNORECASE)
exclude_pattern = re.compile(r"\b(inc|not specified|na)\b", re.IGNORECASE)

for file_name in os.listdir(dir_path):
    file_path = os.path.join(dir_path, file_name)
    if os.path.isfile(file_path):
        print(f"Traitement du fichier : {file_name}")

        with open(file_path, "r") as file:
            lines = file.readlines()

        total_lines = len(lines)
        match_count = 0
        valid_lines = 0

        for line in lines:
            columns = line.strip().split(",")
            if columns:
                last_column = columns[-1].strip()
                if exclude_pattern.search(last_column):
                    continue
                valid_lines += 1
                if pattern.search(last_column):
                    match_count += 1

        no_match_count = valid_lines - match_count
        match_proportion = match_count / valid_lines if valid_lines > 0 else 0
        no_match_proportion = no_match_count / valid_lines if valid_lines > 0 else 0

        print(f"With keywords: {match_proportion:.4f}")
        print(f"Without keywords: {no_match_proportion:.4f}")
        print("-----------------------------------")
