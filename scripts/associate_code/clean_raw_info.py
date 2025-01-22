import re
import os
import pandas as pd

input_file = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/tmp/raw_final_info.txt"
output_file = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/SPECIFIC_RUN_ANALYSIS/final_llm_sample_analysis.csv"


def clean_column(value):
    return re.sub(r"[^a-zA-Z0-9,:-]", " ", value)


if not os.path.exists(input_file):
    raise FileNotFoundError(f"Le fichier d'entrée {input_file} n'existe pas.")

with open(input_file, "r") as file:
    lines = file.readlines()

header = lines[0].strip().split('|')
data = [line.strip().split('|') for line in lines[1:]]
cleaned_data = [[clean_column(cell) for cell in row] for row in data]
cleaned_df = pd.DataFrame(cleaned_data, columns=header)
os.makedirs(os.path.dirname(output_file), exist_ok=True)
cleaned_df.to_csv(output_file, sep="\t", index=False)
