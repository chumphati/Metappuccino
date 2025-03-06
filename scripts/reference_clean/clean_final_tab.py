import pandas as pd
import re
import numpy as np

file_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/store/V1_MANUAL_METAMAP/SPECIFIC_RUN_ANALYSIS/final_llm_sample_analysis.csv"
df = pd.read_csv(file_path, sep="\t", dtype=str)

def clean_text(text):
    if pd.isna(text) or text.strip() == "":
        return "nan"
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\bNA\b", "nan", text)
    return text

df = df.applymap(clean_text)

cleaned_file = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/store/V1_MANUAL_METAMAP/SPECIFIC_RUN_ANALYSIS/final_llm_sample_analysis_cleaned.csv"
df.to_csv(cleaned_file, sep="\t", index=False)
