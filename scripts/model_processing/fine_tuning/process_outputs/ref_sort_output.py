import pandas as pd
import re

prompt_file = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/finetune_data.csv"
output_file = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/ref_sort_output.csv"


def extract_run_accession(text):
    match = re.search(r'Run accession:\s*(\S+)', str(text))
    return match.group(1) if match else None


df = pd.read_csv(prompt_file)
df['run_accession'] = df['prompt'].apply(extract_run_accession)
ref_df = df[['run_accession', 'output']]
ref_df.to_csv(output_file, index=False)
