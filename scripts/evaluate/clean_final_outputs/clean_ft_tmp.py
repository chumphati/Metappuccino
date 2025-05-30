import os
import pandas as pd

csv_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/final_MetaMap_LLM_results/results_mistral7B_FT_Q4_M/SPECIFIC_RUN_ANALYSIS/final_llm_sample_analysis.csv"
txt_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/final_MetaMap_LLM_results/results_mistral7B_FT_Q4_M/SPECIFIC_RUN_ANALYSIS/INFO_BIO_LLM"

df = pd.read_csv(csv_path, sep='\t')

if 'dot_term' not in df.columns:
    df['dot_term'] = ""

for idx, row in df.iterrows():
    run_accession = str(row[0])
    txt_file = os.path.join(txt_dir, f"{run_accession}_bio.txt")
    dot_term_value = ""
    if os.path.isfile(txt_file):
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith("Disease Ontology Term:"):
                    dot_term_value = line.strip().split("Disease Ontology Term:", 1)[1].strip()
                    break
    df.at[idx, 'dot_term'] = dot_term_value

output_path = csv_path.replace(".csv", "_with_dot_term.csv")
df.to_csv(output_path, sep='\t', index=False)
