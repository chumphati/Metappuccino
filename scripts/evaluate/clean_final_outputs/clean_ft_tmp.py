import os
import pandas as pd

csv_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/final_MetaMap_LLM_results/results_mistral7B_v4_FT_original/SPECIFIC_RUN_ANALYSIS/mistral7BFToriginal_final_llm_sample_analysis.csv"
bio_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/final_MetaMap_LLM_results/results_mistral7B_v4_FT_original/SPECIFIC_RUN_ANALYSIS/INFO_BIO_LLM"
study_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/final_MetaMap_LLM_results/results_mistral7B_v4_FT_original/SPECIFIC_RUN_ANALYSIS/INFO_STUDY_LLM"

df = pd.read_csv(csv_path, sep='\t')
if 'dot_term' not in df.columns:
    df['dot_term'] = ""
if 'uberon_term' not in df.columns:
    df['uberon_term'] = ""

for idx, row in df.iterrows():
    run_accession = str(row[0]).strip()
    study_accession = str(row['study_accession']).strip()
    dot_value = ""
    bio_file = os.path.join(bio_dir, f"{run_accession}_bio.txt")
    if os.path.isfile(bio_file):
        with open(bio_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith("Disease Ontology Term:"):
                    dot_value = line.strip().split("Disease Ontology Term:", 1)[1].strip()
                    break

    if not dot_value:
        study_file = os.path.join(study_dir, f"{study_accession}_study.txt")
        if os.path.isfile(study_file):
            with open(study_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip().startswith("DOT term:"):
                        dot_value = line.strip().split("DOT term:", 1)[1].strip()
                        break

    if not dot_value:
        dot_value = "nan"

    df.at[idx, 'dot_term'] = dot_value
    uberon_value = ""

    if os.path.isfile(bio_file):
        with open(bio_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith("UBERON organ and code:"):
                    uberon_value = line.strip().split("UBERON organ and code:", 1)[1].strip()
                    break

    if not uberon_value:
        study_file = os.path.join(study_dir, f"{study_accession}_study.txt")
        if os.path.isfile(study_file):
            with open(study_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip().startswith("UBERON term:"):
                        uberon_value = line.strip().split("UBERON term:", 1)[1].strip()
                        break

    if not uberon_value:
        uberon_value = "nan"
    df.at[idx, 'uberon_term'] = uberon_value

df = df.fillna("nan")
output_path = csv_path.replace(".csv", "_with_all.csv")
df.to_csv(output_path, sep='\t', index=False)
