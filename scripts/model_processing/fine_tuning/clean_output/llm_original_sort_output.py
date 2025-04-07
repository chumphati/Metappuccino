import os
import pandas as pd

input_folder = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/store/MISTRAL7B_RESULTS/SPECIFIC_RUN_ANALYSIS/INFO_BIO_LLM"
output_file = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/llm_original_sort_output.csv"

data = []
for filename in os.listdir(input_folder):
    if filename.endswith("_bio.txt"):
        run_accession = filename.replace("_bio.txt", "")
        file_path = os.path.join(input_folder, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if "UBERON organ and code:" in line:
                    output_text = line.split("UBERON organ and code:")[1].strip()
                    data.append({"run_accession": run_accession, "output": output_text})
                    break

df = pd.DataFrame(data)
df.to_csv(output_file, index=False)
