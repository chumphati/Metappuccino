import os
import re
import pandas as pd

input_folder = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/store/INFO_BIO_LLM_FT"
output_file = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/llm_FT_sort_output.csv"

data = []
pattern = re.compile(r"^[A-Za-z\s]+")

for filename in os.listdir(input_folder):
    if filename.endswith("_bio.txt"):
        run_accession = filename.replace("_bio.txt", "")
        file_path = os.path.join(input_folder, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if "Organ:" in line:
                    raw_output = line.split("Organ:")[1].strip()
                    match = pattern.match(raw_output)
                    output_text = match.group(0).strip() if match else raw_output
                    data.append({"run_accession": run_accession, "output": output_text})
                    break

df = pd.DataFrame(data)
df.to_csv(output_file, index=False)
