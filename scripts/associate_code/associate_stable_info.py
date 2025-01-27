#IMPORT
import os
import re
import csv

##########################################################################################
#PATHS
LLM_OUT = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/SPECIFIC_RUN_ANALYSIS/INFO_BIO_LLM"
OUTPUT_FILE = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/tmp/raw_final_info.txt"
TEMP_FILE = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/tmp/raw_final_info_temp.txt"

if not os.path.exists(LLM_OUT):
    print(f"Error: Directory {LLM_OUT} does not exist.")
    exit(1)
if not os.path.exists(OUTPUT_FILE):
    print(f"Error: File {OUTPUT_FILE} does not exist.")
    exit(1)


##########################################################################################
#FUNCTIONS

def clean_info(info):
    info = re.sub(r"\(.*?(Inferred|based on).*?\)", "", info)
    info = re.sub(r"\b(Inferred|based on)\b.*", "", info, flags=re.IGNORECASE)
    info = re.sub(r"\b(Not specified|estimated|Inferred|based on)\b", "", info, flags=re.IGNORECASE)
    return re.sub(r"[^a-zA-Z0-9\s]", " ", info.strip()).strip()


def normalize_line(line):
    return re.sub(r"[^a-zA-Z0-9:\s]", " ", line).strip()


##########################################################################################
#MAIN

with open(OUTPUT_FILE, 'r') as infile:
    lines = infile.readlines()

header = lines[0].strip().split('|')
data = [line.strip().split('|') for line in lines[1:]]

try:
    tissue_index = header.index("Tissue type")
    cell_line_index = header.index("Cell line")
    cell_type_index = header.index("Cell type")
except ValueError as e:
    print("Error: Required columns are missing in the output file.")
    exit(1)

updated_data = []
for row in data:
    run_accession_number = row[0]

    tissue_type = row[tissue_index] if row[tissue_index] != "NA" else "NA"
    cell_line = row[cell_line_index] if row[cell_line_index] != "NA" else "NA"
    cell_type = row[cell_type_index] if row[cell_type_index] != "NA" else "NA"

    file_path = os.path.join(LLM_OUT, f"{run_accession_number}_bio.txt")
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                normalized_line = normalize_line(line)
                if normalized_line.startswith("Tissue type:"):
                    extracted = clean_info(normalized_line.split("Tissue type:", 1)[1].strip())
                    if extracted:
                        tissue_type = extracted
                elif normalized_line.startswith("Cell line:"):
                    extracted = clean_info(normalized_line.split("Cell line:", 1)[1].strip())
                    if extracted:
                        cell_line = extracted
                elif normalized_line.startswith("Cell type:"):
                    extracted = clean_info(normalized_line.split("Cell type:", 1)[1].strip())
                    if extracted:
                        cell_type = extracted

    row[tissue_index] = tissue_type
    row[cell_line_index] = cell_line
    row[cell_type_index] = cell_type

    updated_data.append(row)

with open(OUTPUT_FILE, 'w', newline='') as outfile:
    writer = csv.writer(outfile, delimiter='|')
    writer.writerow(header)
    writer.writerows(updated_data)

