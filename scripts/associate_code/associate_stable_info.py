import os
import re
import csv
import argparse

##########################################################################################
# PATHS
parser = argparse.ArgumentParser(description="Associate raw LLM information to final output file")
parser.add_argument("--base_path", type=str, required=True, help="Base path to MetaMap")
args = parser.parse_args()

base_path = args.base_path
LLM_OUT = os.path.join(base_path, "INFO_BIO_LLM")
OUTPUT_FILE = os.path.join(base_path, "raw_final_info.txt")
TEMP_FILE = os.path.join(base_path, "raw_final_info_temp.txt")

if not os.path.exists(LLM_OUT):
    print(f"Error: Directory {LLM_OUT} does not exist.")
    exit(1)
if not os.path.exists(OUTPUT_FILE):
    print(f"Error: File {OUTPUT_FILE} does not exist.")
    exit(1)


##########################################################################################
# FUNCTIONS

def clean_info(info):
    info = re.sub(r"\(.*?(Inferred|based on).*?\)", "", info)
    info = re.sub(r"\b(Inferred|based on)\b.*", "", info, flags=re.IGNORECASE)
    info = re.sub(r"\b(Not specified|estimated|Inferred|based on)\b", "", info, flags=re.IGNORECASE)
    return re.sub(r"[^a-zA-Z0-9\s]", " ", info.strip()).strip()


def normalize_line(line):
    return re.sub(r"[^a-zA-Z0-9:\s]", " ", line).strip()


def extract_entropy(line):
    match = re.search(r"([0-9]+\.[0-9]+)$", line)
    return float(match.group(1)) if match else None


##########################################################################################
# MAIN

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

    tissue_type = "NA"
    cell_line = "NA"
    cell_type = "NA"

    tissue_entropy = None
    cell_line_entropy = None
    cell_type_entropy = None

    file_path = os.path.join(LLM_OUT, f"{run_accession_number}_bio.txt")
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                entropy = extract_entropy(line)
                normalized_line = normalize_line(line)

                if "Entropy" in line and entropy is not None:
                    if "Tissue type Entropy" in line:
                        tissue_entropy = entropy
                    elif "Cell line Entropy" in line:
                        cell_line_entropy = entropy
                    elif "Cell type Entropy" in line:
                        cell_type_entropy = entropy
                else:
                    if normalized_line.startswith("Tissue type:") and (
                            tissue_entropy is not None and tissue_entropy < 2.5):
                        tissue_type = clean_info(normalized_line.split("Tissue type:", 1)[1].strip())
                    elif normalized_line.startswith("Cell line:") and (
                            cell_line_entropy is not None and cell_line_entropy < 2.5):
                        cell_line = clean_info(normalized_line.split("Cell line:", 1)[1].strip())
                    elif normalized_line.startswith("Cell type:") and (
                            cell_type_entropy is not None and cell_type_entropy < 2.5):
                        cell_type = clean_info(normalized_line.split("Cell type:", 1)[1].strip())

    row[tissue_index] = tissue_type
    row[cell_line_index] = cell_line
    row[cell_type_index] = cell_type
    updated_data.append(row)

with open(OUTPUT_FILE, 'w', newline='') as outfile:
    writer = csv.writer(outfile, delimiter='|')
    writer.writerow(header)
    writer.writerows(updated_data)
