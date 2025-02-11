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
tt_high_entropy = os.path.join(base_path, "tissue_high_entropy.txt")
cl_high_entropy = os.path.join(base_path, "cellline_high_entropy.txt")
ct_high_entropy = os.path.join(base_path, "celltype_high_entropy.txt")
di_high_entropy = os.path.join(base_path, "donorinfo_high_entropy.txt")

entropy_thresold = 2.5

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


def extract_entropy(line):
    match = re.search(r"([0-9]+\.[0-9]+)$", line)
    return float(match.group(1)) if match else None


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
    donor_information_index = header.index("Donor information")
except ValueError:
    print("Error: Required columns are missing in the output file.")
    exit(1)

updated_data = []
tt_high_entropy_rows = []
cl_high_entropy_rows = []
ct_high_entropy_rows = []

for row in data:
    run_accession_number = row[0]

    tissue_type = "NA"
    cell_line = "NA"
    cell_type = "NA"
    donor_information = "NA"

    tissue_entropy = None
    cell_line_entropy = None
    cell_type_entropy = None
    donor_information_entropy = None

    tissue_full_line = None
    cell_line_full_line = None
    cell_type_full_line = None
    donor_information_full_line = None

    file_path = os.path.join(LLM_OUT, f"{run_accession_number}_bio.txt")
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()

        entropy_dict = {}
        for line in lines:
            entropy = extract_entropy(line)
            if entropy is not None:
                if "Tissue type Entropy" in line:
                    entropy_dict["Tissue type"] = entropy
                elif "Cell line Entropy" in line:
                    entropy_dict["Cell line"] = entropy
                elif "Cell type Entropy" in line:
                    entropy_dict["Cell type"] = entropy
                elif "Donor information Entropy" in line:
                    entropy_dict["Donor information"] = entropy
        # print(run_accession_number)
        # print(entropy_dict)
        for line in lines:
            normalized_line = line.strip()

            if normalized_line.startswith("Tissue type:"):
                entropy = entropy_dict.get("Tissue type", None)
                tissue_full_line = normalized_line
                if entropy is not None and entropy < entropy_thresold:
                    tissue_type = clean_info(normalized_line.split("Tissue type:", 1)[1].strip())
                else:
                    tt_high_entropy_rows.append([run_accession_number, entropy, tissue_full_line])
                    continue

            elif normalized_line.startswith("Cell line:"):
                entropy = entropy_dict.get("Cell line", None)
                cell_line_full_line = normalized_line
                if entropy is not None and entropy < entropy_thresold:
                    cell_line = clean_info(normalized_line.split("Cell line:", 1)[1].strip())
                else:
                    cl_high_entropy_rows.append([run_accession_number, entropy, cell_line_full_line])
                    continue

            elif normalized_line.startswith("Cell type:"):
                entropy = entropy_dict.get("Cell type", None)
                cell_type_full_line = normalized_line
                if entropy is not None and entropy < entropy_thresold:
                    cell_type = clean_info(normalized_line.split("Cell type:", 1)[1].strip())
                else:
                    ct_high_entropy_rows.append([run_accession_number, entropy, cell_type_full_line])
                    continue

            elif normalized_line.startswith("Donor information:"):
                entropy = entropy_dict.get("Donor information", None)
                donor_information_full_line = normalized_line
                donor_information = clean_info(normalized_line.split("Donor information:", 1)[1].strip())


    row[tissue_index] = tissue_type
    row[cell_line_index] = cell_line
    row[cell_type_index] = cell_type
    row[donor_information_index] = donor_information
    updated_data.append(row)

with open(OUTPUT_FILE, 'w', newline='') as outfile:
    writer = csv.writer(outfile, delimiter='|')
    writer.writerow(header)
    writer.writerows(updated_data)

#store high entropy results
#tissue type
with open(tt_high_entropy, 'w') as tt_high_entropy_file:
    tt_high_entropy_file.write("Run accession number|Entropy|Tissue type\n")
    for row in tt_high_entropy_rows:
        tt_high_entropy_file.write('|'.join(map(str, row)) + '\n')
#cell line
with open(cl_high_entropy, 'w') as cl_high_entropy_file:
    cl_high_entropy_file.write("Run accession number|Entropy|Cell line\n")
    for row in cl_high_entropy_rows:
        cl_high_entropy_file.write('|'.join(map(str, row)) + '\n')
#cell type
with open(ct_high_entropy, 'w') as ct_high_entropy_file:
    ct_high_entropy_file.write("Run accession number|Entropy|Cell type\n")
    for row in ct_high_entropy_rows:
        ct_high_entropy_file.write('|'.join(map(str, row)) + '\n')