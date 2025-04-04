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
treatment_high_entropy = os.path.join(base_path, "treatment_high_entropy.txt")
treatmenttime_high_entropy = os.path.join(base_path, "treatmenttime_high_entropy.txt")
res_high_entropy = os.path.join(base_path, "res_high_entropy.txt")
phe_high_entropy = os.path.join(base_path, "phenotype_high_entropy.txt")
libselec_high_entropy = os.path.join(base_path, "libselec_high_entropy.txt")
libsource_high_entropy = os.path.join(base_path, "libsource_high_entropy.txt")

entropy_tt = 1.7
entropy_cl = 1.5
entropy_ct = 2
entropy_treatment = 1.5
entropy_treattime = 1.5
entropy_res = 1.5
entropy_phe = 1.5
entropy_libselec = 1.5
entropy_libsource = 1.5

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
    treatment_index = header.index("Treatment")
    treatmenttime_index = header.index("Treatment Time")
    res_index = header.index("Response")
    phe_index = header.index("Phenotype")
    libselec_index = header.index("Library selection fixed")
    libsource_index = header.index("Library source")
    donor_information_index = header.index("Donor information")
except ValueError:
    print("Error: Required columns are missing in the output file.")
    exit(1)

updated_data = []
tt_high_entropy_rows = []
cl_high_entropy_rows = []
ct_high_entropy_rows = []
treatment_high_entropy_rows = []
treatmenttime_high_entropy_rows = []
res_high_entropy_rows = []
phe_high_entropy_rows = []
libselec_high_entropy_rows = []
libsource_high_entropy_rows = []

for row in data:
    run_accession_number = row[0]
    print(run_accession_number, flush=True)

    tissue_type = "NA"
    cell_line = "NA"
    cell_type = "NA"
    donor_information = "NA"
    treatment = "NA"
    treatmenttime = "NA"
    res = "NA"
    phe = "NA"
    libselec = "NA"
    libsource = "NA"

    tissue_entropy = None
    cell_line_entropy = None
    cell_type_entropy = None
    treatment_entropy = None
    treatmenttime_entropy = None
    res_entropy = None
    phe_entropy = None
    libselec_entropy = None
    libsource_entropy = None
    donor_information_entropy = None

    tissue_full_line = None
    cell_line_full_line = None
    cell_type_full_line = None
    treatment_full_line = None
    treatmenttime_full_line = None
    res_full_line = None
    phe_full_line = None
    libselec_full_line = None
    libsource_full_line = None

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
                elif "Treatment Entropy" in line:
                    entropy_dict["Treatment"] = entropy
                elif "Treatment Time Entropy" in line:
                    entropy_dict["Treatment Time"] = entropy
                elif "Response Entropy" in line:
                    entropy_dict["Response"] = entropy
                elif "Phenotype Entropy" in line:
                    entropy_dict["Phenotype"] = entropy
                elif "Library selection fixed Entropy" in line:
                    entropy_dict["Library selection fixed"] = entropy
                elif "Library source Entropy" in line:
                    entropy_dict["Library source"] = entropy
                elif "Donor information Entropy" in line:
                    entropy_dict["Donor information"] = entropy
        print("totoooooo")
        print(run_accession_number, flush=True)
        # print(entropy_dict)
        for line in lines:
            print(line, flush=True)
            normalized_line = line.strip()
            # normalized_line = re.sub(r"^\d+[\.\)\-]\s*", "", normalized_line)
            normalized_line = re.sub(r"^\d+[\.\)\-\*]*\s*", "", normalized_line)
            print("norm line", flush=True)
            print(normalized_line, flush=True)

            # if normalized_line.startswith("Tissue type:"):
            if re.match(r"^[^a-zA-Z0-9]*Tissue type[^a-zA-Z0-9]*:", normalized_line):
                entropy = entropy_dict.get("Tissue type", None)
                tissue_full_line = normalized_line
                if entropy is not None and entropy < entropy_tt:
                    tissue_info = clean_info(normalized_line.split("Tissue type:", 1)[1].strip())
                    tissue_type = f"{tissue_info} (e={entropy})"
                else:
                    tt_high_entropy_rows.append([run_accession_number, entropy, tissue_full_line])
                    continue

            # elif normalized_line.startswith("Cell line:"):
            elif re.match(r"^[^a-zA-Z0-9]*Cell line[^a-zA-Z0-9]*:", normalized_line):
                entropy = entropy_dict.get("Cell line", None)
                cell_line_full_line = normalized_line
                if entropy is not None and entropy < entropy_cl:
                    cell_line_info = clean_info(normalized_line.split("Cell line:", 1)[1].strip())
                    cell_line = f"{cell_line_info} (e={entropy})"
                else:
                    cl_high_entropy_rows.append([run_accession_number, entropy, cell_line_full_line])
                    continue

            # elif normalized_line.startswith("Cell type:"):
            elif re.match(r"^[^a-zA-Z0-9]*Cell type[^a-zA-Z0-9]*:", normalized_line):
                entropy = entropy_dict.get("Cell type", None)
                cell_type_full_line = normalized_line
                if entropy is not None and entropy < entropy_ct:
                    cell_type_info = clean_info(normalized_line.split("Cell type:", 1)[1].strip())
                    cell_type = f"{cell_type_info} (e={entropy})"
                else:
                    ct_high_entropy_rows.append([run_accession_number, entropy, cell_type_full_line])
                    continue

            # elif normalized_line.startswith("Treatment:"):
            elif re.match(r"^[^a-zA-Z0-9]*Treatment[^a-zA-Z0-9]*:", normalized_line):
                entropy = entropy_dict.get("Treatment", None)
                treatment_full_line = normalized_line
                if entropy is not None and entropy < entropy_treatment:
                    treatment_info = clean_info(normalized_line.split("Treatment:", 1)[1].strip())
                    treatment = f"{treatment_info} (e={entropy})"
                else:
                    treatment_high_entropy_rows.append([run_accession_number, entropy, treatment_full_line])
                    continue

            # elif normalized_line.startswith("Treatment Time:"):
            elif re.match(r"^[^a-zA-Z0-9]*Treatment Time[^a-zA-Z0-9]*:", normalized_line):
                entropy = entropy_dict.get("Treatment Time", None)
                treatmenttime_full_line = normalized_line
                if entropy is not None and entropy < entropy_treattime:
                    treatmenttime_info = clean_info(normalized_line.split("Treatment Time:", 1)[1].strip())
                    treatmenttime = f"{treatmenttime_info} (e={entropy})"
                else:
                    treatmenttime_high_entropy_rows.append([run_accession_number, entropy, treatmenttime_full_line])
                    continue

            # elif normalized_line.startswith("Response:"):
            elif re.match(r"^[^a-zA-Z0-9]*Response[^a-zA-Z0-9]*:", normalized_line):
                entropy = entropy_dict.get("Response", None)
                res_full_line = normalized_line
                if entropy is not None and entropy < entropy_res:
                    res_info = clean_info(normalized_line.split("Response:", 1)[1].strip())
                    res = f"{res_info} (e={entropy})"
                else:
                    res_high_entropy_rows.append([run_accession_number, entropy, res_full_line])
                    continue

            # elif normalized_line.startswith("Phenotype:"):
            elif re.match(r"^[^a-zA-Z0-9]*Phenotype[^a-zA-Z0-9]*:", normalized_line):
                entropy = entropy_dict.get("Phenotype", None)
                phe_full_line = normalized_line
                if entropy is not None and entropy < entropy_phe:
                    phe_info = clean_info(normalized_line.split("Phenotype:", 1)[1].strip())
                    phe = f"{phe_info} (e={entropy})"
                else:
                    phe_high_entropy_rows.append([run_accession_number, entropy, phe_full_line])
                    continue

            # elif normalized_line.startswith("Library selection fixed:"):
            elif re.match(r"^[^a-zA-Z0-9]*Library selection fixed[^a-zA-Z0-9]*:", normalized_line):
                entropy = entropy_dict.get("Library selection fixed", None)
                libselec_full_line = normalized_line
                if entropy is not None and entropy < entropy_libselec:
                    libselec_info = clean_info(normalized_line.split("Library selection fixed:", 1)[1].strip())
                    libselec = f"{libselec_info} (e={entropy})"
                else:
                    libselec_high_entropy_rows.append([run_accession_number, entropy, libselec_full_line])
                    continue

            # elif normalized_line.startswith("Library source:"):
            elif re.match(r"^[^a-zA-Z0-9]*Library source[^a-zA-Z0-9]*:", normalized_line):
                entropy = entropy_dict.get("Library source", None)
                libsource_full_line = normalized_line
                if entropy is not None and entropy < entropy_libsource:
                    libsource_info = clean_info(normalized_line.split("Library source:", 1)[1].strip())
                    libsource = f"{libsource_info} (e={entropy})"
                else:
                    phe_high_entropy_rows.append([run_accession_number, entropy, libsource_full_line])
                    continue

            # elif normalized_line.startswith("Donor information:"):
            elif re.match(r"^[^a-zA-Z0-9]*Donor information[^a-zA-Z0-9]*:", normalized_line):
                entropy = entropy_dict.get("Donor information", None)
                donor_info = normalized_line.split("Donor information:", 1)[1].strip()
                if entropy is not None:
                    donor_information = f"{donor_info} (e={entropy})"
                else:
                    donor_information = donor_info

    row[tissue_index] = tissue_type
    row[cell_line_index] = cell_line
    row[cell_type_index] = cell_type
    row[treatment_index] = treatment
    row[treatmenttime_index] = treatmenttime
    row[res_index] = res
    row[phe_index] = phe
    row[libselec_index] = libselec
    row[libsource_index] = libsource
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
#treatment
with open(treatment_high_entropy, 'w') as treatment_high_entropy_file:
    treatment_high_entropy_file.write("Run accession number|Entropy|Treatment\n")
    for row in treatment_high_entropy_rows:
        treatment_high_entropy_file.write('|'.join(map(str, row)) + '\n')
#treatment time
with open(treatmenttime_high_entropy, 'w') as treatmenttime_high_entropy_file:
    treatmenttime_high_entropy_file.write("Run accession number|Entropy|Treatment Time\n")
    for row in treatmenttime_high_entropy_rows:
        treatmenttime_high_entropy_file.write('|'.join(map(str, row)) + '\n')
#response
with open(res_high_entropy, 'w') as res_high_entropy_file:
    res_high_entropy_file.write("Run accession number|Entropy|Response\n")
    for row in res_high_entropy_rows:
        res_high_entropy_file.write('|'.join(map(str, row)) + '\n')
#phenotype
with open(phe_high_entropy, 'w') as phe_high_entropy_file:
    phe_high_entropy_file.write("Run accession number|Entropy|Phenotype\n")
    for row in phe_high_entropy_rows:
        phe_high_entropy_file.write('|'.join(map(str, row)) + '\n')
#library selection fixed
with open(libselec_high_entropy, 'w') as libselec_high_entropy_file:
    libselec_high_entropy_file.write("Run accession number|Entropy|Library selection fixed\n")
    for row in libselec_high_entropy_rows:
        libselec_high_entropy_file.write('|'.join(map(str, row)) + '\n')
#library source
with open(libsource_high_entropy, 'w') as libsource_high_entropy_file:
    libsource_high_entropy_file.write("Run accession number|Entropy|Library source\n")
    for row in libsource_high_entropy_rows:
        libsource_high_entropy_file.write('|'.join(map(str, row)) + '\n')
