##########################################################################################
# IMPORT
import os
import csv
from collections import defaultdict
import argparse

##########################################################################################
# PATHS
parser = argparse.ArgumentParser(description="Split specific, study and donor analysis")
parser.add_argument("--base_path", type=str, required=True, help="Base path to MetaMap")
args = parser.parse_args()

base_path = args.base_path
INPUT_FILE = os.path.join(base_path, "cleaned_metadata_sra.txt")
OUTPUT_DIR = os.path.join(base_path, "tmp")
FLAG_FILE = os.path.join(base_path, "STEP2_2.flag")

#outputs
if not os.path.isfile(INPUT_FILE):
    raise FileNotFoundError(f"Error: input file '{INPUT_FILE}' not found.")
os.makedirs(OUTPUT_DIR, exist_ok=True)

#files
# donor_info_file = os.path.join(OUTPUT_DIR, "donor_info.txt")
sample_info_file = os.path.join(OUTPUT_DIR, "sample_info.txt")
study_info_file = os.path.join(OUTPUT_DIR, "study_info.txt")

##########################################################################################
# MAIN

with open(INPUT_FILE, 'r') as infile:
    reader = csv.DictReader(infile, delimiter='\t')
    rows = list(reader)


def get_value(row, column_name):
    return row.get(column_name, "").strip() or "unknown"


#donor_info.txt
# with open(donor_info_file, 'w', newline='') as outfile:
#     writer = csv.writer(outfile, delimiter=';')
#     writer.writerow(["run_accession", "sample_metadata_ncbi", "age"])
#     for row in rows:
#         writer.writerow([
#             get_value(row, "run_accession"),
#             get_value(row, "sample metadata ncbi"),
#             get_value(row, "age")
#         ])


#sample_info.txt
with open(sample_info_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile, delimiter=';')
    writer.writerow(["run_accession", "sample_title", "sample_description", "description", "study_title", "sample_metadata_ncbi", "age"])
    for row in rows:
        writer.writerow([
            get_value(row, "run_accession"),
            get_value(row, "sample_title"),
            get_value(row, "sample_description"),
            get_value(row, "description"),
            get_value(row, "study_title"),
            get_value(row, "sample metadata ncbi"),
            get_value(row, "age")
        ])

#study_info.txt
study_data = defaultdict(lambda: {"run_accession_list": [], "library_construction_protocol": "unknown", "study_metadata_ncbi": "unknown"})
for row in rows:
    study_accession = get_value(row, "study_accession")
    study_data[study_accession]["run_accession_list"].append(get_value(row, "run_accession"))
    study_data[study_accession]["library_construction_protocol"] = get_value(row, "library_construction_protocol")
    study_data[study_accession]["study_metadata_ncbi"] = get_value(row, "study metadata ncbi")

with open(study_info_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile, delimiter=';')
    writer.writerow(["study_accession", "run_accession_list", "library_construction_protocol", "study_metadata_ncbi"])
    for study_accession, data in study_data.items():
        writer.writerow([
            study_accession,
            ",".join(data["run_accession_list"]),
            data["library_construction_protocol"],
            data["study_metadata_ncbi"]
        ])

# create flag end process before cleaning
open(FLAG_FILE, 'w').close()