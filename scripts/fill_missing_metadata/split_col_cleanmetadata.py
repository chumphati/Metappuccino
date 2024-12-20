import os
import csv
from collections import defaultdict

INPUT_FILE = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/CLEAN_METADATA_SRA.txt"
OUTPUT_DIR = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/LLM_METADATA_READY"

#outputs
if not os.path.isfile(INPUT_FILE):
    raise FileNotFoundError(f"Input file '{INPUT_FILE}' not found.")
os.makedirs(OUTPUT_DIR, exist_ok=True)

#files
donor_info_file = os.path.join(OUTPUT_DIR, "donor_info.txt")
sample_info_file = os.path.join(OUTPUT_DIR, "sample_info.txt")
study_info_file = os.path.join(OUTPUT_DIR, "study_info.txt")

with open(INPUT_FILE, 'r') as infile:
    reader = csv.DictReader(infile, delimiter='\t')
    rows = list(reader)

#donor_info.txt
with open(donor_info_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile, delimiter='\t')
    writer.writerow(["run_accession", "sample_metadata_ncbi", "age"])
    for row in rows:
        writer.writerow([row["run_accession"], row["sample metadata ncbi"], row["age"]])

#sample_info.txt
with open(sample_info_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile, delimiter='\t')
    writer.writerow(["run_accession", "sample_title", "sample_description", "description", "study_title"])
    for row in rows:
        writer.writerow([row["run_accession"], row["sample_title"], row["sample_description"], row["description"], row["study_title"]])

#study_info.txt
study_data = defaultdict(lambda: {"run_accession_list": [], "library_construction_protocol": "", "study_metadata_ncbi": ""})
for row in rows:
    study_accession = row["study_accession"]
    study_data[study_accession]["run_accession_list"].append(row["run_accession"])
    study_data[study_accession]["library_construction_protocol"] = row["library_construction_protocol"]
    study_data[study_accession]["study_metadata_ncbi"] = row["study metadata ncbi"]

with open(study_info_file, 'w', newline='') as outfile:
    writer = csv.writer(outfile, delimiter='\t')
    writer.writerow(["study_accession", "run_accession_list", "library_construction_protocol", "study_metadata_ncbi"])
    for study_accession, data in study_data.items():
        writer.writerow([
            study_accession,
            ",".join(data["run_accession_list"]),
            data["library_construction_protocol"],
            data["study_metadata_ncbi"]
        ])
