##########################################################################################
#IMPORT
import os
import csv
import argparse

parser = argparse.ArgumentParser(description="Process metadata with LLM")
parser.add_argument("--base_path", type=str, required=True, help="Base path to Metappuccino")
args = parser.parse_args()
base_path = args.base_path

ref_cleaned_metadata = os.path.join(base_path, "cleaned_metadata_sra.txt")
final_csv = os.path.join(base_path, "final_llm_sample_analysis.csv")
final_toout = os.path.join(base_path, "tmp_final_llm_sample_analysis.csv")

##########################################################################################
#ADD STUDY ACCESSION

#ref study file
metadata_dict = {}
with open(ref_cleaned_metadata, 'r', newline='') as metadata_file:
    reader = csv.DictReader(metadata_file, delimiter='\t')
    for row in reader:
        metadata_dict[row['run_accession']] = row['study_accession']

#final output
with open(final_csv, 'r', newline='') as analysis_file:
    reader = csv.reader(analysis_file, delimiter='\t')
    header = next(reader)
    data = list(reader)

#add study col
header.insert(1, 'study accession')
for row in data:
    run_accession_number = row[0]
    row.insert(1, metadata_dict.get(run_accession_number, ''))

#save and update header
new_header = "run_accession_number\tstudy_accession\tnumber_base_pairs\ttissue_type\tcell_line\tcell_type\tuberon_code\tuberon_term\tdot_code\tdot_term\ttreatment\ttreatment_time\tresponse\tphenotype\tlibrary_strategy\tlibrary_selection_fixed\tlibrary_source\tinstrument_platform\tdonor_information"
with open(final_toout, 'w', newline='') as output_file:
    writer = csv.writer(output_file, delimiter='\t')
    writer.writerow(new_header.split('\t'))
    writer.writerows(data)
