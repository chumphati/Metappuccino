########################################################################################################################
#IMPORT
import csv
import argparse
import os

########################################################################################################################
#PATHS
parser = argparse.ArgumentParser(description="Fetch information with Cellosaurus")
parser.add_argument("--base_path", type=str, required=True, help="Base path to Metappuccino")
parser.add_argument("--input_logan_path", type=str, required=True, help="Path to logan csv file, sample_acc = runs accessions")
args = parser.parse_args()

base_path = args.base_path
rpl27a_path = args.input_logan_path
metadata_sra_path = os.path.join(base_path, "cleaned_metadata_sra.txt")
metadata_sra_out_path = os.path.join(base_path, "metadata_sra_with_logan.txt")

########################################################################################################################
#MAIN
print("Logan seach analysis", flush=True)
with open(rpl27a_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    sample_accs = []
    for row in reader:
        # print(row['ID'])
        sample_accs.append(row['ID'])

runs = []
infos = []

with open(metadata_sra_path, newline='') as tsvfile:
    reader = csv.DictReader(tsvfile, delimiter='\t')
    for row in reader:
        run_accession = row['run_accession']
        # print(run_accession)
        if run_accession in sample_accs:
            info = []
            for k in row:
                if k != 'run_accession':
                    info.append(str(row[k]))
            runs.append(run_accession)
            infos.append('\t'.join(info))

logan_map = {}
with open(rpl27a_path, newline='') as loganfile:
    reader = csv.DictReader(loganfile)
    for row in reader:
        run_acc = row['ID']
        info = ';'.join([str(row[k]) for k in row if k != 'ID'])
        logan_map[run_acc] = info


with open(metadata_sra_path, newline='') as infile, open(metadata_sra_out_path, 'w', newline='') as outfile:
    reader = csv.DictReader(infile, delimiter='\t')
    fieldnames = reader.fieldnames + ['logan_info']
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    for row in reader:
        run_acc = row['run_accession']
        row['logan_info'] = logan_map.get(run_acc, '')
        writer.writerow(row)