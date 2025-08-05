########################################################################################################################
#IMPORT
import csv
import argparse

########################################################################################################################
#PATHS
parser = argparse.ArgumentParser(description="Fetch information with Cellosaurus")
parser.add_argument("--base_path", type=str, required=True, help="Base path to Metappuccino")
parser.add_argument("--input_logan_path", type=str, required=True, help="Path to logan csv file, sample_acc = runs accessions")
args = parser.parse_args()

base_path = args.base_path
rpl27a_path = args.input_logan_path
metadata_sra_path = os.path.join(base_path, "metadata_sra.txt")
output_path = os.path.join(base_path, "logan_comp.csv")

# rpl27a_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/RPL27A_Seq3.csv"
# metadata_sra_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_mistral7B_Q4M/ORIGINAL_METADATA/metadata_sra.txt"
# output_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_mistral7B_Q4M/ORIGINAL_METADATA/logan_comp.csv"

########################################################################################################################
#MAIN
with open(rpl27a_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    sample_accs = []
    for row in reader:
        # print(row['sample_acc'])
        sample_accs.append(row['sample_acc'])

runs = []
infos = []

with open(metadata_sra_path, newline='') as tsvfile:
    reader = csv.DictReader(tsvfile, delimiter='\t')
    for row in reader:
        run_accession = row['run_accession']
        print(run_accession)
        if run_accession in sample_accs:
            info = []
            for k in row:
                if k != 'run_accession':
                    info.append(str(row[k]))
            runs.append(run_accession)
            infos.append('\t'.join(info))

with open(output_path, 'w', newline='') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(['run_accession', 'logan_info'])
    for r, i in zip(runs, infos):
        writer.writerow([r, i])
