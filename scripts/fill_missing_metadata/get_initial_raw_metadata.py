########################################################################################################################
#IMPORT LIB
import pandas as pd
import os
import argparse

########################################################################################################################
#PATHS
parser = argparse.ArgumentParser(description="Process raw matadata information directly from database")
parser.add_argument("--base_path", type=str, required=True, help="Base path to MetaMap")
args = parser.parse_args()

base_path = args.base_path
input_file = os.path.join(base_path, "cleaned_metadata_sra.txt")
output_file = os.path.join(base_path, "initial_raw_metadata.txt")

# input_file = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/METADATA/cleaned_metadata_sra.txt"
# output_file = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/tmp/initial_raw_metadata.txt"
# BASE_PATH = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results"

########################################################################################################################
#DECLARE COLUMNS
#column to extract in input and map output
columns_mapping = {
    "run_accession": "Run accession number",
    "base_count": "Number of base pairs",
    "tissue_type": "Tissue type",
    "cell_line": "Cell line",
    "cell_type": "Cell type",
    "library_strategy": "Library strategy",
    "instrument_platform": "Instrument platform",
    "host_phenotype": "Phenotype",
}

#final columns out
final_columns = [
    "Run accession number",
    "Number of base pairs",
    "Tissue type",
    "Cell line",
    "Cell type",
    "UBERON organ and code",
    "Disease Ontology Term",
    "Treatment",
    "Treatment Time",
    "Response",
    "Phenotype",
    "Library strategy",
    "Library selection fixed",
    "Library source",
    "Instrument platform",
    "Donor information"
]

########################################################################################################################
#PROCESS METADATA

#get metadata
df = pd.read_csv(input_file, sep="\t")
#select columns with good header
final_df = pd.DataFrame()

#get infos
for original_col, new_col in columns_mapping.items():
    if original_col in df.columns:
        final_df[new_col] = df[original_col]
    else:
        final_df[new_col] = "NA"  #if column doesn't exist = NA

#add missed columns with NA
for col in final_columns:
    if col not in final_df.columns:
        final_df[col] = "NA"

#reorganise columns ans save
final_df = final_df.fillna("NA")
final_df = final_df[final_columns]
final_df.to_csv(output_file, index=False, sep="|")
