##########################################################################################
# IMPORT
import re
import os
import pandas as pd
import argparse

##########################################################################################
# PATHS
parser = argparse.ArgumentParser(description="Clean raw final info and override with initial metadata")
parser.add_argument("--base_path", type=str, required=True, help="Base path to MetaMap")
args = parser.parse_args()

base_dir = args.base_path

raw_final_info_path = os.path.join(base_dir, "raw_final_info.txt")
initial_raw_metadata_path = os.path.join(base_dir, "initial_raw_metadata.txt")
output_file_path = os.path.join(base_dir, "final_llm_sample_analysis.csv")
FLAG_PATH = os.path.join(base_dir, "STEP3.flag")

# base_dir="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap"
# raw_final_info_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/tmp/raw_final_info.txt"
# initial_raw_metadata_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/tmp/initial_raw_metadata.txt"
# output_file_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/SPECIFIC_RUN_ANALYSIS/final_llm_sample_analysis.csv"

##########################################################################################
# FUNCTIONS

def clean_text(text):
    return re.sub(r"[^a-zA-Z0-9,:-]", " ", text)

##########################################################################################
# MAIN


#charge files
raw_final_info_df = pd.read_csv(raw_final_info_path, sep='|', dtype=str)
raw_final_info_df = raw_final_info_df.map(lambda x: clean_text(x) if isinstance(x, str) else x)
initial_raw_metadata_df = pd.read_csv(initial_raw_metadata_path, sep='|', dtype=str)
initial_raw_metadata_df = initial_raw_metadata_df.map(lambda x: clean_text(x) if isinstance(x, str) else x)

#update col if != NA
columns_to_update = ["Tissue type", "Cell line", "Cell type", "Library strategy", "Instrument platform", "Donor information"]

#data fusion
for col in columns_to_update:
    mask = (initial_raw_metadata_df[col] != 'NA') & (initial_raw_metadata_df[col].notna())
    raw_final_info_df.loc[mask, col] = initial_raw_metadata_df.loc[mask, col]

#save final output file
raw_final_info_df = raw_final_info_df.astype(str)
raw_final_info_df.fillna('NA', inplace=True)
raw_final_info_df.to_csv(output_file_path, sep='\t', index=False)
