##########################################################################################
# IMPORT
import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Process metadata with LLM")
parser.add_argument("--base_path", type=str, required=True, help="Base path to MetaMap")
args = parser.parse_args()
base_path = args.base_path

final_file = os.path.join(base_path, "final_llm_sample_analysis.csv")
ref_file = os.path.join(base_path, "CELLOSAURUS_CLEAN.csv")

##########################################################################################
#FUNCTIONS


def update_cell_line_names(base_path):
    df_final = pd.read_csv(final_file)
    df_ref = pd.read_csv(ref_file)

    #convert synonyms in dict
    synonym_dict = {}
    for _, row in df_ref.iterrows():
        name = row["name"]
        synonyms = str(row["synonym"]).split(';')
        for synonym in synonyms:
            synonym_dict[synonym.strip()] = name

    #update cell_line
    def update_cell_line(cell_line):
        return synonym_dict.get(cell_line, cell_line)

    df_final["cell_line"] = df_final["cell_line"].apply(update_cell_line)

    #save
    updated_file = os.path.join(base_path, "tmp_final_llm_sample_analysis.csv")
    df_final.to_csv(updated_file, index=False)

##########################################################################################
#MAIN


if __name__ == "__main__":
    update_cell_line_names(args.base_path)
