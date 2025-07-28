##########################################################################################
# IMPORT
import os
import argparse
import csv

parser = argparse.ArgumentParser(description="Process metadata with LLM")
parser.add_argument("--base_path", type=str, required=True, help="Base path to Metappuccino")
args = parser.parse_args()
base_path = args.base_path

final_file = os.path.join(base_path, "final_llm_sample_analysis.csv")
ref_file = os.path.join(base_path, "CELLOSAURUS_CLEAN.csv")

##########################################################################################
#FUNCTIONS


def update_cell_line_names(base_path):
    with open(final_file, "r", newline="", encoding="utf-8") as f_final:
        final_reader = csv.DictReader(f_final, delimiter="\t")
        final_rows = list(final_reader)
        fieldnames = final_reader.fieldnames

    with open(ref_file, "r", newline="", encoding="utf-8") as f_ref:
        ref_reader = csv.DictReader(f_ref)
        ref_rows = list(ref_reader)

    #convert synonyms in dict
    synonym_dict = {}
    for row in ref_rows:
        name = row["name"]
        synonyms = str(row["synonym"]).split(';')
        for synonym in synonyms:
            synonym_dict[synonym.strip()] = name

    #update cell_line
    def update_cell_line(cell_line):
        return synonym_dict.get(cell_line, cell_line)

    for row in final_rows:
        row["Cell line"] = update_cell_line(row["Cell line"])

    #save
    updated_file = os.path.join(base_path, "tmp_final_llm_sample_analysis.csv")
    with open(updated_file, "w", newline="", encoding="utf-8") as f_updated:
        writer = csv.DictWriter(f_updated, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(final_rows)

##########################################################################################
#MAIN


if __name__ == "__main__":
    update_cell_line_names(args.base_path)
