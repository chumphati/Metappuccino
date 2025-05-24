##########################################################################################
# IMPORT
import os
import csv
from collections import defaultdict
import argparse
import re

##########################################################################################
# PATHS
parser = argparse.ArgumentParser(description="Process metadata with LLM")
parser.add_argument("--base_path", type=str, required=True, help="Base path to MetaMap")
args = parser.parse_args()

base_path = args.base_path
study_info_path = os.path.join(base_path, "study_info.txt")
llm_results_dir = os.path.join(base_path, "INFO_STUDY_LLM")
high_entropy_dir = os.path.join(base_path, "high_entropy")
output_file = os.path.join(base_path, "best_inferences_per_run.csv")
categories = ["Cell type", "UBERON term", "Tissue type", "Cell line", "DOT term", "Treatment", "Treatment Time", "Response", "Phenotype", "Library selection fixed", "Library source"]

##########################################################################################
# MAIN
category_to_filename_prefix = {
    "Cell type": "celltype",
    "UBERON term": "uberon",
    "Tissue type": "tissue",
    "Cell line": "cellline",
    "DOT term": "dot",
    "Treatment": "treatment",
    "Treatment Time": "treatment_time",
    "Response": "response",
    "Phenotype": "phenotype",
    "Library selection fixed": "libselect",
    "Library source": "libsource"
}

#load study_info.txt
study_to_runs = {}
with open(study_info_path, "r") as f:
    next(f)
    for line in f:
        parts = line.strip().split(";")
        if len(parts) < 2:
            continue
        study_accession = parts[0]
        run_accessions = parts[1].split(",")
        study_to_runs[study_accession] = run_accessions

print(study_accession)
print(study_to_runs)
#store each category for each run
high_entropy_data = {cat: {} for cat in categories}
for cat in categories:
    file_path = os.path.join(high_entropy_dir, f"{category_to_filename_prefix[cat]}_high_entropy.txt")
    if not os.path.exists(file_path):
        continue
    with open(file_path, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            run_accession, entropy_str, value_str = parts
            try:
                entropy_value = float(entropy_str.strip())
                high_entropy_data[cat][run_accession] = {"entropy": entropy_value, "value": value_str.strip()}
            except ValueError:
                continue

# print(high_entropy_data)

#proces study files
llm_results = defaultdict(lambda: defaultdict(dict))
for filename in os.listdir(llm_results_dir):
    if not filename.endswith("_study.txt"):
        continue

    #get study accession
    study_accession = filename.replace("_study.txt", "")
    # print(study_accession)
    if study_accession not in study_to_runs:
        continue

    with open(os.path.join(llm_results_dir, filename), "r") as f:
        content = f.readlines()

    current_category = None
    for line in content:
        line = line.strip()
        if not line:
            continue

        # Remove preceding numbering, stars, or similar characters before the category name
        line = re.sub(r'^[^A-Za-z]+', '', line)

        #get value
        for cat in categories:
            if line.startswith(f"{cat}:"):
                if "Entropy" not in line:
                    value = line.split(":", 1)[1].strip()
                    llm_results[study_accession][cat]["value"] = value
                    current_category = cat
                break

        #get entropy per category
        for cat in categories:
            if line.startswith(f"{cat} Entropy:"):
                try:
                    entropy_value = float(line.split(":", 1)[1].strip())
                    llm_results[study_accession][cat]["entropy"] = entropy_value
                    current_category = None
                except ValueError:
                    pass
                break


final_results = []
# for study_accession, category_data in llm_results.items():
for study_accession, run_list in study_to_runs.items():
    category_data = llm_results.get(study_accession, {})
    #get list run for a study from study_info
    print("\nStudy accession from study.txt:", study_accession, flush=True)
    run_list = study_to_runs.get(study_accession, [])
    #process run in study
    for run_accession in run_list:
        print("\nruns in study:", run_accession, flush=True)
        best_values = {}
        for cat in categories:
            #get study entropy
            print("cat:", cat, flush=True)
            llm_entry = category_data.get(cat, {})
            llm_entropy = llm_entry.get("entropy", float("inf"))
            print("llm_entropy:", llm_entropy, flush=True)
            llm_value = llm_entry.get("value", None)
            print("llm_value:", llm_value, flush=True)
            #get entropy and run value
            print("high_entropy_data", high_entropy_data)
            hr_entry = high_entropy_data.get(cat, {}).get(run_accession, None)
            print("hr_entry:", hr_entry, flush=True)
            if hr_entry:
                hr_entropy = hr_entry.get("entropy", float("inf"))
                hr_value = hr_entry.get("value", None)
            else:
                hr_entropy = float("inf")
                hr_value = None
            #keep lowest entropy
            if llm_value is None and hr_value is None:
                best = "nan"
                best_entropy = "nan"
            elif llm_value is None:
                best = hr_value
                best_entropy = hr_entropy
            elif hr_value is None:
                best = llm_value
                best_entropy = llm_entropy
            else:
                if llm_entropy == float("inf") and hr_entropy == float("inf"):
                    if str(llm_value).lower() != "nan":
                        best, best_entropy = llm_value, llm_entropy
                    elif str(hr_value).lower() != "nan":
                        best, best_entropy = hr_value, hr_entropy
                    else:
                        best, best_entropy = "nan", "nan"
                elif llm_entropy <= hr_entropy:
                    best = llm_value
                    best_entropy = llm_entropy
                else:
                    best = hr_value
                    best_entropy = hr_entropy
            #remove the category prefix and any existing entropy marker from the best value
            if best != "nan":
                pattern_cat = re.compile(rf"^{re.escape(cat)}\s*[:\-]?\s*", re.IGNORECASE)
                best = pattern_cat.sub("", best)
                best = re.sub(r'\s*\(e=\d+(?:\.\d+)?\)', '', best)
            best_values[cat] = f"{best} (e={best_entropy})" if best != "nan" else "nan"

        #if at least one categorie as value != than nan, add line for this run
        print("DEBUG", study_accession, run_accession, best_values)
        if any(best_values[cat] != "nan" for cat in categories):
            final_results.append([run_accession] + [best_values.get(cat, "nan") for cat in categories])

#write final result
#warning: if study written, initial sample llm can be none, so it doesnt mean it will last in the final file
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(["Run Accession"] + categories)
    writer.writerows(final_results)
