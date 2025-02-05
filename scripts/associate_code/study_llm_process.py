import os
import csv
from collections import defaultdict

base_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results"
study_info_path = os.path.join(base_path, "tmp/study_info.txt")
llm_results_dir = os.path.join(base_path, "SPECIFIC_RUN_ANALYSIS/INFO_STUDY_LLM")
high_entropy_dir = os.path.join(base_path, "tmp/high_entropy")
output_file = os.path.join(base_path, "SPECIFIC_RUN_ANALYSIS/best_inferences_per_run.csv")
categories = ["Cell type", "UBERON term", "Tissue type", "Cell line", "DOT term"]

category_to_filename_prefix = {
    "Cell type": "celltype",
    "UBERON term": "uberon",
    "Tissue type": "tissue",
    "Cell line": "cellline",
    "DOT term": "dot"
}

#charge mapping study_accession → run_accessions
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

#charge entropies from high_entropy
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
            run_accession, entropy_value, _ = parts
            try:
                high_entropy_data[cat][run_accession] = float(entropy_value)
            except ValueError:
                continue

#get llm files + values
llm_results = defaultdict(lambda: defaultdict(dict))
# print(study_to_runs)

for filename in os.listdir(llm_results_dir):
    if not filename.endswith("_study.txt"):
        continue

    study_accession = filename.replace("_study.txt", "")
    if study_accession not in study_to_runs:
        continue

    with open(os.path.join(llm_results_dir, filename), "r") as f:
        content = f.readlines()

    current_category = None
    for line in content:
        line = line.strip()
        if not line:
            continue

        for cat in categories:
            if line.startswith(f"{cat}:"):
                value = line.split(":", 1)[1].strip()
                llm_results[study_accession][cat]["value"] = value
                current_category = cat
                # print(value)
                # print(current_category)
                break

        if current_category and "Entropy:" in line:
            print(line)
            try:
                entropy_value = float(line.split(":")[1].strip())
                llm_results[study_accession][current_category]["entropy"] = entropy_value
            except ValueError:
                continue

final_results = []

for study_accession, category_data in llm_results.items():
    for run_accession in study_to_runs.get(study_accession, []):
        best_values = {}

        for category, data in category_data.items():
            llm_entropy = data.get("entropy", float("inf"))
            high_entropy = high_entropy_data.get(category, {}).get(run_accession, float("inf"))

            if llm_entropy < high_entropy:
                best_values[category] = data["value"]

        if best_values:
            final_results.append([run_accession] + [best_values.get(cat, "NA") for cat in categories])

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f, delimiter=";")
    writer.writerow(["Run Accession"] + categories)
    writer.writerows(final_results)
