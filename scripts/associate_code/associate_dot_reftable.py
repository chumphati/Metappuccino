##########################################################################################
#IMPORT
import csv
import os
import re
import argparse

##########################################################################################
#PATHS
parser = argparse.ArgumentParser(description="Associate DOT codes to terms")
parser.add_argument("--base_path", type=str, required=True, help="Base path to MetaMap")
args = parser.parse_args()

base_path = args.base_path
dot_ref = os.path.join(base_path, "DOT_TABLE_CLEAN.csv")
llm_out = os.path.join(base_path, "INFO_BIO_LLM")
output_file = os.path.join(base_path, "raw_final_info.txt")
high_entropy_output = os.path.join(base_path, "dot_high_entropy.txt")

# dot_ref = os.path.join("/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/data/DOT_TABLE_CLEAN.csv")
# llm_out = os.path.join("/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/SPECIFIC_RUN_ANALYSIS/INFO_BIO_LLM")
# output_file = os.path.join("/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/tmp/raw_final_info.txt")
# FLAG_FILE = os.path.join("/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/tmp/STEP3_1.flag")

#check files
if not os.path.exists(dot_ref):
    print(f"error: {dot_ref} doesn't exist.")
    exit()
if not os.path.exists(llm_out):
    print(f"error: {llm_out} doesn't exist.")
    exit()
if not os.path.exists(output_file):
    print(f"error: {output_file} doesn't exist.")
    exit()

##########################################################################################
#FUNCTIONS


#clean llm output for comparison
def clean_string(s):
    return re.sub(r'[^a-zA-Z\s]', '', s).lower().strip()


#extract entropy
def extract_entropy(line):
    match = re.search(r"([0-9]+\.[0-9]+)$", line)
    return float(match.group(1)) if match else None

##########################################################################################
#SORT DOT REF
dot_data = []
with open(dot_ref, 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        synonyms = row['synonym'].strip('"').split(';') if row['synonym'] else []
        dot_data.append({
            'code_dot': row['code_dot'],
            'name': row['name'],
            'synonyms': [clean_string(row['name'])] + [clean_string(s) for s in synonyms]
        })

##########################################################################################
#OUTPUT TREATMENT
with open(output_file, 'r') as raw_file:
    lines = raw_file.readlines()

header = lines[0].strip().split('|')
rows = [line.strip().split('|') for line in lines[1:]]

# Update header indices
header_mapping = {
    "Run accession number": header.index("Run accession number"),
    "DOT code": header.index("DOT code"),
    "DOT term": header.index("DOT term")
}

rows_dict = {row[header_mapping["Run accession number"]]: row for row in rows}

#get dot terms
patterns = [
    r"DOID:\d+\s*\(([^)]+)\)\s*[-+]?\s*(.*?)\s*(?:\)|\(|,|\+|$)",
    r"DOID:\d+\s+-\s+(.*?)(?:\)|\(|,|\+|$)",
    r"DOID:\d+\s+\+\s+(.*?)(?:\)|\(|,|\+|$)",
    r"DOID:\d+[:+-]\s+(.*?)(?:\)|\(|,|\+|$)",
    r"DOID:\d+\s*\(([^)]+)\)",
    r"DOID:\d+\s*\(([^)]+)\)\s*\*\*.*?\*\*",
    r"\bEstimated\s*-\s*(.*?)\b",
    r"Estimated:\s*(.*?)\b(?:,|\)|\.|$)",
    r"Estimated:\s*DOID:\d+\s*[+\-]\s*(.*?)(?:\)|,|\.|\$)",
    r"[+\-]\s*([^;]+)(?:;|$)",
    r"Disease Ontology Term:\s*\*\*(.*?)\*\*(?:\s*\(([^)]+)\))?"
]
compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

##########################################################################################
#MAIN

high_entropy_rows = []

for fichier in os.listdir(llm_out):
    path = os.path.join(llm_out, fichier)
    if os.path.isfile(path) and fichier.endswith("_bio.txt"):
        run_accession = fichier.replace("_bio.txt", "")
        if run_accession in rows_dict:
            row = rows_dict[run_accession]
            dot_codes = set()
            dot_terms = set()
            dot_entropy = None
            dot_full_line = None

            with open(path, 'r') as f:

                for ligne in f:
                    if "Disease Ontology Term Entropy" in ligne:
                        dot_entropy = extract_entropy(ligne)
                    if "Disease Ontology Term:" in ligne:
                        dot_full_line = ligne.strip()

                if dot_entropy is None or dot_entropy < 2.5:
                    f.seek(0)
                    for ligne in f:
                        if "Disease Ontology Term:" in ligne:
                            medical_terms = []
                            # print(ligne)
                            for pattern in compiled_patterns:
                                matches = pattern.findall(ligne)
                                for match in matches:
                                    if isinstance(match, tuple):
                                        for sub_match in match:
                                            if sub_match and not any(keyword in sub_match.lower() for keyword in ["requires", "applicable", "estimated", "validated", "suggested", "validation", "based", "inferred", "context"]):
                                                split_terms = sub_match.split(" and ") if " and " in sub_match else [sub_match]
                                                for term in split_terms:
                                                    medical_terms.append(clean_string(term))
                                    elif match and not any(keyword in match.lower() for keyword in ["requires", "applicable", "estimated", "validated", "suggested", "validation", "based", "inferred", "context"]):
                                        medical_terms.append(clean_string(match))

                            for term in medical_terms:
                                matched = False
                                for entry in dot_data:
                                    if term in entry['synonyms']:
                                        dot_codes.add(entry['code_dot'])
                                        dot_terms.add(entry['name'])
                                        matched = True
                                        break
                                if not matched:
                                    split_further = term.split(" ")
                                    for sub_term in split_further:
                                        for entry in dot_data:
                                            if sub_term in entry['synonyms']:
                                                dot_codes.add(entry['code_dot'])
                                                dot_terms.add(entry['name'])
                                                matched = True
                                                break
                                    if not matched:
                                        print(run_accession, term)
                                        dot_terms.add(f"{term} (!)")
                else:
                    high_entropy_rows.append([run_accession, dot_entropy, dot_full_line])
                    continue

            row[header_mapping["DOT code"]] = ', '.join(sorted(dot_codes)) if dot_codes else 'normal'
            dot_term_val = ', '.join(sorted(dot_terms)) if dot_terms else 'normal'
            if dot_entropy is not None:
                dot_term_val += f" (e={dot_entropy})"
            row[header_mapping["DOT term"]] = dot_term_val

#update raw_final_info
with open(output_file, 'w') as output_file:
    output_file.write('|'.join(header) + '\n')
    for row in rows:
        output_file.write('|'.join(row) + '\n')

#store high entropy results
with open(high_entropy_output, 'w') as high_entropy_file:
    high_entropy_file.write("Run accession number|Entropy|DOT term\n")
    for row in high_entropy_rows:
        high_entropy_file.write('|'.join(map(str, row)) + '\n')
