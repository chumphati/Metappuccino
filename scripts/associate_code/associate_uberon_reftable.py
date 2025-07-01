##########################################################################################
#IMPORT
import csv
import os
import re
import argparse

##########################################################################################
#PATHS
parser = argparse.ArgumentParser(description="Associate UBERON codes to terms")
parser.add_argument("--base_path", type=str, required=True, help="Base path to Metappuccino")
args = parser.parse_args()

base_path = args.base_path
uberon_ref = os.path.join(base_path, "UBERON_TABLE_CLEAN.csv")
llm_out = os.path.join(base_path, "INFO_BIO_LLM")
output_file = os.path.join(base_path, "raw_final_info.txt")
high_entropy_output = os.path.join(base_path, "uberon_high_entropy.txt")


#check files
if not os.path.exists(uberon_ref):
    print(f"error: {uberon_ref} doesn't exist.")
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
#SORT UBERON REF
uberon_data = []
with open(uberon_ref, 'r') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        synonyms = row['synonym'].strip('"').split(';') if row['synonym'] else []
        uberon_data.append({
            'code_uberon': row['code_uberon'],
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
    "UBERON code": header.index("UBERON code"),
    "UBERON term": header.index("UBERON term")
}

rows_dict = {row[header_mapping["Run accession number"]]: row for row in rows}

#get uberon terms
patterns = [
    r"UBERON:\d{4,5,6,7}\s*\(([^)]+)\)\s*[-+]?\s*(.*?)\s*(?:\)|\(|,|\+|$)",
    r"UBERON:\d{4,5,6,7}\s*[-+]\s*(.*?)(?:\)|\(|,|\+|$)",
    r"UBERON:\d{4,5,6,7}\s+-\s+(.*?)(?:\)|\(|,|\+|$)",
    r"UBERON:\d{4,5,6,7}\s+\+\s+(.*?)(?:\)|\(|,|\+|$)",
    r"UBERON:\d{4,5,6,7}[:+-]\s+(.*?)(?:\)|\(|,|\+|$)",
    r"UBERON:\d{4,5,6,7}[:+-]?\s+(.*?)(?:\)|\(|,|\+|$)",
    r"UBERON:\d{4,5,6,7}\s*\(([^)]+)\)",
    r"UBERON:\d{4,5,6,7}\s*\(([^)]+)\)\s*\*\*.*?\*\*",
    r"UBERON organ and code:\s*UBERON:\d{4,5,6,7}\s*[-+]?\s*(.*?)$",
    r"\bEstimated\s*-\s*(.*?)\b",
    r"[+\-]\s*([^;]+)(?:;|$)",
    r"UBERON organ and code:\s*UBERON:\d{4,7}\s+(.+)$",
    r"UBERON organ and code:\s*\*\*UBERON:(\d{4,7})\s*\+\s*(.*?)\*\*(?:\s*\(([^)]+)\))?",
    r"^(?:\d+[.)-]\s*)?UBERON organ and code:\s*UBERON:\d{4,7}\s+(.+)$",
    r"^(?:\d+[.)-]\s*)?UBERON organ and code:\s*\*\*UBERON:(\d{4,7})\s*\+\s*(.*?)\*\*(?:\s*\(([^)]+)\))?",
    r"^(?:\d+[.)-]\s*)?UBERON organ and code:\s*UBERON:\d{4,7}\s*(?:[-+])?\s*(.*?)(?=\s*\(|$)",
    r"^(?:\d+[.)-]\s*)?UBERON organ and code:\s*\*\*UBERON:\d{4,7}\s*\+\s*([^(]+)(?:\s*\(.*?\))?\*\*",
    r"UBERON:\d{4,7}\s*[+-]\s*([\w\s]+?)(?:\s*\(.*?\))?(?:\s*\(e=[\d.]+\))?",
    r"UBERON:\d{4,7}\s*\+\s*(\w[\w\s]*?)(?:\s*\(.*?\))?(?:\s*\(e=[\d.]+\))?",
    r"^(?:\d+[.)-]\s*)?UBERON organ and code:\s*(?:(?:UBERON:\d{4,7}\s*[-+]\s*)?(.+))$"
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
            uberon_codes = set()
            uberon_terms = set()
            uberon_entropy = None
            uberon_full_line = None

            with open(path, 'r') as f:
                for ligne in f:
                    if "UBERON organ and code Entropy" in ligne:
                        uberon_entropy = extract_entropy(ligne)
                    if "UBERON organ and code:" in ligne:
                        uberon_full_line = ligne.strip()

                if uberon_entropy is None or uberon_entropy < 2.2:
                    f.seek(0)
                    for ligne in f:
                        if "UBERON organ and code:" in ligne:
                            medical_terms = []
                            # print(ligne)
                            for pattern in compiled_patterns:
                                # print(compiled_patterns)
                                matches = pattern.findall(ligne)
                                for match in matches:
                                    # print(f"Terms {run_accession} :")
                                    # print(match)
                                    if isinstance(match, tuple):
                                        for sub_match in match:
                                            if sub_match and not any(keyword in sub_match.lower() for keyword in ["requires", "applicable", "estimated", "validated", "suggested", "validation", "based", "inferred", "context"]):
                                                split_terms = sub_match.split(" and ") if " and " in sub_match else [sub_match]
                                                for term in split_terms:
                                                    medical_terms.append(clean_string(term))
                                    elif match and not any(keyword in match.lower() for keyword in ["requires", "applicable", "estimated", "validated", "suggested", "validation", "based", "inferred", "context"]):
                                        medical_terms.append(clean_string(match))

                            medical_terms = [term for term in medical_terms if len(term) >= 3]
                            for term in medical_terms:
                                matched = False
                                for entry in uberon_data:
                                    if term in entry['synonyms']:
                                        uberon_codes.add(entry['code_uberon'])
                                        uberon_terms.add(entry['name'])
                                        matched = True
                                        break
                                if not matched:
                                    split_further = term.split(" ")
                                    for sub_term in split_further:
                                        for entry in uberon_data:
                                            if sub_term in entry['synonyms']:
                                                uberon_codes.add(entry['code_uberon'])
                                                uberon_terms.add(entry['name'])
                                                matched = True
                                                break
                                    if not matched:
                                        uberon_terms.add(f"{term} (!)")
                else:
                    high_entropy_rows.append([run_accession, uberon_entropy, uberon_full_line])
                    # For high entropy, assign the full line as the term if available.
                    if uberon_full_line:
                        uberon_terms.add(uberon_full_line)
                    else:
                        uberon_terms.add("NA")

            row[header_mapping["UBERON code"]] = ', '.join(sorted(uberon_codes)) if uberon_codes else 'NA'
            row[header_mapping["UBERON term"]] = ', '.join(sorted(uberon_terms)) if uberon_terms else 'NA'
            if uberon_entropy is not None:
                row[header_mapping["UBERON term"]] += f" (e={uberon_entropy})"

#update raw_final_info
with open(output_file, 'w') as output_file:
    output_file.write('|'.join(header) + '\n')
    for row in rows:
        output_file.write('|'.join(row) + '\n')


#get all run treated
rows = list(rows_dict.values())
all_runs = set(rows_dict.keys())

def complete_high_entropy(rows_list, category_name):
    seen = {run for run, _, _ in rows_list}
    missing = all_runs - seen
    for run in missing:
        rows_list.append([
            run,
            float("inf"),
            f"{category_name}: no run-level prediction"
        ])


complete_high_entropy(high_entropy_rows,"UBERON term")

#store high entropy results
with open(high_entropy_output, 'w') as high_entropy_file:
    high_entropy_file.write("Run accession number|Entropy|UBERON term\n")
    for row in high_entropy_rows:
        high_entropy_file.write('|'.join(map(str, row)) + '\n')
