##########################################################################################
#IMPORT
import csv
import os
import re
import argparse

##########################################################################################
#PATHS
parser = argparse.ArgumentParser(description="Associate UBERON codes to terms")
parser.add_argument("--base_path", type=str, required=True, help="Base path to MetaMap")
args = parser.parse_args()

base_path = args.base_path
uberon_ref = os.path.join(base_path, "UBERON_TABLE_CLEAN.csv")
llm_out = os.path.join(base_path, "INFO_BIO_LLM")
output_file = os.path.join(base_path, "raw_final_info.txt")

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
    r"UBERON:\d{7}\s*\(([^)]+)\)\s*[-+]?\s*(.*?)\s*(?:\)|\(|,|\+|$)",
    r"UBERON:\d{7}\s+-\s+(.*?)(?:\)|\(|,|\+|$)",
    r"UBERON:\d{7}\s+\+\s+(.*?)(?:\)|\(|,|\+|$)",
    r"UBERON:\d{7}[:+-]\s+(.*?)(?:\)|\(|,|\+|$)",
    r"UBERON:\d{7}\s*\(([^)]+)\)",
    r"UBERON:\d{7}\s*\(([^)]+)\)\s*\*\*.*?\*\*",
    r"\bEstimated\s*-\s*(.*?)\b"
]
compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

##########################################################################################
#MAIN
for fichier in os.listdir(llm_out):
    path = os.path.join(llm_out, fichier)
    if os.path.isfile(path) and fichier.endswith("_bio.txt"):
        run_accession = fichier.replace("_bio.txt", "")
        if run_accession in rows_dict:
            row = rows_dict[run_accession]
            uberon_codes = set()
            uberon_terms = set()

            with open(path, 'r') as f:
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

            row[header_mapping["UBERON code"]] = ', '.join(sorted(uberon_codes)) if uberon_codes else 'NA'
            row[header_mapping["UBERON term"]] = ', '.join(sorted(uberon_terms)) if uberon_terms else 'NA'

# update raw_final_info
with open(output_file, 'w') as output_file:
    output_file.write('|'.join(header) + '\n')
    for row in rows:
        output_file.write('|'.join(row) + '\n')

