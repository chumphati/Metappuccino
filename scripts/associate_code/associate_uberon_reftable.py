##########################################################################################
#IMPORT
import csv
import os
import re

##########################################################################################
#PATHS
uberon_ref = "/Users/fionahak/Documents/phd/phd_code/MetaMap/data/UBERON_TABLE_CLEAN.csv"
llm_out = "/Users/fionahak/Documents/phd/phd_code/MetaMap/results/INFO_BIO_LLM"
output_file = "/Users/fionahak/Documents/phd/phd_code/MetaMap/results/RAW_FINAL_INFO.txt"

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

#dictionary to get run accession
rows_dict = {row[header.index("Run accession number")]: row for row in rows}

#get uberon terms
patterns = [
    r"UBERON:\d{7}\s*\(([^)]+)\)\s*[-+]?\s*(.*?)\s*(?:\)|\(|,|\+|$)",
    r"UBERON:\d{7}\s+-\s+(.*?)(?:\)|\(|,|\+|$)",
    r"UBERON:\d{7}\s+\+\s+(.*?)(?:\)|\(|,|\+|$)",
    r"UBERON:\d{7}[:+-]\s+(.*?)(?:\)|\(|,|\+|$)",
    r"UBERON:\d{7}\s*\(([^)]+)\)"
]
compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in patterns]

##########################################################################################
#MAIN
#for all the runs in llm process
for fichier in os.listdir(llm_out):
    path = os.path.join(llm_out, fichier)
    #find accession
    if os.path.isfile(path) and fichier.endswith("_bio.txt"):
        run_accession = fichier.replace("_bio.txt", "")
        #find good
        if run_accession in rows_dict:
            # print(f"{run_accession}:")
            row = rows_dict[run_accession]
            uberon_index = header.index("UBERON organ and code")
            found_terms = set()
            with open(path, 'r') as f:
                for ligne in f:
                    if "UBERON organ and code:" in ligne:
                        # print(ligne)
                        #get uberon terms
                        medical_terms = []
                        for pattern in compiled_patterns:
                            matches = pattern.findall(ligne)
                            # print(matches)
                            for match in matches:
                                if isinstance(match, tuple):
                                    for sub_match in match:
                                        if sub_match and "requires" not in sub_match.lower():
                                            medical_terms.append(clean_string(sub_match))
                                elif match and "requires" not in match.lower():
                                    medical_terms.append(clean_string(match))
                        #compare llm medical term to reference
                        for term in medical_terms:
                            matched = False
                            for entry in uberon_data:
                                if term in entry['synonyms']:
                                    found_terms.add(f"{entry['code_uberon']}, {entry['name']}")
                                    matched = True
                                    break
                            #no match
                            if not matched and "requires" not in term.lower() and "validation" not in term.lower():
                                found_terms.add(f"{term} not found for {run_accession}")

            if found_terms:
                row[uberon_index] = '; '.join(sorted(found_terms))
            else:
                row[uberon_index] = 'NA'

#update raw_final_info
with open(output_file, 'w') as output_file:
    output_file.write('|'.join(header) + '\n')
    for row in rows:
        output_file.write('|'.join(row) + '\n')
