import re
import csv
import os
import pandas as pd
import numpy as np
import argparse

##########################################################################################
# PATHS
parser = argparse.ArgumentParser(description="Process metadata with LLM")
parser.add_argument("--base_path", type=str, required=True, help="Base path to MetaMap")
args = parser.parse_args()

base_path = args.base_path
uberon_ref_file = os.path.join(base_path, "UBERON_TABLE_CLEAN.csv")
doid_ref_file = os.path.join(base_path, "DOT_TABLE_CLEAN.csv")
final_file = os.path.join(base_path, "final_llm_sample_analysis.csv")
best_inf_study = os.path.join(base_path, "best_inferences_per_run.csv")
FLAG_FILE = os.path.join(base_path, "STEP4_2.flag")

# uberon_ref_file = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/data/UBERON_TABLE_CLEAN.csv"
# doid_ref_file = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/data/DOT_TABLE_CLEAN.csv"
# final_file = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/SPECIFIC_RUN_ANALYSIS/final_llm_sample_analysis.csv"
# best_inf_study = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/SPECIFIC_RUN_ANALYSIS/best_inferences_per_run.csv"

##########################################################################################
# FUNCTIONS


def clean_generic(val):
    if not isinstance(val, str):
        return val
    val = re.sub(r'\b(?:inferred|simulated|estimated)\b', '', val, flags=re.IGNORECASE)

    def replace_parentheses(match):
        group = match.group(0)
        if re.fullmatch(r'\(e=\d+(?:\.\d+)?\)', group):
            return group
        else:
            return ''

    val = re.sub(r'\(.*?\)', replace_parentheses, val)
    val = re.sub(r'[^\w\s\(\)=\.]', '', val, flags=re.UNICODE)
    return val.strip()


def clean_string(s):
    if not isinstance(s, str):
        return ""

    def replace_parentheses(match):
        group = match.group(0)
        if re.fullmatch(r'\(e=\d+(?:\.\d+)?\)', group):
            return group
        else:
            return ''

    s = re.sub(r'\(.*?\)', replace_parentheses, s)
    s = re.sub(r'[^a-zA-Z0-9\s\(\)=\.]', '', s)
    return s.lower().strip()


uberon_patterns = [
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
    r"UBERON:\d{4,7}\s*\+\s*(\w[\w\s]*?)(?:\s*\(.*?\))?(?:\s*\(e=[\d.]+\))?"
]
uberon_compiled = [re.compile(p, re.IGNORECASE) for p in uberon_patterns]


doid_patterns = [
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
    r"Disease Ontology Term:\s*\*\*(.*?)\*\*(?:\s*\(([^)]+)\))?",
    r"^(?:\d+[.)-]\s*)?Disease Ontology Term:\s*\*\*(.*?)\*\*(?:\s*\(([^)]+)\))?"
]
doid_compiled = [re.compile(p, re.IGNORECASE) for p in doid_patterns]


def process_uberon_term(val, uberon_data):
    if not isinstance(val, str) or pd.isna(val):
        return None, "nan"
    candidates = []
    for pattern in uberon_compiled:
        matches = pattern.findall(val)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    for part in match:
                        candidate = part.strip()
                        candidate = re.sub(r"UBERON:\d{7}", "", candidate, flags=re.IGNORECASE)
                        candidate = candidate.strip(" ;,")
                        if candidate and not any(kw in candidate.lower() for kw in [
                            "requires", "applicable", "estimated", "validated",
                            "suggested", "validation", "based", "inferred", "context"
                        ]):
                            candidates.append(candidate)
                else:
                    candidate = match.strip()
                    candidate = re.sub(r"UBERON:\d{7}", "", candidate, flags=re.IGNORECASE)
                    candidate = candidate.strip(" ;,")
                    if candidate and not any(kw in candidate.lower() for kw in [
                        "requires", "applicable", "estimated", "validated",
                        "suggested", "validation", "based", "inferred", "context"
                    ]):
                        candidates.append(candidate)

    seen = set()
    cleaned_candidates = []
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            cleaned_candidates.append(cand)
    if not cleaned_candidates:
        return None, "nan"
    codes = set()
    terms = set()
    for cand in cleaned_candidates:
        comp_val = clean_string(cand)
        matched = False
        for entry in uberon_data:
            if comp_val in entry['synonyms']:
                codes.add(entry['code'])
                terms.add(entry['name'])
                matched = True
                break
        if not matched:
            terms.add(cand)
    final_term = ", ".join(sorted(terms)) if terms else ""
    if not final_term.strip():
        final_term = "nan"
    return (", ".join(sorted(codes)) if codes else None, final_term)


def process_doid_term(val, doid_data):
    if not isinstance(val, str) or pd.isna(val):
        return None, "nan"
    candidates = []
    for pattern in doid_compiled:
        matches = pattern.findall(val)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    for part in match:
                        candidate = part.strip()
                        candidate = re.sub(r"DOID:\d+", "", candidate, flags=re.IGNORECASE)
                        candidate = candidate.strip(" +;,")
                        if candidate and not any(kw in candidate.lower() for kw in [
                            "requires", "applicable", "estimated", "validated",
                            "suggested", "validation", "based", "inferred", "context"
                        ]):
                            candidates.append(candidate)
                else:
                    candidate = match.strip()
                    candidate = re.sub(r"DOID:\d+", "", candidate, flags=re.IGNORECASE)
                    candidate = candidate.strip(" +;,")  # on retire également le '+' ici
                    if candidate and not any(kw in candidate.lower() for kw in [
                        "requires", "applicable", "estimated", "validated",
                        "suggested", "validation", "based", "inferred", "context"
                    ]):
                        candidates.append(candidate)

    seen = set()
    cleaned_candidates = []
    for cand in candidates:
        if cand not in seen:
            seen.add(cand)
            cleaned_candidates.append(cand)
    if not cleaned_candidates:
        return None, "nan"
    codes = set()
    terms = set()
    for cand in cleaned_candidates:
        comp_val = clean_string(cand)
        matched = False
        for entry in doid_data:
            if comp_val in entry['synonyms']:
                codes.add(entry['code'])
                terms.add(entry['name'])
                matched = True
                break
        if not matched:
            terms.add(cand)
    final_term = ", ".join(sorted(terms)) if terms else ""
    if not final_term.strip():
        final_term = "nan"
    return (", ".join(sorted(codes)) if codes else None, final_term)

##########################################################################################
# LOAD REFERENCE TABLES


uberon_data = []
with open(uberon_ref_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        synonyms = row['synonym'].strip('"').split(';') if row.get('synonym') else []
        uberon_data.append({
            'code': row['code_uberon'],
            'name': row['name'],
            'synonyms': [clean_string(row['name'])] + [clean_string(s) for s in synonyms]
        })

doid_data = []
if os.path.exists(doid_ref_file):
    with open(doid_ref_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            synonyms = row['synonym'].strip('"').split(';') if row.get('synonym') else []
            doid_data.append({
                'code': row['code_dot'],
                'name': row['name'],
                'synonyms': [clean_string(row['name'])] + [clean_string(s) for s in synonyms]
            })

##########################################################################################
# LOAD FILES TO ANALYSE

df1 = pd.read_csv(final_file, sep='\t')
df2 = pd.read_csv(best_inf_study, sep=';')

df2.rename(columns={"Run Accession": "Run accession number"}, inplace=True)
columns_to_check = ["Cell type", "UBERON term", "Tissue type", "Cell line", "DOT term"]
merged_df = df1.merge(df2, on="Run accession number", suffixes=("", "_new"), how="left")

for col in columns_to_check:
    new_col = col + "_new"
    merged_df[col] = merged_df.apply(lambda row: row[new_col] if pd.isna(row[col]) and row[new_col] != "nan" else row[col], axis=1)
    merged_df.drop(columns=[new_col], inplace=True)

##########################################################################################
# ADDITIONAL FILTERING AND CLEANING

df1_indexed = df1.set_index("Run accession number")


for col in ["Cell type", "Tissue type", "Cell line"]:
    merged_df[col] = merged_df.apply(
        lambda row: clean_generic(row[col]) if pd.isna(df1_indexed.loc[row["Run accession number"], col]) else row[col],
        axis=1
    )


def update_uberon(row):
    code, term = process_uberon_term(row["UBERON term"], uberon_data)
    if not isinstance(term, str) or term.strip() == "":
        term = "nan"
    row["UBERON term"] = term
    if code is not None:
        row["UBERON code"] = code
    return row


merged_df = merged_df.apply(
    lambda row: update_uberon(row) if pd.isna(df1_indexed.loc[row["Run accession number"], "UBERON term"]) else row,
    axis=1
)


def update_doid(row):
    code, term = process_doid_term(row["DOT term"], doid_data)

    if not isinstance(term, str) or term.strip() == "":
        term = "nan"
    row["DOT term"] = term
    dot_term_val = row["DOT term"]

    if isinstance(dot_term_val, str) and dot_term_val.strip().lower() == "nan":
        return row
    if code is not None:
        row["DOT code"] = code
    else:
        row["DOT code"] = "normal"
    return row


merged_df = merged_df.apply(
    lambda row: update_doid(row) if pd.isna(df1_indexed.loc[row["Run accession number"], "DOT term"]) else row,
    axis=1
)

merged_df.fillna("nan", inplace=True)

##########################################################################################
# OUTPUT

merged_df.to_csv(final_file, sep='\t', index=False)

# create flag end process before cleaning
open(FLAG_FILE, 'w').close()
