##########################################################################################
# IMPORT
import os
import re
import json
import pandas as pd
import argparse

##########################################################################################
#PATHS
parser = argparse.ArgumentParser(description="Fetch information with Cellosaurus")
parser.add_argument("--base_path", type=str, required=True, help="Base path to Metappuccino")
parser.add_argument("--verbose", action="store_true", help="Verbose output")
args = parser.parse_args()

base_path = args.base_path
CSV_INPUT = os.path.join(base_path, "database_metadata_curated.csv")
INFERENCE_DIR = os.path.join(base_path, "METADATA_LLM_INFERENCE")
CELLOSAURUS = os.path.join(base_path, "CELLOSAURUS_CLEAN.csv")
DOT_FILE = os.path.join(base_path, "DOT_TABLE_CLEAN.csv")
UBERON_FILE = os.path.join(base_path, "UBERON_TABLE_CLEAN.csv")
CSV_OUTPUT = os.path.join(base_path, "completed_metadata.csv")
FLAG_FILE = os.path.join(base_path, "STEP4_1.flag")

INVALID_ENTRIES = {"unknown", "missing", "n/a", "na", "none", ""}
STOPWORDS = {"for", "to", "and", "in", "with", "via", "on", "of", "the", "a", "an", "by"}

VERBOSE = args.verbose
vprint = print if VERBOSE else (lambda *a, **k: None)

##########################################################################################
#FUNCTIONS
def load_syn(csv, sc, nc, cc):
    with open(csv, "r", encoding="utf-8") as f:
        sep = "\t" if "\t" in f.readline() else ","
    df = pd.read_csv(csv, sep=sep, dtype=str, on_bad_lines='skip').fillna('')
    sd, cd, names = {}, {}, set()
    for _, r in df.iterrows():
        name = r[nc].strip()
        if not name:
            continue
        names.add(name.lower())
        syns = [s.strip() for s in r[sc].split(';')] if r[sc] else []
        for s in syns:
            sd[s.lower()] = name
        if cc and r[cc].strip():
            cd[name] = r[cc].strip()
    return names, sd, cd

def normalize_term(raw_value, names_set, syn_dict):
    to_remove = [
        r"\bcell line\b", r"\bcells?\b", r"\bspecimen\b", r"\bsample\b",
        r"\btype\b", r"\btissue\b", r"\bderived from\b", r"\bunknown\b",
        r"\ba\b", r"\ban\b", r"\bthe\b"
    ]

    def clean(val):
        original = val.strip()
        val = original.lower()
        for expr in to_remove:
            val = re.sub(expr, "", val, flags=re.IGNORECASE)
        val = re.sub(r"^[\s\-\–\—\:\.\,;]+", "", val)  # début
        val = re.sub(r"[\s\-\–\—\:\.\,;]+$", "", val)  # fin
        return val.strip(), original.strip()

    results = []
    for part in re.split(r';|,', raw_value):
        cleaned, original = clean(part)
        if cleaned in names_set:
            results.append(cleaned)
        elif cleaned in syn_dict:
            results.append(syn_dict[cleaned])
        else:
            results.append(original)
    return results

def fmt_codes(x):
    if not isinstance(x, str) or not x:
        return x
    return x.replace('_', ':').replace('+', ';')

def infer_from_cell_line(cell_line, cell_df):
    row = cell_df[cell_df["name"] == cell_line].iloc[0]
    output = {}
    for f in ["disease", "age", "sex", "ethnicity", "localization", "biopsy_type", "biopsy_site", "uberon_code", "cell_type"]:
        val = row.get(f, "")
        if val and val.strip():
            output[f] = val.strip()
    return output

def clean_cell_line_name(raw):
    if not isinstance(raw, str):
        return ""
    cleaned = raw.lower()
    cleaned = re.sub(r'\bcells?\b', '', cleaned)
    cleaned = re.sub(r'[\.\,\-\_\;\:\(\)\[\]\'\"]', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned.strip()

##########################################################################################
#MAIN
df = pd.read_csv(CSV_INPUT, sep='\t', dtype=str, on_bad_lines='skip').fillna('')
df.columns = df.columns.str.strip()
assert "run_accession" in df.columns, "Colonne 'run_accession' manquante"

disease_names, disease_syn, disease_code = load_syn(DOT_FILE, "synonym", "name", "code_dot")
organ_names, organ_syn, organ_code = load_syn(UBERON_FILE, "synonym", "name", "code_uberon")
cell_df = pd.read_csv(CELLOSAURUS, dtype=str, on_bad_lines='skip').fillna('')
cell_syn = {}
for _, r in cell_df.iterrows():
    name = r["name"].strip()
    if not name:
        continue
    syns = [s.strip() for s in r["synonym"].split(";")] if r["synonym"] else []
    for s in syns + [name]:
        cell_syn[s.lower()] = name

fields = ["run_accession", "study_accession", "instrument_platform", "library_selection", "library_strategy",
          "base_count", "sequencing_source", "biopsy_site", "bs_uberon_code", "biopsy_type", "cell_line",
          "cell_type", "organ", "organ_uberon_code", "disease", "do_code", "is_cancer", "treatment", "treatment_time",
          "response", "age", "sex", "ethnicity", "localization"]

no_entropy_fields = {"do_code", "organ_uberon_code", "bs_uberon_code"}
entropy_cols = [f"confidence_entropy_{f}" for f in fields if f not in no_entropy_fields and f != "run_accession"]
output_cols = []
for f in fields:
    output_cols.append(f)
    if f not in no_entropy_fields and f != "run_accession":
        output_cols.append(f"confidence_entropy_{f}")

augmented_data = []
for _, row in df.iterrows():
    run = row["run_accession"]

    entry = {}
    locked_fields = set()
    for f in fields:
        value = row.get(f, "").strip()
        if value and value.lower() not in INVALID_ENTRIES:
            entry[f] = value
            if f != "run_accession" and f not in no_entropy_fields:
                entry[f"confidence_entropy_{f}"] = "0"
                locked_fields.add(f)
        else:
            entry[f] = ""
            if f != "run_accession" and f not in no_entropy_fields:
                entry[f"confidence_entropy_{f}"] = "0"

    json_file = os.path.join(INFERENCE_DIR, f"{run}.json")
    if os.path.exists(json_file):
        with open(json_file) as jf:
            js = json.load(jf)
            if run in js:
                for k in js[run]:
                    if k in fields and k not in locked_fields:
                        val = js[run][k]
                        if isinstance(val, list):
                            val = "; ".join(map(str, val))
                        val = val.strip() if isinstance(val, str) else str(val)
                        entry[k] = val if val else "unknown"
                        entry[f"confidence_entropy_{k}"] = str(js.get("entropy", {}).get(k, "unknown"))

    if "cell_line" not in locked_fields and entry["cell_line"].lower() not in INVALID_ENTRIES:
        cleaned = clean_cell_line_name(entry["cell_line"])
        canonical = cell_syn.get(cleaned, None)
        if canonical:
            entry["cell_line"] = canonical
            if canonical in cell_df["name"].values:
                inferred = infer_from_cell_line(canonical, cell_df)
                for k, v in inferred.items():
                    if k not in locked_fields:
                        if k == "uberon_code":
                            entry["bs_uberon_code"] = v
                            entry["confidence_entropy_bs_uberon_code"] = "0"
                        else:
                            entry[k] = v
                            ent_k = f"confidence_entropy_{k}"
                            if ent_k in entry:
                                entry[ent_k] = "0"
        else:
            entry["cell_line"] = cleaned

    if "disease" not in locked_fields:
        norm = normalize_term(entry["disease"], disease_names, disease_syn)
        entry["disease"] = "; ".join(norm)
        entry["do_code"] = "; ".join([disease_code.get(x, "unknown") for x in norm])

    if "organ" not in locked_fields:
        norm = normalize_term(entry["organ"], organ_names, organ_syn)
        entry["organ"] = "; ".join(norm)
        entry["organ_uberon_code"] = "; ".join([organ_code.get(x, "unknown") for x in norm])

    if "biopsy_site" not in locked_fields:
        norm = normalize_term(entry["biopsy_site"], organ_names, organ_syn)
        entry["biopsy_site"] = "; ".join(norm)
        entry["bs_uberon_code"] = "; ".join([organ_code.get(x, "unknown") for x in norm])

    entry["bs_uberon_code"] = fmt_codes(entry["bs_uberon_code"])
    entry["organ_uberon_code"] = fmt_codes(entry["organ_uberon_code"])

    for k in fields:
        if not entry.get(k) or entry[k].strip().lower() in INVALID_ENTRIES:
            entry[k] = "unknown"
            if k != "run_accession":
                entry[f"confidence_entropy_{k}"] = "unknown"

    augmented_data.append(entry)

out_df = pd.DataFrame(augmented_data, columns=output_cols)

exclude_cols = {"cell_line", "treatment_time", "response"}
for col in out_df.columns:
    out_df[col] = out_df[col].replace("None", "unknown")
    if col in exclude_cols:
        continue
    out_df[col] = out_df[col].replace("not applicable", "unknown")
    
def _normalize_sex_value(x: str) -> str:
    if not isinstance(x, str) or not x.strip():
        return "unknown"
    parts = re.split(r'[;,/|]', x)
    mapped = []
    for p in parts:
        s = p.strip().lower()
        if s in INVALID_ENTRIES or s == "unknown":
            mapped.append("unknown")
            continue
        if re.fullmatch(r"m|male|masculin|mâle|man", s):
            mapped.append("male")
        elif re.fullmatch(r"f|female|féminin|femelle|woman", s):
            mapped.append("female")
        else:
            mapped.append("unknown")
    dedup = []
    for val in mapped:
        if val not in dedup:
            dedup.append(val)
    if all(v == "unknown" for v in dedup):
        return "unknown"
    return "; ".join(dedup)

if "sex" in out_df.columns:
    out_df["sex"] = out_df["sex"].apply(_normalize_sex_value)

##########################################################################################
#SAVE
#.csv
out_df.to_csv(CSV_OUTPUT, index=False)
#.xlsx
excel_output = CSV_OUTPUT.replace('.csv', '.xlsx')
out_df.to_excel(excel_output, index=False)
#.parquet
parquet_output = CSV_OUTPUT.replace('.csv', '.parquet')
out_df.to_parquet(parquet_output, index=False)
#.json
json_output = CSV_OUTPUT.replace('.csv', '.json')
out_df.to_json(json_output, orient='records', lines=True, force_ascii=False)
#.tsv
tsv_output = CSV_OUTPUT.replace('.csv', '.tsv')
out_df.to_csv(tsv_output, sep='\t', index=False)
#.feather
feather_output = CSV_OUTPUT.replace('.csv', '.feather')
out_df.reset_index(drop=True).to_feather(feather_output)

open(FLAG_FILE, 'w').close()
