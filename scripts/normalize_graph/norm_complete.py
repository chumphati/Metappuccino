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
args = parser.parse_args()

base_path = args.base_path
CSV_INPUT = os.path.join(base_path, "database_metadata_curated.csv")
INFERENCE_DIR = os.path.join(base_path, "METADATA_LLM_INFERENCE")
CELLOSAURUS = os.path.join(base_path, "CELLOSAURUS_CLEAN.csv")
DOT_FILE = os.path.join(base_path, "DOT_TABLE_CLEAN.csv")
UBERON_FILE = os.path.join(base_path, "UBERON_TABLE_CLEAN.csv")
CSV_OUTPUT = os.path.join(base_path, "completed_metadata.csv")
FLAG_FILE = os.path.join(base_path, "STEP4_1.flag")

# CSV_INPUT = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/tmp/database_metadata_curated.csv"
# INFERENCE_DIR = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/SPECIFIC_RUN_ANALYSIS/METADATA_LLM_INFERENCE"
# CELLOSAURUS = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/CELLOSAURUS_CLEAN.csv"
# DOT_FILE = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/DOT_TABLE_CLEAN.csv"
# UBERON_FILE = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/UBERON_TABLE_CLEAN.csv"
# CSV_OUTPUT = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/tmp/metadata_augmented_with_entropy.csv"

INVALID_ENTRIES = {"unknown", "missing", "n/a", "na", "none", ""}
STOPWORDS = {"for", "to", "and", "in", "with", "via", "on", "of", "the", "a", "an", "by"}

##########################################################################################
#FUNCTIONS
def load_syn(csv, sc, nc, cc):
    with open(csv, "r", encoding="utf-8") as f:
        sep = "\t" if "\t" in f.readline() else ","
    df = pd.read_csv(csv, sep=sep, dtype=str, on_bad_lines='skip').fillna('')
    sd, cd = {}, {}
    for _, r in df.iterrows():
        name = r[nc].strip()
        if not name: continue
        syns = [s.strip() for s in r[sc].split(';')] if r[sc] else []
        for s in syns + [name]:
            sd[s.lower()] = name
        if cc and r[cc].strip():
            cd[name] = r[cc].strip()
    return sd, cd

def fuzzy_lookup(v, mapping_dict):
    v = v.lower()
    if v in mapping_dict:
        return mapping_dict[v]
    for key in mapping_dict:
        if key in v:
            return mapping_dict[key]
    return v

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

def fuzzy_lookup_strict(value, mapping_dict):
    value = value.lower().strip()
    if not value:
        return value
    if value in mapping_dict:
        return mapping_dict[value]
    words = re.findall(r'\b[a-zA-Z]+\b', value)
    for word in words:
        if word in STOPWORDS:
            continue
        if word in mapping_dict:
            return mapping_dict[word]
    return value

##########################################################################################
#MAIN
df = pd.read_csv(CSV_INPUT, sep='\t', dtype=str, on_bad_lines='skip').fillna('')
df.columns = df.columns.str.strip()
assert "run_accession" in df.columns, "Colonne 'run_accession' manquante"

disease_syn, disease_code = load_syn(DOT_FILE, "synonym", "name", "code_dot")
organ_syn, organ_code = load_syn(UBERON_FILE, "synonym", "name", "code_uberon")
cell_df = pd.read_csv(CELLOSAURUS, dtype=str, on_bad_lines='skip').fillna('')
cell_syn = {}
for _, r in cell_df.iterrows():
    name = r["name"].strip()
    if not name: continue
    syns = [s.strip() for s in r["synonym"].split(";")] if r["synonym"] else []
    for s in syns + [name]:
        cell_syn[s.lower()] = name

fields = ["run_accession", "study_accession", "instrument_platform", "library_selection", "library_strategy",
          "base_count", "sequencing_source", "biopsy_site", "bs_uberon_code", "biopsy_type", "cell_line",
          "cell_type", "organ", "organ_uberon_code", "disease", "do_code", "treatment", "treatment_time",
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

    #initial values prio
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

    #search in inference json
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

    #search cellosaurus
    if "cell_line" not in locked_fields and entry["cell_line"].lower() not in INVALID_ENTRIES:
        canonical = cell_syn.get(entry["cell_line"].lower(), entry["cell_line"])
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

    #norm DOT/UBERON
    if "disease" not in locked_fields:
        entry["disease"] = fuzzy_lookup_strict(entry["disease"], disease_syn)
        entry["do_code"] = disease_code.get(entry["disease"], "") or "unknown"
    if "organ" not in locked_fields:
        entry["organ"] = fuzzy_lookup_strict(entry["organ"], organ_syn)
        entry["organ_uberon_code"] = organ_code.get(entry["organ"], "") or "unknown"
    if "biopsy_site" not in locked_fields:
        entry["biopsy_site"] = fuzzy_lookup_strict(entry["biopsy_site"], organ_syn)
        entry["bs_uberon_code"] = organ_code.get(entry["biopsy_site"], entry["bs_uberon_code"]) or "unknown"

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
    if col in exclude_cols:
        continue
    out_df[col] = out_df[col].replace("not applicable", "unknown")

out_df.to_csv(CSV_OUTPUT, index=False)

open(FLAG_FILE, 'w').close()
