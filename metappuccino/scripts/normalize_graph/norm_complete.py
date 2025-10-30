##########################################################################################
# IMPORT
import os
import re
import json
import pandas as pd
import argparse

##########################################################################################
# PATHS
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
SUMMARIES_FILE = os.path.join(base_path, "metadata_sra_summarized.txt")
FLAG_FILE = os.path.join(base_path, "STEP4_1.flag")
NLL_OUTPUT = os.path.join(base_path, "nll_inference.csv")
PPL_OUTPUT = os.path.join(base_path, "ppl_inference.csv")

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
        val = re.sub(r"^[\s\-\–\—\:\.\,;]+", "", val)
        val = re.sub(r"[\s\-\–\—\:\.\,;]+$", "", val)
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
    for f in ["disease", "age", "sex", "ethnicity", "biopsy_type", "biopsy_site", "uberon_code",
              "cell_type"]:
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


def _compute_codes_with_partial_tokens(raw_value, normalized_parts, names_set, syn_dict, code_map, canon_case_map):
    if not isinstance(raw_value, str):
        raw_value = ""
    raw_parts = [p.strip() for p in re.split(r';|,', raw_value)] if raw_value else []
    codes = []
    for i, norm_val in enumerate(normalized_parts):
        code_for_part = None
        key = norm_val.strip().lower()
        if key in canon_case_map:
            canonical = canon_case_map[key]
            code_for_part = code_map.get(canonical, None)
        if not code_for_part:
            part_text = raw_parts[i] if i < len(raw_parts) else raw_value
            tokens = re.findall(r"\b[A-Za-z0-9][A-Za-z0-9\-\+']*\b", part_text)
            for tok in tokens:
                t = tok.lower()
                if t in syn_dict:
                    canonical = syn_dict[t]
                    code_for_part = code_map.get(canonical, None)
                    if code_for_part:
                        break
                elif t in names_set:
                    canonical = canon_case_map.get(t, None)
                    if canonical:
                        code_for_part = code_map.get(canonical, None)
                        if code_for_part:
                            break
        codes.append(code_for_part if code_for_part else "unknown")
    return "; ".join(codes)


def _norm_cell_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _words_with_uc_and_digit(text: str):
    if not isinstance(text, str):
        return []
    spans = re.findall(r"[A-Za-z0-9][A-Za-z0-9\-\._/']*[A-Za-z0-9]", text)
    out = []
    for sp in spans:
        core = sp.strip("()[]'\".,;: ")
        if not core:
            continue
        if any(c.isupper() for c in core) and any(c.isdigit() for c in core):
            out.append(core)
    return out


def _is_official_name(name_str: str, cell_df) -> bool:
    return str(name_str) in set(cell_df["name"].astype(str))


##########################################################################################
#MAIN
df = pd.read_csv(CSV_INPUT, sep='\t', dtype=str, on_bad_lines='skip').fillna('')
df.columns = df.columns.str.strip()
assert "run_accession" in df.columns, "'run_accession' missing"

disease_names, disease_syn, disease_code = load_syn(DOT_FILE, "synonym", "name", "code_dot")
organ_names, organ_syn, organ_code = load_syn(UBERON_FILE, "synonym", "name", "code_uberon")
cell_df = pd.read_csv(CELLOSAURUS, dtype=str, on_bad_lines='skip').fillna('')

cell_lookup = {}
for _, r in cell_df.iterrows():
    name = r["name"].strip()
    if not name:
        continue
    lo = name.lower()
    nk = _norm_cell_key(name)
    cell_lookup[lo] = name
    cell_lookup[nk] = name
    syns = [s.strip() for s in r["synonym"].split(";")] if r["synonym"] else []
    for s in syns + [name]:
        if not s:
            continue
        lo2 = s.lower()
        nk2 = _norm_cell_key(s)
        cell_lookup[lo2] = name
        cell_lookup[nk2] = name

disease_canon_case = {k.lower(): k for k in disease_code.keys()}
organ_canon_case = {k.lower(): k for k in organ_code.keys()}

fields = ["run_accession", "study_accession", "instrument_platform", "library_selection", "library_strategy",
          "base_count", "sequencing_source", "biopsy_site", "bs_uberon_code", "biopsy_type", "cell_line",
          "cell_type", "organ", "organ_uberon_code", "disease", "do_code", "is_cancer", "treatment", "treatment_time",
          "response", "age", "sex", "ethnicity"]

no_entropy_fields = {"do_code", "organ_uberon_code", "bs_uberon_code"}
output_cols = list(fields)

augmented_data = []
nll_rows = []
ppl_rows = []
sources_per_run = {}

for _, row in df.iterrows():
    run = row["run_accession"]

    entry = {}
    locked_fields = set()
    source = {}
    llm_nll = {}
    llm_ppl = {}

    for f in fields:
        value = row.get(f, "").strip()
        if value and value.lower() not in INVALID_ENTRIES:
            entry[f] = value
            source[f] = "db"
            if f != "run_accession" and f not in no_entropy_fields:
                locked_fields.add(f)
        else:
            entry[f] = ""

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
                        source[k] = "llm"
                for k, v in js.get("nll", {}).items():
                    if k in fields:
                        llm_nll[k] = str(v)
                for k, v in js.get("ppl", {}).items():
                    if k in fields:
                        llm_ppl[k] = str(v)

    if entry.get("cell_line", "").strip().lower() not in INVALID_ENTRIES:
        original = entry["cell_line"]
        cleaned = clean_cell_line_name(original)
        canonical = None
        k_lower = cleaned.lower()
        k_norm = _norm_cell_key(original)
        if k_lower in cell_lookup:
            canonical = cell_lookup[k_lower]
        elif k_norm in cell_lookup:
            canonical = cell_lookup[k_norm]
        if canonical:
            entry["cell_line"] = canonical
            source["cell_line"] = source.get("cell_line", "llm")
            inferred = infer_from_cell_line(canonical, cell_df)
            for k, v in inferred.items():
                if v and v.strip().lower() not in INVALID_ENTRIES:
                    if k == "uberon_code":
                        if not entry.get("bs_uberon_code") or entry["bs_uberon_code"].lower() in INVALID_ENTRIES:
                            entry["bs_uberon_code"] = v
                            source["bs_uberon_code"] = "cello"
                    else:
                        if not entry.get(k) or entry[k].lower() in INVALID_ENTRIES:
                            entry[k] = v
                            source[k] = "cello"

    cl_val = entry.get("cell_line", "")
    if cl_val and not _is_official_name(cl_val, cell_df):
        found_canonical = None
        for cand in _words_with_uc_and_digit(cl_val):
            k1 = cand.lower()
            k2 = _norm_cell_key(cand)
            k3 = _norm_cell_key(cand.replace("-", "").replace("_", ""))
            if k1 in cell_lookup:
                found_canonical = cell_lookup[k1]
                break
            if k2 in cell_lookup:
                found_canonical = cell_lookup[k2]
                break
            if k3 in cell_lookup:
                found_canonical = cell_lookup[k3]
                break
        if not found_canonical:
            raw_candidates = []
            raw_candidates.append(cl_val)
            for extra in [row.get("cell_line", ""), entry.get("cell_line", "")]:
                if isinstance(extra, str) and extra:
                    raw_candidates.append(extra)
            for raw in raw_candidates:
                for cand in _words_with_uc_and_digit(raw):
                    k1 = cand.lower()
                    k2 = _norm_cell_key(cand)
                    k3 = _norm_cell_key(cand.replace("-", "").replace("_", ""))
                    if k1 in cell_lookup:
                        found_canonical = cell_lookup[k1]; break
                    if k2 in cell_lookup:
                        found_canonical = cell_lookup[k2]; break
                    if k3 in cell_lookup:
                        found_canonical = cell_lookup[k3]; break
                if found_canonical:
                    break
        if found_canonical:
            entry["cell_line"] = found_canonical
            source["cell_line"] = source.get("cell_line", "cello")
            inferred = infer_from_cell_line(found_canonical, cell_df)
            for k, v in inferred.items():
                if v and v.strip().lower() not in INVALID_ENTRIES:
                    if k == "uberon_code":
                        if not entry.get("bs_uberon_code") or entry["bs_uberon_code"].lower() in INVALID_ENTRIES:
                            entry["bs_uberon_code"] = v
                            source["bs_uberon_code"] = "cello"
                    else:
                        if not entry.get(k) or entry[k].lower() in INVALID_ENTRIES:
                            entry[k] = v
                            source[k] = "cello"

    if "disease" not in set():
        raw_val = entry["disease"]
        norm = normalize_term(raw_val, disease_names, disease_syn)
        entry["disease"] = "; ".join(norm)
        entry["do_code"] = _compute_codes_with_partial_tokens(raw_val, norm, disease_names, disease_syn, disease_code,
                                                              {k.lower(): k for k in disease_code.keys()})

    if "organ" not in set():
        raw_val = entry["organ"]
        norm = normalize_term(raw_val, organ_names, organ_syn)
        entry["organ"] = "; ".join(norm)
        entry["organ_uberon_code"] = _compute_codes_with_partial_tokens(raw_val, norm, organ_names, organ_syn,
                                                                        organ_code, organ_canon_case)

    if "biopsy_site" not in set():
        raw_val = entry["biopsy_site"]
        norm = normalize_term(raw_val, organ_names, organ_syn)
        entry["biopsy_site"] = "; ".join(norm)
        entry["bs_uberon_code"] = _compute_codes_with_partial_tokens(raw_val, norm, organ_names, organ_syn, organ_code,
                                                                     organ_canon_case)

    entry["bs_uberon_code"] = fmt_codes(entry["bs_uberon_code"])
    entry["organ_uberon_code"] = fmt_codes(entry["organ_uberon_code"])

    for k in fields:
        if not entry.get(k) or entry[k].strip().lower() in INVALID_ENTRIES:
            entry[k] = "unknown"

    if entry.get("treatment", "").strip().lower() == "unknown":
        entry["treatment_time"] = "not applicable"
        entry["response"] = "not applicable"

    for k in ("age", "sex", "ethnicity"):
        base_val = row.get(k, "").strip()
        if base_val and base_val.lower() not in INVALID_ENTRIES:
            entry[k] = base_val
            source[k] = "db"

    augmented_data.append(entry)
    sources_per_run[run] = source

    metric_fields = [f for f in fields if f not in no_entropy_fields and f not in {"run_accession", "study_accession", "instrument_platform", "base_count", "library_strategy"}]
    nll_row = {"run_accession": run}
    ppl_row = {"run_accession": run}
    for f in metric_fields:
        if sources_per_run[run].get(f) == "llm":
            nll_row[f] = llm_nll.get(f, "not applicable")
            ppl_row[f] = llm_ppl.get(f, "not applicable")
        else:
            nll_row[f] = "not applicable"
            ppl_row[f] = "not applicable"
    nll_rows.append(nll_row)
    ppl_rows.append(ppl_row)

out_df = pd.DataFrame(augmented_data, columns=output_cols)

exclude_cols = {"cell_line", "treatment_time", "response", "age"}
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
#VERIF AGE

summories_map_alias = {}
summaries_map = {}
if os.path.exists(SUMMARIES_FILE):
    try:
        _df_sum = pd.read_csv(SUMMARIES_FILE, sep="\t", dtype=str, on_bad_lines="skip").fillna("")
        if {"run_accession", "summary"}.issubset(set(_df_sum.columns)):
            summaries_map = dict(zip(_df_sum["run_accession"].astype(str), _df_sum["summary"].astype(str)))
    except Exception:
        summaries_map = {}


_AGE_UNIT_MAP = {
    "y": "y", "yr": "y", "yrs": "y", "year": "y", "years": "y",
    "mo": "mo", "mos": "mo", "month": "mo", "months": "mo",
    "d": "d", "day": "d", "days": "d",
    "wk": "wk", "wks": "wk", "week": "wk", "weeks": "wk",
}

_AGE_PATTERNS = [
    re.compile(r"(?i)\b(?:age|aged|y\/o|yo|years?\s+old|months?\s+old|weeks?\s+old|days?\s+old|gestational\s+age)\s*[:=]?\s*(\d{1,3}(?:\.\d+)?)\s*([a-z]+)?\b"),
    re.compile(r"(?i)\b(\d{1,3}(?:\.\d+)?)\s*(y|yr|yrs|year|years|mo|mos|month|months|wk|wks|week|weeks|d|day|days)\b"),
    re.compile(r"(?i)\b(\d{1,2})\s*(?:y\/o|yo)\b"),
]

def _parse_age_value(raw: str):
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower()
    if not s or s in INVALID_ENTRIES:
        return None
    m = re.search(r"^(\d{1,3}(?:\.\d+)?)(?:\s*([a-z]+))?$", s)
    if not m:
        return None
    num = m.group(1)
    unit = m.group(2) or "y"
    unit = _AGE_UNIT_MAP.get(unit, unit)
    if unit not in {"y", "mo", "wk", "d"}:
        if re.fullmatch(r"\d{1,3}(?:\.\d+)?", s):
            unit = "y"
        else:
            return None
    return (num, unit)

def _extract_age_mentions_from_text(text: str):
    out = []
    if not isinstance(text, str) or not text:
        return out
    for pat in _AGE_PATTERNS:
        for m in pat.finditer(text):
            num = m.group(1)
            unit = m.group(2) if m.lastindex and m.lastindex >= 2 else None
            if unit is None and pat is _AGE_PATTERNS[2]:
                unit = "y"
            if num:
                unit = (unit or "y").lower()
                unit = _AGE_UNIT_MAP.get(unit, unit)
                if unit in {"y", "mo", "wk", "d"}:
                    out.append((num, unit))
    return out

def _nums_equal(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except Exception:
        return a == b

def _age_is_confirmed_in_summary(age_value: str, summary: str):
    parsed = _parse_age_value(age_value)
    if not parsed:
        return True
    num_v, unit_v = parsed
    mentions = _extract_age_mentions_from_text(summary or "")
    for num_s, unit_s in mentions:
        if unit_s == unit_v and _nums_equal(num_s, num_v):
            return True
        if unit_v == "y" and unit_s == "y" and _nums_equal(num_s, num_v):
            return True
    return False

if "age" in out_df.columns and "run_accession" in out_df.columns:
    _validated_ages = []
    for _i, _r in out_df.iterrows():
        _run = str(_r["run_accession"])
        _age = str(_r["age"])
        if sources_per_run.get(_run, {}).get("age") != "llm":
            _validated_ages.append(_age)
            continue
        _summary = summaries_map.get(_run, "")
        if _age.strip().lower() not in INVALID_ENTRIES and not _age.strip().lower() == "unknown":
            if not _age_is_confirmed_in_summary(_age, _summary):
                _age = "unknown"
        _validated_ages.append(_age)
    out_df["age"] = _validated_ages

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

pd.DataFrame(nll_rows).to_csv(NLL_OUTPUT, index=False)
pd.DataFrame(ppl_rows).to_csv(PPL_OUTPUT, index=False)

open(FLAG_FILE, 'w').close()
