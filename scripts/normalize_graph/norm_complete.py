##########################################################################################
# IMPORT
import os
import re
import json
import pandas as pd
import argparse
import math

##########################################################################################
# PATHS
parser = argparse.ArgumentParser(description="Fetch information with Cellosaurus")
parser.add_argument("--base_path", type=str, required=True, help="Base path to Metappuccino")
parser.add_argument("--verbose", action="store_true", help="Verbose output")
parser.add_argument("--without_cellosaurus", action="store_true", help="Disable Cellosaurus-based propagation (cell line completion + enrichment).")
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

WITHOUT_CELLOSAURUS = args.without_cellosaurus

VERBOSE = args.verbose
vprint = print if VERBOSE else (lambda *a, **k: None)

##########################################################################################
#FUNCTIONS
def load_syn(csv, sc, nc, cc):
    GENERIC_SYNS = {
        "node", "lymph", "lymph node", "lymph nodes", "tissue", "organ", "gland", "blood", "plasma",
        "cell", "cells", "cell line", "sample", "specimen", "biopsy", "organism", "organ part",
        "anatomical location", "body", "human", "mouse", "rat",
        "disease", "cancer", "tumor", "carcinoma", "infection", "normal", "control"
    }
    def _norm_simple(s: str) -> str:
        s = str(s).replace("\u00a0", " ").strip().lower()
        s = re.sub(r"[^a-z0-9]+", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s
    def _valid_syn(s: str) -> bool:
        if not s:
            return False
        s = str(s).replace("\u00a0", " ").strip()
        if not s or s.lower() in {"nan", "none", "null"}:
            return False
        has_digit = any(ch.isdigit() for ch in s)
        letters = sum(ch.isalpha() for ch in s)
        if not has_digit and letters < 3:
            return False
        if sum(ch.isalnum() for ch in s) < 2:
            return False
        if _norm_simple(s) in GENERIC_SYNS:
            return False
        return True

    with open(csv, "r", encoding="utf-8") as f:
        sep = "\t" if "\t" in f.readline() else ","
    df = pd.read_csv(csv, sep=sep, dtype=str, on_bad_lines='skip').fillna('')
    sd, cd, names = {}, {}, set()

    for _, r in df.iterrows():
        name = str(r.get(nc, "") or "").replace("\u00a0", " ").strip()
        if not name or name.lower() == "nan":
            continue

        names.add(name.lower().strip())
        sd[name.lower().strip()] = name.strip()

        raw_syn = str(r.get(sc, "") or "").replace("\u00a0", " ").strip()
        if raw_syn.lower() in {"nan", "none", "null"}:
            raw_syn = ""

        syns = [s.strip() for s in raw_syn.split(';')] if raw_syn else []
        syns = [s for s in syns if _valid_syn(s)]

        for s in syns:
            sd[s.lower().strip()] = name.strip()

        if cc and str(r.get(cc, "") or "").strip() and str(r.get(cc, "") or "").strip().lower() != "nan":
            cd[name.strip()] = str(r.get(cc, "")).strip()

    return names, sd, cd

def normalize_term(raw_value, names_set, syn_dict):
    def _exact_lookup(s: str):
        s0 = str(s or "").replace("\u00a0", " ").strip().lower()
        if not s0:
            return None
        if s0 in names_set:
            return s0
        if s0 in syn_dict:
            return str(syn_dict[s0]).strip().lower()
        return None

    raw_value = "" if raw_value is None else str(raw_value).replace("\u00a0", " ")
    results = []
    for part in re.split(r';|,', raw_value):
        original = str(part or "").replace("\u00a0", " ").strip()
        if not original:
            continue
        ex = _exact_lookup(original)
        if ex is not None:
            results.append(ex)
            continue
        results.append(original.strip().lower())
    return results


def fmt_codes(x):
    if not isinstance(x, str) or not x:
        return x
    x = x.replace('_', ':').replace('+', ';')
    x = re.sub(r"\bUBERON:unknown\b", "unknown", x)
    return x


def _uberon_fix(code: str) -> str:
    if not isinstance(code, str):
        return ""
    s = code.strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    s = s.replace("UBERON_", "UBERON:").replace("uberon_", "UBERON:").replace("uberon:", "UBERON:")
    s = re.sub(r"\s+", "", s)
    return s

def infer_from_cell_line(cell_line, cell_df):
    if cell_df is None or cell_df.empty:
        return {}
    sub = cell_df[cell_df["name"] == cell_line]
    if sub.empty:
        return {}
    row = sub.iloc[0]
    output = {}
    for f in ["disease", "age", "sex", "ethnicity", "biopsy_type", "biopsy_site", "uberon_code",
              "cell_type", "organ"]:
        val = row.get(f, "")
        if val and str(val).strip():
            output[f] = str(val).strip()
    return output

def _metric_or_na(v):
    try:
        x = float(v)
        return str(x) if math.isfinite(x) else "not applicable"
    except Exception:
        return "not applicable"

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
        key = str(norm_val or "").strip().lower()
        if key in canon_case_map:
            canonical = canon_case_map[key]
            code_for_part = code_map.get(canonical, None)
        if not code_for_part:
            part_text = raw_parts[i] if i < len(raw_parts) else raw_value
            part_text0 = str(part_text or "").replace("\u00a0", " ").strip().lower()
            if part_text0 in syn_dict:
                canonical = syn_dict[part_text0]
                code_for_part = code_map.get(canonical, None)
            elif part_text0 in names_set:
                canonical = canon_case_map.get(part_text0, None)
                if canonical:
                    code_for_part = code_map.get(canonical, None)
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
    if cell_df is None or cell_df.empty:
        return False
    return str(name_str) in set(cell_df["name"].astype(str))


def _normalize_cell_type_value(x: str) -> str:
    if not isinstance(x, str) or not x.strip():
        return "unknown"
    parts = re.split(r'[;,/|]', x)
    mapped = []
    for p in parts:
        s = p.strip().lower()
        if not s or s in INVALID_ENTRIES or s == "unknown":
            mapped.append("unknown")
            continue
        s_norm = re.sub(r"[^a-z0-9]+", " ", s)
        s_norm = re.sub(r"\s+", " ", s_norm).strip()
        if re.fullmatch(r"(t)\s*(cell|cells|lymphocyte|lymphocytes)", s_norm) or s_norm in {"t cell", "t cells", "t lymphocyte", "t lymphocytes", "tcell", "tcells"}:
            mapped.append("t-cell")
        elif re.fullmatch(r"(b)\s*(cell|cells|lymphocyte|lymphocytes)", s_norm) or s_norm in {"b cell", "b cells", "b lymphocyte", "b lymphocytes", "bcell", "bcells"}:
            mapped.append("b-cell")
        else:
            mapped.append(s_norm)
    dedup = []
    for val in mapped:
        if val not in dedup:
            dedup.append(val)
    if all(v == "unknown" for v in dedup):
        return "unknown"
    return "; ".join(dedup)


def _normalize_age_value(x: str) -> str:
    if not isinstance(x, str):
        return "unknown"
    s = x.strip()
    if not s or s.lower() in INVALID_ENTRIES or s.lower() == "unknown":
        return "unknown"
    s0 = s.strip().lower()
    s0 = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\-]+", " ", s0)
    s0 = re.sub(r"\s+", " ", s0).strip()
    m = re.search(
        r"^(?P<y>\d{1,3}(?:\.\d+)?)\s*(?:y|yr|yrs|year|years)\s*(?P<m>\d{1,2}(?:\.\d+)?)?\s*(?:mo|mos|month|months|m)?\b",
        s0
    )
    if m:
        y = m.group("y")
        mo = m.group("m")
        if mo and float(mo) != 0.0:
            return f"{y}Y{mo}MO"
        return f"{y}Y"
    m = re.search(
        r"^(?P<y>\d{1,3}(?:\.\d+)?)\s*y\s*(?P<m>\d{1,2}(?:\.\d+)?)\s*(?:mo|mos|m)\b",
        s0
    )
    if m:
        y = m.group("y")
        mo = m.group("m")
        return f"{y}Y{mo}MO"
    m = re.search(r"^(\d{1,3}(?:\.\d+)?)\s*(mo|mos|month|months)\b", s0)
    if m:
        return f"{m.group(1)}MO"
    m = re.search(r"^(\d{1,3}(?:\.\d+)?)\s*(wk|wks|week|weeks)\b", s0)
    if m:
        return f"{m.group(1)}WK"
    m = re.search(r"^(\d{1,3}(?:\.\d+)?)\s*(d|day|days)\b", s0)
    if m:
        return f"{m.group(1)}D"
    m = re.search(r"^(\d{1,3}(?:\.\d+)?)\s*(y|yr|yrs|year|years)\b", s0)
    if m:
        return f"{m.group(1)}Y"
    m = re.search(r"\b(\d{1,3}(?:\.\d+)?)\s*(y|yr|yrs|year|years)\s*old\b", s0)
    if not m:
        m = re.search(r"\b(\d{1,3}(?:\.\d+)?)\s*yo\b", s0)
        if m:
            return f"{m.group(1)}Y"
        m = re.search(r"\b(\d{1,3}(?:\.\d+)?)\s*y\s*/\s*o\b", s0)
        if m:
            return f"{m.group(1)}Y"
    if m:
        return f"{m.group(1)}Y"
    m = re.search(r"^(\d{1,3}(?:\.\d+)?)$", s0)
    if m:
        return f"{m.group(1)}Y"
    return "unknown"

def _is_assay_like_treatment(x: str) -> bool:
    if not isinstance(x, str) or not x.strip():
        return False
    s = x.strip().lower()
    if s in INVALID_ENTRIES or s == "unknown":
        return False
    s_norm = re.sub(r"[^a-z0-9]+", " ", s)
    s_norm = re.sub(r"\s+", " ", s_norm).strip()
    s_compact = re.sub(r"[^a-z0-9]+", "", s)
    bad_phrases = [
        "single cell", "singlecell", "rna seq", "rnaseq", "rna sequencing", "sequencing",
        "scrna", "sc rna", "sc rnaseq", "transcriptome", "transcriptomics"
    ]
    for bp in bad_phrases:
        bp_norm = re.sub(r"[^a-z0-9]+", " ", bp).strip()
        bp_compact = re.sub(r"[^a-z0-9]+", "", bp)
        if bp_norm and bp_norm in s_norm:
            return True
        if bp_compact and bp_compact in s_compact:
            return True
    return False


def _lower_text_value(x: str) -> str:
    if not isinstance(x, str):
        return x
    return x.lower()


##########################################################################################
#MAIN
df = pd.read_csv(CSV_INPUT, sep='\t', dtype=str, on_bad_lines='skip').fillna('')
df.columns = df.columns.str.strip()
assert "run_accession" in df.columns, "'run_accession' missing"

disease_names, disease_syn, disease_code = load_syn(DOT_FILE, "synonym", "name", "code_dot")
organ_names, organ_syn, organ_code = load_syn(UBERON_FILE, "synonym", "name", "code_uberon")

cell_df_clean = None
if not WITHOUT_CELLOSAURUS and os.path.exists(CELLOSAURUS):
    cell_df_clean = pd.read_csv(CELLOSAURUS, dtype=str, on_bad_lines='skip').fillna('')
    cell_df = cell_df_clean
else:
    cell_df = pd.DataFrame(columns=["name", "synonym", "disease", "age", "sex", "ethnicity", "biopsy_type", "biopsy_site", "uberon_code", "cell_type", "organ"])
    cell_df_clean = cell_df

cell_lookup = {}
if not WITHOUT_CELLOSAURUS and cell_df is not None and not cell_df.empty:
    for _, r in cell_df.iterrows():
        name = str(r.get("name", "") or "").strip()
        if not name:
            continue
        lo = name.lower()
        nk = _norm_cell_key(name)
        cell_lookup[lo] = name
        cell_lookup[nk] = name
        syns = [s.strip() for s in str(r.get("synonym", "") or "").split(";")] if str(r.get("synonym", "") or "").strip() else []
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
    js = None
    if os.path.exists(json_file):
        try:
            if os.path.getsize(json_file) == 0:
                raise ValueError("empty json file")
            with open(json_file, "r", encoding="utf-8") as jf:
                raw = jf.read().strip()
                if not raw:
                    raise ValueError("empty json content")
                js = json.loads(raw)
        except Exception as e:
            vprint(f"[WARN] Bad JSON for run {run}: {json_file} ({e})")
            js = None

    if isinstance(js, dict) and run in js:
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
                llm_nll[k] = _metric_or_na(v)
        for k, v in js.get("ppl", {}).items():
            if k in fields:
                llm_ppl[k] = _metric_or_na(v)

    if not WITHOUT_CELLOSAURUS and entry.get("cell_line", "").strip().lower() not in INVALID_ENTRIES:
        original = entry["cell_line"]
        canonical = None
        k_lower = str(original or "").replace("\u00a0", " ").strip().lower()
        k_norm = _norm_cell_key(str(original or ""))
        cleaned = clean_cell_line_name(original)
        if k_lower in cell_lookup:
            canonical = cell_lookup[k_lower]
        elif k_norm in cell_lookup:
            canonical = cell_lookup[k_norm]
        elif cleaned.lower() in cell_lookup:
            canonical = cell_lookup[cleaned.lower()]
        if canonical:
            entry["cell_line"] = canonical
            source["cell_line"] = source.get("cell_line", "llm")
            inferred = infer_from_cell_line(canonical, cell_df)
            for k, v in inferred.items():
                if v and str(v).strip().lower() not in INVALID_ENTRIES:
                    if k == "uberon_code":
                        vfix = _uberon_fix(v)
                        if vfix:
                            if entry.get("biopsy_site", "").strip().lower() in INVALID_ENTRIES:
                                entry["biopsy_site"] = str(row.get("biopsy_site", "") or entry.get("biopsy_site", "")).strip()
                            if not entry.get("bs_uberon_code") or entry["bs_uberon_code"].lower() in INVALID_ENTRIES:
                                entry["bs_uberon_code"] = vfix
                                source["bs_uberon_code"] = "cello"
                    elif k == "organ":
                        if not entry.get("organ") or entry["organ"].lower() in INVALID_ENTRIES:
                            entry["organ"] = v
                            source["organ"] = "cello"
                    elif k == "biopsy_site":
                        if not entry.get("biopsy_site") or entry["biopsy_site"].lower() in INVALID_ENTRIES:
                            entry["biopsy_site"] = v
                            source["biopsy_site"] = "cello"
                    else:
                        if not entry.get(k) or entry[k].lower() in INVALID_ENTRIES:
                            entry[k] = v
                            source[k] = "cello"

            if not WITHOUT_CELLOSAURUS and cell_df_clean is not None and not cell_df_clean.empty:
                subc = cell_df_clean[cell_df_clean["name"] == canonical]
                if not subc.empty:
                    rowc = subc.iloc[0]
                    uc = _uberon_fix(str(rowc.get("uberon_code", "") or "").strip())
                    if uc and uc.lower() not in {"unknown"}:
                        if not entry.get("bs_uberon_code") or entry["bs_uberon_code"].strip().lower() in INVALID_ENTRIES:
                            entry["bs_uberon_code"] = uc
                            source["bs_uberon_code"] = "cello"

    if not WITHOUT_CELLOSAURUS:
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
                    if v and str(v).strip().lower() not in INVALID_ENTRIES:
                        if k == "uberon_code":
                            vfix = _uberon_fix(v)
                            if vfix:
                                if not entry.get("bs_uberon_code") or entry["bs_uberon_code"].lower() in INVALID_ENTRIES:
                                    entry["bs_uberon_code"] = vfix
                                    source["bs_uberon_code"] = "cello"
                        elif k == "organ":
                            if not entry.get("organ") or entry["organ"].lower() in INVALID_ENTRIES:
                                entry["organ"] = v
                                source["organ"] = "cello"
                        elif k == "biopsy_site":
                            if not entry.get("biopsy_site") or entry["biopsy_site"].lower() in INVALID_ENTRIES:
                                entry["biopsy_site"] = v
                                source["biopsy_site"] = "cello"
                        else:
                            if not entry.get(k) or entry[k].lower() in INVALID_ENTRIES:
                                entry[k] = v
                                source[k] = "cello"

                if not WITHOUT_CELLOSAURUS and cell_df_clean is not None and not cell_df_clean.empty:
                    subc = cell_df_clean[cell_df_clean["name"] == found_canonical]
                    if not subc.empty:
                        rowc = subc.iloc[0]
                        uc = _uberon_fix(str(rowc.get("uberon_code", "") or "").strip())
                        if uc and uc.lower() not in {"unknown"}:
                            if not entry.get("bs_uberon_code") or entry["bs_uberon_code"].strip().lower() in INVALID_ENTRIES:
                                entry["bs_uberon_code"] = uc
                                source["bs_uberon_code"] = "cello"

    if "disease" not in set():
        raw_val = entry["disease"]
        norm = normalize_term(raw_val, disease_names, disease_syn)
        entry["disease"] = "; ".join(norm) if norm else "unknown"
        entry["do_code"] = _compute_codes_with_partial_tokens(raw_val, norm, disease_names, disease_syn, disease_code,
                                                              {k.lower(): k for k in disease_code.keys()})

    if "organ" not in set():
        raw_val = entry["organ"]
        norm = normalize_term(raw_val, organ_names, organ_syn)
        norm_canon = [organ_syn.get(str(t or "").strip().lower(), str(t or "").strip().lower()) for t in norm]
        entry["organ"] = "; ".join(norm_canon) if norm_canon else "unknown"
        entry["organ_uberon_code"] = _compute_codes_with_partial_tokens(raw_val, norm_canon, organ_names, organ_syn,
                                                                        organ_code, organ_canon_case)

    if "biopsy_site" not in set():
        raw_val = entry["biopsy_site"]
        norm = normalize_term(raw_val, organ_names, organ_syn)
        norm_canon = [organ_syn.get(str(t or "").strip().lower(), str(t or "").strip().lower()) for t in norm]
        entry["biopsy_site"] = "; ".join(norm_canon) if norm_canon else "unknown"
        entry["bs_uberon_code"] = _compute_codes_with_partial_tokens(raw_val, norm_canon, organ_names, organ_syn, organ_code,
                                                                     organ_canon_case)

    entry["cell_type"] = _normalize_cell_type_value(entry.get("cell_type", ""))

    entry["age"] = _normalize_age_value(entry.get("age", ""))

    if entry.get("treatment", "").strip() and entry.get("treatment", "").strip().lower() not in INVALID_ENTRIES:
        if _is_assay_like_treatment(entry.get("treatment", "")):
            entry["treatment"] = "unknown"

    lower_fields = {"library_selection", "library_strategy", "sequencing_source", "biopsy_site", "biopsy_type", "cell_type",
                    "organ", "disease", "treatment", "treatment_time", "response", "sex", "ethnicity", "is_cancer"}
    for lf in lower_fields:
        if lf in entry and isinstance(entry[lf], str):
            entry[lf] = _lower_text_value(entry[lf]).strip()

    entry["bs_uberon_code"] = fmt_codes(_uberon_fix(entry.get("bs_uberon_code", "")) or entry.get("bs_uberon_code", ""))
    entry["organ_uberon_code"] = fmt_codes(_uberon_fix(entry.get("organ_uberon_code", "")) or entry.get("organ_uberon_code", ""))

    for k in ["organ", "biopsy_site", "disease"]:
        if not entry.get(k) or str(entry.get(k)).strip().lower() in INVALID_ENTRIES:
            entry[k] = "unknown"

    if str(entry.get("bs_uberon_code", "")).strip().lower() not in INVALID_ENTRIES:
        if str(entry.get("biopsy_site", "")).strip().lower() in INVALID_ENTRIES:
            entry["bs_uberon_code"] = "unknown"
    if str(entry.get("organ_uberon_code", "")).strip().lower() not in INVALID_ENTRIES:
        if str(entry.get("organ", "")).strip().lower() in INVALID_ENTRIES:
            entry["organ_uberon_code"] = "unknown"
    if str(entry.get("do_code", "")).strip().lower() not in INVALID_ENTRIES:
        if str(entry.get("disease", "")).strip().lower() in INVALID_ENTRIES:
            entry["do_code"] = "unknown"

    for k in fields:
        if not entry.get(k) or str(entry[k]).strip().lower() in INVALID_ENTRIES:
            entry[k] = "unknown"

    if entry.get("treatment", "").strip().lower() == "unknown":
        entry["treatment_time"] = "not applicable"
        entry["response"] = "not applicable"

    for k in ("age", "sex", "ethnicity"):
        base_val = row.get(k, "").strip()
        if base_val and base_val.lower() not in INVALID_ENTRIES:
            if k == "age":
                entry[k] = _normalize_age_value(base_val)
            else:
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
        if re.fullmatch(r"m|male|man", s):
            mapped.append("male")
        elif re.fullmatch(r"f|female|woman", s):
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

#return or (y, "y") for nY, ((y, "y"), (m, "mo")) for nYmMO, (n, unit) for MO/WK/D
def _parse_age_value(raw: str):
    if not isinstance(raw, str):
        return None
    s = raw.strip().lower()
    if not s or s in INVALID_ENTRIES:
        return None
    m = re.fullmatch(r"(\d{1,3}(?:\.\d+)?)y(\d{1,2}(?:\.\d+)?)mo", s)
    if m:
        return ((m.group(1), "y"), (m.group(2), "mo"))
    m = re.fullmatch(r"(\d{1,3}(?:\.\d+)?)(y|mo|wk|d)", s)
    if m:
        return (m.group(1), m.group(2))
    m = re.fullmatch(r"(\d{1,3}(?:\.\d+)?)", s)
    if m:
        return (m.group(1), "y")
    return None

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
    mentions = _extract_age_mentions_from_text(summary or "")
    if isinstance(parsed, tuple) and len(parsed) == 2 and isinstance(parsed[0], tuple) and isinstance(parsed[1], tuple):
        (y_num, y_unit), (m_num, m_unit) = parsed
        has_y = any(unit == y_unit and _nums_equal(num, y_num) for num, unit in mentions)
        has_m = any(unit == m_unit and _nums_equal(num, m_num) for num, unit in mentions)
        return has_y and has_m
    num_v, unit_v = parsed
    for num_s, unit_s in mentions:
        if unit_s == unit_v and _nums_equal(num_s, num_v):
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
try:
    out_df.to_excel(excel_output, index=False)
except Exception as e:
    vprint(f"[WARN] Excel export failed: {excel_output} ({e})")
#.parquet
parquet_output = CSV_OUTPUT.replace('.csv', '.parquet')
try:
    out_df.to_parquet(parquet_output, index=False)
except ImportError as e:
    vprint(f"[WARN] Parquet export skipped (missing engine): {parquet_output} ({e})")
except Exception as e:
    vprint(f"[WARN] Parquet export failed: {parquet_output} ({e})")
#.json
json_output = CSV_OUTPUT.replace('.csv', '.json')
out_df.to_json(json_output, orient='records', lines=True, force_ascii=False)
#.tsv
tsv_output = CSV_OUTPUT.replace('.csv', '.tsv')
out_df.to_csv(tsv_output, sep='\t', index=False)
#.feather
feather_output = CSV_OUTPUT.replace('.csv', '.feather')
try:
    out_df.reset_index(drop=True).to_feather(feather_output)
except ImportError as e:
    vprint(f"[WARN] Feather export skipped (missing engine): {feather_output} ({e})")
except Exception as e:
    vprint(f"[WARN] Feather export failed: {feather_output} ({e})")

pd.DataFrame(nll_rows).to_csv(NLL_OUTPUT, index=False)
pd.DataFrame(ppl_rows).to_csv(PPL_OUTPUT, index=False)

open(FLAG_FILE, 'w').close()
