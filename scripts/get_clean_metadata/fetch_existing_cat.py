########################################################################################################################
# IMPORT LIB
import pandas as pd
import re
import glob
import os
import argparse
import spacy
from spacy.matcher import PhraseMatcher
from spacy.cli import download
from concurrent.futures import ProcessPoolExecutor

########################################################################################################################
#PATHS
parser = argparse.ArgumentParser(description="Fetch information with Cellosaurus")
parser.add_argument("--base_path", type=str, required=True, help="Base path to Metappuccino")
parser.add_argument("--cpu_number", type=int, default=os.cpu_count(), help="Number of CPU used to parallelize the metadata fetching")
parser.add_argument("--verbose", action="store_true", help="Verbose output")
args = parser.parse_args()

base_path = args.base_path
xml_dir = os.path.join(base_path, "metadata")
cell_df_file = os.path.join(base_path, "CELLOSAURUS_PRECUT.csv")
dot_file = os.path.join(base_path, "DOT_TABLE_CLEAN.csv")
uberon_file = os.path.join(base_path, "UBERON_TABLE_CLEAN.csv")
input_file = os.path.join(base_path, "metadata_sra.txt")
output_file_df = os.path.join(base_path, "database_metadata_curated.csv")
FLAG_FILE = os.path.join(base_path, "STEP2_1.flag")
AMBIG_FILE = os.path.join(base_path, "ambiguous_cell_lines.csv")
cpu_number = args.cpu_number

invalid_entries = {"unknown", "not applicable", "missing", "n/a", "na", "none", ""}

VERBOSE = args.verbose
vprint = print if VERBOSE else (lambda *a, **k: None)

########################################################################################################################
#FUNCTIONS

download("en_core_web_sm")
nlp_match = spacy.load("en_core_web_sm")

def ensure_maxlen(nlp, text):
    if len(text) + 1 > nlp.max_length:
        nlp.max_length = len(text) + 1

nlp_match.max_length = max(nlp_match.max_length, 5_000_000)

def load_syn(csv, sc, nc, cc):
    df = pd.read_csv(csv, dtype=str, on_bad_lines='skip').fillna('')
    sd, cd = {}, {}
    for _, r in df.iterrows():
        name = r[nc].strip()
        if not name:
            continue
        syns = [s.strip() for s in r[sc].split(';')] if r[sc] else []
        for s in syns + [name]:
            sd[s.lower()] = name
        if cc and r[cc].strip():
            cd[name] = r[cc].strip()
    return sd, cd

def extract_tag(ctx: str, tag: str) -> str:
    pattern = tag.replace("_", "[_ ]")
    m = re.search(rf"<TAG>{pattern}</TAG>\s*<VALUE>([^<]+)</VALUE>", ctx, re.IGNORECASE)
    return m.group(1).strip() if m else ""

def extract_sample_title(ctx: str) -> str:
    m = re.search(r'sample_title="([^"]+)"', ctx, re.IGNORECASE)
    return m.group(1).strip() if m else ""

def normalize(v, d):
    return d.get(v.strip().lower(), v.strip()) if isinstance(v, str) and v.strip() else ""

def extract_libsel(ctx):
    if re.search(r"poly[.\s-]?a|oligo[.\s-]?d[.\s-]?t|truseq[.\s-]?mrna|smarter[.\s-]?mrna|polyadenylated|polyadenylation|nebnext poly[.\s-]?a|magnetic poly[.\s-]?a", ctx, re.IGNORECASE):
        return "polyA"
    if re.search(r"ribo[.\s-]?(minus|dep|zero)|deplete[.\s-]?ribosom|rrna[.\s-]?deple|rrna[.\s-]?minus|rrna removal|rrna depletion kit|ribominus|ribodepletion", ctx, re.IGNORECASE):
        return "inverse rRNA"
    if re.search(r"hybrid[.\s-]?selection|exon[.\s-]?capture|exome[.\s-]?capture|rna[.\s-]?exome|bait|capture-based|targeted transcriptome|myBaits|SureSelect", ctx, re.IGNORECASE):
        return "hybrid selection"
    if re.search(r"truseq[.\s-]?small|size[.\s-]?fraction|small[.\s-]?rna|mirna|micro[.\s-]?rna|small RNA-Seq|smallRNA|miRNA-Seq|small RNA library", ctx, re.IGNORECASE):
        return "small RNA"
    return ""

def extract_src(ctx):
    if re.search(r"(spatial.{0,10}(transcriptomic|sequencing|rna.?seq|visium|geomx|merfish|hdst|slide[-\s]?seq|nanostring|st-seq))|(visium|geomx|merfish|hdst|slide[-\s]?seq|nanostring|spatial transcriptomics)", ctx, re.IGNORECASE):
        return "spatial"
    if re.search(r"single[\s\-]?cell|scrna|10x|drop[-\s]?seq|smart[-\s]?seq|fluidigm|inDrop|seq[-\s]?well|c1 platform|smartseq|smart-seq|microwell|scRNA(-seq)?|single[-\s]?nucleus", ctx, re.IGNORECASE):
        return "single cell"
    if re.search(r"bulk(\s*rna)?(\s*-?\s*seq)?|conventional rna[\s\-]?seq|whole[\s\-]?transcriptome|wta|standard rna[\s\-]?seq|total rna[\s\-]?seq|rna[\s\-]?seq", ctx, re.IGNORECASE):
        return "bulk"
    return ""

def enrich_from_cell_df(record: dict, mapped: str):
    if mapped in cell_df["name"].values:
        row = cell_df[cell_df["name"] == mapped].iloc[0]
        for f in ["disease", "age", "sex", "ethnicity", "biopsy_type", "biopsy_site", "uberon_code", "cell_type"]:
            v = row.get(f, "")
            if v and v.strip():
                if f == "biopsy_site" and v.strip().lower() == "not specified":
                    continue
                if f == "uberon_code":
                    record["bs_uberon_code"] = v.strip()
                    record["bs_uberon_code_source"] = "cellosaurus"
                else:
                    record[f] = v.strip()
                    record[f + "_source"] = "cellosaurus"

def valid_regex_candidate(tok: str) -> bool:
    letters = sum(c.isalpha() for c in tok)
    digits = sum(c.isdigit() for c in tok)
    return letters >= 3 and digits >= 3

def uniq_preserve_order(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

disease_syn, disease_code = load_syn(dot_file, "synonym", "name", "code_dot")
organ_syn, organ_code = load_syn(uberon_file, "synonym", "name", "code_uberon")

cell_df = pd.read_csv(cell_df_file, dtype=str, on_bad_lines='skip').fillna('')

official_names_lower = set(cell_df["name"].str.lower())

def resolve_cell_line(term: str) -> str:
    if term.lower() in official_names_lower:
        return term
    return cell_syn.get(term.lower(), term)

cell_syn = {}
for _, r in cell_df.iterrows():
    name = r['name'].strip()
    if not name:
        continue
    syns = [s.strip() for s in r['synonym'].split(';')] if r['synonym'] else []
    for s in syns + [name]:
        cell_syn[s.lower()] = name

cols = [
    "run_accession", "library_selection", "sequencing_source",
    "biopsy_site", "bs_uberon_code", "biopsy_type", "cell_line", "cell_type",
    "organ", "organ_uberon_code", "disease", "do_code",
    "treatment", "treatment_time", "response",
    "age", "sex", "ethnicity"
]

stopwords = r"\b(?:for|to|and|in|with|via|on|of|the)\b"

cell_options = set()
for _, row in cell_df.iterrows():
    cell_options.add(row["name"].strip())
    if row["synonym"]:
        cell_options.update(s.strip() for s in row["synonym"].split(";") if s.strip())

GENERIC_WORDS = {
    "cell", "cells", "line", "cancer", "tumor", "normal", "human",
    "mouse", "rat", "wildtype", "wt", "control", "sample", "treated",
    "treated cells", "immortalized", "immortal", "in vitro", "primary",
    "junior", "jake", "ears", "bob", "princess", "ashes", "fisher", "center"
}

def keep_token(tok: str) -> bool:
    if tok.lower() in GENERIC_WORDS:
        return False
    letters = sum(ch.isalpha() for ch in tok)
    has_digit = any(ch.isdigit() for ch in tok)
    return (letters >= 2 and has_digit) or letters >= 6

codes_selected = {t for t in cell_options if keep_token(t)}
matcher = PhraseMatcher(nlp_match.vocab, attr="LOWER")
matcher.add("CELL_LINE_DICT", [nlp_match.make_doc(t) for t in sorted(codes_selected)])

regex_pattern = re.compile(r"\b[A-Za-z0-9]{4,}\b")

def _bs_norm(s):
    return re.sub(r"[^a-z0-9]+", "_", (s or "").strip().lower()).strip("_")

BS_CACHE = {}
def _load_biosample_attrs(run):
    if run in BS_CACHE:
        return BS_CACHE[run]
    p = os.path.join(xml_dir, f"{run}_biosample.xml")
    d = {}
    if os.path.exists(p):
        try:
            txt = open(p, "r", encoding="utf-8", errors="ignore").read()
            attrs = re.findall(r'<Attribute[^>]*?(?:attribute_name|harmonized_name)="([^"]+)"[^>]*>(.*?)</Attribute>', txt, flags=re.IGNORECASE | re.DOTALL)
            for k, v in attrs:
                nk = _bs_norm(k)
                val = re.sub(r"\s+", " ", v).strip()
                if nk and val and nk not in d:
                    d[nk] = val
        except Exception:
            pass
    BS_CACHE[run] = d
    return d

def _bs_get(run, tag, extra_aliases=None):
    d = _load_biosample_attrs(run)
    alts = [tag, tag.replace("_", "-"), tag.replace("_", " ")]
    if extra_aliases:
        alts.extend(extra_aliases)
    keys = {_bs_norm(a) for a in alts}
    for k in keys:
        if k in d:
            return d[k]
    for alias in alts:
        parts = [p for p in re.split(r"[\s_\-]+", alias.strip().lower()) if p]
        if not parts:
            continue
        pat = r"[_\-\s]+".join(re.escape(p) for p in parts)
        for k in d:
            if re.fullmatch(pat, k):
                return d[k]
    return ""

def process_path(path):
    ambiguous_rows_local = []
    out_local = None
    run = os.path.basename(path).split("_metadata.xml")[0]
    vprint(run)
    ctx = open(path, "r", encoding="utf-8", errors="ignore").read()
    o = {c: "" for c in cols}
    is_ambiguous = False
    o["run_accession"] = run
    found = False
    raw_cl = extract_tag(ctx, "cell_line")
    if raw_cl and raw_cl.lower() not in invalid_entries:
        vprint("tag")
        mapped = resolve_cell_line(raw_cl.strip())
        o["cell_line"] = re.sub(r"\bcell(s)?\b", "", mapped, flags=re.IGNORECASE).strip()
        vprint(o["cell_line"])
        enrich_from_cell_df(o, o["cell_line"])
        found = True
    if not found:
        bs_cl = _bs_get(run, "cell_line", ["cell type", "cell-type", "cell_type"])
        if bs_cl and bs_cl.lower() not in invalid_entries:
            vprint("biosample_cell_line")
            mapped = resolve_cell_line(bs_cl.strip())
            o["cell_line"] = re.sub(r"\bcell(s)?\b", "", mapped, flags=re.IGNORECASE).strip()
            enrich_from_cell_df(o, o["cell_line"])
            found = True
    if not found:
        src_name = extract_tag(ctx, "source_name")
        sample_title = extract_sample_title(ctx)
        for origin, src in (("source_name", src_name), ("sample_title", sample_title)):
            if not src:
                continue
            tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9._/\-]*", src)
            tokens = [t for t in tokens if len(t) >= 4]
            hits = [t for t in tokens if t.lower() in cell_syn]
            if hits:
                hits = uniq_preserve_order(hits)
                if len(hits) >= 2:
                    ambiguous_rows_local.append({"run_accession": run, "candidates": ";".join(hits)})
                    is_ambiguous = True
                    vprint(f"{origin}>=2")
                    found = True
                    break
                else:
                    mapped = resolve_cell_line(hits[0].strip())
                    vprint(origin)
                    o["cell_line"] = re.sub(r"\bcell(s)?\b", "", mapped, flags=re.IGNORECASE).strip()
                    vprint(o["cell_line"])
                    enrich_from_cell_df(o, o["cell_line"])
                    found = True
                    break
    if not found:
        ensure_maxlen(nlp_match, ctx)
        doc_m = nlp_match(ctx)
        matches = matcher(doc_m)
        if matches:
            cand_texts = []
            seen = set()
            for _, s, e in matches:
                t = doc_m[s:e].text
                if t not in seen:
                    seen.add(t)
                    cand_texts.append(t)
            cand_texts = uniq_preserve_order(cand_texts)
            if cand_texts:
                ambiguous_rows_local.append({"run_accession": run, "candidates": ";".join(cand_texts)})
                is_ambiguous = True
                vprint("PhraseMatcher")
                found = True
    if not found:
        all_tokens = regex_pattern.findall(ctx)
        regex_cands = [t for t in all_tokens if valid_regex_candidate(t) and t.lower() in cell_syn]
        regex_cands = uniq_preserve_order(regex_cands)
        if regex_cands:
            ambiguous_rows_local.append({"run_accession": run, "candidates": ";".join(regex_cands)})
            is_ambiguous = True
            vprint("regex")
            found = True
    for tag in ["sex", "treatment", "treatment_time", "response", "age", "ethnicity", "biopsy_site", "biopsy_type", "organ", "disease"]:
        if tag in ["treatment", "treatment_time"]:
            continue
        if o.get(tag) and o.get(tag + "_source") == "cellosaurus":
            continue
        val = extract_tag(ctx, tag)
        if not val:
            aliases = {
                "sex": ["gender"],
                "age": ["host_age", "subject_age"],
                "ethnicity": ["race", "population"],
                "biopsy_site": ["host body site", "host_body_site", "body site", "organism part", "organism_part", "tissue", "tissue type", "tissue_type", "organ"],
                "biopsy_type": ["biopsy type", "sample type", "sample_type", "specimen type", "specimen_type"],
                "organ": ["organism part", "organism_part", "tissue", "tissue type", "tissue_type", "anatomical location", "anatomical_location"],
                "disease": ["disease state", "phenotype", "diagnosis"]
            }.get(tag, [])
            val = _bs_get(run, tag, aliases)
            if val:
                vprint(f"biosample::{tag}")
        if val and val.lower() not in invalid_entries:
            o[tag] = val
    if not o["library_selection"]:
        o["library_selection"] = extract_libsel(ctx)
    if not o["sequencing_source"]:
        o["sequencing_source"] = extract_src(ctx)
    if o["treatment"]:
        times = re.findall(r"\b\d+\s*(?:h|hr|hrs|hours?|d|day|days?)\b", o["treatment"], re.IGNORECASE)
        if times:
            for t in times:
                if o["treatment_time"]:
                    o["treatment_time"] += "; " + t
                else:
                    o["treatment_time"] = t
            o["treatment"] = re.sub(r"\b\d+\s*(?:h|hr|hrs|hours?|d|day|days?)\b", "", o["treatment"], flags=re.IGNORECASE).strip(" ,;")
        o["treatment"] = re.sub(stopwords, "", o["treatment"], flags=re.IGNORECASE).strip()
        o["treatment"] = re.sub(r"\s{2,}", " ", o["treatment"])
    allowed_libsel = {"polyA", "inverse rRNA", "hybrid selection", "small RNA", "other"}
    allowed_src = {"spatial", "bulk", "single cell"}
    allowed_biopsy_type = {"metastasis", "blood", "primary"}
    if o.get("library_selection") and o["library_selection"] not in allowed_libsel:
        o["library_selection"] = ""
    if o.get("sequencing_source") and o["sequencing_source"] not in allowed_src:
        o["sequencing_source"] = ""
    if o.get("biopsy_type") and o["biopsy_type"] not in allowed_biopsy_type and o.get("biopsy_type_source") != "cellosaurus":
        o["biopsy_type"] = ""
    o["disease"] = normalize(o["disease"], disease_syn)
    o["do_code"] = disease_code.get(o["disease"], "")
    if o.get("organ_source") != "cellosaurus":
        o["organ"] = normalize(o["organ"], organ_syn)
    if not o.get("organ_uberon_code"):
        o["organ_uberon_code"] = organ_code.get(o["organ"], "")
    if o.get("biopsy_site_source") != "cellosaurus":
        o["biopsy_site"] = normalize(o["biopsy_site"], organ_syn)
    if (not o.get("bs_uberon_code")) or (o.get("bs_uberon_code_source") != "cellosaurus"):
        o["bs_uberon_code"] = organ_code.get(o["biopsy_site"], o.get("bs_uberon_code", ""))
    is_official = bool(o.get("cell_line")) and (o["cell_line"].strip().lower() in official_names_lower)
    if is_ambiguous:
        vprint("ambiguous")
        out_local = o
    elif not is_official:
        ambiguous_rows_local.append({"run_accession": run, "candidates": o.get("cell_line", "")})
        vprint("unresolved_or_not_official")
        out_local = o
    else:
        out_local = o
    return out_local, ambiguous_rows_local

if __name__ == "__main__":
    paths = glob.glob(os.path.join(xml_dir, "*_metadata.xml"))
    results = []
    with ProcessPoolExecutor(max_workers=cpu_number) as ex:
        for r in ex.map(process_path, paths, chunksize=100):
            results.append(r)
    out = [r[0] for r in results]
    ambiguous_rows = []
    for r in results:
        ambiguous_rows.extend(r[1])

    df_conf = pd.DataFrame(out, columns=cols)

    def fmt_codes(x):
        if not isinstance(x, str) or not x:
            return x
        return x.replace('_', ':').replace('+', ';')

    if not df_conf.empty:
        df_conf["bs_uberon_code"] = df_conf["bs_uberon_code"].apply(fmt_codes)
        df_conf["organ_uberon_code"] = df_conf["organ_uberon_code"].apply(fmt_codes)

    sra_df = pd.read_csv(input_file, sep='\t', dtype=str, on_bad_lines='warn').fillna('')

    runs_xml = {os.path.basename(p).split("_metadata.xml")[0] for p in glob.glob(os.path.join(xml_dir, "*_metadata.xml"))}
    runs_sra = set(sra_df['run_accession'].astype(str).str.strip())

    vprint(f"[DEBUG] XML files: {len(runs_xml)} | SRA runs: {len(runs_sra)}")

    missing_in_sra = sorted(runs_xml - runs_sra)
    missing_in_xml = sorted(runs_sra - runs_xml)

    if missing_in_sra:
        vprint("[DEBUG] Not in metadata_sra.txt: " + ", ".join(missing_in_sra[:10]) + (" ..." if len(missing_in_sra) > 10 else ""))

    if missing_in_xml:
        vprint("[DEBUG] In SRA but no corresponding XML: " + ", ".join(missing_in_xml[:10]) + (" ..." if len(missing_in_xml) > 10 else ""))

    dupes = sra_df['run_accession'][sra_df['run_accession'].duplicated(keep=False)]
    if not dupes.empty:
        vprint("[DEBUG] Duplicates SRA (run_accession): " + ", ".join(sorted(dupes.unique())))

    if 'run_accession' not in sra_df.columns:
        for alt in ['Run', 'run', 'RUN']:
            if alt in sra_df.columns:
                sra_df = sra_df.rename(columns={alt: 'run_accession'})
                break

    sra_df['run_accession'] = sra_df['run_accession'].astype(str).str.strip()

    cols_to_extract = ['run_accession', 'base_count', 'library_strategy', 'instrument_platform', 'study_accession']
    cols_exist = [c for c in cols_to_extract if c in sra_df.columns]
    sra_base = sra_df[cols_exist].copy()

    df_conf['run_accession'] = df_conf['run_accession'].astype(str).str.strip()

    df = sra_base.merge(df_conf, on='run_accession', how='left')

    for c in ['base_count', 'library_strategy', 'instrument_platform', 'study_accession']:
        if c in df.columns:
            df[c] = df[c].fillna('')

    conf_runs = set(df_conf['run_accession'].tolist())

    for i, row in df.iterrows():
        if row['run_accession'] not in conf_runs:
            continue
        if not row['biopsy_type']:
            ctx_path = os.path.join(xml_dir, f"{row['run_accession']}_metadata.xml")
            if not os.path.exists(ctx_path):
                continue
            ctx = open(ctx_path, "r", encoding="utf-8", errors="ignore").read()
            if re.search(r"\bmetasta(?:sis|ses|tic)\b", ctx, re.IGNORECASE):
                df.at[i, 'biopsy_type'] = 'metastasis'
            elif re.search(r"\bblood|plasma|venous|whole[-\s]?blood\b", ctx, re.IGNORECASE):
                df.at[i, 'biopsy_type'] = 'blood'

    df = df.fillna('')

    df.to_csv(output_file_df, sep="\t", index=False)

    if ambiguous_rows:
        pd.DataFrame(ambiguous_rows, columns=["run_accession", "candidates"]).to_csv(AMBIG_FILE, index=False)

    open(FLAG_FILE, 'w').close()
