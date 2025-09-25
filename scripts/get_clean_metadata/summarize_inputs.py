##########################################################################################
#IMPORT
import csv
import os
import argparse
import re
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from spacy.cli import download
from transformers import AutoTokenizer
from collections import Counter
from nltk.corpus import stopwords

##########################################################################################
#PATHS
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download("stopwords", quiet=True)
download('en_core_web_md')
nlp = spacy.load('en_core_web_md')

# Initialize STOPWORDS
try:
    STOPWORDS = set(w.lower() for w in stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(w.lower() for w in stopwords.words("english"))

parser = argparse.ArgumentParser(description="Fetch information with Cellosaurus")
parser.add_argument("--base_path", type=str, required=True, help="Base path to Metappuccino")
parser.add_argument("--verbose", action="store_true", help="Verbose output")
args = parser.parse_args()

base_path = args.base_path
INPUT_FILE = os.path.join(base_path, "cleaned_metadata_sra.txt")
OUTPUT_FILE = os.path.join(base_path, "metadata_sra_summarized.txt")
FLAG_FILE = os.path.join(base_path, "STEP2_2.flag")
AMBIG_FILE = os.path.join(base_path, "ambiguous_cell_lines.csv")

VERBOSE = args.verbose
vprint = print if VERBOSE else (lambda *a, **k: None)

MAX_TOKENS = 2000
TOKEN_THRESHOLD = MAX_TOKENS
CHUNK_SIZE = 100
SEMANTIC_THRESHOLD = 0.4
RELEVANCE_BOOST = 2.0
TIME_BOOST = 2.0
NEG_BOOST = 1.0

tok = None
tok_id = None
for mid in [
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-v0.1"
]:
    try:
        tok = AutoTokenizer.from_pretrained(mid, use_fast=True)
        tok_id = mid
        break
    except Exception:
        continue
if tok is None:
    try:
        tok = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=True)
        tok_id = "meta-llama/Llama-2-7b-hf (fallback)"
    except Exception:
        tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        tok_id = "gpt2 (fallback)"
vprint(f"Tokenizer loaded: {tok_id}", flush=True)

##########################################################################################
#FUNCTIONS

def mistral_tokens(text: str) -> int:
    return len(tok.encode(text, add_special_tokens=False))


def _safe_split_candidates(s):
    if s is None:
        return []
    s = s.strip().strip("[](){}")
    s = re.sub(r"[;|/]", ",", s)
    parts = [p.strip().strip("'").strip('"') for p in s.split(",")]
    seen, out = set(), []
    for p in parts:
        if p and p not in seen:
            seen.add(p); out.append(p)
    return out

def _candidate_regex(cand):
    toks = re.findall(r"[A-Za-z]+|\d+", cand)
    if not toks:
        toks = [cand]
    pattern = r"\b" + r"[\s\-_\/]*".join(map(re.escape, toks)) + r"\b"
    return re.compile(pattern, flags=re.IGNORECASE)

def _best_candidate_from_context(ctx_text, candidates):
    if not candidates:
        return None, "none"
    counts = []
    for c in candidates:
        rgx = _candidate_regex(c)
        counts.append(len(re.findall(rgx, ctx_text)))
    max_cnt = max(counts) if counts else 0
    if max_cnt > 0:
        best_idx = counts.index(max_cnt)
        return candidates[best_idx], "match"
    short_ctx = ctx_text[:10000]
    try:
        doc_ctx = nlp(short_ctx)
        sims = [doc_ctx.similarity(nlp(c)) for c in candidates]
        best_idx = int(np.argmax(sims))
        return candidates[best_idx], "similarity"
    except Exception:
        return candidates[0], "fallback"

def _load_ambiguous_rows(path):
    rows = []
    if not os.path.exists(path):
        return rows, {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            dr = csv.DictReader(f)
            if dr.fieldnames and "run_accession" in dr.fieldnames and "candidates" in dr.fieldnames:
                for r in dr:
                    rows.append(dict(r))
                mapping = {r["run_accession"].strip(): _safe_split_candidates(r.get("candidates")) for r in rows if r.get("run_accession")}
                return rows, mapping
        except Exception:
            pass
        f.seek(0)
        rr = csv.reader(f)
        for i, row in enumerate(rr):
            if not row:
                continue
            if i == 0 and len(row) >= 2 and row[0].lower() == "run_accession":
                continue
            if len(row) >= 2:
                rows.append({"run_accession": row[0], "candidates": row[1]})
        mapping = {r["run_accession"].strip(): _safe_split_candidates(r.get("candidates")) for r in rows if r.get("run_accession")}
        return rows, mapping

def _write_ambiguous_rows(path, rows):
    fieldnames = ["run_accession", "candidates", "chosen", "note"]
    if rows:
        extra = [k for k in rows[0].keys() if k not in fieldnames]
        fieldnames = ["run_accession", "candidates"] + extra + ["chosen", "note"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        dw = csv.DictWriter(f, fieldnames=fieldnames)
        dw.writeheader()
        for r in rows:
            if "chosen" not in r: r["chosen"] = ""
            if "note" not in r: r["note"] = ""
            dw.writerow(r)

_SRA_ID_PATTERNS = [
    re.compile(r"^(SRR|DRR|ERR)\d+", re.IGNORECASE),   # runs
    re.compile(r"^(SRP|DRP|ERP)\d+", re.IGNORECASE),   # studies/projects SRA
    re.compile(r"^PRJ[A-Z]{2}\d+", re.IGNORECASE),     # BioProject: PRJNA/PRJEB/PRJDB
]

def _remove_sra_ids(text: str) -> str:
    words = text.split()
    kept = []
    for w in words:
        uw = w.upper().strip('.,;:()[]{}<>"\'')
        if any(p.match(uw) for p in _SRA_ID_PATTERNS):
            continue
        if uw.startswith("SRR") or uw.startswith("DRR") or uw.startswith("ERR"):
            continue
        kept.append(w)
    return " ".join(kept)

def trim_context_to_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    cur_tokens = mistral_tokens(text)
    if cur_tokens <= max_tokens:
        vprint(f"trim_context_to_tokens: within limit ({cur_tokens} tokens)", flush=True)
        return text

    #Step 1: remove SRA IDs
    text = _remove_sra_ids(text)
    cur_tokens = mistral_tokens(text)
    if cur_tokens <= max_tokens:
        vprint(f"trim_context_to_tokens: reduced by removing SRA IDs ({cur_tokens} tokens)", flush=True)
        return text

    #Step 2: remove frequent/stopwords first
    words = text.split()
    if len(words) > 50:
        freq = Counter(w.lower().strip(",.;:!?") for w in words)
        scored = [
            (w, (1 if w.lower() in STOPWORDS else 0) + 0.5 * freq[w.lower()])
            for w in words
        ]
        scored_sorted = sorted(enumerate(scored), key=lambda x: x[1][1], reverse=True)
        keep_mask = [True] * len(words)
        i = 0
        while mistral_tokens(" ".join([w for (w,m), k in zip(scored, keep_mask) if k])) > max_tokens and i < len(scored_sorted):
            idx, _ = scored_sorted[i]
            keep_mask[idx] = False
            i += 1
        reduced = " ".join([w for (w,m), k in zip(scored, keep_mask) if k])
        cur_tokens = mistral_tokens(reduced)
        if cur_tokens <= max_tokens:
            vprint(f"trim_context_to_tokens: reduced by stopwords/frequent removal ({cur_tokens} tokens)", flush=True)
            return reduced
        text = reduced
        vprint(f"trim_context_to_tokens: stopwords/frequent removal not enough ({cur_tokens} tokens)", flush=True)

    #Step 3: cut by sentences
    sents = sent_tokenize(text)
    while sents and mistral_tokens(" ".join(sents)) > max_tokens:
        sents.pop()
    if sents:
        cur_tokens = mistral_tokens(" ".join(sents))
        vprint(f"trim_context_to_tokens: reduced by sentence trimming ({cur_tokens} tokens)", flush=True)
        return " ".join(sents)

    #Step 4: fallback proportional cut
    words = text.split()
    ratio = max_tokens / (cur_tokens + 1e-9)
    new_len = max(1, int(len(words) * ratio))
    final_text = " ".join(words[:new_len])
    vprint(f"trim_context_to_tokens: reduced by proportional cut ({mistral_tokens(final_text)} tokens)", flush=True)
    return final_text


_TIME_REGEXES = [
    re.compile(r"\b\d+(\.\d+)?\s?(h|hr|hrs|hour|hours|min|mins|minute|minutes|sec|secs|second|seconds|d|day|days|wk|wks|week|weeks)\b", re.IGNORECASE),
    re.compile(r"\b(\d+(\.\d+)?)(h|m|min|d|w|wk)\b", re.IGNORECASE),
    re.compile(r"\b(hpi|dpi|wpi)\b", re.IGNORECASE),
    re.compile(r"\btime\s*point\b", re.IGNORECASE),
    re.compile(r"\bpre[-\s]?treatment|\bpost[-\s]?treatment|\bbaseline\b|\bfollow[-\s]?up\b", re.IGNORECASE),
]
_NEG_WORDS = re.compile(r"\b(no|not|without|lack|absence|absent|never|none|neither|nor|negative for)\b", re.IGNORECASE)

def time_score(text):
    s = 0
    for rgx in _TIME_REGEXES:
        if rgx.search(text):
            s += 1.0
    return s

def neg_score(text):
    return len(re.findall(_NEG_WORDS, text))

amb_rows, ambiguous_map = _load_ambiguous_rows(AMBIG_FILE)
updates = {}

##########################################################################################
#MAIN
with open(INPUT_FILE, 'r', encoding='utf-8') as fin, open(OUTPUT_FILE, 'w', encoding='utf-8', newline="") as fout:
    rdr = csv.reader(fin, delimiter='\t')
    wtr = csv.writer(fout, delimiter='\t')
    hdr = next(rdr)
    wtr.writerow([hdr[0], 'summary'])
    for row in tqdm(rdr, desc='Processing rows'):
        run_acc = row[0]
        raw = ' '.join(f for f in row[1:] if not any(x in f.lower() for x in ['run accession', 'study accession', 'experiment accession', 'sample accession']))
        ctx = re.sub(r'\S+?\.fastq\.gz', '', raw).strip()
        orig_tokens = mistral_tokens(ctx)
        vprint(f"{run_acc} original_tokens={orig_tokens}", flush=True)

        if orig_tokens > MAX_TOKENS:
            summ = trim_context_to_tokens(ctx, max_tokens=MAX_TOKENS)
            vprint(f"{run_acc} trimmed_tokens={mistral_tokens(summ)}", flush=True)
        else:
            summ = ctx
            vprint(f"{run_acc} kept_tokens={mistral_tokens(summ)}", flush=True)

        if run_acc in ambiguous_map and ambiguous_map[run_acc]:
            cands = ambiguous_map[run_acc]
            chosen, method = _best_candidate_from_context(ctx if ctx else summ, cands)
            method_tag = "by match" if method == "match" else ("by similarity" if method == "similarity" else "by fallback")
            note = f" Cell line disambiguation: {chosen} (candidates: {' | '.join(cands)}) — {method_tag}."
            tentative = (summ + " " + note).strip()
            if mistral_tokens(tentative) > MAX_TOKENS:
                sents = sent_tokenize(summ)
                while sents and mistral_tokens(' '.join(sents) + " " + note) > MAX_TOKENS:
                    sents.pop()
                summ = (' '.join(sents)).strip()
            summ = (summ + " " + note).strip()
            updates[run_acc] = {"chosen": chosen, "note": note.strip()}
            vprint(f"{run_acc} ambiguous_resolved={chosen} method={method}", flush=True)

        wtr.writerow([run_acc, summ])
        vprint(f"{run_acc} final_tokens={mistral_tokens(summ)}", flush=True)

if amb_rows:
    rows_by_run = {r.get("run_accession", "").strip(): r for r in amb_rows}
    for run, up in updates.items():
        if run in rows_by_run:
            rows_by_run[run]["chosen"] = up["chosen"]
            rows_by_run[run]["note"] = up["note"]
        else:
            rows_by_run[run] = {"run_accession": run, "candidates": "", "chosen": up["chosen"], "note": up["note"]}
    _write_ambiguous_rows(AMBIG_FILE, list(rows_by_run.values()))

open(FLAG_FILE, 'w').close()
