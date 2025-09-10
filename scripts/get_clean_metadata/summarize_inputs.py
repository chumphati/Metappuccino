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

##########################################################################################
#PATHS
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
download('en_core_web_md')
nlp = spacy.load('en_core_web_md')

parser = argparse.ArgumentParser(description="Fetch information with Cellosaurus")
parser.add_argument("--base_path", type=str, required=True, help="Base path to Metappuccino")
parser.add_argument("--verbose", action="store_true", help="Verbose output")
args = parser.parse_args()

base_path = args.base_path
INPUT_FILE = os.path.join(base_path, "cleaned_metadata_sra.txt")
OUTPUT_FILE = os.path.join(base_path, "metadata_sra_summarized.txt")
FLAG_FILE = os.path.join(base_path, "STEP2_2.flag")
AMBIG_FILE = os.path.join(base_path, "ambiguous_cell_lines.csv")

# INPUT_FILE = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/DATA_PER_CAT/tmp/cleaned_metadata_sra.txt"
# OUTPUT_FILE = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/DATA_PER_CAT/cell_line/metadata_sra_summarized.txt"
# FLAG_FILE = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/DATA_PER_CAT/tmp/STEP2_2.flag"
# AMBIG_FILE = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/DATA_PER_CAT/tmp/ambiguous_cell_lines.csv"

VERBOSE = args.verbose
# VERBOSE = False
vprint = print if VERBOSE else (lambda *a, **k: None)

MAX_TOKENS = 1200
TOKEN_THRESHOLD = 1200
CHUNK_SIZE = 100
CATEGORY_KEYWORDS = [
    'cell type', 'tissue type', 'cell line', 'organ', 'disease',
    'host phenotype', 'library selection', 'library source',
    'treatment', 'treatment time', 'response', 'donor information', 'instrument platform'
]
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

category_docs = [nlp(cat) for cat in CATEGORY_KEYWORDS]

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

def extract_clauses(text):
    tokens = word_tokenize(text)
    chunks = []
    i, n = 0, len(tokens)
    while i < n:
        j = min(i + CHUNK_SIZE, n)
        k = j
        while k > i and tokens[k-1] not in '.?!;,':
            k -= 1
        if k == i:
            k = j
        chunks.append({'text': ' '.join(tokens[i:k]), 'orig_idx': i})
        i = k
    return chunks

def score_clauses(clauses):
    texts = [c['text'] for c in clauses] or [""]
    vect = TfidfVectorizer(stop_words='english').fit(texts)
    X = vect.transform(texts)
    tfidf_scores = np.array(X.sum(axis=1)).ravel()
    cat_scores = []
    for t in texts:
        doc = nlp(t)
        s = 0.0
        for cdoc in category_docs:
            sim = doc.similarity(cdoc)
            if sim > SEMANTIC_THRESHOLD:
                s += sim
        cat_scores.append(s)
    times = np.array([time_score(t) for t in texts])
    negs = np.array([neg_score(t) for t in texts])
    combined = tfidf_scores + RELEVANCE_BOOST * np.array(cat_scores) + TIME_BOOST * times + NEG_BOOST * negs
    return combined

def summarize_by_clauses(text):
    clauses = extract_clauses(text)
    if not clauses:
        return text.strip()
    scores = score_clauses(clauses)
    order = np.argsort(-scores)
    summary_chunks = []
    used = set()
    token_total = 0
    for i in order:
        t = clauses[i]['text']
        t_tokens = mistral_tokens(t)
        if token_total + t_tokens <= MAX_TOKENS:
            summary_chunks.append((clauses[i]['orig_idx'], t))
            used.add(i)
            token_total += t_tokens
        if token_total >= MAX_TOKENS:
            break
    for cdoc in category_docs:
        best_sim, best_idx = 0.0, -1
        for i, clause in enumerate(clauses):
            if i in used:
                continue
            doc = nlp(clause['text'])
            sim = doc.similarity(cdoc)
            if sim > best_sim:
                best_sim, best_idx = sim, i
        if best_sim > SEMANTIC_THRESHOLD and best_idx >= 0:
            t = clauses[best_idx]['text']
            t_tokens = mistral_tokens(t)
            if token_total + t_tokens <= MAX_TOKENS:
                summary_chunks.append((clauses[best_idx]['orig_idx'], t))
                used.add(best_idx)
                token_total += t_tokens
    summary_chunks.sort(key=lambda x: x[0])
    texts = [s for _, s in summary_chunks]
    summary = ' '.join(t if t.endswith(('.', '?', '!')) else t + '.' for t in texts).strip()
    if mistral_tokens(summary) > MAX_TOKENS:
        sents = sent_tokenize(summary)
        while sents and mistral_tokens(' '.join(sents)) > MAX_TOKENS:
            sents.pop()
        summary = ' '.join(sents).strip()
    return summary

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
        if orig_tokens > TOKEN_THRESHOLD:
            summ = summarize_by_clauses(ctx)
            vprint(f"{run_acc} summarized_tokens={mistral_tokens(summ)}", flush=True)
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
