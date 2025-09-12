##########################################################################################
#IMPORT
import os
import psutil
from llama_cpp import Llama
import torch
import sys
import argparse
import time
import numpy as np
import re
import json
import csv
from collections import defaultdict

parser = argparse.ArgumentParser(description="Process metadata with LLM (corrected)")
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--input_metadata_path", type=str, required=True)
parser.add_argument("--error_file_path", type=str, required=True)
parser.add_argument("--log_file_path", type=str, required=True)
parser.add_argument("--flag_file", type=str, required=True)
parser.add_argument("--initial_n_ctx", type=int, default=3500)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--max_value_tokens", type=int, default=128)
parser.add_argument("--strict_match_training", action="store_true", help="Per-category prompting aligned with training format")
parser.add_argument("--verbose", action="store_true", help="Verbose output")
args = parser.parse_args()

base_path           = args.base_path
input_metadata_path = args.input_metadata_path
error_file_path     = args.error_file_path
log_file_path       = args.log_file_path
FLAG_FILE           = args.flag_file
initial_n_ctx       = args.initial_n_ctx
model               = args.model
max_value_tokens    = args.max_value_tokens
STRICT_MATCH_TRAINING = args.strict_match_training

raw_final_info_path = os.path.join(base_path, "database_metadata_curated.csv")
output_dir          = os.path.join(base_path, "METADATA_LLM_INFERENCE")
skipped_runs_path   = os.path.join(base_path, "skipped_runs.txt")
ambi_cl_path        = os.path.join(base_path, "ambiguous_cell_lines.csv")
model_path          = os.path.join(base_path, model)
error_file_header   = "run_accession\tsummary"

sys.stdout = open(log_file_path, "a", encoding="utf-8")
sys.stderr = sys.stdout
VERBOSE = args.verbose
vprint = print if VERBOSE else (lambda *a, **k: None)

##########################################################################################
#FUNCTIONS

# def print_memory_usage(proc):
#     m = proc.memory_info()
#     v = psutil.virtual_memory()
#     vprint(f"rss: {m.rss/1024**2:.2f} MB, virt used: {v.used/1024**2:.2f} MB")

def get_llama_model(path, ctx):
    return Llama(
        model_path=path,
        n_ctx=ctx,
        n_gpu_layers=-1,
        use_mmap=True,
        n_threads=4,
        logits_all=True,
        flash_attn=True,
    )


def write_reload_file(fp, header, parts):
    dirpath = os.path.dirname(fp)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    entry = "\t".join(parts)
    if not os.path.exists(fp):
        with open(fp, "w", encoding="utf-8") as f:
            f.write(header + "\n")
    # Avoid duplicates
    with open(fp, "r+", encoding="utf-8") as f:
        content = f.read()
        if entry not in content:
            f.write(entry + "\n")


def load_ambiguous_map(fp):
    ambi = defaultdict(list)
    if not fp or not os.path.isfile(fp):
        return {}
    with open(fp, "r", encoding="utf-8", newline="") as f:
        sample = f.read(1024)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t;")
        except csv.Error:
            class _D: delimiter = ","  # fallback
            dialect = _D()
        reader = csv.reader(f, dialect)
        header_peek = next(reader, None)
        if header_peek is None:
            return {}

        def _push(row):
            if not row or len(row) < 2:
                return
            run_id = row[0].strip()
            val = row[1].strip()
            if run_id and val:
                ambi[run_id].append(val)

        if header_peek and not re.search(r"run|accession", " ".join(header_peek), re.I):
            _push(header_peek)
        for row in reader:
            _push(row)
    return {k: "; ".join(v) for k, v in ambi.items()}


def clean_val(t: str) -> str:
    t = (t or "").replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    if '"' in t:
        t = t.split('"', 1)[0]
    t = re.sub(r"\s*[,;}\]]\s*$", "", t)
    t = t.strip()
    return t if t else "unknown"


def calc_entropy_from_logprobs(lp_list):
    vals = [x for x in (lp_list or []) if x is not None]
    if not vals:
        return 0.0
    arr = np.array(vals, dtype=np.float64)
    arr -= np.max(arr)
    probs = np.exp(arr)
    probs /= probs.sum()
    ent = -np.sum(probs * np.log(probs + 1e-12))
    return float(ent)

##########################################################################################
# DATA & CAT

categories = [
    "library_selection", "sequencing_source", "biopsy_site", "biopsy_type",
    "cell_line", "cell_type", "organ", "disease", "treatment",
    "treatment_time", "response", "age", "sex", "ethnicity", "localization", "is_cancer"
]

definitions = {
    "library_selection": "one of: 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or extract other rare value (exclude cDNA or similar that are previous steps before real library selection). IF not in those categories, state 'other'",
    "sequencing_source": "one of: 'spatial', 'bulk', 'single cell'. search for transcriptomics information in context",
    "biopsy_site": "organ, body part or fluid WHERE TISSUE WAS SAMPLED",
    "biopsy_type": "state 'metastasis' IF CANCER AND METASTASIS MENTIONNED, OR 'blood' if no metastasis and blood related information mentionned, OTHERWISE state 'primary'. CAN ONLY STATE THOSE THREE INFORMATION, YOU SHOULD ALWAYS BE CAPABLE TO DETERMINE ONE OF THE 3 VALUES",
    "cell_line": "exact cell line, ie standardized names of cultured/immortalized laboratory cell populations. Get the cell line code anywhere in the text",
    "cell_type": "extract cell type: if known, specify it (e.g., 'T cell', 'fibroblast', etc). state specific cell type; otherwise, write 'primary tissue'. If the cell type is not directly available, TRY to deduce it from the organ before answering 'primary tissue'.",
    "organ": "organ studied or affected (not where the sample is from, very different from biopsy_site)",
    "disease": "report associated disease (BE SPECIFIC) or 'healthy' status (be careful to specific vocabulary that could indicate that the sample is healthy, for eg. adjacent is something next to the disease, or normal, etc...)",
    "treatment": "treatment applied treatment applied = may be molecules or drug treatments or surgical operations, or something implanted, or events altering the state of the organism, etc. if not explicitly stated, infer from the context. DON'T STATE the disease, get info just from treatment",
    "treatment_time": "time or phase relative to treatment (qualitative or quantitative information, but favour quantitative data). BE CAREFUL TO GET THE TIME RELATED TO THE DEDUCED TREATMENT(S)",
    "response": "treatment response, state of the cell after treatment, without mention again the treatment any kind of event after treatment if applicable. if no clear statement, try to infer from context the stage of the disease after treatment if possible",
    "age": "sample donor age. Can be quantitative (range or exact age) or qualitative (eg: child, teenage, adult, senior, ETC)",
    "sex": "sample donor sex",
    "ethnicity": "sample donor ethnicity (origins, genetics)",
    "localization": "all geographical information available, if several list them all",
    "is_cancer": "return 'True' if the disease is cancer related, 'False' otherwise"
}

##########################################################################################
#MAIN
process = psutil.Process(os.getpid())

use_gpu = torch.cuda.is_available()
vprint(f"use_gpu: {use_gpu}")
gpu_count = torch.cuda.device_count() if use_gpu else 0
vprint(f"Used GPU count: {gpu_count}")

if use_gpu and gpu_count > 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))
    vprint(f"Using {gpu_count} GPU(s): {os.environ['CUDA_VISIBLE_DEVICES']}")
else:
    vprint("No GPU detected → using CPU only")

llm = get_llama_model(model_path, initial_n_ctx)

with open(input_metadata_path, "r", encoding="utf-8") as mf:
    header = mf.readline()
    metadata_lines = mf.readlines()

with open(raw_final_info_path, "r", encoding="utf-8") as rf:
    raw = rf.readlines()
    raw_headers = raw[0].rstrip("\n").split("\t")
    raw_data = {r.split("\t")[0]: r.rstrip("\n").split("\t") for r in raw[1:]}
    vprint(f"Loaded curated rows: {len(raw_data)}")

ambi_map = load_ambiguous_map(ambi_cl_path)

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

skipped_runs = []

vprint("[INFO] Prompt mode:", "STRICT_MATCH_TRAINING" if STRICT_MATCH_TRAINING else "GENERAL_MULTI_CATEGORIES")
vprint("[INFO] Model (gguf):", model_path)

# Helper to build prompts (single-category when STRICT_MATCH_TRAINING, otherwise multi-cat JSON like production)

def build_prompt_single_category(run, summary, extra_info_str, key):
    return f"""Run accession: {run}
Summary: {summary} {extra_info_str}

Categories and definitions:
- {key}: {definitions[key]}

For each category below:
- Extract information from the summary if possible
- If one value is impossible to extract, even by deducing it, return "unknown"

BE CAREFUL: Sometimes the information concerns several samples from the same study. It is important to distinguish between them and semantically extract what applies to the current run, so everything must be consistent.
FOR EACH CATEGORY SEVERAL ANSWERS CAN BE POSSIBLE, CITE THEM ALL WITH A ',' OR ';' SEPARATOR

Respond strictly in a few words with valid JSON (double quotes around keys and values), no extra keys. ONLY THE CATEGORIES CITED UNDER 'Categories and definitions'

Here is the output:
{{\n"{key}": """  # we will stop at the closing quote


def build_prompt_multi_category(run, summary, extra_info_str):
    inst_lines = "\n".join(f"- {c}: {definitions[c]}" for c in categories)
    return f"""Run accession: {run}
Summary: {summary} {extra_info_str}

Categories and definitions:
{inst_lines}

For each category below:
- Infer from the summary if possible
- The value can be not applicable ONLY FOR: treatment_time and response (if treatment = no treatment) AND cell_line (if cell_type = primary tissue), RETURN "not applicable" for those categories. CAN'T BE NOT APPLICABLE FOR THE OTHER CATEGORIES.
- If one value is impossible to infer, return "unknown", applicable for all categories ALWAYS BETTER THAN FALSE ANSWER ESPECIALLY FOR SPECIFIC DONOR INFORMATION (AGE, SEX, etc)

BE CAREFUL: Sometimes the information concerns several samples from the same study. It is important to distinguish between them and semantically extract what applies to the current run, so everything must be consistent.
FOR EACH CATEGORY SEVERAL ANSWERS CAN BE POSSIBLE, CITE THEM ALL WITH A ',' OR ';' SEPARATOR

Respond strictly in a few words with valid JSON (double quotes around keys and values), no extra keys. ONLY THE CATEGORIES CITED UNDER 'Categories and definitions'

Here is the output:
"""


# Process each run
for idx, line in enumerate(metadata_lines):
    if not line.strip():
        write_reload_file(error_file_path, error_file_header, [f"LINE_{idx}", "empty line"])
        skipped_runs.append(f"LINE_{idx}")
        continue

    parts = line.rstrip("\n").split("\t", 1)
    if len(parts) < 2 or not parts[0].strip():
        bad_run = parts[0].strip() if parts and parts[0].strip() else f"LINE_{idx}"
        write_reload_file(error_file_path, error_file_header, [bad_run, "malformed: missing tab/summary"])
        skipped_runs.append(bad_run)
        continue

    run, summary = parts[0].strip(), parts[1].strip()
    vprint(run)

    if run not in raw_data:
        skipped_runs.append(run)
        continue

    raw_vals = raw_data[run]
    extra_info = []
    for col, val in zip(raw_headers, raw_vals):
        if val:
            extra_info.append(f"- {col}: {val}")
    ambi_val = ambi_map.get(run)
    if ambi_val:
        extra_info.append(f"{ambi_val}")
    extra_info_str = " ".join(extra_info) if extra_info else ""

    vprint(f"\n[{idx+1}/{len(metadata_lines)}] {run}", flush=True)
    # print_memory_usage(process)

    if STRICT_MATCH_TRAINING:
        out = {}
        entropies = {}
        for key in categories:
            prefix_text = build_prompt_single_category(run, summary, extra_info_str, key)
            vprint(f"[PEFT] {run} | {key:16s} -> active: (none; llama.cpp)")
            vprint(f"[PROMPT] {run} | {key}\n{prefix_text}")
            tcat0 = time.perf_counter()
            resp = llm(prefix_text, max_tokens=max_value_tokens, stop=['"'], logprobs=True, echo=False)
            text = resp["choices"][0]["text"]
            logp = resp["choices"][0].get("logprobs", {}).get("token_logprobs", [])
            out[key] = clean_val(text)
            entropies[key] = calc_entropy_from_logprobs(logp)
            vprint(f"[OUT] {run} | {key}: {out[key]} | H={entropies[key]:.6f}")
            vprint(f"[TIMING][cat] {run} | {key}: {time.perf_counter() - tcat0:.4f}s")
            vprint("----------------------------------------------------------------------------")

        output_payload = {run: out, "entropy": entropies}

    else:
        prompt = build_prompt_multi_category(run, summary, extra_info_str)
        vprint("PROMPT:", flush=True)
        vprint(prompt, flush=True)
        vprint("BEGIN:", flush=True)
        resp = llm(prompt, max_tokens=350, logprobs=True)
        vprint("ANSWER:", flush=True)
        vprint(resp["choices"][0]["text"])
        text = resp["choices"][0]["text"].strip()

        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            vprint("No json bloc in the answer")
            write_reload_file(error_file_path, error_file_header, [run, summary])
            skipped_runs.append(run)
            continue

        json_str = m.group(0)
        try:
            parsed_json = json.loads(json_str)
            vprint("Json good format: ", parsed_json)
        except json.JSONDecodeError:
            vprint("Json format error")
            write_reload_file(error_file_path, error_file_header, [run, summary])
            skipped_runs.append(run)
            continue

        tokens = resp["choices"][0]["logprobs"]["tokens"]
        logprobs = resp["choices"][0]["logprobs"]["token_logprobs"]
        ordered_keys = list(parsed_json.keys())
        entropy_dict = {}

        def tokenize_pieces(llm_obj, text_):
            ids = llm_obj.tokenize(text_.encode("utf-8"), add_bos=False)
            return [llm_obj.detokenize([tid]).decode("utf-8", errors="ignore") for tid in ids]

        def build_category_token_patterns(keys_, llm_obj):
            patterns_ = {}
            suffixes = ['"', '":', '": ', '": "']
            prefixes = ['', ' ', '\n', ', ', '\n  ']
            for k in keys_:
                variants = []
                for suf in suffixes:
                    s = f'"{k}{suf}'
                    variants.append(tokenize_pieces(llm_obj, s))
                    for pref in prefixes[1:]:
                        variants.append(tokenize_pieces(llm_obj, pref + s))
                dedup = []
                seen = set()
                for v in variants:
                    tup = tuple(v)
                    if tup not in seen:
                        seen.add(tup)
                        dedup.append(v)
                patterns_[k] = dedup
            return patterns_

        def find_subsequence(full, sub):
            L, l = len(full), len(sub)
            for i in range(L - l + 1):
                if full[i:i + l] == sub:
                    return i
            return -1

        def find_subsequence_any(full_tokens, list_of_patterns):
            for pat in list_of_patterns:
                idx = find_subsequence(full_tokens, pat)
                if idx >= 0:
                    return idx, len(pat)
            return -1, 0

        category_token_patterns = build_category_token_patterns(categories, llm)

        for key in ordered_keys:
            pats = category_token_patterns.get(key, [])
            if not pats:
                entropy_dict[key] = None
                continue

            idx, matched_len = find_subsequence_any(tokens, pats)
            if idx < 0:
                entropy_dict[key] = None
                continue
            start = idx + matched_len

            end = len(tokens)
            for next_key in ordered_keys[ordered_keys.index(key) + 1:]:
                next_pats = category_token_patterns.get(next_key, [])
                if not next_pats:
                    continue
                ni, _ = find_subsequence_any(tokens, next_pats)
                if ni >= 0:
                    end = ni
                    break

            if end <= start:
                entropy_dict[key] = None
                continue

            segment_logprobs = logprobs[start:end]
            entropy_dict[key] = calc_entropy_from_logprobs(segment_logprobs)

        output_payload = {run: parsed_json, "entropy": entropy_dict}

    out_fp = os.path.join(output_dir, f"{run}.json")
    with open(out_fp, "w", encoding="utf-8") as of:
        json.dump(output_payload, of, indent=2, ensure_ascii=False)

with open(skipped_runs_path, "w", encoding="utf-8") as sf:
    for r in skipped_runs:
        sf.write(r + "\n")

sys.stdout.close()

del llm
import gc
gc.collect()
open(FLAG_FILE, "w").close()
