##########################################################################################
#IMPORT
import os
import psutil
from llama_cpp import Llama
import torch
import sys
import argparse
import math
import numpy as np
import re
import json
import csv
from collections import defaultdict

parser = argparse.ArgumentParser(description="Process metadata with LLM")
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--input_metadata_path", type=str, required=True)
parser.add_argument("--error_file_path", type=str, required=True)
parser.add_argument("--log_file_path", type=str, required=True)
parser.add_argument("--flag_file", type=str, required=True)
parser.add_argument("--initial_n_ctx", type=int, default=3500)
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--verbose", action="store_true", help="Verbose output")
args = parser.parse_args()

base_path           = args.base_path
input_metadata_path = args.input_metadata_path
error_file_path     = args.error_file_path
log_file_path       = args.log_file_path
FLAG_FILE           = args.flag_file
initial_n_ctx       = args.initial_n_ctx
model               = args.model

raw_final_info_path = os.path.join(base_path, "database_metadata_curated.csv")
output_dir          = os.path.join(base_path, "METADATA_LLM_INFERENCE")
skipped_runs_path   = os.path.join(base_path, "skipped_runs.txt")
ambi_cl_path   = os.path.join(base_path, "ambiguous_cell_lines.csv")
model_path          = os.path.join(base_path, model)
error_file_header   = "run_accession\tsummary"

sys.stdout = open(log_file_path, "a")
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
        flash_attn=True
    )

def write_reload_file(fp, header, parts):
    dirpath = os.path.dirname(fp)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    entry = "\t".join(parts)
    if not os.path.exists(fp):
        with open(fp, "w") as f:
            f.write(header + "\n")
    with open(fp, "r+") as f:
        content = f.read()
        if entry not in content:
            f.write(entry + "\n")

def calculate_entropy_optimized(logprobs):
    arr = np.array(logprobs)
    arr -= np.max(arr)
    probs = np.exp(arr)
    probs /= probs.sum()
    return float(-np.sum(probs * np.log(probs + 1e-10)))

def find_subsequence(full, sub):
    L, l = len(full), len(sub)
    for i in range(L - l + 1):
        if full[i:i+l] == sub:
            return i
    return -1

def get_token_spans(text, tokens):
    spans = []
    offset = 0
    for tok in tokens:
        decoded = llm.detokenize([tok]).decode("utf-8", errors="ignore")
        offset = text.find(decoded, offset)
        if offset == -1:
            spans.append(None)
        else:
            spans.append(offset)
            offset += len(decoded)
    return spans


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
            class _D: delimiter = ","
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

with open(input_metadata_path) as mf:
    header = mf.readline()
    metadata_lines = mf.readlines()

with open(raw_final_info_path) as rf:
    raw = rf.readlines()
    raw_headers = raw[0].strip().split("\t")
    raw_data = {r.split("\t")[0]: r.strip().split("\t") for r in raw[1:]}
    vprint(raw_data)

ambi_map = load_ambiguous_map(ambi_cl_path)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

skipped_runs = []

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

    na_columns = categories.copy()
    vprint(na_columns)

    raw_vals = raw_data[run]
    extra_info = []
    for col, val in zip(raw_headers, raw_vals):
        if val and val != "":
            extra_info.append(f"- {col}: {val}")

    ambi_val = ambi_map.get(run)
    if ambi_val:
        extra_info.append(f"{ambi_val}")

    extra_info_str = " ".join(extra_info) if extra_info else ""

    vprint(f"\n[{idx+1}/{len(metadata_lines)}] {run}", flush=True)
    # print_memory_usage(process)

    inst_lines = "\n".join(f"- {c}: {definitions[c]}" for c in na_columns)
    fmt_keys   = ", ".join(f'"{c}": "<value>"' for c in na_columns)

    prompt = f"""Run accession: {run}
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

    vprint("PROMPT:", flush=True)
    vprint(prompt, flush=True)

    vprint("BEGIN:", flush=True)
    resp      = llm(prompt, max_tokens=350, logprobs=True)
    vprint("ANSWER:", flush=True)
    vprint(resp["choices"][0]["text"])
    text      = resp["choices"][0]["text"].strip()

    m = re.search(r'\{.*\}', text, flags=re.DOTALL)
    if m:
        json_str = m.group(0)
        try:
            parsed_json = json.loads(json_str)
            vprint("Json good format: ", parsed_json)
        except json.JSONDecodeError:
            vprint("Json format error")
            write_reload_file(error_file_path, error_file_header, [run, summary])
            skipped_runs.append(run)
            continue
    else:
        vprint("No json bloc in the answer")
        write_reload_file(error_file_path, error_file_header, [run, summary])
        skipped_runs.append(run)
        continue

    #mistral
    # category_token_patterns = {
    #     "library_selection": ['library', '_', 'selection', '":', ' "'],
    #     "sequencing_source": [' "', 'sequ', 'encing', '_', 'source', '":', ' "'],
    #     "biopsy_site": [' "', 'bi', 'ops', 'y', '_', 'site', '":', ' "'],
    #     "biopsy_type": [' "', 'bi', 'ops', 'y', '_', 'type', '":', ' "'],
    #     "cell_line": [' "', 'cell', '_', 'line', '":', ' "'],
    #     "cell_type": [' "', 'cell', '_', 'type', '":', ' "'],
    #     "organ": [' "', 'organ', '":', ' "'],
    #     "disease": [' "', 'd', 'ise', 'ase', '":', ' "'],
    #     "treatment": [' "', 't', 'reat', 'ment', '":', ' "'],
    #     "treatment_time": [' "', 't', 'reat', 'ment', '_', 'time', '":', ' "'],
    #     "response": [' "', 'response', '":', ' "'],
    #     "age": [' "', 'age', '":', ' "'],
    #     "sex": [' "', 'sex', '":', ' "'],
    #     "ethnicity": [' "', 'eth', 'nic', 'ity', '":', ' "'],
    #     "localization": [' "', 'local', 'ization', '":', ' "'],
    #     "is_cancer": [' "', 'is', '_', 'c', 'ancer', '":', ' "'],
    # }

    #llama
    # category_token_patterns = {
    #     "library_selection": [' {"', 'library', '_selection', '":', ' "'],
    #     "sequencing_source": [' "', 'sequ', 'encing', '_source', '":', ' "'],
    #     "biopsy_site": [' "', 'bi', 'opsy', '_site', '":', ' "'],
    #     "biopsy_type": [' "', 'bi', 'opsy', '_type', '":', ' "'],
    #     "cell_line": [' "', 'cell', '_line', '":', ' "'],
    #     "cell_type": [' "', 'cell', '_type', '":', ' "'],
    #     "organ": [' "', 'organ', '":', ' "'],
    #     "disease": [' "', 'd', 'isease', '":', ' "'],
    #     "treatment": [' "', 't', 'reatment', '":', ' "'],
    #     "treatment_time": [' "', 't', 'reatment', '_time', '":', ' "'],
    #     "response": [' "', 'response', '":', ' "'],
    #     "age": [' "', 'age', '":', ' "'],
    #     "sex": [' "', 'sex', '":', ' "'],
    #     "ethnicity": [' "', 'ethnic', 'ity', '":', ' "'],
    #     "localization": [' "', 'local', 'ization', '":', ' "'],
    #     "is_cancer": [' "', 'is', '_c', 'ancer', '":', ' "'],
    # }

    #gemma
    # category_token_patterns = {
    #     "library_selection": ['"', 'library', '_', 'selection', '":', '▁"'],
    #     "sequencing_source": ['"', 'sequ', 'encing', '_', 'source', '":', '▁"'],
    #     "biopsy_site": ['"', 'bio', 'psy', '_', 'site', '":', '▁"'],
    #     "biopsy_type": ['"', 'bio', 'psy', '_', 'type', '":', '▁"'],
    #     "cell_line": ['"', 'cell', '_', 'line', '":', '▁"'],
    #     "cell_type": ['"', 'cell', '_', 'type', '":', '▁"'],
    #     "organ": ['"', 'organ', '":', '▁"'],
    #     "disease": ['"', 'disease', '":', '▁"'],
    #     "treatment": ['"', 'treatment', '":', '▁"'],
    #     "treatment_time": ['"', 'treatment', '_', 'time', '":', '▁"'],
    #     "response": ['"', 'response', '":', '▁"'],
    #     "age": ['"', 'age', '":', '▁"'],
    #     "sex": ['"', 'sex', '":', '▁"'],
    #     "ethnicity": ['"', 'ethnicity', '":', '▁"'],
    #     "localization": ['"', 'localization', '":', '▁"'],
    #     "is_cancer": ['"', 'is', '_', 'cancer', '":', '▁"'],
    # }

    #deepseek
    category_token_patterns = {
        "library_selection": ['"', 'library', '_', 'selection', '":'],
        "sequencing_source": ['"', 'sequ', 'encing', '_', 'source', '":'],
        "biopsy_site": ['"', 'bi', 'opsy', '_', 'site', '":'],
        "biopsy_type": ['"', 'bi', 'opsy', '_', 'type', '":'],
        "cell_line": ['"', 'cell', '_', 'line', '":'],
        "cell_type": ['"', 'cell', '_', 'type', '":'],
        "organ": ['"', 'organ', '":'],
        "disease": ['"', 'disease', '":'],
        "treatment": ['"', 'treatment', '":'],
        "treatment_time": ['"', 'treatment', '_', 'time', '":'],
        "response": ['"', 'response', '":'],
        "age": ['"', 'age', '":'],
        "sex": ['"', 'sex', '":'],
        "ethnicity": ['"', 'eth', 'nic', 'ity', '":'],
        "localization": ['"', 'local', 'ization', '":'],
        "is_cancer": ['"', 'is', '_', 'c', 'ancer', '":'],
    }

    tokens = resp["choices"][0]["logprobs"]["tokens"]
    logprobs = resp["choices"][0]["logprobs"]["token_logprobs"]

    ordered_keys = list(parsed_json.keys())
    entropy_dict = {}

    for key in ordered_keys:
        pat = category_token_patterns.get(key)
        if pat is None:
            continue

        idx = find_subsequence(tokens, pat)
        start = idx + len(pat) if idx >= 0 else None
        if start is None:
            entropy_dict[key] = None
            continue

        end = len(tokens)
        next_idx = ordered_keys.index(key) + 1
        while next_idx < len(ordered_keys):
            next_pat = category_token_patterns.get(ordered_keys[next_idx])
            if next_pat is not None:
                ni = find_subsequence(tokens, next_pat)
                if ni >= 0:
                    end = ni
                break
            next_idx += 1

        if end <= start:
            entropy_dict[key] = None
            continue

        segment_token_ids = tokens[start:end]
        segment_text = ''.join(segment_token_ids)
        segment_logits = logprobs[start:end]
        vprint(f"-- Key «{key}»:")
        vprint(f"    All tokens : \t{tokens}")
        vprint(f"   Tokens ids : {segment_token_ids}")
        vprint(f"   Tokens text: {segment_text!r}")
        vprint(f"   Logits     : {segment_logits}")

        segment_logits = logprobs[start:end]
        entropy_dict[key] = calculate_entropy_optimized(segment_logits)

    # for i, key in enumerate(ordered_keys):
    #     pat = category_token_patterns[key]
    #     idx = find_subsequence(tokens, pat)
    #     start = idx + len(pat) if idx >= 0 else None
    #     if start is None:
    #         entropy_dict[key] = None
    #         continue
    #
    #     if i + 1 < len(ordered_keys):
    #         next_pat = category_token_patterns[ordered_keys[i + 1]]
    #         next_idx = find_subsequence(tokens, next_pat)
    #         end = next_idx if next_idx >= 0 else len(tokens)
    #     else:
    #         end = len(tokens)
    #
    #     if end <= start:
    #         entropy_dict[key] = None
    #         continue
    #
    #     segment_token_ids = tokens[start:end]
    #     segment_text = ''.join(segment_token_ids)
    #     segment_logits = logprobs[start:end]
    #     vprint(f"-- Key «{key}»:")
    #     vprint(f"    All tokens : \t{tokens}")
    #     vprint(f"   Tokens ids : {segment_token_ids}")
    #     vprint(f"   Tokens text: {segment_text!r}")
    #     vprint(f"   Logits     : {segment_logits}")
    #
    #     segment = logprobs[start:end]
    #     entropy_dict[key] = calculate_entropy_optimized(segment)

    output = {run: parsed_json, "entropy": entropy_dict}
    vprint("Final output:", flush=True)
    vprint(output)
    out_fp = os.path.join(output_dir, f"{run}.json")
    with open(out_fp, "w") as of:
        json.dump(output, of, indent=2)

with open(skipped_runs_path, "w") as sf:
    for r in skipped_runs:
        sf.write(r + "\n")


sys.stdout.close()

del llm
import gc
gc.collect()
open(FLAG_FILE, "w").close()