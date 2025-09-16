##########################################################################################
# IMPORT
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
parser.add_argument("--verbose", action="store_true", help="Verbose output")
args = parser.parse_args()

base_path = args.base_path
input_metadata_path = args.input_metadata_path
error_file_path = args.error_file_path
log_file_path = args.log_file_path
FLAG_FILE = args.flag_file
initial_n_ctx = args.initial_n_ctx
base_model_path = args.model
max_value_tokens = args.max_value_tokens

raw_final_info_path = os.path.join(base_path, "database_metadata_curated.csv")
output_dir = os.path.join(base_path, "METADATA_LLM_INFERENCE")
skipped_runs_path = os.path.join(base_path, "skipped_runs.txt")
ambi_cl_path = os.path.join(base_path, "ambiguous_cell_lines.csv")
error_file_header = "run_accession\tsummary"

sys.stdout = open(log_file_path, "a", encoding="utf-8")
sys.stderr = sys.stdout
VERBOSE = args.verbose
vprint = print if VERBOSE else (lambda *a, **k: None)

##########################################################################################
#HELPPER FUNCTIONS
def ensure_file_exists(path: str, empty: bool = False):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        mode = "w" if empty else "a"
        with open(path, mode, encoding="utf-8") as _:
            pass


def load_ambiguous_map(fp):
    ensure_file_exists(fp, empty=True)
    result = {}
    try:
        with open(fp, "r", encoding="utf-8", newline="") as f:
            rdr = csv.reader(f)
            for row in rdr:
                if not row:
                    continue
                if row[0].strip().lower() == "run_accession":
                    continue
                if len(row) < 2:
                    continue
                ra = row[0].strip()
                cand = row[1].strip()
                result[ra] = cand
    except Exception as e:
        vprint(f"[WARN] Could not read ambiguous file: {e}")
    return result


def append_skipped_and_error(run: str, summary: str):
    try:
        with open(skipped_runs_path, "a", encoding="utf-8") as sf:
            sf.write(run + "\n")
        need_header = (not os.path.exists(error_file_path)) or (os.path.getsize(error_file_path) == 0)
        with open(error_file_path, "a", encoding="utf-8") as ef:
            if need_header:
                ef.write(error_file_header + "\n")
            ef.write(f"{run}\t{summary}\n")
        vprint(f"[ERROR] Marked run as skipped & logged for retry: {run}")
    except Exception as e:
        vprint(f"[ERROR] Failed to write skip/error logs for {run}: {e}")


##########################################################################################
# MAIN

os.makedirs(output_dir, exist_ok=True)

STRICT_MATCH_TRAINING = True

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


def read_runs(fp):
    rows = []
    with open(fp, "r", encoding="utf-8", newline="") as f:
        sample = f.read(1024)
        f.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
        except csv.Error:
            class _D:
                delimiter = "\t"

            dialect = _D()
        rdr = csv.reader(f, dialect)
        header = next(rdr, None)
        if header is None:
            return rows
        header_lower = [h.strip().lower() for h in header]
        try:
            i_run = header_lower.index("run_accession")
            i_sum = header_lower.index("summary")
        except ValueError:
            if len(header) >= 2:
                i_run, i_sum = 0, 1
            else:
                return rows
        for r in rdr:
            if not r or len(r) <= max(i_run, i_sum):
                continue
            ra = r[i_run].strip()
            su = r[i_sum].strip()
            if ra and su:
                rows.append((ra, su))
    return rows


def clean_val(t):
    t = t.replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    if '"' in t:
        t = t.split('"', 1)[0]
    t = re.sub(r"\s*[,;}\]]\s*$", "", t)
    t = t.strip()
    if t == "":
        t = "unknown"
    return t


def build_prompt_single_category_train_like(run, summary, key, definition):
    return f"""Run accession: {run}
                Summary: {summary}

                Categories and definitions:
                - {key}: {definition}

                For each category below:
                - Extract information from the summary if possible
                - If one value is impossible to extract, even by deducing it, return "unknown"

                BE CAREFUL: Sometimes the information concerns several samples from the same study. It is important to distinguish between them and semantically extract what applies to the current run, so everything must be consistent.
                FOR EACH CATEGORY SEVERAL ANSWERS CAN BE POSSIBLE, CITE THEM ALL WITH A ',' OR ';' SEPARATOR

                Respond strictly in a few words with valid JSON (double quotes around keys and values), no extra keys. ONLY THE CATEGORIES CITED UNDER 'Categories and definitions'

                Here is the output:
                """


try:
    import torch
    use_gpu = torch.cuda.is_available()
except Exception:
    use_gpu = False

if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
    use_gpu = False

n_gpu_layers_val = -1 if use_gpu else 0
try:
    n_threads_val = psutil.cpu_count(logical=True) or os.cpu_count() or 4
except Exception:
    n_threads_val = os.cpu_count() or 4

llm = Llama(
    model_path=base_model_path,
    n_ctx=initial_n_ctx,
    n_gpu_layers=n_gpu_layers_val,
    n_threads=n_threads_val,
    use_mmap=True,
    logits_all=True,
)

vprint(f"[LLM] use_gpu={use_gpu} n_gpu_layers={n_gpu_layers_val} n_threads={n_threads_val}")


runs = read_runs(input_metadata_path)

vprint("[INFO] Mode prompt:", "STRICT_MATCH_TRAINING" if STRICT_MATCH_TRAINING else "GENERAL_MULTI_CATEGORIES")
vprint("[INFO] Model (gguf):", base_model_path)

ambiguous_map = load_ambiguous_map(ambi_cl_path)


def calc_entropy_from_logprobs(lp_list):
    vals = [x for x in lp_list if x is not None]
    if not vals:
        return 0.0
    arr = np.array(vals, dtype=np.float64)
    arr -= np.max(arr)
    probs = np.exp(arr)
    probs /= probs.sum()
    ent = -np.sum(probs * np.log(probs + 1e-12))
    return float(ent)


total_t0 = time.perf_counter()

for run, summary in runs:
    t0 = time.perf_counter()
    out = {}
    entropies = {}
    run_failed = False

    for key in categories:
        try:
            if STRICT_MATCH_TRAINING:
                prompt_key = build_prompt_single_category_train_like(run, summary, key, definitions[key])
                extra_ctx = ambiguous_map.get(run, "")
                if extra_ctx and extra_ctx.strip().lower() != "null":
                    prompt_key = prompt_key + f"\nAdditional context (cell line candidates): {extra_ctx}\n"
                prefix_text = prompt_key + "{\n" + f"\"{key}\": \""
            else:
                inst_lines = "\n".join(f"- {c}: {definitions[c]}" for c in categories)
                prompt_general = f"""Run accession: {run}
                                Summary: {summary}

                                Categories and definitions:
                                {inst_lines}

                                For each category below:
                                - Extract information from the summary if possible
                                - The value can be not applicable ONLY FOR: treatment_time and response (if treatment = no treatment) AND cell_line (if cell_type = primary tissue), RETURN "not applicable" for those categories. CAN'T BE NOT APPLICABLE FOR THE OTHER CATEGORIES.
                                - If one value is impossible to extract, even by deducing it, return "unknown", applicable for all categories

                                BE CAREFUL: Sometimes the information concerns several samples from the same study. It is important to distinguish between them and semantically extract what applies to the current run, so everything must be consistent.
                                FOR EACH CATEGORY SEVERAL ANSWERS CAN BE POSSIBLE, CITE THEM ALL WITH A ',' OR ';' SEPARATOR

                                Respond strictly in a few words with valid JSON (double quotes around keys and values), no extra keys. ONLY THE CATEGORIES CITED UNDER 'Categories and definitions'

                                Here is the output:
                                """
                extra_ctx = ambiguous_map.get(run, "")
                if extra_ctx and extra_ctx.strip().lower() != "null":
                    prompt_general = prompt_general + f"\nAdditional context (cell line candidates): {extra_ctx}\n"
                prefix_text = prompt_general + "{\n" + f"\"{key}\": \""

            vprint(f"[PEFT] {run} | {key:16s} -> actif: (none; llama.cpp)")
            vprint(f"[PROMPT] {run} | {key}\n{prefix_text}")

            cat_t0 = time.perf_counter()

            resp = llm(prefix_text, max_tokens=max_value_tokens, stop=['"'], logprobs=True, echo=False)

            try:
                text = resp["choices"][0]["text"]
                logp = resp["choices"][0].get("logprobs", {}).get("token_logprobs", [])
            except Exception as e:
                vprint(f"[ERROR] {run} | {key}: malformed LLM response: {e}")
                text = ""
                logp = []
                run_failed = True

            if text is None or str(text).strip() == "":
                run_failed = True

            out[key] = clean_val(text or "")
            entropies[key] = calc_entropy_from_logprobs(logp)

            vprint(f"[OUT] {run} | {key}: {out[key]} | H={entropies[key]:.6f}")
            vprint(f"[TIMING][cat] {run} | {key}: {time.perf_counter() - cat_t0:.4f}s")
            vprint("----------------------------------------------------------------------------")

        except Exception as e:
            run_failed = True
            vprint(f"[ERROR] {run} | {key}: {e}")
            vprint("----------------------------------------------------------------------------")
            continue

    try:
        with open(os.path.join(output_dir, f"{run}.json"), "w") as f:
            json.dump({run: out, "entropy": entropies}, f, indent=2)
    except Exception as e:
        vprint(f"[ERROR] Failed to write JSON for {run}: {e}")

    vprint(f"[TIMING] run {run}: {time.perf_counter() - t0:.4f}s")

    if run_failed:
        append_skipped_and_error(run, summary)

vprint("[INFO] Mode prompt:", "STRICT_MATCH_TRAINING" if STRICT_MATCH_TRAINING else "GENERAL_MULTI_CATEGORIES")
vprint(f"[TIMING] total: {time.perf_counter() - total_t0:.4f}s")

open(FLAG_FILE, "w").close()
