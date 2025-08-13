##########################################################################################
#IMPORT
import os
import psutil
import torch
import sys
import argparse
import math
import numpy as np
import re
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

parser = argparse.ArgumentParser(description="Process metadata with LLM")
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--input_metadata_path", type=str, required=True)
parser.add_argument("--error_file_path", type=str, required=True)
parser.add_argument("--log_file_path", type=str, required=True)
parser.add_argument("--flag_file", type=str, required=True)
parser.add_argument("--initial_n_ctx", type=int, default=3500)
parser.add_argument("--model", type=str, required=True)
args = parser.parse_args()

base_path = args.base_path
input_metadata_path = args.input_metadata_path
error_file_path = args.error_file_path
log_file_path = args.log_file_path
FLAG_FILE = args.flag_file
initial_n_ctx = args.initial_n_ctx
model = args.model

raw_final_info_path = os.path.join(base_path, "database_metadata_curated.csv")
output_dir = os.path.join(base_path, "METADATA_LLM_INFERENCE")
skipped_runs_path = os.path.join(base_path, "skipped_runs.txt")
model_peft_dir = os.path.join(base_path, model)
model_base_dir = os.path.join(base_path, "Mistral-7B-Instruct-v0.3")
error_file_header = "run_accession\tsummary"

sys.stdout = open(log_file_path, "a")
sys.stderr = sys.stdout


##########################################################################################
#FUNCTIONS

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
    arr = np.array(logprobs, dtype=np.float64)
    if arr.size == 0:
        return None
    arr -= np.max(arr)
    probs = np.exp(arr)
    s = probs.sum()
    if s <= 0:
        return None
    probs /= s
    return float(-np.sum(probs * np.log(probs + 1e-10)))


def tok_ids(s):
    return tokenizer(s, add_special_tokens=False, return_tensors=None)["input_ids"]


def greedy_step(model, input_ids, attention_mask, past, temperature=0.0):
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        if temperature and temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            lp = torch.log(probs.gather(-1, next_id))
            return next_id, float(lp.item()), out.past_key_values
        else:
            probs = torch.log_softmax(logits, dim=-1)
            next_id = torch.argmax(probs, dim=-1, keepdim=True)
            lp = probs.gather(-1, next_id)
            return next_id, float(lp.item()), out.past_key_values


def prime_tokens(model, seq_ids, past):
    with torch.no_grad():
        out = model(input_ids=seq_ids, past_key_values=past, use_cache=True)
        return out.past_key_values


##########################################################################################
#MAIN
process = psutil.Process(os.getpid())

use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count() if use_gpu else 0

if use_gpu and gpu_count > 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))
    print(f"Using {gpu_count} GPU(s): {os.environ['CUDA_VISIBLE_DEVICES']}")
else:
    print("No GPU detected → using CPU only")

tokenizer = AutoTokenizer.from_pretrained(model_base_dir)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    tokenizer.pad_token = '<pad>'
model_base = AutoModelForCausalLM.from_pretrained(model_base_dir,
                                                  torch_dtype=torch.float16 if use_gpu else torch.float32,
                                                  device_map="auto")

categories = [
    "library_selection", "sequencing_source", "biopsy_site", "biopsy_type",
    "cell_line", "cell_type", "organ", "disease", "treatment",
    "treatment_time", "response", "age", "sex", "ethnicity", "localization", "is_cancer"
]

def load_all_category_adapters(base_model, peft_root, cats):
    peft_model = None
    for c in cats:
        path = os.path.join(peft_root, f"cat_{c}")
        if os.path.isdir(path):
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(base_model, path, adapter_name=f"cat_{c}")
            else:
                peft_model.load_adapter(path, adapter_name=f"cat_{c}")
            print(f"loaded adapter: cat_{c}")
    if peft_model is None:
        peft_model = PeftModel.from_pretrained(base_model, peft_root)
        print(f"loaded single adapter from: {peft_root}")
    return peft_model

model = load_all_category_adapters(model_base, model_peft_dir, categories)
model.eval()

with open(input_metadata_path) as mf:
    header = mf.readline()
    metadata_lines = mf.readlines()

with open(raw_final_info_path) as rf:
    raw = rf.readlines()
    raw_headers = raw[0].strip().split("\t")
    raw_data = {r.split("\t")[0]: r.strip().split("\t") for r in raw[1:]}
    print(raw_data)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

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

quote_ids = tok_ids('"')
quote_id = quote_ids[0] if len(quote_ids) > 0 else None
comma_ids = tok_ids('", ')
if len(comma_ids) == 0:
    comma_ids = tok_ids('",')
brace_close_ids = tok_ids('"}')
if len(brace_close_ids) == 0:
    brace_close_ids = tok_ids('" }')
max_value_tokens = 64

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
    print(run)

    if run not in raw_data:
        skipped_runs.append(run)
        continue

    na_columns = categories.copy()
    print(na_columns)

    raw_vals = raw_data[run]
    extra_info = []
    for col, val in zip(raw_headers, raw_vals):
        if val and val != "":
            extra_info.append(f"- {col}: {val}")

    extra_metadata_block = ""
    if extra_info:
        extra_metadata_block = (
                "\nWARNING: The complementary following information are information on the sample that you have to consider to be true. Use them to infer properly from the context the other categories:\n"
                + "\n".join(extra_info)
        )

    print(f"\n[{idx + 1}/{len(metadata_lines)}] {run}", flush=True)

    inst_lines = "\n".join(f"- {c}: {definitions[c]}" for c in na_columns)
    fmt_keys = ", ".join(f'"{c}": "<value>"' for c in na_columns)

    prompt = f"""Run accession: {run}
            Summary: {summary}

            Categories and definitions:
            {inst_lines}

            For each category below:
            - Infer from the summary if possible
            - The value can be not applicable ONLY FOR: treatment_time and response (if treatment = no treatment) AND cell_line (if cell_type = primary tissue), RETURN "not applicable" for those categories. CAN'T BE NOT APPLICABLE FOR THE OTHER CATEGORIES.
            - If one value is impossible to infer, return "unknown", applicable for all categories ALWAYS BETTER THAN FALSE ANSWER ESPECIALLY FOR SPECIFIC DONOR INFORMATION (AGE, SEX, etc)
            {extra_metadata_block} 

            BE CAREFUL: Sometimes the information concerns several samples from the same study. It is important to distinguish between them and semantically extract what applies to the current run, so everything must be consistent.
            FOR EACH CATEGORY SEVERAL ANSWERS CAN BE POSSIBLE, CITE THEM ALL WITH A ',' OR ';' SEPARATOR

            Respond strictly in a few words with valid JSON (double quotes around keys and values), no extra keys. ONLY THE CATEGORIES CITED UNDER 'Categories and definitions'

            Here is the output:
            """

    print("PROMPT:", flush=True)
    print(prompt, flush=True)

    device = model.device
    with torch.no_grad():
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=initial_n_ctx).to(device)
        input_ids = ids["input_ids"]
        attention_mask = ids["attention_mask"]
        out0 = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past = out0.past_key_values

        generated_map = {}
        entropy_dict = {}

        ks = tok_ids("{")
        if len(ks) == 0:
            ks = tok_ids("{\n")
        if len(ks) == 0:
            ks = tok_ids("{ ")
        ks_t = torch.tensor([ks], device=device)
        input_ids = torch.cat([input_ids, ks_t], dim=1)
        attention_mask = torch.ones_like(input_ids, device=device)
        past = prime_tokens(model, ks_t, past)

        for ci, key in enumerate(categories):
            model.set_adapter(f"cat_{key}")

            prefix = f'"{key}": "'
            key_ids = tok_ids(prefix)
            if len(key_ids) == 0:
                key_ids = tok_ids(f'"{key}": "')
            kid = torch.tensor([key_ids], device=device)
            input_ids = torch.cat([input_ids, kid], dim=1)
            attention_mask = torch.ones_like(input_ids, device=device)
            past = prime_tokens(model, kid, past)

            val_tokens = []
            val_logprobs = []
            t = 0
            while t < max_value_tokens:
                next_id, lp, past = greedy_step(model, input_ids[:, -1:], attention_mask[:, -1:], past, temperature=0.0)
                input_ids = torch.cat([input_ids, next_id], dim=1)
                attention_mask = torch.ones_like(input_ids, device=device)
                tok = next_id.item()
                if quote_id is not None and tok == quote_id:
                    break
                val_tokens.append(tok)
                val_logprobs.append(lp)
                t += 1

            text_val = tokenizer.decode(val_tokens, skip_special_tokens=True).strip()
            generated_map[key] = text_val
            entropy_dict[key] = calculate_entropy_optimized(val_logprobs)

            if ci < len(categories) - 1:
                cid = torch.tensor([comma_ids], device=device)
                input_ids = torch.cat([input_ids, cid], dim=1)
                attention_mask = torch.ones_like(input_ids, device=device)
                past = prime_tokens(model, cid, past)
            else:
                bid = torch.tensor([brace_close_ids], device=device)
                input_ids = torch.cat([input_ids, bid], dim=1)
                attention_mask = torch.ones_like(input_ids, device=device)
                past = prime_tokens(model, bid, past)

    parsed_json = generated_map
    print("Json good format: ", parsed_json)

    output = {run: parsed_json, "entropy": entropy_dict}
    print("Final output:", flush=True)
    print(output)
    out_fp = os.path.join(output_dir, f"{run}.json")
    with open(out_fp, "w") as of:
        json.dump(output, of, indent=2)

with open(skipped_runs_path, "w") as sf:
    for r in skipped_runs:
        sf.write(r + "\n")

sys.stdout.close()
import gc

gc.collect()
open(FLAG_FILE, "w").close()
