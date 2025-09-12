##########################################################################################
# IMPORT
import os
import psutil
import torch
import sys
import argparse
import time
import numpy as np
import re
import json
import csv
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel

parser = argparse.ArgumentParser(description="Process metadata with LLM (HF+PEFT, corrigé)")
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--input_metadata_path", type=str, required=True)
parser.add_argument("--error_file_path", type=str, required=True)
parser.add_argument("--log_file_path", type=str, required=True)
parser.add_argument("--flag_file", type=str, required=True)
parser.add_argument("--initial_n_ctx", type=int, default=3500)
parser.add_argument("--model", type=str, required=True, help="Root folder containing PEFT adapters (cat_* or cat_all)")
parser.add_argument("--base_model_dir", type=str, required=True, help="Base HF causal LM directory")
parser.add_argument("--max_value_tokens", type=int, default=128)
parser.add_argument("--strict_match_training", action="store_true", help="Per-category prompting aligned with training: stop at first quote")
parser.add_argument("--verbose", action="store_true", help="Verbose output")
args = parser.parse_args()

base_path           = args.base_path
input_metadata_path = args.input_metadata_path
error_file_path     = args.error_file_path
log_file_path       = args.log_file_path
FLAG_FILE           = args.flag_file
initial_n_ctx       = args.initial_n_ctx
adapters_root       = args.model
base_model_dir      = args.base_model_dir
max_value_tokens    = args.max_value_tokens
STRICT_MATCH_TRAINING = args.strict_match_training

raw_final_info_path = os.path.join(base_path, "database_metadata_curated.csv")
output_dir          = os.path.join(base_path, "METADATA_LLM_INFERENCE")
skipped_runs_path   = os.path.join(base_path, "skipped_runs.txt")
ambi_cl_path        = os.path.join(base_path, "ambiguous_cell_lines.csv")
error_file_header   = "run_accession\tsummary"

sys.stdout = open(log_file_path, "a", encoding="utf-8")
sys.stderr = sys.stdout

VERBOSE = args.verbose
vprint = print if VERBOSE else (lambda *a, **k: None)

##########################################################################################
# FUNCTIONS

def write_reload_file(fp, header, parts):
    dirpath = os.path.dirname(fp)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    entry = "\t".join(parts)
    if not os.path.exists(fp):
        with open(fp, "w", encoding="utf-8") as f:
            f.write(header + "\n")
    with open(fp, "r+", encoding="utf-8") as f:
        content = f.read()
        if entry not in content:
            f.write(entry + "\n")


def calculate_entropy_optimized(logprobs):
    arr = np.array(logprobs, dtype=np.float64)
    if arr.size == 0:
        return 0.0
    arr -= np.max(arr)
    probs = np.exp(arr)
    s = probs.sum()
    if s <= 0:
        return 0.0
    probs /= s
    return float(-np.sum(probs * np.log(probs + 1e-12)))


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


def clean_val(t: str) -> str:
    t = (t or "").replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    if '"' in t:
        t = t.split('"', 1)[0]
    t = re.sub(r"\s*[,;}\]]\s*$", "", t)
    t = t.strip()
    return t if t else "unknown"


def tok_ids(tokenizer, s):
    return tokenizer(s, add_special_tokens=False, return_tensors=None)["input_ids"]


def prime_tokens(model, seq_ids, past):
    with torch.no_grad():
        out = model(input_ids=seq_ids, past_key_values=past, use_cache=True)
        return out.past_key_values

##########################################################################################
# CATEGORIES & DEFINITIONS

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
# MAIN
process = psutil.Process(os.getpid())

use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count() if use_gpu else 0
if use_gpu and gpu_count > 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))
    vprint(f"Using {gpu_count} GPU(s): {os.environ['CUDA_VISIBLE_DEVICES']}")
else:
    vprint("No GPU detected → using CPU only")

try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    tokenizer.pad_token = '<pad>'

dtype = (
    torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
        torch.float16 if torch.cuda.is_available() else torch.float32
    )
)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_dir,
    torch_dtype=dtype,
    device_map="auto",
)
try:
    base_model.resize_token_embeddings(len(tokenizer))
except Exception:
    pass


def load_all_category_adapters(base_model, peft_root, cats):
    merged_dirs = [d for d in os.listdir(peft_root)
                   if d.startswith("merged_cat_") and os.path.isdir(os.path.join(peft_root, d))]
    if merged_dirs:
        raise RuntimeError(
            "Merged checkpoints in --model."
        )

    peft_model = None
    loaded_names = set()

    shared_path = os.path.join(peft_root, "shared")
    if os.path.isdir(shared_path):
        peft_model = PeftModel.from_pretrained(base_model, shared_path, adapter_name="shared")
        loaded_names.add("shared")

    for c in cats:
        path = os.path.join(peft_root, f"cat_{c}")
        if os.path.isdir(path):
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(base_model, path, adapter_name=f"cat_{c}")
            else:
                peft_model.load_adapter(path, adapter_name=f"cat_{c}")
            loaded_names.add(f"cat_{c}")

    if peft_model is None:
        root_cfg = os.path.join(peft_root, "adapter_config.json")
        if os.path.isfile(root_cfg):
            peft_model = PeftModel.from_pretrained(base_model, peft_root, adapter_name="cat_all")
            loaded_names.add("cat_all")
        else:
            raise RuntimeError(f"No adapter PEFT founded in {peft_root}")

    try:
        setattr(peft_model, "_loaded_adapters", loaded_names)
    except Exception:
        pass

    return peft_model

model = load_all_category_adapters(base_model, adapters_root, categories)
model.eval()

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

vprint("[INFO] Prompt mode:", "STRICT_MATCH_TRAINING" if STRICT_MATCH_TRAINING else "GENERAL_MULTI_CATEGORIES")
vprint("[PEFT] Charged adaptaters:", sorted(list(getattr(model, "_loaded_adapters", set()))))

quote_ids = tok_ids(tokenizer, '"')
quote_id = quote_ids[0] if len(quote_ids) > 0 else None

class StopOnQuote(StoppingCriteria):
    def __init__(self, quote_id):
        self.quote_id = quote_id
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.quote_id is None:
            return False
        return input_ids[0, -1].item() == self.quote_id


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
            {{\n"{key}": """


def build_prompt_multi_category(run, summary, extra_info_str):
    inst_lines = "\n".join(f"- {c}: {definitions[c]}" for c in categories)
    return f"""Run accession: {run}
            Summary: {summary} {extra_info_str}
            
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


skipped_runs = []
total_t0 = time.perf_counter()

with torch.inference_mode():
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

        vprint(f"\n[{idx + 1}/{len(metadata_lines)}] {run}", flush=True)

        device = model.device
        out_map = {}
        entropy_map = {}

        prompt_txt_multi = build_prompt_multi_category(run, summary, extra_info_str)
        ids_prompt = tokenizer(prompt_txt_multi, return_tensors="pt", truncation=True, max_length=initial_n_ctx)
        prompt_input_ids = ids_prompt["input_ids"].to(device)
        prompt_attention_mask = ids_prompt["attention_mask"].to(device)

        past_by_adapter = {}
        loaded = getattr(model, "_loaded_adapters", set())

        for key in categories:
            peft_names = list(getattr(model, "peft_config", {}).keys())
            cand = f"cat_{key}"
            if cand in peft_names:
                name = cand
            elif "cat_all" in peft_names:
                name = "cat_all"
            else:
                raise RuntimeError(f"Aucun adaptateur PEFT disponible pour la catégorie '{key}'")

            if "shared" in loaded:
                try:
                    model.set_adapter(["shared", name])
                except Exception:
                    model.set_adapter(name)
            else:
                model.set_adapter(name)
            active = getattr(model, "active_adapter", None)
            vprint(f"[PEFT] {run} | {key:16s} -> active: {active}")

            if STRICT_MATCH_TRAINING:
                if name not in past_by_adapter:
                    out_prompt = model(input_ids=prompt_input_ids, attention_mask=prompt_attention_mask, use_cache=True)
                    past_by_adapter[name] = out_prompt.past_key_values
                pkv = past_by_adapter[name]

                ks = tok_ids(tokenizer, "{")
                if len(ks) == 0:
                    ks = tok_ids(tokenizer, "{\n")
                if len(ks) == 0:
                    ks = tok_ids(tokenizer, "{ ")
                ks_t = torch.tensor([ks], device=device)
                pkv = prime_tokens(model, ks_t, pkv)

                prefix = f'"{key}": "'
                key_ids = tok_ids(tokenizer, prefix)
                if len(key_ids) == 0:
                    key_ids = tok_ids(tokenizer, f'"{key}": "')
                kid = torch.tensor([key_ids], device=device)
                pkv = prime_tokens(model, kid, pkv)

                stop_criteria = StoppingCriteriaList([StopOnQuote(quote_id)])
                gen = model.generate(
                    input_ids=torch.empty((1, 0), dtype=torch.long, device=device),
                    max_new_tokens=max_value_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=None,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    stopping_criteria=stop_criteria,
                    past_key_values=pkv,
                )

                gen_ids = gen.sequences
                if gen_ids.size(1) > 0 and quote_id is not None and gen_ids[0, -1].item() == quote_id:
                    val_ids = gen_ids[0, :-1]
                else:
                    val_ids = gen_ids[0]

                text_val = tokenizer.decode(val_ids, skip_special_tokens=True).strip()
                text_val = clean_val(text_val)

                lps = []
                seq_tokens = val_ids.tolist()
                for t, scores in enumerate(gen.scores[:len(seq_tokens)]):
                    lp = torch.log_softmax(scores, dim=-1)[0, seq_tokens[t]].item()
                    lps.append(lp)
                ent_val = calculate_entropy_optimized(lps)

                out_map[key] = text_val
                entropy_map[key] = ent_val

            else:
                prompt_txt = build_prompt_multi_category(run, summary, extra_info_str)
                ids = tokenizer(prompt_txt, return_tensors="pt", truncation=True, max_length=initial_n_ctx)
                input_ids = ids["input_ids"].to(device)
                attention_mask = ids["attention_mask"].to(device)

                stop_criteria = None
                gen = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=350,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=None,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    stopping_criteria=stop_criteria,
                )
                new_txt = tokenizer.decode(gen.sequences[0, input_ids.size(1):], skip_special_tokens=True)
                m = re.search(r"\{.*\}", new_txt, flags=re.DOTALL)
                if not m:
                    write_reload_file(error_file_path, error_file_header, [run, summary])
                    skipped_runs.append(run)
                    break
                try:
                    parsed = json.loads(m.group(0))
                except json.JSONDecodeError:
                    write_reload_file(error_file_path, error_file_header, [run, summary])
                    skipped_runs.append(run)
                    break
                out_map = parsed
                entropy_map = {k: 0.0 for k in out_map.keys()}

        payload = {run: out_map, "entropy": entropy_map}
        out_fp = os.path.join(output_dir, f"{run}.json")
        with open(out_fp, "w", encoding="utf-8") as of:
            json.dump(payload, of, indent=2, ensure_ascii=False)
        vprint(f"[OUT] {run} saved → {out_fp}")

with open(skipped_runs_path, "w", encoding="utf-8") as sf:
    for r in skipped_runs:
        sf.write(r + "\n")

sys.stdout.close()
del model
import gc
gc.collect()
open(FLAG_FILE, "w").close()
