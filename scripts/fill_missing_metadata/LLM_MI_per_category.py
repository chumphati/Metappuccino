##########################################################################################
#IMPORT
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
parser.add_argument("--max_value_tokens", type=int, default=24)
parser.add_argument("--verbose", action="store_true", help="Verbose output")
args = parser.parse_args()

base_path = args.base_path
input_metadata_path = args.input_metadata_path
error_file_path = args.error_file_path
log_file_path = args.log_file_path
FLAG_FILE = args.flag_file
initial_n_ctx = args.initial_n_ctx
adapters_root = args.model
base_model_dir = args.base_model_dir
max_value_tokens = args.max_value_tokens

MAX_TOKENS_BY_CAT = {
    "age": 8,
    "sex": 80,
    "is_cancer": 80,
    "library_selection": 80,
    "sequencing_source": 80,
    "biopsy_site": 80,
    "biopsy_type": 8,
    "cell_line": 80,
    "cell_type": 80,
    "organ": 80,
    "disease": 80,
    "treatment": 80,
    "treatment_time": 80,
    "response": 80,
    "ethnicity": 80,
}
default_max = max_value_tokens

raw_final_info_path = os.path.join(base_path, "database_metadata_curated.csv")
output_dir = os.path.join(base_path, "METADATA_LLM_INFERENCE")
skipped_runs_path = os.path.join(base_path, "skipped_runs.txt")
ambi_cl_path = os.path.join(base_path, "ambiguous_cell_lines.csv")
error_file_header = "run_accession\tsummary"

sys.stdout = open(log_file_path, "a", encoding="utf-8")
sys.stderr = sys.stdout

VERBOSE = args.verbose
vprint = print if VERBOSE else (lambda *a, **k: None)

os.makedirs(output_dir, exist_ok=True)

STRICT_MATCH_TRAINING = True

categories = [
    "library_selection", "sequencing_source", "biopsy_site", "biopsy_type",
    "cell_line", "cell_type", "organ", "disease", "treatment",
    "treatment_time", "response", "age", "sex", "ethnicity", "is_cancer"
]

definitions = {
    "library_selection": "based on the given context, determine the library selection fixed category by searching for specific keywords or synonyms that match one of the five strict categories: 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or 'other'. Assign 'polyA' if the context contains any of the following terms OR SIMIILAR MEANING THAT CAN BE INFERRED: 'PolyA', 'poly.A', 'oligo.dT', 'oligodT', 'truseq.mrna', 'truseq.stranded.mrna', 'truseq.standard.mrna', 'smarter.mRNA', 'stran ded.mRNA'. Assign 'inverse rRNA' if the context mentions depletion of ribosomal RNA with any of these terms OR SIMIILAR MEANING THAT CAN BE INFERRED: 'ribominus', 'ribodep', 'ribozero', 'ribo.zero', 'riboerase', 'ribogone', 'ribocop', 'ribo-dep', 'ribo-mi', 'ribo minus', 'depleted ribosom', 'remove ribosom', 'TruSeq.Stranded.Total', 'TruSeq.Total', 'SMARTer.Stranded.Total', 'SMARTer.Total'. Assign 'hybrid selection' if the context refers to hybrid capture or exon selection using any of these terms OR SIMIILAR MEANING THAT CAN BE INFERRED: 'Hybrid.Selection', 'Exon.capture', 'Exome.capture', 'RNA.Exome', 'geoMX'. Assign 'small RNA' if the context refers to small RNA isolation with keywords such as 'TruSeq.Small', 'size.fraction' OR SIMIILAR MEANING THAT CAN BE INFERRED. Assign 'other' if none of the above terms are found. Return only the exact category name: 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or 'other', with no additional text.",
    "sequencing_source": "one of: 'spatial', 'bulk', 'single cell'. Spatial refers to transcriptomics that preserves in-tissue localization of molecules, mapping expression directly onto the tissue architecture. Bulk means sequencing a mixture of cells together, producing an aggregate average signal across the population. Single cell captures RNA from individual cells, yielding per-cell expression profiles and cellular heterogeneity.",
    "biopsy_site": "organ, body part or fluid WHERE TISSUE WAS SAMPLED. same as organ if not cancer. If it is a xenograft mention it here. Must not be related to the disease, just the tissue sampled. If possible DEDUCE IT FROM WHERE THE CELL LINE COMES FROM",
    "biopsy_type": "state 'metastasis' IF CANCER AND METASTASIS MENTIONNED, OR 'blood' if no metastasis and blood related information mentionned, OTHERWISE state 'primary'. DO NOT STATE METASTASIS OR BLOOD IF NOT EXPLICITELY IN THE CONTEXT. CAN ONLY STATE THOSE THREE INFORMATION, YOU SHOULD ALWAYS BE CAPABLE TO DETERMINE ONE OF THE 3 VALUES",
    "cell_line": "exact cell line, ie standardized names of cultured/immortalized laboratory cell populations. Specify the cell line, or state 'Primary tissue' if the sample is from a primary tissue and not a cell line.",
    "cell_type": "The type of cell in the sample (e.g., neuron, fibroblast, CD8 T cell, CD4 T cell, monocyte NK cell, mast cell, melanocyte, dendritic cell, etc...). If not provided, deduce based on context. otherwise, state 'primary tissue'",
    "organ": "organ studied or affected (not where the sample is from, very different from biopsy_site). Must not be related to the disease, just the tissue concerned.",
    "disease": "report associated disease (BE SPECIFIC) or 'healthy' status (be careful to specific vocabulary that could indicate that the sample is healthy, for eg. adjacent is something next to the disease, or normal, etc...)",
    "treatment": "treatment applied treatment applied = may be molecules or drug treatments or surgical operations, or something implanted, or events altering the state of the organism, etc. if not explicitly stated, infer from the context. DON'T STATE the disease, get info just from treatment. Also state a treatment if it's planed to be done, as a pre-treatment step.",
    "treatment_time": "time or phase relative to treatment. if you state a quantitative information state if it it post, pre or on treatment",
    "response": "type of reaction to the treatment, can be: no treatment, unknown, stable, progressive, success",
    "age": "sample donor ageif human. if not possible to infer, state 'unknown'. Can be quantitative (range or exact age) or qualitative (eg: child, teenage, adult, senior, ETC). careful to find an age not just a random number",
    "sex": "sample donor sex if human. if not possible to infer, state 'unknown'",
    "ethnicity": "sample donor ethnicity if human (e.g. caucasian, black, asian). if not possible to infer, state 'unknown'",
    "is_cancer": "return 'True' if the disease is cancer related, 'False' otherwise"
}


def ensure_file_exists(path: str) -> None:
    if not os.path.exists(path):
        with open(path, "a", encoding="utf-8"):
            pass


def ensure_error_with_header(path: str, header_line: str) -> None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        with open(path, "w", encoding="utf-8") as f:
            f.write(header_line + "\n")


def append_error_line(path: str, run: str, summary: str) -> None:
    ensure_error_with_header(path, error_file_header)
    with open(path, "a", encoding="utf-8") as f:
        f.write(f"{run}\t{summary}\n")


def load_ambiguous_candidates(fp: str):
    if not os.path.exists(fp) or os.path.getsize(fp) == 0:
        return {}
    mapping = {}
    with open(fp, "r", encoding="utf-8", newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr, None)
        idx_run, idx_cand = 0, 1
        if header:
            hl = [h.strip().lower() for h in header]
            if "run_accession" in hl:
                idx_run = hl.index("run_accession")
            if "candidates" in hl:
                idx_cand = hl.index("candidates")
        else:
            f.seek(0)
            rdr = csv.reader(f)
        for row in rdr:
            if not row or len(row) <= max(idx_run, idx_cand):
                continue
            mapping[row[idx_run].strip()] = row[idx_cand].strip()
    return mapping


ensure_error_with_header(error_file_path, error_file_header)
ensure_file_exists(skipped_runs_path)
if not os.path.exists(ambi_cl_path):
    with open(ambi_cl_path, "a", encoding="utf-8"):
        pass

ambiguous_map = load_ambiguous_candidates(ambi_cl_path)


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
            if not r or len(r) <= max(i_run, i_sum): continue
            ra = r[i_run].strip()
            su = r[i_sum].strip()
            if ra and su:
                rows.append((ra, su))
    return rows


def load_all_category_adapters(base_model, peft_root, cats):
    peft_model = None
    loaded_names = set()
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
            raise RuntimeError(f"Aucun adapter PEFT trouvé dans {peft_root}")
    try:
        setattr(peft_model, "_loaded_adapters", loaded_names)
    except Exception:
        pass
    return peft_model


class StopOnQuote(StoppingCriteria):
    def __init__(self, quote_id): self.quote_id = quote_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.quote_id is None: return False
        return input_ids[0, -1].item() == self.quote_id


def clean_val(t):
    t = t.replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    if '"' in t: t = t.split('"', 1)[0]
    t = re.sub(r'\s*[,;}\]]\s*$', '', t)
    t = t.strip()
    if t == "": t = "unknown"
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

            Here is the output:
            """


torch.backends.cuda.matmul.allow_tf32 = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '<pad>'})
    tokenizer.pad_token = '<pad>'

dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else (
    torch.float16 if torch.cuda.is_available() else torch.float32)
base_model = AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype=dtype, device_map="auto")
try:
    base_model.resize_token_embeddings(len(tokenizer))
except Exception:
    pass

model = load_all_category_adapters(base_model, adapters_root, categories)
model.eval()

peft_names = list(getattr(model, "peft_config", {}).keys())
vprint("[INFO] Prompt mode:", "STRICT_MATCH_TRAINING" if STRICT_MATCH_TRAINING else "GENERAL_MULTI_CATEGORIES")
vprint("[PEFT] Adaptators", peft_names if peft_names else "(aucun)")
loaded_set = getattr(model, "_loaded_adapters", set())
vprint("[PEFT] Internal marks:", sorted(list(loaded_set)))

quote_ids = tokenizer('"', add_special_tokens=False)["input_ids"]
quote_id = quote_ids[0] if len(quote_ids) > 0 else None

runs = read_runs(input_metadata_path)

total_t0 = time.perf_counter()

for run, summary in runs:
    try:
        t0 = time.perf_counter()
        dev = model.device
        out = {}
        nlls = {}
        ppls = {}
        entropies = {}

        for key in categories:
            cand = f"cat_{key}"
            if cand in peft_names:
                name = cand
            elif "cat_all" in peft_names:
                name = "cat_all"
            else:
                raise RuntimeError(f"No PEFT adaptator PEFT '{key}'")
            model.set_adapter(name)
            active = getattr(model, "active_adapter", None)
            vprint(f"[PEFT] {run} | {key:16s} -> active: {active}")

            if STRICT_MATCH_TRAINING:
                prompt_key = build_prompt_single_category_train_like(run, summary, key, definitions[key])
                if key == "cell_line":
                    cand_val = ambiguous_map.get(run)
                    if cand_val is not None and cand_val.strip() != "" and cand_val.strip().lower() != "null":
                        prompt_key = prompt_key.rstrip("\n") + f"\nContext value: {cand_val}\n"
                prefix_text = prompt_key + "{\n" + f"\"{key}\": \""
            else:
                prompt_general = f"""Run accession: {run}
                Summary: {summary}

                Categories and definitions:
                - library_selection: {definitions['library_selection']}
                - sequencing_source: {definitions['sequencing_source']}
                - biopsy_site: {definitions['biopsy_site']}
                - biopsy_type: {definitions['biopsy_type']}
                - cell_line: {definitions['cell_line']}
                - cell_type: {definitions['cell_type']}
                - organ: {definitions['organ']}
                - disease: {definitions['disease']}
                - treatment: {definitions['treatment']}
                - treatment_time: {definitions['treatment_time']}
                - response: {definitions['response']}
                - age: {definitions['age']}
                - sex: {definitions['sex']}
                - ethnicity: {definitions['ethnicity']}
                - is_cancer: {definitions['is_cancer']}

                For each category below:
                - Extract information from the summary if possible
                - The value can be not applicable ONLY FOR: treatment_time and response (if treatment = no treatment) AND cell_line (if cell_type = primary tissue), RETURN "not applicable" for those categories. CAN'T BE NOT APPLICABLE FOR THE OTHER CATEGORIES.
                - If one value is impossible to extract, even by deducing it, return "unknown", applicable for all categories

                BE CAREFUL: Sometimes the information concerns several samples from the same study. It is important to distinguish between them and semantically extract what applies to the current run, so everything must be consistent.
                FOR EACH CATEGORY SEVERAL ANSWERS CAN BE POSSIBLE, CITE THEM ALL WITH A ',' OR ';' SEPARATOR

                Respond strictly in a few words with valid JSON (double quotes around keys and values), no extra keys. ONLY THE CATEGORIES CITED UNDER 'Categories and definitions'

                Here is the output:
                """
                if key == "cell_line":
                    cand_val = ambiguous_map.get(run)
                    if cand_val is not None and cand_val.strip() != "" and cand_val.strip().lower() != "null":
                        prompt_general = prompt_general.rstrip("\n") + f"\nContext value: {cand_val}\n"
                prefix_text = prompt_general + "{\n" + f"\"{key}\": \""

            vprint(f"[PROMPT] {run} | {key}\n{prefix_text}")

            cat_t0 = time.perf_counter()

            ids = tokenizer(prefix_text, return_tensors="pt", truncation=True, max_length=initial_n_ctx)
            input_ids = ids["input_ids"].to(dev)
            attention_mask = ids["attention_mask"].to(dev)

            stop_criteria = StoppingCriteriaList([StopOnQuote(quote_id)])

            max_new = MAX_TOKENS_BY_CAT.get(key, default_max)
            with torch.no_grad():
                gen = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=None,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    stopping_criteria=stop_criteria
                )

            new_tokens_full = gen.sequences[0, input_ids.size(1):]
            trim_last = bool(
                quote_id is not None and new_tokens_full.size(0) > 0 and new_tokens_full[-1].item() == quote_id)
            gen_tokens = new_tokens_full[:-1] if trim_last else new_tokens_full
            scores_eff = gen.scores[:-1] if trim_last else gen.scores

            if len(gen_tokens) == 0:
                nll_val = float("inf")
                ppl_val = float("inf")
            else:
                nll_steps = []
                for t, logits in enumerate(scores_eff[:len(gen_tokens)]):
                    log_probs = torch.log_softmax(logits, dim=-1)[0]
                    y_t = gen_tokens[t].item()
                    nll_steps.append(float(-log_probs[y_t]))
                nll_val = float(np.mean(nll_steps))
                ppl_val = float(np.exp(nll_val))

            text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            out[key] = clean_val(text)
            nlls[key] = nll_val
            ppls[key] = ppl_val

            vprint(f"[OUT] {run} | {key}: {out[key]} | NLL={nll_val:.4f} | PPL={ppl_val:.3f}")
            vprint(f"[TIMING][cat] {run} | {key}: {time.perf_counter() - cat_t0:.4f}s")
            vprint("----------------------------------------------------------------------------")

        with open(os.path.join(output_dir, f"{run}.json"), "w") as f:
            json.dump({run: out, "nll": nlls, "ppl": ppls}, f, indent=2)
        vprint(f"[TIMING] run {run}: {time.perf_counter() - t0:.4f}s")

    except Exception as e:
        vprint(f"[ERROR] {run} | {repr(e)}")
        with open(skipped_runs_path, "a", encoding="utf-8") as sf:
            sf.write(run + "\n")
        append_error_line(error_file_path, run, summary)
        continue

vprint(f"[INFO] Prompt mode:", "STRICT_MATCH_TRAINING" if STRICT_MATCH_TRAINING else "GENERAL_MULTI_CATEGORIES")
vprint(f"[TIMING] total: {time.perf_counter() - total_t0:.4f}s")

open(FLAG_FILE, "w").close()