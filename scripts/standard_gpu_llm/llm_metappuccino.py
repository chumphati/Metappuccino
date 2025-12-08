import os
import time
import csv
import re
import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
from peft import PeftModel
from contextlib import contextmanager
import tempfile

base_model_dir = "MetappuccinoLLModel/Mistral-7B-Instruct-v0.3"
adapters_root = "MetappuccinoLLModel"
input_metadata_path = "results_dir/ORIGINAL_METADATA/metadata_sra_summarized.csv"
output_dir = "results_dir/COMPLETED_INFERENCE/METADATA_LLM_INFERENCE"
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

initial_n_ctx = 3500

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
default_max = 24

def read_runs(fp):
    rows = []
    with open(fp, "r", encoding="utf-8", newline="") as f:
        sample = f.read(1024)
        f.seek(0)
        default_delim = "," if fp.lower().endswith(".csv") else "\t"
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
        except csv.Error:
            class _D:
                delimiter = default_delim
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


def load_all_category_adapters(base_model, peft_root, cats, dtype):
    from collections import Counter
    peft_model = None
    loaded_names = set()
    for c in cats:
        path = os.path.join(peft_root, f"cat_{c}")
        if os.path.isdir(path):
            if peft_model is None:
                peft_model = PeftModel.from_pretrained(
                    base_model, path, adapter_name=f"cat_{c}", torch_dtype=dtype
                )
            else:
                peft_model.load_adapter(path, adapter_name=f"cat_{c}", torch_dtype=dtype)
            loaded_names.add(f"cat_{c}")
    if peft_model is None:
        root_cfg = os.path.join(peft_root, "adapter_config.json")
        if os.path.isfile(root_cfg):
            peft_model = PeftModel.from_pretrained(
                base_model, peft_root, adapter_name="cat_all", torch_dtype=dtype
            )
            loaded_names.add("cat_all")
        else:
            raise RuntimeError(f"Aucun adapter PEFT trouvé dans {peft_root}")
    for n, p in peft_model.named_parameters():
        if "lora_" in n and p.dtype != dtype:
            p.data = p.data.to(dtype)
    try:
        setattr(peft_model, "_loaded_adapters", loaded_names)
    except Exception:
        pass
    from collections import Counter
    print("[DTYPE LoRA]", Counter(
        p.dtype for n, p in peft_model.named_parameters() if "lora_" in n
    ))
    return peft_model


class StopOnQuote(StoppingCriteria):
    def __init__(self, quote_id):
        self.quote_id = quote_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.quote_id is None:
            return False
        return input_ids[0, -1].item() == self.quote_id


def clean_val(t):
    t = t.replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t)
    if '"' in t:
        t = t.split('"', 1)[0]
    t = re.sub(r'\s*[,;}\]]\s*$', '', t)
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

model = load_all_category_adapters(base_model, adapters_root, categories, dtype)
model.eval()

peft_names = list(getattr(model, "peft_config", {}).keys())
print("[INFO] Mode prompt       :", "STRICT_MATCH_TRAINING" if STRICT_MATCH_TRAINING else "GENERAL_MULTI_CATEGORIES")
print("[PEFT] Adaptateurs dispo :", peft_names if peft_names else "(aucun)")
loaded_set = getattr(model, "_loaded_adapters", set())
print("[PEFT] Marqueurs internes:", sorted(list(loaded_set)))

quote_ids = tokenizer('"', add_special_tokens=False)["input_ids"]
quote_id = quote_ids[0] if len(quote_ids) > 0 else None

runs_all = read_runs(input_metadata_path)

completed = {
    fn[:-5] for fn in os.listdir(output_dir) if fn.endswith(".json")
}

SHARD_ID = int(os.environ.get("METAPP_SHARD_ID", "0"))
NUM_SHARDS = int(os.environ.get("METAPP_NUM_SHARDS", "1"))
max_seconds_env = os.environ.get("METAPP_MAX_SECONDS", "").strip()
MAX_SECONDS = int(max_seconds_env) if max_seconds_env else None

print(f"[CFG] Shard {SHARD_ID+1}/{NUM_SHARDS}, MAX_SECONDS={MAX_SECONDS}")

runs = [
    r for i, r in enumerate(runs_all)
    if (i % NUM_SHARDS) == SHARD_ID and r[0] not in completed
]

print(f"[RESUME] {len(completed)} runs déjà présents, {len(runs)} à traiter dans CE shard.")


def load_existing_result(path):
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data
    except Exception:
        return None


@contextmanager
def atomic_write_json(path):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=d)
    os.close(fd)
    try:
        with open(tmp, "w") as f:
            yield f
        os.replace(tmp, path)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


total_t0 = time.perf_counter()

for idx, (run, summary) in enumerate(runs):
    if MAX_SECONDS is not None:
        elapsed = time.perf_counter() - total_t0
        if elapsed > MAX_SECONDS:
            print(f"[GUARD] Shard {SHARD_ID+1}/{NUM_SHARDS} : temps max atteint ({elapsed:.1f}s > {MAX_SECONDS}s), arrêt propre.")
            break

    t0 = time.perf_counter()
    json_path = os.path.join(output_dir, f"{run}.json")

    existing = load_existing_result(json_path)
    if existing is not None and run in existing and isinstance(existing[run], dict):
        out = dict(existing[run])
        entropies = dict(existing.get("entropy", {}))
        print(f"[RESUME] {run}: reprise, catégories déjà présentes: {sorted(out.keys())}")
    else:
        out, entropies = {}, {}

    if all(k in out and str(out[k]).strip() != "" for k in categories):
        print(f"[SKIP] {run}: déjà complet → on saute.")
        continue

    dev = model.device
    peft_names = list(getattr(model, "peft_config", {}).keys())

    for key in categories:
        if key in out and str(out[key]).strip() != "":
            print(f"[SKIP] {run} | {key}: déjà présent: {out[key]}")
            continue

        cand = f"cat_{key}"
        if cand in peft_names:
            name = cand
        elif "cat_all" in peft_names:
            name = "cat_all"
        else:
            raise RuntimeError(f"Aucun adaptateur PEFT disponible pour la catégorie '{key}'")
        model.set_adapter(name)
        active = getattr(model, "active_adapter", None)
        print(f"[PEFT] {run} | {key:16s} -> actif: {active}")

        if STRICT_MATCH_TRAINING:
            prompt_key = build_prompt_single_category_train_like(run, summary, key, definitions[key])
            prefix_text = prompt_key + "{\n" + f"\"{key}\": \""
        else:
            prompt_key = build_prompt_single_category_train_like(run, summary, key, definitions[key])
            prefix_text = prompt_key + "{\n" + f"\"{key}\": \""

        print(f"[PROMPT] {run} | {key}\n{prefix_text}")

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

        if len(gen.scores) > 0:
            scores_eff = gen.scores[:-1] if trim_last else gen.scores
            ents = []
            for s in scores_eff:
                e = torch.distributions.Categorical(logits=s).entropy()[0]
                ents.append(e.item())
            ent_val = float(np.mean(ents)) if len(ents) > 0 else 0.0
        else:
            ent_val = 0.0

        text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        out[key] = clean_val(text)
        entropies[key] = ent_val

        print(f"[OUT] {run} | {key}: {out[key]} | H={entropies[key]:.6f}")
        print(f"[TIMING][cat] {run} | {key}: {time.perf_counter() - cat_t0:.4f}s")
        print("----------------------------------------------------------------------------")

    data_to_save = {run: out, "entropy": entropies}
    with atomic_write_json(json_path) as f:
        json.dump(data_to_save, f, indent=2)

    print(f"[TIMING] run {run}: {time.perf_counter() - t0:.4f}s")

print(f"[INFO] Mode prompt       :", "STRICT_MATCH_TRAINING" if STRICT_MATCH_TRAINING else "GENERAL_MULTI_CATEGORIES")
print(f"[TIMING] total: {time.perf_counter() - total_t0:.4f}s")
