##########################################################################################
#IMPORT
import random
import math
import torch
import re
import json
import os
import optuna
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, StoppingCriteria, AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorWithPadding, TrainerCallback, TrainerControl, TrainerState, EvalPrediction, StoppingCriteriaList, \
    MaxLengthCriteria
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import argparse
from peft import PeftModel
from torch.utils.tensorboard import SummaryWriter
from transformers import EarlyStoppingCallback, TrainerCallback
import torch.nn.functional as F
from sklearn.model_selection import KFold, train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch.nn import CrossEntropyLoss
from collections import deque
import copy

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
torch.backends.cuda.matmul.allow_tf32 = True

##########################################################################################
#SEEDS
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

##########################################################################################
#PATHS
parser = argparse.ArgumentParser(description="Fine-tune model")
parser.add_argument("--base_path", type=str, required=True, help="Base path to Metappuccino")
parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds")
args = parser.parse_args()

base_path = args.base_path
n_splits = args.n_splits

prompt_train_file = os.path.join(base_path, "finetune_data_train_corrected.csv")
prompt_val_file = os.path.join(base_path, "finetune_data_val_corrected.csv")
prompt_test_file = os.path.join(base_path, "finetune_data_test_corrected.csv")
prompt_test_oov_file = os.path.join(base_path, "finetune_data_test_oov_corrected.csv")

train_model = os.path.join(base_path, "mistral7B_train")
output_model = os.path.join(base_path, "mistral7B_fine_tuned")
merged_model_path = os.path.join(base_path, "mistral7B_full_finetuned")
model_name = os.path.join(base_path, "Mistral-7B-Instruct-v0.3")
tensorboard_log_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/tensorboard"

#parameters for semantic matching
sem_model_name = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
model_sem = SentenceTransformer(sem_model_name)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

##########################################################################################
#MODEL
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '<pad>'})
tokenizer.pad_token = '<pad>'
print(f"debug/tokenizer_is_fast={getattr(tokenizer, 'is_fast', False)}", flush=True)

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if use_bf16 else torch.float16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Load model", flush=True)
model_base = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    low_cpu_mem_usage=True
)
model_base.resize_token_embeddings(len(tokenizer))
model_base.config.use_cache = False
model_base.gradient_checkpointing_enable()
model_base.to(device)

print("Config LoRA", flush=True)

##########################################################################################
#FUNCTIONS

IS_TRAINING = False
LABEL_DROPOUT_PROB = 0.0

def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(s).lower())).strip()

CLOSED_SETS = {
    "library_selection": {"polya", "inverse rrna", "hybrid selection", "small rna", "other"},
    "sequencing_source": {"spatial", "bulk", "single cell"},
    "biopsy_type": {"metastasis", "blood", "primary"},
    "sex": {"male", "female", "unknown"},
    "is_cancer": {"True", "False"},
}
OPEN_CATS = ["cell_line","disease","treatment","organ","cell_type","localization","ethnicity","biopsy_site"]
ALIASES = {
    "library_selection": {
    "poly a":"polya","poly-a":"polya","poly a+":"polya",
    "rrna depletion":"inverse rrna","ribominus":"inverse rrna","ribo minus":"inverse rrna",
    "capture hybrid":"hybrid selection","hybrid capture":"hybrid selection",
    },
    "sex": {"m":"male","man":"male","male":"male","f":"female","woman":"female","M":"male","F":"female"},
    "biopsy_type": {"primary tumor":"primary","metastatic":"metastasis","primary tissue":"primary"},
    "sequencing_source": {
        "bulk": "bulk", "bulk rna": "bulk", "bulk rna seq": "bulk",
        "bulk transcriptome": "bulk", "bulk transcriptom": "bulk",
        "single cell": "single cell", "single-cell": "single cell",
        "sc rna": "single cell", "scrna": "single cell",
        "single cell sequencing": "single cell", "single cell rna": "single cell",
        "spatial": "spatial", "spatial transcriptomics": "spatial",
        "spatial transcriptome": "spatial", "spatial transcriptom": "spatial"
      },
      "is_cancer": {
        "true":"True","false":"False",
        "cancer":"True","oncologic":"True","tumor":"True","tumour":"True",
        "non cancer":"False","non-cancer":"False","non oncologic":"False",
        "benign":"False","healthy":"False","control":"False"
      }
}

def _canonicalize_closed_value(cat: str, raw_val: str):
    v_raw = raw_val or ""
    v_norm = _normalize_text(v_raw)
    v_norm = v_norm.replace("singlecell", "single cell")
    v_alias = ALIASES.get(cat, {}).get(v_norm, v_norm)
    allowed = CLOSED_SETS.get(cat, set())
    # normalized allowed lookup to accept case-insensitive and substring matches
    allowed_norm_map = { _normalize_text(a): a for a in allowed }

    def _return_canon(x):
        if cat == "is_cancer":
            # ensure canonical capitalization
            return {"true":"True","false":"False","True":"True","False":"False"}.get(x, x)
        return x

    def _canon_candidate(x):
        # accept direct allowed (case-sensitive)
        if x in allowed:
            return _return_canon(x)
        # accept case-insensitive against allowed
        x_norm = _normalize_text(x)
        if x_norm in allowed_norm_map:
            return _return_canon(allowed_norm_map[x_norm])
        return None

    # direct checks on alias and normalized
    c = _canon_candidate(v_alias)
    if c is not None:
        return c

    # try heads / simplified tokens (keep original intent)
    parts = v_norm.split()
    heads = []
    if parts: heads.append(parts[0])
    if len(parts) >= 2: heads.append(" ".join(parts[:2]))
    drops = {"sequencing","rna","seq","kit","library","transcriptom","transcriptome","study","cells","cell"}
    head_clean = " ".join([w for w in parts if w not in drops][:2]).strip()
    if head_clean: heads.append(head_clean)

    for h in heads:
        h_alias = ALIASES.get(cat, {}).get(h, h)
        c = _canon_candidate(h_alias)
        if c is not None:
            return c

    # accept substring matches of any allowed token (prefer longest)
    for key_norm, a in sorted(allowed_norm_map.items(), key=lambda kv: -len(kv[0])):
        if key_norm and key_norm in v_norm:
            return _return_canon(a)

    return None

def remap_closed_sets(d):
    print("debug/remap_closed_sets: entering", flush=True)
    for k, allowed in CLOSED_SETS.items():
        if k in d:
            v_raw = d[k]
            v = _normalize_text(v_raw)
            v = v.replace("singlecell","single cell")
            v = ALIASES.get(k, {}).get(v, v)
            canon = _canonicalize_closed_value(k, v)
            if canon is not None:
                d[k] = canon
                print(f"debug/remap_closed_sets: '{k}' canonicalized → '{canon}'", flush=True)
                continue

            if v not in allowed and _normalize_text(v) not in {_normalize_text(a) for a in allowed}:
                if k == "library_selection":
                    d[k] = "other"
                elif k == "is_cancer":
                    d[k] = "unknown"
                else:
                    d[k] = "unknown"
                print(f"debug/remap_closed_sets: '{k}' value '{v}' not in allowed → mapped to '{d[k]}'", flush=True)
            else:
                for a in allowed:
                    if v == a or _normalize_text(v) == _normalize_text(a):
                        d[k] = {"true":"True","false":"False","True":"True","False":"False"}.get(a, a)
                        print(f"debug/remap_closed_sets: '{k}' normalized passthrough → '{d[k]}'", flush=True)
                        break
    return d

def find_subseq(seq, subseq):
    n, m = len(seq), len(subseq)
    if m == 0 or m > n:
        return -1
    for i in range(n - m + 1):
        if seq[i:i+m] == subseq:
            return i
    return -1

def make_cat_masks_offsets(output_text, prompt_ids, categories, tokenizer, max_len=2048, mark_all_occurrences=True, output_enc=None):
    masks = np.zeros((len(categories), max_len), dtype=np.float32)
    if output_enc is None:
        print("debug/make_cat_masks_offsets: WARNING no output_enc provided; falling back to independent encoding", flush=True)
        available_len = max(1, max_len - len(prompt_ids))
        enc = tokenizer(
            output_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=available_len
        )
    else:
        enc = output_enc
        if "offset_mapping" not in enc or enc["offset_mapping"] is None:
            print("debug/make_cat_masks_offsets: missing offset_mapping -> returning zero mask", flush=True)
            return masks

    offsets = enc["offset_mapping"]
    eff_out_len = len(enc["input_ids"])
    print(f"debug/make_cat_masks_offsets: effective_output_token_length={eff_out_len}", flush=True)

    def char_span_to_token_span(char_start, char_end):
        t_start, t_end = None, None
        for ti, (s, e) in enumerate(offsets):
            if e <= char_start:
                continue
            if s >= char_end:
                break
            if t_start is None:
                t_start = ti
            t_end = ti + 1
        return t_start, t_end

    for j, cat in enumerate(categories):
        prefix = f'"{cat}": "'
        k = output_text.find(prefix)
        if k == -1:
            continue
        start_val = k + len(prefix)
        i = start_val
        end_val = None
        while i < len(output_text):
            if output_text[i] == '"' and output_text[i-1] != '\\':
                end_val = i
                break
            i += 1
        if end_val is None or end_val <= start_val:
            continue

        t_start, t_end = char_span_to_token_span(start_val, end_val)
        if t_start is None or t_end is None:
            continue

        s = len(prompt_ids) + min(t_start, eff_out_len)
        e = len(prompt_ids) + min(t_end, eff_out_len)
        if s < max_len:
            e = min(e, max_len)
            if e > s:
                masks[j, s:e] = 1.0

    if random.random() < 0.05:
        nonzeros = {categories[idx]: int(masks[idx].sum()) for idx in range(len(categories))}
        print(f"debug/masks_per_sample_nonzero_tokens={nonzeros}", flush=True)

    return masks

def apply_label_dropout(output_text):
    if not IS_TRAINING:
        return output_text
    try:
        obj = json.loads(output_text)
    except Exception:
        return output_text
    for c in OPEN_CATS:
        if c in obj:
            v = str(obj[c]).strip().lower()
            if v and v not in ["unknown","not applicable"] and random.random() < LABEL_DROPOUT_PROB:
                obj[c] = "unknown"
    return json.dumps(obj, ensure_ascii=False)

def tokenize_function(example):
    prompt = example["prompt"].strip()
    output = example["output"].strip()
    output = apply_label_dropout(output)
    max_length = 2048
    max_out_len = 350
    out_enc = tokenizer(output, truncation=True, max_length=max_out_len, add_special_tokens=False, return_offsets_mapping=True)
    output_ids = out_enc["input_ids"]

    max_prompt_len = max_length - len(output_ids)
    if max_prompt_len <= 0:
        max_prompt_len = 1
        output_ids = output_ids[: max_length - 1]
        # also trim out_enc consistently
        out_enc["input_ids"] = output_ids
        if "offset_mapping" in out_enc and out_enc["offset_mapping"] is not None:
            out_enc["offset_mapping"] = out_enc["offset_mapping"][:len(output_ids)]

    prompt_enc = tokenizer(prompt, truncation=True, max_length=max_prompt_len, add_special_tokens=False)
    prompt_ids = prompt_enc["input_ids"]

    input_ids = prompt_ids + output_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + output_ids

    padding_length = max_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    attention_mask += [0] * padding_length
    labels += [-100] * padding_length

    cat_masks = make_cat_masks_offsets(
        output_text=output,
        prompt_ids=prompt_ids,
        categories=categories,
        tokenizer=tokenizer,
        max_len=max_length,
        mark_all_occurrences=False,
        output_enc=out_enc
    )
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "cat_masks": cat_masks.tolist()}

def parse_pred_block(raw_pred):
    try:
        return json.loads(raw_pred)
    except Exception:
        pass
    start = raw_pred.find('{')
    if start == -1:
        print("Invalid JSON format in prediction:", raw_pred)
        return {}
    depth = 0
    i = start
    in_str = False
    escape = False
    while i < len(raw_pred):
        ch = raw_pred[i]
        if in_str:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw_pred[start:i+1])
                    except Exception:
                        break
        i += 1
    print("Invalid JSON format in prediction:", raw_pred)
    return {}

normal_accuracy_categories = {
    'is_cancer', 'biopsy_type', 'sequencing_source', 'sex'
}

def normalize(x):
    return re.sub(r'[-_]', ' ', str(x)).strip().lower()

def norm_val(x):
    if isinstance(x, list):
        x = x[0] if x else ""
    return str(x).strip()

def _parse_duration_days(s: str):
    s = str(s).lower().strip()
    m = re.search(r'(\d+(?:\.\d+)?)\s*([a-z]+)', s)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2)
    unit = {
        'd':'day', 'day':'day', 'days':'day',
        'w':'week', 'wk':'week', 'wks':'week', 'week':'week', 'weeks':'week',
        'mo':'month','mos':'month','mon':'month','month':'month','months':'month',
        'y':'year','yr':'year','yrs':'year','year':'year','years':'year',
        'h':'hour','hr':'hour','hrs':'hour','hour':'hour','hours':'hour'
    }.get(unit, unit)
    mult = {'hour': 1/24.0, 'day': 1.0, 'week': 7.0, 'month': 30.0, 'year': 365.0}.get(unit)
    if mult is None:
        return None
    return val * mult

THRESH = {
    "age": 0.30,
    "biopsy_site": 0.42,
    "cell_type": 0.38,
    "disease": 0.36,
    "ethnicity": 0.38,
    "localization": 0.38,
    "organ": 0.42,
    "treatment": 0.40,
    "response": 0.34,
    "treatment_time": 0.35,
    "cell_line": 0.4,
    "library_selection": 0.35,
}

def compute_categorical_metrics(pred_texts, ref_texts, categories):
    metrics = {}
    print("List categories: ", categories, flush=True)
    NA_EQUIV = {"not applicable", "not_applicable"}

    for cat in categories:
        accs = []
        for pred, ref in zip(pred_texts, ref_texts):
            print("--------------------")
            print("raw pred: ", pred, flush=True)
            print("raw ref: ", ref, flush=True)

            p_dict = parse_pred_block(pred)
            r_dict = json.loads(ref)

            print("--------------------")
            print("p_dict: ", p_dict, flush=True)
            print("r_dict: ", r_dict, flush=True)

            p_val = norm_val(p_dict.get(cat, ""))
            r_val = norm_val(r_dict.get(cat, ""))

            print("--------------------")
            print("category: ", cat, flush=True)
            print("pred val: ", p_val, flush=True)
            print("ref val: ", r_val, flush=True)

            if str(r_val).lower() == 'nan':
                accs.append(str(p_val).lower() == 'nan')
                continue

            if not r_val:
                continue

            if cat == "cell_line" and normalize(r_val) == "unknown" and normalize(p_val) in NA_EQUIV:
                accs.append(True)
                print("match (NA vs unknown override): True", flush=True)
                continue
            if cat in {"treatment_time", "response"}:
                if normalize(r_dict.get("treatment","")) == "no treatment" and normalize(p_val) in NA_EQUIV:
                    accs.append(True)
                    print("match (NA vs unknown override, no treatment): True", flush=True)
                    continue

            if not p_val or str(p_val).lower() == 'nan':
                acc = False
            elif cat == "treatment_time":
                d_ref = _parse_duration_days(r_val)
                d_pred = _parse_duration_days(p_val)
                if d_ref is not None and d_pred is not None:
                    acc = abs(d_ref - d_pred) <= 1.0
                    print("match (duration): ", acc, flush=True)
                else:
                    emb_ref = model_sem.encode([r_val], convert_to_tensor=True)
                    emb_pred = model_sem.encode([p_val], convert_to_tensor=True)
                    cos = cosine_similarity(emb_ref.cpu().numpy(), emb_pred.cpu().numpy())[0][0]
                    acc = cos > THRESH[cat]
            elif cat in normal_accuracy_categories:
                acc = (normalize(p_val) == normalize(r_val))
            else:
                emb_ref = model_sem.encode([r_val], convert_to_tensor=True)
                emb_pred = model_sem.encode([p_val], convert_to_tensor=True)
                cos = cosine_similarity(emb_ref.cpu().numpy(), emb_pred.cpu().numpy())[0][0]
                acc = cos > THRESH[cat]

            accs.append(acc)
            print("match: ", acc, flush=True)
            print("--------------------")
            if cat == "cell_type" and not acc:
                print("Pred failed for cell_type →")
                print(f"Predicted: {p_val}")
                print(f"Expected: {r_val}")

        metrics[f"accuracy_{cat.lower()}"] = (sum(accs) / len(accs)) if accs else 0.0

    metrics["accuracy_overall"] = (
        sum(metrics[f"accuracy_{c.lower()}"] for c in categories) / len(categories)
        if categories else 0.0
    )
    return metrics

def compute_metrics(eval_preds: EvalPrediction):
    gen_ids, label_ids = eval_preds.predictions, eval_preds.label_ids
    decoded_preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    cat_metrics = compute_categorical_metrics(decoded_preds, decoded_labels, categories)
    return {f"eval_{k}": v for k,v in cat_metrics.items()}

def _no_repeat_ngram_bans(gen_ids, no_repeat_ngram_size, vocab_size):
    if no_repeat_ngram_size <= 0 or len(gen_ids) < no_repeat_ngram_size - 1:
        return set()
    n = no_repeat_ngram_size
    idx = {}
    for i in range(len(gen_ids) - n + 1):
        prev = tuple(gen_ids[i:i+n-1])
        nxt  = gen_ids[i+n-1]
        idx.setdefault(prev, set()).add(nxt)
    prev = tuple(gen_ids[-(n-1):])
    return idx.get(prev, set())

class GenerationEarlyStoppingCallback(TrainerCallback):
    def __init__(self, metric_name: str, patience: int=3, verbose: bool=True):
        self.metric_name = metric_name
        self.patience = patience
        self.verbose = verbose
        self.best = -math.inf
        self.num_bad=0
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        # Hugging Face passes metrics via kwargs["metrics"] for on_evaluate
        metrics = logs if logs is not None else kwargs.get("metrics")
        if metrics is None:
            return control
        current = metrics.get(self.metric_name)
        if current and current>self.best:
            self.best=current; self.num_bad=0
            control.should_save=True
        else:
            self.num_bad+=1
            if self.num_bad>=self.patience:
                control.should_early_stop=True; control.should_save=True
        return control

def add_category_adapters(peft_model: PeftModel, categories, base_config: LoraConfig):
    for cat in categories:
        name = f"cat_{cat}"
        if name not in peft_model.peft_config:
            peft_model.add_adapter(name, base_config)

def set_active_adapter(peft_model: PeftModel, adapter_name: str):
    peft_model.set_adapter(adapter_name)

def get_adapter_state_dict(peft_model: PeftModel, adapter_name: str):
    sd = peft_model.state_dict()
    filt = {k: v.clone() for k, v in sd.items() if f".{adapter_name}." in k}
    return filt

def load_adapter_state_dict_(peft_model: PeftModel, adapter_name: str, adapter_sd: dict):
    full_sd = peft_model.state_dict()
    for k in full_sd.keys():
        if f".{adapter_name}." in k and k in adapter_sd:
            full_sd[k] = adapter_sd[k].to(full_sd[k].device, dtype=full_sd[k].dtype)
    peft_model.load_state_dict(full_sd, strict=False)

def tok_ids(s):
    return tokenizer(s, add_special_tokens=False, return_tensors=None)["input_ids"]

def stop_token_id_set():
    return set()

@torch.no_grad()
def greedy_step(model, input_ids, attention_mask, past, temperature=0.0):
    out = model(input_ids=input_ids, attention_mask=attention_mask, past_key_values=past, use_cache=True)
    logits = out.logits[:, -1, :]
    if temperature and temperature > 0.0:
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        lp = torch.log(probs.gather(-1, next_id))
        return next_id, float(lp.item()), out.past_key_values
    else:
        logp = torch.log_softmax(logits, dim=-1)
        next_id = torch.argmax(logp, dim=-1, keepdim=True)
        lp = logp.gather(-1, next_id)
        return next_id, float(lp.item()), out.past_key_values

@torch.no_grad()
def prime_tokens(model, seq_ids, past):
    out = model(input_ids=seq_ids, past_key_values=past, use_cache=True)
    return out.past_key_values

def _sanitize_value(text: str) -> str:
    text = re.sub(r'[\n\r`]+', ' ', str(text))
    text = text.strip().strip('",}').strip()
    text = re.sub(r'\s+', ' ', text)
    return text

QUOTE_CHARS = ['"', '”', '“']
BAD_VALUE_CHARS_NO_QUOTE = [',', '}', '\n', '`', ':', ';', '(', ')', '[', ']']
_BAD_NOQUOTE_IDS = None
_QUOTE_ANY_IDS = None
_ALWAYS_BANNED_IDS = None
_CAT_BANNED_IDS = {}
_CAT_MAX_TOKENS = {
    "sex": 2,
    "is_cancer": 2,
    "biopsy_type": 6,
    "library_selection": 6,
    "sequencing_source": 6,
    "response": 6,
    "age": 6,
    "treatment_time": 6,
    "organ": 6,
    "biopsy_site": 4,
    "cell_type": 8,
    "disease": 8,
    "ethnicity": 6,
    "localization": 10,
    "cell_line": 8,
    "treatment": 8,
}
CURRENT_CATEGORY = None
_DISCOURAGED_IDS = None
_CAT_HINT_IDS = {}
_GEN_DEBUG_ONCE = False

def _ensure_vocab_masks():
    global _BAD_NOQUOTE_IDS, _QUOTE_ANY_IDS, _ALWAYS_BANNED_IDS, _CAT_BANNED_IDS, _DISCOURAGED_IDS, _CAT_HINT_IDS, _GEN_DEBUG_ONCE
    if _BAD_NOQUOTE_IDS is not None and _ALWAYS_BANNED_IDS is not None and _DISCOURAGED_IDS is not None and _CAT_HINT_IDS:
        return
    bad_noquote = set()
    quote_any = set()
    always_banned = set()
    cat_banned = {k: set() for k in _CAT_MAX_TOKENS.keys()}
    discouraged = set()
    cat_hints = {k: set() for k in _CAT_MAX_TOKENS.keys()}

    vocab_size = getattr(tokenizer, 'vocab_size', None) or len(tokenizer)
    acc_substrings = [
        "SRS", "SRR", "SRX", "SRP", "SAMN", "SAMD", "SAME",
        "GSM", "GSE", "GPL",
        "PRJ", "ERP", "DRR", "ERR", "E-",
    ]

    def piece_has_accession(piece: str) -> bool:
        if any(s in piece for s in acc_substrings):
            return True
        if re.search(r'[A-Z]{2,}\d{2,}', piece):
            return True
        if re.search(r'\d{5,}', piece):
            return True
        return False

    numeric_ban_cats = set(list(CLOSED_SETS.keys()) + ["biopsy_type"])

    hint_map = {
        "sex": ["female","male"],
        "is_cancer": ["true","false"],
        "sequencing_source": ["bulk","spatial","single"],
        "biopsy_type": ["metastasis","primary","blood"],
        "library_selection": ["polya","inverse","rrna","hybrid","small","other"],
    }

    for tid in range(vocab_size):
        piece = tokenizer.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if not piece:
            continue
        pl = piece.lower()
        if any(q in piece for q in QUOTE_CHARS):
            quote_any.add(tid)
        if any(ch in piece for ch in BAD_VALUE_CHARS_NO_QUOTE):
            bad_noquote.add(tid)

        if piece_has_accession(piece):
            always_banned.add(tid)

        if "unknown" in pl or "not applicable" in pl or pl.strip() in {"n/a","na","none","unspecified"}:
            discouraged.add(tid)

        if re.search(r'\d', piece):
            for k in numeric_ban_cats:
                cat_banned[k].add(tid)

        for k, hints in hint_map.items():
            if any(h in pl for h in hints):
                cat_hints[k].add(tid)

    _BAD_NOQUOTE_IDS = bad_noquote
    _QUOTE_ANY_IDS = quote_any
    _ALWAYS_BANNED_IDS = always_banned
    _CAT_BANNED_IDS = cat_banned
    _DISCOURAGED_IDS = discouraged
    _CAT_HINT_IDS = cat_hints

    if not _GEN_DEBUG_ONCE:
        _GEN_DEBUG_ONCE = True
        print(f"debug/vocab_masks: discouraged_tokens={len(_DISCOURAGED_IDS)} quote_tokens={len(_QUOTE_ANY_IDS)} "
              f"always_banned={len(_ALWAYS_BANNED_IDS)}", flush=True)
        for k in sorted(_CAT_BANNED_IDS.keys()):
            print(f"debug/vocab_masks: numeric_banned_in[{k}]={len(_CAT_BANNED_IDS[k])}", flush=True)

@torch.no_grad()
def gen_until_quote_fullctx(model, tokenizer, context_ids, max_value_tokens=64, no_repeat_ngram_size=3):
    _ensure_vocab_masks()
    device = context_ids.device
    input_ids = context_ids
    gen_buf = []
    has_content = False

    cat = CURRENT_CATEGORY if CURRENT_CATEGORY is not None else ""
    cat_max = _CAT_MAX_TOKENS.get(cat, max_value_tokens)
    steps_limit = min(max_value_tokens, cat_max)

    for _ in range(steps_limit):
        out = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids, device=device),
            use_cache=False
        )
        logits = out.logits[:, -1, :]

        if gen_buf:
            logits[:, gen_buf[-1]] -= 2.0
            import numpy as _np
            uniq, counts = _np.unique(gen_buf, return_counts=True)
            for u, c in zip(uniq.tolist(), counts.tolist()):
                logits[:, u] -= 0.25 * max(0, c - 1)

        banned = set(_BAD_NOQUOTE_IDS)
        if gen_buf:
            banned.add(gen_buf[-1])
        if not has_content:
            banned |= _QUOTE_ANY_IDS
        banned |= _ALWAYS_BANNED_IDS
        banned |= _no_repeat_ngram_bans(gen_buf, no_repeat_ngram_size, logits.size(-1))
        if cat in _CAT_BANNED_IDS:
            banned |= _CAT_BANNED_IDS[cat]
        if banned:
            logits[:, list(banned)] = -float("inf")

        if model.config.eos_token_id is not None:
            logits[:, model.config.eos_token_id] = -float("inf")

        if has_content and cat in CLOSED_SETS and _QUOTE_ANY_IDS:
            for tid_q in _QUOTE_ANY_IDS:
                logits[:, tid_q] += 1.5

        if not has_content and cat in _CAT_HINT_IDS and _CAT_HINT_IDS[cat]:
            for tid in _CAT_HINT_IDS[cat]:
                logits[:, tid] += 0.6

        if _DISCOURAGED_IDS:
            if cat in CLOSED_SETS:
                for tid in _DISCOURAGED_IDS:
                    logits[:, tid] -= 1.5
            else:
                # stronger discouragement on open categories to avoid "unknown"/"not applicable" bias
                for tid in _DISCOURAGED_IDS:
                    logits[:, tid] -= 5.0

        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        tid = next_id.item()
        piece = tokenizer.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False)

        if any(q in piece for q in QUOTE_CHARS) or any(ch in piece for ch in BAD_VALUE_CHARS_NO_QUOTE):
            break
        if re.search(r'(SRS|SRR|SRX|SRP|SAMN|SAMD|SAME|GSM|GSE|GPL|PRJ|ERP|DRR|ERR|E-)[A-Za-z0-9_-]*', piece):
            break
        if re.search(r'[A-Z]{2,}\d{2,}', piece):
            break

        input_ids = torch.cat([input_ids, next_id], dim=1)
        gen_buf.append(tid)
        if any(c.isalnum() for c in piece):
            has_content = True

        if cat in CLOSED_SETS:
            partial = tokenizer.decode(gen_buf, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            canon = _canonicalize_closed_value(cat, partial)
            if canon is not None:
                return canon, input_ids

    value = tokenizer.decode(gen_buf, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    value = re.split(r'[";\n`,}:()\[\]]', value)[0]
    value = re.sub(r'\s+', ' ', value).strip(' ,;.-').strip()
    return value, input_ids

##########################################################################################
#CUSTOMED TRAINER

class MyTrainer(Trainer):
    def __init__(self, *args, sem_model=None, sem_loss_weight=0.05, label_smoothing=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        self.sem_model = sem_model.eval()
        for p in self.sem_model.parameters():
            p.requires_grad = False
        self.sem_loss_weight = sem_loss_weight
        self.label_smoothing = label_smoothing
        self.loss_fct = CrossEntropyLoss(label_smoothing=self.label_smoothing, ignore_index=-100)
        self._tot_sum = 0.0
        self._count = 0
        self.loss_ma = deque(maxlen=50)
        self.hidden_to_sem = torch.nn.Linear(
            self.model.config.hidden_size,
            self.sem_model.get_sentence_embedding_dimension()
        ).to(self.model.device)
        self.categories = getattr(self.model.config, "categories", [])
        self._cat_sum = torch.zeros(len(self.categories), dtype=torch.float64)
        self._cat_count = torch.zeros(len(self.categories), dtype=torch.float64)
        self.guard_ema = torch.zeros(len(self.categories), dtype=torch.float32)
        self.guard_initialized = torch.zeros(len(self.categories), dtype=torch.bool)
        self.guard_lambda = 0.001
        self.guard_margin = 0.01
        self.guard_ema_beta = 0.9
        self.best_cat_acc = {c: -1.0 for c in self.categories}
        self.best_cat_adapter_sd = {c: None for c in self.categories}
        self.accept_revert_tolerance = 0.01
        self.min_revert_step = 100
        self.cat2adapter = {c: f"cat_{c}" for c in self.categories}
        print(f"debug/trainer_init: label_smoothing={self.label_smoothing} guard_lambda={self.guard_lambda}", flush=True)

    def log(self, logs, *args, **kwargs):
        return super().log(logs)

    def _print_masks_debug(self, inputs, cat_masks):
        try:
            input_ids = inputs["input_ids"]
            for b in range(min(input_ids.size(0), 1)):
                for j, cat in enumerate(self.categories):
                    m = cat_masks[b, j].cpu().numpy()
                    idx = np.where(m > 0.5)[0]
                    if idx.size == 0:
                        continue
                    groups = []
                    start = idx[0]
                    prev = idx[0]
                    for k in idx[1:]:
                        if k == prev + 1:
                            prev = k
                        else:
                            groups.append((start, prev + 1))
                            start = k
                            prev = k
                    groups.append((start, prev + 1))
                    for (s, e) in groups[:3]:
                        toks = input_ids[b, s:e].tolist()
                        text = tokenizer.decode([t for t in toks if t != tokenizer.pad_token_id], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        except Exception as e:
            print("debug/mask_print_error:", str(e), flush=True)

    def _forward_with_adapter(self, adapter_name, inputs):
        set_active_adapter(self.model, adapter_name)
        return self.model(**inputs, output_hidden_states=False)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        cat_masks = inputs.pop("cat_masks", None)
        labels = inputs['labels']
        total_loss = torch.zeros([], device=labels.device)
        if cat_masks is not None:
            cat_masks = torch.as_tensor(cat_masks, device=labels.device, dtype=torch.float32)
            self._print_masks_debug(inputs, cat_masks)

        if cat_masks is not None:
            masks_shift_all = cat_masks[:, :, 1:].contiguous()
            B, C, L = masks_shift_all.size()
            tok_counts = masks_shift_all.sum(dim=(0, 2))
            present = (tok_counts > 0).nonzero(as_tuple=False).flatten().tolist()

            active_idx = present
            any_valid = False
            per_cat_valid_tokens = []
            cat_terms = []

            for j in active_idx:
                cat = self.categories[j]
                m = masks_shift_all[:, j, :].reshape(B * L) > 0.5
                if not m.any():
                    per_cat_valid_tokens.append((cat, 0))
                    continue
                outputs_cat = self._forward_with_adapter(self.cat2adapter[cat], inputs)
                logits_cat = outputs_cat.logits
                logits_cat_shift = logits_cat[:, :-1, :].contiguous()
                labels_shift = labels[:, 1:].contiguous()
                V = logits_cat_shift.size(-1)
                flat_logits = logits_cat_shift.view(B * L, V)
                flat_labels = labels_shift.view(B * L)
                valid = (flat_labels != -100) & m
                vcount = int(valid.sum().item())
                per_cat_valid_tokens.append((cat, vcount))
                if valid.any():
                    any_valid = True
                    loss_j = self.loss_fct(flat_logits[valid], flat_labels[valid])
                    if not self.guard_initialized[j]:
                        self.guard_ema[j] = loss_j.detach().float()
                        self.guard_initialized[j] = True
                    else:
                        self.guard_ema[j] = self.guard_ema_beta * self.guard_ema[j] + (
                                    1 - self.guard_ema_beta) * loss_j.detach().float()
                    penalty = torch.relu(loss_j - (self.guard_ema[j].to(loss_j.device) + self.guard_margin))
                    cat_terms.append((loss_j, penalty, vcount, cat))

            try:
                summary = {c: v for c, v in per_cat_valid_tokens}
                total_valid = sum(v for _, v in per_cat_valid_tokens)
                num_present = len(active_idx)
                print(
                    f"debug/compute_loss: present_cats={num_present}/{C}, total_valid_tokens={total_valid}, per_cat_valid_tokens={summary}",
                    flush=True)
            except Exception:
                total_valid = sum(v for _, v in per_cat_valid_tokens)

            num_terms = len(cat_terms)
            if num_terms > 0:
                per_cat_scale = 1.0 / num_terms
                for (loss_j, penalty, vcount, cat) in cat_terms:
                    total_loss = total_loss + per_cat_scale * loss_j + self.guard_lambda * per_cat_scale * penalty

            if not any_valid:
                if len(self.categories) > 0:
                    fallback_cat = random.choice(self.categories)
                    print(f"debug/fallback_used: no valid category tokens; using '{fallback_cat}'", flush=True)
                    outputs_cat = self._forward_with_adapter(self.cat2adapter[fallback_cat], inputs)
                    logits_cat = outputs_cat.logits
                    logits_shift = logits_cat[:, :-1, :].contiguous()
                    labels_shift = labels[:, 1:].contiguous()
                    valid = (labels_shift != -100).view(-1)
                    if valid.any():
                        total_loss = total_loss + self.loss_fct(logits_shift.view(-1, logits_shift.size(-1))[valid],
                                                                labels_shift.view(-1)[valid])
        else:
            if len(self.categories) > 0:
                fallback_cat = random.choice(self.categories)
                print(f"debug/fallback_used: no cat_masks provided; using '{fallback_cat}'", flush=True)
                outputs_cat = self._forward_with_adapter(self.cat2adapter[fallback_cat], inputs)
                logits_cat = outputs_cat.logits
                logits_shift = logits_cat[:, :-1, :].contiguous()
                labels_shift = labels[:, 1:].contiguous()
                valid = (labels_shift != -100).view(-1)
                if valid.any():
                    total_loss = total_loss + self.loss_fct(logits_shift.view(-1, logits_shift.size(-1))[valid],
                                                            labels_shift.view(-1)[valid])

        step_loss = float(total_loss.detach().cpu())
        print(f"train/loss_step_{self.state.global_step}: {step_loss}", flush=True)
        self.loss_ma.append(step_loss)
        if (self.state.global_step + 1) % max(1, self.args.logging_steps) == 0:
            ma = float(np.mean(self.loss_ma)) if len(self.loss_ma) > 0 else step_loss
            self.log({"train/loss_ma": ma})
        if return_outputs:
            outputs_dummy = type("obj", (), {})()
            outputs_dummy.loss_total = total_loss
            return total_loss, outputs_dummy
        return total_loss

    def _generate_with_adapter(self, adapter_name, prompt):
        set_active_adapter(self.model, adapter_name)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(self.model.device)
        with torch.no_grad():
            out_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=350,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                early_stopping=True,
                do_sample=False,
                top_p=0.9,
                temperature=0.0,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
            )
        raw = tokenizer.decode(out_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return raw

    def _eval_per_category_generate(self, df_eval, max_rows=None):
        device = self.model.device
        _ = stop_token_id_set()
        comma_ids = tok_ids('", ')
        if max_rows is not None and len(df_eval) > max_rows:
            df_eval = df_eval.sample(max_rows, random_state=SEED)
        if len(comma_ids) == 0:
            comma_ids = tok_ids('",')
        brace_close_ids = tok_ids('"}')
        if len(brace_close_ids) == 0:
            brace_close_ids = tok_ids('" }')

        preds_json_str, refs_json_str = [], []
        for _, row in df_eval.iterrows():
            prompt = row['prompt'].strip()
            expected = row['output'].strip()
            ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
            context_ids = ids["input_ids"]
            ks = tok_ids("{") or tok_ids("{\n") or tok_ids("{ ")
            context_ids = torch.cat([context_ids, torch.tensor([ks], device=device)], dim=1)
            merged = {}
            for ci, cat in enumerate(self.categories):
                prefix_ids = tok_ids(f'"{cat}": "')
                context_ids = torch.cat([context_ids, torch.tensor([prefix_ids], device=device)], dim=1)
                set_active_adapter(self.model, f"cat_{cat}")
                global CURRENT_CATEGORY
                CURRENT_CATEGORY = cat
                text_val, context_ids = gen_until_quote_fullctx(
                    self.model, tokenizer, context_ids, max_value_tokens=64
                )
                merged[cat] = text_val

                if ci < len(self.categories) - 1:
                    context_ids = torch.cat([context_ids, torch.tensor([comma_ids], device=device)], dim=1)
                else:
                    context_ids = torch.cat([context_ids, torch.tensor([brace_close_ids], device=device)], dim=1)

            preds_json_str.append(json.dumps(remap_closed_sets(merged), ensure_ascii=False))
            refs_json_str.append(expected)

        return preds_json_str, refs_json_str

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kwargs):
        was_training = self.model.training
        self.model.eval()
        try:
            eval_dataloader = self.get_eval_dataloader(eval_dataset)
            total_loss, count = 0.0, 0
            for inputs in eval_dataloader:
                inputs = self._prepare_inputs(inputs)
                with torch.no_grad():
                    loss, outputs = self.compute_loss(self.model, inputs, return_outputs=True)
                total_loss += outputs.loss_total.item()
                count += 1
            avg_total = total_loss / count if count else 0.0
            results = {f"{metric_key_prefix}_loss": avg_total}
            self.log(results)

            DO_FULL = (self.state.global_step % 500 == 0) or (self.state.global_step == 0)
            max_rows = None if DO_FULL else 64
            preds, refs = self._eval_per_category_generate(df_val_final, max_rows=max_rows)
            cat_metrics = compute_categorical_metrics(preds, refs, self.categories)
            for k, v in cat_metrics.items():
                key = f"{metric_key_prefix}_{k}"
                self.log({key: v})
                results[key] = v

            for cat in self.categories:
                acc_key = f"accuracy_{cat.lower()}"
                cur = cat_metrics.get(acc_key, 0.0)
                best = self.best_cat_acc.get(cat, -1.0)
                if cur > best + self.accept_revert_tolerance:
                    self.best_cat_acc[cat] = cur
                    sd = get_adapter_state_dict(self.model, self.cat2adapter[cat])
                    self.best_cat_adapter_sd[cat] = {k: v.detach().cpu().clone() for k, v in sd.items()}
                    print(f"debug/accept_adapter: {self.cat2adapter[cat]} new_best={cur:.4f}", flush=True)
                elif cur < best - self.accept_revert_tolerance and self.best_cat_adapter_sd.get(cat) is not None and self.state.global_step >= getattr(self, "min_revert_step", 0):
                    load_adapter_state_dict_(self.model, self.cat2adapter[cat], self.best_cat_adapter_sd[cat])
                    print(f"debug/revert_adapter: {self.cat2adapter[cat]} revert_to_best={best:.4f} cur={cur:.4f}",
                          flush=True)
                else:
                    print(f"debug/no_updates: new_best={cur:.4f}, best = {best:.4f}", flush=True)

            # make sure callbacks see these eval metrics (early stopping/saving)
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, results)
            return results
        finally:
            if was_training:
                self.model.train()


##########################################################################################
#MAIN
print("Load dataset with prompts", flush=True)
df_train = pd.read_csv(prompt_train_file)
df_val = pd.read_csv(prompt_val_file)
df_test = pd.read_csv(prompt_test_file)

df_train_final = df_train
df_val_final = df_val

print(f"Training set size: {len(df_train_final)}, Validation set size: {len(df_val_final)}, Test set size: {len(df_test)}", flush=True)

all_cats = set()
for out in df_train_final["output"].dropna():
    try:
        obj = json.loads(out)
        all_cats.update(obj.keys())
    except json.JSONDecodeError:
        continue
categories = sorted(all_cats)
print("Detected categories:", categories)

metric_names = [f"eval_accuracy_{c.lower()}" for c in categories]

print("Content of test set:\n", flush=True)
run_accessions_list = []
for i, row in df_test.iterrows():
    prompt = row["prompt"].strip()
    first_line = prompt.splitlines()[0]
    match = re.search(r"Run accession:\s*(\S+)", first_line)
    run_accession = match.group(1) if match else "N/A"
    run_accessions_list.append(run_accession)
print(run_accessions_list, flush=True)

##########################################################################################
#FINAL TRAINING ON ALL DATA (TRAIN+VAL)

train_dataset_final = Dataset.from_pandas(df_train_final)
val_dataset_final = Dataset.from_pandas(df_val_final)
IS_TRAINING = True
tokenized_train_final = train_dataset_final.map(
    tokenize_function,
    remove_columns=train_dataset_final.column_names
)
IS_TRAINING = False
tokenized_val_final = val_dataset_final.map(
    tokenize_function,
    remove_columns=val_dataset_final.column_names
)

final_peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,
    lora_alpha=2*8,
    lora_dropout=0.05,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']
)

model_final = get_peft_model(model_base, final_peft_config)
add_category_adapters(model_final, categories, final_peft_config)
model_final.resize_token_embeddings(len(tokenizer))
model_final.config.categories = categories
model_final.to(device)

trainable = [(n, p) for n, p in model_final.named_parameters() if p.requires_grad]
print("trainable_params_total:", len(trainable), flush=True)
by_dev_dtype = {}
for n, p in trainable:
    key = (str(p.device), str(p.dtype))
    by_dev_dtype[key] = by_dev_dtype.get(key, 0) + p.numel()
print("trainable_params_by_device_dtype:", by_dev_dtype, flush=True)

training_args_final = TrainingArguments(
    output_dir=train_model + "_final",
    eval_strategy="steps",
    eval_steps=50,
    learning_rate=2e-4,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=4,
    weight_decay=0.0,
    save_strategy='steps',
    save_steps=50,
    logging_strategy='steps',
    logging_steps=50,
    fp16=(not use_bf16),
    bf16=use_bf16,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    report_to=["tensorboard"],
    logging_dir=os.path.join(tensorboard_log_dir, "final_training"),
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy_overall",
    remove_unused_columns=False,
    greater_is_better=True,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    max_grad_norm=0.3,
    adam_beta2=0.95
)
print(f"debug/training_args: learning_rate={training_args_final.learning_rate}, adam_beta2={training_args_final.adam_beta2}", flush=True)

trainer_final = MyTrainer(
    model=model_final,
    args=training_args_final,
    train_dataset=tokenized_train_final,
    eval_dataset=tokenized_val_final,
    label_smoothing=0.1,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt", padding=True),
    callbacks=[
            GenerationEarlyStoppingCallback("eval_accuracy_overall", patience=2),
        ],
    sem_model=model_sem,
    compute_metrics=compute_metrics
)

final_writer_training = SummaryWriter(os.path.join(tensorboard_log_dir, "final_training"))
final_writer_training.add_scalar("train/size", len(df_train_final), 0)
final_writer_training.add_scalar("val/size",   len(df_val_final),   0)

init_val = trainer_final.evaluate(eval_dataset=tokenized_val_final, metric_key_prefix="eval")
print("Initial val metrics - original model (step=0) →", init_val)

for cat in categories:
    key = f"eval_accuracy_{cat.lower()}"
    final_writer_training.add_scalar(key, init_val.get(key, 0.0), 0)

trainer_final.train()
final_writer_training.close()

print("Save new model", flush=True)
trainer_final.save_model(output_model)

print("Merge and save full model", flush=True)
os.makedirs(merged_model_path, exist_ok=True)
for cat in categories:
    adapter_name = f"cat_{cat}"
    set_active_adapter(model_final, adapter_name)
    model_merged = model_final.merge_and_unload()
    save_dir = os.path.join(merged_model_path, f"merged_{adapter_name}")
    os.makedirs(save_dir, exist_ok=True)
    model_merged.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    model_final = get_peft_model(model_base, final_peft_config)
    add_category_adapters(model_final, categories, final_peft_config)
    for c in categories:
        best_sd = trainer_final.best_cat_adapter_sd.get(c)
        if best_sd is not None:
            load_adapter_state_dict_(model_final, f"cat_{c}", best_sd)

##########################################################################################
#EVAL TEST

print("\nGenerating predictions on final test set", flush=True)
model_final.eval()

test_df = df_test
final_writer = SummaryWriter(os.path.join(tensorboard_log_dir, "final_predictions"))
test_preds, test_refs = [], []

for i, row in test_df.iterrows():
    prompt = row["prompt"].strip()
    print(prompt, flush=True)
    expected = row["output"].strip()
    merged = {}
    device = model_final.device
    ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    context_ids = ids["input_ids"]

    ks = tok_ids("{") or tok_ids("{\n") or tok_ids("{ ")
    context_ids = torch.cat([context_ids, torch.tensor([ks], device=device)], dim=1)

    comma_ids = tok_ids('", ')
    if len(comma_ids) == 0:
        comma_ids = tok_ids('",')
    brace_close_ids = tok_ids('"}')
    if len(brace_close_ids) == 0:
        brace_close_ids = tok_ids('" }')

    for ci, cat in enumerate(categories):
        prefix_ids = tok_ids(f'"{cat}": "')
        context_ids = torch.cat([context_ids, torch.tensor([prefix_ids], device=device)], dim=1)

        set_active_adapter(model_final, f"cat_{cat}")
        CURRENT_CATEGORY = cat
        text_val, context_ids = gen_until_quote_fullctx(
            model_final, tokenizer, context_ids, max_value_tokens=64
        )
        merged[cat] = text_val

        if ci < len(categories) - 1:
            context_ids = torch.cat([context_ids, torch.tensor([comma_ids], device=device)], dim=1)
        else:
            context_ids = torch.cat([context_ids, torch.tensor([brace_close_ids], device=device)], dim=1)

    clean_prediction = json.dumps(remap_closed_sets(merged), ensure_ascii=False)
    print(f"--- Predicted output {i + 1}: {clean_prediction}", flush=True)
    print(f"--- Expected output  {i + 1}: {expected}", flush=True)
    print("-" * 50, flush=True)
    test_preds.append(clean_prediction)
    test_refs.append(expected)
    final_writer.add_text(
        f"Prediction/Test_Example_{i + 1}",
        f"Prompt: {prompt}\nPred: {clean_prediction} | Label: {expected}",
        global_step=i
    )

final_writer.close()

metrics_test = compute_categorical_metrics(test_preds, test_refs, categories)
print("\nCategory SMA - test set:", flush=True)
for cat in categories:
    print(f"{cat}: {metrics_test[f'accuracy_{cat.lower()}']:.4f}", flush=True)

final_test_writer = SummaryWriter(os.path.join(tensorboard_log_dir, "final_test"))
for cat in categories:
    final_test_writer.add_scalar(f"test/{cat}", metrics_test[f"accuracy_{cat.lower()}"] , 0)
final_test_writer.close()

##########################################################################################
#EVAL TEST OOV

if os.path.exists(prompt_test_oov_file):
    print("\nGenerating predictions on final OOV test set", flush=True)
    df_test_oov = pd.read_csv(prompt_test_oov_file)
    with torch.no_grad():
        model_gen_oov = copy.deepcopy(model_final)
    test_oov_preds, test_oov_refs = [], []
    oov_writer = SummaryWriter(os.path.join(tensorboard_log_dir, "final_predictions_oov"))

    for i, row in df_test_oov.iterrows():
        prompt = row["prompt"].strip()
        print(prompt, flush=True)
        expected = row["output"].strip()
        merged = {}
        device = model_gen_oov.device
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
        context_ids = ids["input_ids"]

        ks = tok_ids("{") or tok_ids("{\n") or tok_ids("{ ")
        context_ids = torch.cat([context_ids, torch.tensor([ks], device=device)], dim=1)

        comma_ids = tok_ids('", ')
        if len(comma_ids) == 0:
            comma_ids = tok_ids('",')
        brace_close_ids = tok_ids('"}')
        if len(brace_close_ids) == 0:
            brace_close_ids = tok_ids('" }')

        for ci, cat in enumerate(categories):
            prefix_ids = tok_ids(f'"{cat}": "')
            context_ids = torch.cat([context_ids, torch.tensor([prefix_ids], device=device)], dim=1)

            set_active_adapter(model_gen_oov, f"cat_{cat}")
            CURRENT_CATEGORY = cat
            text_val, context_ids = gen_until_quote_fullctx(
                model_gen_oov, tokenizer, context_ids, max_value_tokens=64
            )
            merged[cat] = text_val

            if ci < len(categories) - 1:
                context_ids = torch.cat([context_ids, torch.tensor([comma_ids], device=device)], dim=1)
            else:
                context_ids = torch.cat([context_ids, torch.tensor([brace_close_ids], device=device)], dim=1)

        clean_prediction = json.dumps(remap_closed_sets(merged), ensure_ascii=False)
        print(f"--- Predicted output {i + 1}: {clean_prediction}", flush=True)
        print(f"--- Expected output  {i + 1}: {expected}", flush=True)
        print("-" * 50, flush=True)
        test_oov_preds.append(clean_prediction)
        test_oov_refs.append(expected)
        oov_writer.add_text(
            f"Prediction_OOV/Test_Example_{i + 1}",
            f"Prompt: {prompt}\nPred: {clean_prediction} | Label: {expected}",
            global_step=i
        )

    oov_writer.close()

    metrics_test_oov = compute_categorical_metrics(test_oov_preds, test_oov_refs, categories)
    print("\nCategory SMA - oov test set:", flush=True)
    for cat in categories:
        print(f"{cat}: {metrics_test_oov[f'accuracy_{cat.lower()}']:.4f}", flush=True)

    final_test_oov_writer = SummaryWriter(os.path.join(tensorboard_log_dir, "final_test_oov"))
    for cat in categories:
        final_test_oov_writer.add_scalar(f"test_oov/{cat}", metrics_test_oov[f"accuracy_{cat.lower()}"], 0)
    final_test_oov_writer.close()
else:
    print(f"\n[Info] OOV test file not found: {prompt_test_oov_file}", flush=True)
