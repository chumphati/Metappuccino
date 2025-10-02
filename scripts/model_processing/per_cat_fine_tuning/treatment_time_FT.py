import os, re, json, math, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorWithPadding, EarlyStoppingCallback, TrainerCallback
)
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description="Fine-tune treatment_time (multitask) with soft semantic accuracy + duration normalization")
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--cls_loss_weight", type=float, default=0.4)
parser.add_argument("--reg_loss_weight", type=float, default=0.5)
parser.add_argument("--sem_loss_weight", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
base_path = args.base_path

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

train_file = os.path.join(base_path, "finetune_data_train.csv")
val_file   = os.path.join(base_path, "finetune_data_val.csv")
test_file  = os.path.join(base_path, "finetune_data_test.csv")
model_name = os.path.join(base_path, "Mistral-7B-Instruct-v0.3")

tb_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/logs16/tensorboard_treatment_time"
adapter_out_dir = os.path.join(base_path, "cat_treatment_time")
os.makedirs(tb_dir, exist_ok=True); os.makedirs(adapter_out_dir, exist_ok=True)

os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if use_bf16 else torch.float16

print("Load base model", flush=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    device_map="auto",
    low_cpu_mem_usage=True
)
base_model.config.pad_token_id = tokenizer.pad_token_id
if tokenizer.pad_token_id is not None and base_model.get_input_embeddings().num_embeddings != len(tokenizer):
    base_model.resize_token_embeddings(len(tokenizer))

print("Configure LoRA", flush=True)
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.2,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
)
peft_model = get_peft_model(base_model, peft_config)
peft_model.train()

print("Load datasets", flush=True)

def read_two_col_csv(path):
    try:
        df = pd.read_csv(path, engine="python", sep=None, dtype=str, keep_default_na=False)
        if "prompt" in df.columns and "output" in df.columns:
            return df[["prompt","output"]].astype(str)
    except Exception:
        pass
    try:
        df = pd.read_csv(path, engine="python", sep=";", dtype=str, keep_default_na=False)
        if "prompt" in df.columns and "output" in df.columns:
            return df[["prompt","output"]].astype(str)
    except Exception:
        pass
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            line = line.rstrip("\n\r")
            if not line:
                continue
            if "\t" in line:
                parts = line.split("\t"); prompt = "\t".join(parts[:-1]); out = parts[-1]
            elif ";" in line:
                parts = line.split(";"); prompt = ";".join(parts[:-1]); out = parts[-1]
            else:
                idx = line.rfind(",")
                if idx == -1:
                    prompt, out = line, ""
                else:
                    prompt, out = line[:idx], line[idx+1:]
            rows.append({"prompt": prompt, "output": out})
    return pd.DataFrame(rows)

df_train = read_two_col_csv(train_file)
df_val   = read_two_col_csv(val_file)
df_test  = read_two_col_csv(test_file)

df_train = df_train.drop_duplicates(subset=["prompt"]).copy()
df_train = df_train[df_train["output"].astype(str).str.strip()!=""].copy()

print(f"Train/Val/Test sizes: {len(df_train)}/{len(df_val)}/{len(df_test)}", flush=True)

def _escape_json_val(v):
    return str(v).replace('\\','\\\\').replace('"','\\"')

def _extract_value(txt):
    try:
        j = json.loads(txt)
        v = j.get("treatment_time","")
        return "" if v is None else str(v)
    except Exception:
        m = re.search(r'"treatment_time"\s*:\s*"([^"]*)"', str(txt))
        return m.group(1) if m else str(txt)

FR_NUM = {
    "un":1, "une":1, "deux":2, "trois":3, "quatre":4, "cinq":5, "six":6, "sept":7, "huit":8, "neuf":9,
    "dix":10, "onze":11, "douze":12, "treize":13, "quatorze":14, "quinze":15, "seize":16,
    "dix-sept":17, "dix sept":17, "dix-huit":18, "dix huit":18, "dix-neuf":19, "dix neuf":19, "vingt":20,
    "quelques":3
}
EN_NUM = {
    "one":1, "two":2, "three":3, "four":4, "five":5, "six":6, "seven":7, "eight":8, "nine":9,
    "ten":10, "eleven":11, "twelve":12, "thirteen":13, "fourteen":14, "fifteen":15, "sixteen":16,
    "seventeen":17, "eighteen":18, "nineteen":19, "twenty":20, "few":3, "couple":2
}

def _to_float_num(tok):
    tok = tok.replace(",", ".")
    try:
        return float(tok)
    except:
        pass
    return FR_NUM.get(tok, EN_NUM.get(tok, None))

def _normalize_duration_days(text):
    if text is None: return None
    s = re.sub(r'\s+', ' ', str(text).strip().lower())
    s = s.replace("½","0.5").replace("demi","0.5")
    if "fortnight" in s or "quinzaine" in s:
        return 14.0
    num = None
    m = re.search(r'(\d+(?:[\.,]\d+)?)\s*(semaines?|weeks?|w)\b', s)
    if m: num = _to_float_num(m.group(1)); return float(num)*7.0 if num is not None else None
    m = re.search(r'(\d+(?:[\.,]\d+)?)\s*(jours?|days?|d)\b', s)
    if m: num = _to_float_num(m.group(1)); return float(num) if num is not None else None
    m = re.search(r'(\d+(?:[\.,]\d+)?)\s*(mois|months?|m)\b', s)
    if m: num = _to_float_num(m.group(1)); return float(num)*30.0 if num is not None else None
    m = re.search(r'(\d+(?:[\.,]\d+)?)\s*(heures?|hours?|h)\b', s)
    if m: num = _to_float_num(m.group(1)); return float(num)/24.0 if num is not None else None
    m = re.search(r'\b(un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|dix-sept|dix sept|dix-huit|dix huit|dix-neuf|dix neuf|vingt|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|few|couple)\s*(semaines?|weeks?)\b', s)
    if m:
        num = _to_float_num(m.group(1)); return float(num)*7.0 if num is not None else None
    m = re.search(r'\b(un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|dix-sept|dix sept|dix-huit|dix huit|dix-neuf|dix neuf|vingt|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|few|couple)\s*(jours?|days?)\b', s)
    if m:
        num = _to_float_num(m.group(1)); return float(num) if num is not None else None
    m = re.search(r'\b(un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|dix-sept|dix sept|dix-huit|dix huit|dix-neuf|dix neuf|vingt|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|few|couple)\s*(mois|months?)\b', s)
    if m:
        num = _to_float_num(m.group(1)); return float(num)*30.0 if num is not None else None
    m = re.search(r'\b(un|une|deux|trois|quatre|cinq|six|sept|huit|neuf|dix|onze|douze|treize|quatorze|quinze|seize|dix-sept|dix sept|dix-huit|dix huit|dix-neuf|dix neuf|vingt|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|few|couple)\s*(heures?|hours?)\b', s)
    if m:
        num = _to_float_num(m.group(1)); return float(num)/24.0 if num is not None else None
    if "long cours" in s or "long-term" in s or "long term" in s: return None
    return None

def canonicalize_time_text(text):
    s = (text or "").strip()
    d = _normalize_duration_days(s)
    if d is None:
        return s
    if d < 2:
        h = round(d*24)
        return f"{h} hours"
    elif d < 21:
        return f"{int(round(d))} days"
    elif d < 90:
        w = round(d/7)
        return f"{int(w)} weeks"
    else:
        m = round(d/30)
        return f"{int(m)} months"

def has_duration_flag(text):
    return 1.0 if _normalize_duration_days(text) is not None else 0.0

for _df in (df_train, df_val, df_test):
    _df["output_raw"]  = _df["output"].astype(str)
    _df["output_raw"]  = _df["output_raw"].map(canonicalize_time_text)
    _df["duration_days"] = _df["output_raw"].map(_normalize_duration_days)
    _df["has_dur"] = _df["output_raw"].map(has_duration_flag)
    _df["output_json"] = _df["output_raw"].apply(lambda v: f'{{"treatment_time": "{_escape_json_val(v)}"}}')

labels_train = sorted(set(df_train["output_raw"].tolist()))
label2id = {lbl:i for i,lbl in enumerate(labels_train)}
id2label = {i:lbl for lbl,i in label2id.items()}
print(f"Label space (train) size = {len(label2id)}", flush=True)

for _df in (df_train, df_val, df_test):
    _df["cls_label"] = _df["output_raw"].map(lambda x: label2id.get(x, -1))

sem_model_name = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
model_sem = SentenceTransformer(sem_model_name)

for _df in (df_train, df_val, df_test):
    vals = _df["output_json"].tolist()
    refs = [_extract_value(v).strip() for v in vals]
    vecs = model_sem.encode(refs, convert_to_numpy=True, normalize_embeddings=True)
    _df["sem_emb"] = [v.astype("float32").tolist() for v in vecs]

make_cols = ["prompt","output_json","cls_label","duration_days","has_dur","sem_emb"]
datasets = {
    "train": Dataset.from_pandas(df_train[make_cols]),
    "validation": Dataset.from_pandas(df_val[make_cols]),
    "test": Dataset.from_pandas(df_test[make_cols])
}
print({k: len(v) for k,v in datasets.items()}, flush=True)

for name, ds in datasets.items():
    df = ds.to_pandas()
    if "sem_emb" in df.columns:
        bad = df["sem_emb"].isna()
        if hasattr(bad, "any") and bad.any():
            n_bad = int(bad.sum())
            print(f"warn/{name}: {n_bad} rows have sem_emb=None; they’ll be zero-filled at runtime", flush=True)

SIM_THRESH = 0.38
SEM_DIM = 768

def tokenize_fn(example):
    prompt   = example["prompt"].strip()
    out_json = example["output_json"].strip()
    m = re.search(r'"treatment_time"\s*:\s*"([^"]*)"', out_json)
    value_str  = m.group(1) if m else ""
    prefix_str = '{"treatment_time": "'
    suffix_str = '"}'
    max_in, max_out, max_total = 2048, 64, 2112
    in_enc  = tokenizer(prompt, truncation=True, padding=False, max_length=max_in, add_special_tokens=False)
    pref_ids = tokenizer(prefix_str, add_special_tokens=False)["input_ids"]
    val_ids  = tokenizer(value_str, truncation=True, padding=False, max_length=max_out, add_special_tokens=False)["input_ids"]
    suff_ids = tokenizer(suffix_str, add_special_tokens=False)["input_ids"]
    out_ids = pref_ids + val_ids + suff_ids
    ids  = in_enc["input_ids"] + out_ids
    attn = [1]*len(ids)
    labels_out = [-100]*len(pref_ids) + val_ids
    if len(suff_ids) > 0:
        labels_out += [suff_ids[0]] + [-100]*(len(suff_ids)-1)
    labels = [-100]*len(in_enc["input_ids"]) + labels_out
    if len(ids) > max_total:
        ids = ids[:max_total]; attn = attn[:max_total]; labels = labels[:max_total]
    dur = example.get("duration_days", None)
    has_dur = 1 if (dur is not None) else 0
    dur_val = float(dur) if dur is not None else 0.0
    sem = example.get("sem_emb", None)
    if sem is None or (isinstance(sem, float) and math.isnan(sem)):
        sem = [0.0] * SEM_DIM
    else:
        sem = list(sem)
        if len(sem) < SEM_DIM:
            sem = sem + [0.0] * (SEM_DIM - len(sem))
        elif len(sem) > SEM_DIM:
            sem = sem[:SEM_DIM]
    return {
        "input_ids": ids,
        "attention_mask": attn,
        "labels": labels,
        "split_idx": len(in_enc["input_ids"]),
        "cls_label": int(example.get("cls_label", -1)),
        "duration_mask": has_dur,
        "duration_value": math.log1p(dur_val),
        "sem_emb": sem
    }

class CausalLMPadCollator:
    def __init__(self, tokenizer):
        self._padder = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    def __call__(self, features):
        labels    = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        split_idx = torch.tensor([int(f.get("split_idx", 0)) for f in features], dtype=torch.long)
        cls_label = torch.tensor([int(f.get("cls_label", -1)) for f in features], dtype=torch.long)
        duration_mask = torch.tensor([int(f.get("duration_mask",0)) for f in features], dtype=torch.float32)
        duration_value = torch.tensor([float(f.get("duration_value",0.0)) for f in features], dtype=torch.float32)
        sem_list = []
        for f in features:
            sem = f.get("sem_emb", None)
            if sem is None or (isinstance(sem, float) and math.isnan(sem)):
                sem = [0.0] * SEM_DIM
            else:
                sem = list(sem)
                if len(sem) < SEM_DIM:
                    sem = sem + [0.0] * (SEM_DIM - len(sem))
                elif len(sem) > SEM_DIM:
                    sem = sem[:SEM_DIM]
            sem_list.append(sem)
        sem_emb = torch.tensor(sem_list, dtype=torch.float32)
        for f in features:
            f.pop("labels", None); f.pop("split_idx", None); f.pop("cls_label", None)
            f.pop("duration_mask", None); f.pop("duration_value", None); f.pop("sem_emb", None)
        batch = self._padder(features)
        max_len = batch["input_ids"].size(1)
        padded = []
        for l in labels:
            if l.numel() < max_len:
                pad = torch.full((max_len - l.numel(),), -100, dtype=torch.long)
                l = torch.cat([l, pad], dim=0)
            else:
                l = l[:max_len]
            padded.append(l)
        batch["labels"]    = torch.stack(padded, dim=0)
        batch["split_idx"] = split_idx
        batch["cls_label"] = cls_label
        batch["duration_mask"] = duration_mask
        batch["duration_value"] = duration_value
        batch["sem_emb"] = sem_emb
        return batch

data_collator = CausalLMPadCollator(tokenizer)

def _cos_sim(a, b):
    an = a / max(np.linalg.norm(a), 1e-12)
    bn = b / max(np.linalg.norm(b), 1e-12)
    return float(np.dot(an, bn))

def _duration_match(pred_text, ref_text):
    pd = _normalize_duration_days(pred_text)
    rd = _normalize_duration_days(ref_text)
    if pd is None or rd is None:
        return False
    tol = max(0.5, 0.15 * max(abs(rd), 1.0))
    return abs(pd - rd) <= tol

def gen_eval_soft_accuracy(model, tokenizer, hf_dataset, max_examples=None, tag="eval"):
    model.eval()
    dev = next(model.parameters()).device
    n_ok, n_all = 0, 0
    rows = hf_dataset.to_pandas()
    if max_examples is not None:
        rows = rows.sample(min(max_examples, len(rows)), random_state=42)
    ref_texts = [ _extract_value(r["output_json"]).strip() for _, r in rows.iterrows() ]
    ref_embs = model_sem.encode(ref_texts, convert_to_numpy=True, normalize_embeddings=True)
    for idx_row, r in enumerate(rows.itertuples(index=False)):
        prompt   = getattr(r, "prompt")
        expected = getattr(r, "output_json")
        ref_val  = ref_texts[idx_row]
        prefix = prompt + '{"treatment_time": "'
        inp = tokenizer(prefix, return_tensors="pt").to(dev)
        with torch.no_grad():
            out_ids = model.generate(
                **inp,
                max_new_tokens=64,
                do_sample=False,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        cont = tokenizer.decode(out_ids[0][inp["input_ids"].size(1):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        pred_val = re.split(r'["\n,}]', cont)[0].strip()
        pred_emb = model_sem.encode(pred_val, convert_to_numpy=True, normalize_embeddings=True)
        sim = _cos_sim(pred_emb, ref_embs[idx_row])
        ok = (sim >= SIM_THRESH) or _duration_match(pred_val, ref_val)
        n_ok += int(ok); n_all += 1
        print(f'{tag}/pair {n_all}: pred="{pred_val}" ref="{ref_val}" cos_sim={sim:.4f} dur_match={_duration_match(pred_val, ref_val)} match={ok}', flush=True)
    model.train()
    return (n_ok / max(1, n_all))

writer = SummaryWriter(tb_dir)

class TBCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        step = state.global_step
        for k in ("loss","eval_loss"):
            if k in logs and isinstance(logs[k], (int, float)):
                writer.add_scalar(f"trainer/{k}", float(logs[k]), step)

class PrintProgressCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % max(1, args.logging_steps) == 0:
            print(f"debug/train_step={state.global_step} completed", flush=True)

class GradNormLogger(TrainerCallback):
    def __init__(self, every=50): self.every = every
    def on_backward_end(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None: return
        total_sq, max_g, n = 0.0, 0.0, 0
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                gn = float(g.float().norm(2).item())
                total_sq += gn * gn
                max_g = max(max_g, gn); n += 1
        total = math.sqrt(total_sq) if n > 0 else 0.0
        if state.global_step % self.every == 0:
            print(f"debug/grad_norm_total={total:.6f} max_grad_norm={max_g:.6f} params_with_grad={n} step={state.global_step}", flush=True)
            writer.add_scalar("grads/total_norm", total, state.global_step)
            writer.add_scalar("grads/max_norm",   max_g,   state.global_step)

print("Initial evaluation on validation with BASE model (semantic soft accuracy + duration match)", flush=True)
base_model.eval()
init_val_acc = gen_eval_soft_accuracy(base_model, tokenizer, datasets["validation"], max_examples=None, tag="initial")
print(f"initial/validation_accuracy_treatment_time={init_val_acc:.4f}", flush=True)
writer.add_scalar("initial/validation_accuracy_treatment_time", init_val_acc, 0)
writer.flush()

hidden_size = getattr(peft_model.config, "hidden_size", None) or getattr(peft_model.config, "n_embd", None)
num_classes = len(label2id)
cls_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_size, num_classes)
)
reg_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size//2),
    nn.ReLU(),
    nn.Linear(hidden_size//2, 1)
)
proj_head = nn.Linear(hidden_size, SEM_DIM)
cls_head.to(next(peft_model.parameters()).device)
reg_head.to(next(peft_model.parameters()).device)
proj_head.to(next(peft_model.parameters()).device)

class MultiTaskTrainer(Trainer):
    def __init__(self, cls_head: nn.Module, reg_head: nn.Module, proj_head: nn.Module, cls_weight: float = 0.7, reg_weight: float = 0.5, sem_weight: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.cls_head = cls_head
        self.reg_head = reg_head
        self.proj_head = proj_head
        self.cls_weight = float(cls_weight)
        self.reg_weight = float(reg_weight)
        self.sem_weight = float(sem_weight)
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.l1 = nn.L1Loss(reduction='none')
        self.cos = nn.CosineEmbeddingLoss(margin=0.0, reduction='mean')

    def create_optimizer(self):
        if self.optimizer is None:
            model_params = [p for p in self.model.parameters() if p.requires_grad]
            head_params  = [p for p in self.cls_head.parameters() if p.requires_grad] + [p for p in self.reg_head.parameters() if p.requires_grad] + [p for p in self.proj_head.parameters() if p.requires_grad]
            optimizer_grouped_parameters = [
                {"params": model_params, "weight_decay": self.args.weight_decay},
                {"params": head_params,  "weight_decay": 0.05},
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        return self.optimizer

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        try:
            ds = datasets.get("validation", None)
            acc = gen_eval_soft_accuracy(self.model, self.tokenizer, ds if ds is not None else datasets["validation"], max_examples=128, tag=metric_key_prefix)
            metrics[f"{metric_key_prefix}_accuracy_treatment_time"] = float(acc)
            step = getattr(self.state, "global_step", -1)
            print(f"{metric_key_prefix}/accuracy_treatment_time={acc:.4f} at step={step}", flush=True)
        except Exception as e:
            print(f"warn/eval_callback_exception: {e}", flush=True)
        return metrics

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        split_idx = inputs.pop("split_idx")
        cls_label = inputs.pop("cls_label")
        duration_mask = inputs.pop("duration_mask")
        duration_value = inputs.pop("duration_value")
        sem_emb = inputs.pop("sem_emb")
        outputs   = model(**inputs, output_hidden_states=True)
        lm_loss   = outputs.loss
        hidden    = outputs.hidden_states[-1]
        B = hidden.size(0)
        idx = torch.clamp(split_idx - 1, min=0)
        gather = hidden[torch.arange(B, device=hidden.device), idx, :]
        if next(self.cls_head.parameters()).device != gather.device:
            self.cls_head.to(gather.device)
            self.reg_head.to(gather.device)
            self.proj_head.to(gather.device)
        logits = self.cls_head(gather)
        mask = (cls_label >= 0)
        if mask.any():
            cls_loss = self.ce(logits[mask], cls_label[mask])
        else:
            cls_loss = torch.tensor(0.0, device=hidden.device)
        pred_log_days = self.reg_head(gather).squeeze(-1)
        reg_loss_all = self.l1(pred_log_days, duration_value.to(pred_log_days.device))
        if duration_mask.any():
            reg_loss = (reg_loss_all * duration_mask.to(pred_log_days.device)).sum() / duration_mask.to(pred_log_days.device).sum()
        else:
            reg_loss = torch.tensor(0.0, device=hidden.device)
        pred_sem = nn.functional.normalize(self.proj_head(gather), dim=-1)
        tgt_sem = nn.functional.normalize(sem_emb.to(pred_sem.device), dim=-1)
        y = torch.ones(pred_sem.size(0), device=pred_sem.device)
        sem_loss = self.cos(pred_sem, tgt_sem, y)
        loss = lm_loss + self.cls_weight * cls_loss + self.reg_weight * reg_loss + self.sem_weight * sem_loss
        if return_outputs:
            return loss, outputs
        return loss

training_args = TrainingArguments(
    output_dir=os.path.join(base_path, "checkpoints_treatment_time"),
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=100,
    save_steps=100,
    save_total_limit=3,
    learning_rate=3e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.05,
    logging_strategy="steps",
    logging_steps=50,
    fp16=(not use_bf16),
    bf16=use_bf16,
    gradient_accumulation_steps=8,
    report_to=["tensorboard"],
    logging_dir=tb_dir,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy_treatment_time",
    greater_is_better=True,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
)

train_tok = datasets["train"].map(tokenize_fn, remove_columns=datasets["train"].column_names)
val_tok   = datasets["validation"].map(tokenize_fn, remove_columns=datasets["validation"].column_names)

trainer = MultiTaskTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=val_tok,
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),
        TBCallback(),
        PrintProgressCallback(),
        GradNormLogger(every=50)
    ],
    cls_head=cls_head,
    reg_head=reg_head,
    proj_head=proj_head,
    cls_weight=args.cls_loss_weight,
    reg_weight=args.reg_loss_weight,
    sem_weight=args.sem_loss_weight
)

print("Begin training", flush=True)
trainer.train()
writer.flush()

print("Save adapter only", flush=True)
trainer.model.save_pretrained(adapter_out_dir)
tokenizer.save_pretrained(adapter_out_dir)

print("Evaluate on validation with generation (semantic soft accuracy + duration match)", flush=True)
val_acc = gen_eval_soft_accuracy(trainer.model, tokenizer, datasets["validation"], max_examples=None, tag="final_val")
print(f"final/validation_accuracy_treatment_time={val_acc:.4f}", flush=True)
writer.add_scalar("final/validation_accuracy_treatment_time", val_acc, trainer.state.global_step)
writer.flush()

print("Generate on test set (semantic soft accuracy + duration match)", flush=True)
dev = next(trainer.model.parameters()).device
rows = datasets["test"].to_pandas()
n_ok, n_all = 0, 0
ref_texts = [ _extract_value(r["output_json"]).strip() for _, r in rows.iterrows() ]
ref_embs = model_sem.encode(ref_texts, convert_to_numpy=True, normalize_embeddings=True)
for i, r in enumerate(rows.itertuples(index=False)):
    prompt   = getattr(r, "prompt")
    expected = getattr(r, "output_json")
    ref_val  = ref_texts[i]
    prefix = prompt + '{"treatment_time": "'
    inp = tokenizer(prefix, return_tensors="pt").to(dev)
    with torch.no_grad():
        out_ids = trainer.model.generate(
            **inp,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    cont = tokenizer.decode(out_ids[0][inp["input_ids"].size(1):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    pred_val = re.split(r'["\n,}]', cont)[0].strip()
    pred_json = f'{{"treatment_time": "{_escape_json_val(pred_val)}"}}'
    pred_emb = model_sem.encode(pred_val, convert_to_numpy=True, normalize_embeddings=True)
    sim = _cos_sim(pred_emb, ref_embs[i])
    ok = (sim >= SIM_THRESH) or _duration_match(pred_val, ref_val)
    print(f"--- Predicted output {i+1}: {pred_json}", flush=True)
    print(f"--- Expected output  {i+1}: {expected}", flush=True)
    print(f"--- cos_sim={sim:.4f} dur_match={_duration_match(pred_val, ref_val)} match={ok}", flush=True)
    print("-"*50, flush=True)
    n_ok += int(ok); n_all += 1

test_acc = (n_ok / max(1, n_all))
print(f"final/test_accuracy_treatment_time={test_acc:.4f}", flush=True)
writer.add_scalar("final/test_accuracy_treatment_time", test_acc, trainer.state.global_step)
writer.close()
