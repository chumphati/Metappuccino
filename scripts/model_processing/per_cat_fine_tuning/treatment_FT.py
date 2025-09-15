import os, re, json, argparse, random
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description="2-epoch simple fine-tune with anti-overfit tests")
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
base_path = args.base_path

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

train_file = os.path.join(base_path, "finetune_data_train.csv")
val_file   = os.path.join(base_path, "finetune_data_val.csv")
test_file  = os.path.join(base_path, "finetune_data_test.csv")
model_name = os.path.join(base_path, "Mistral-7B-Instruct-v0.3")
adapter_out_dir = os.path.join(base_path, "adapter_treatment_simple")
os.makedirs(adapter_out_dir, exist_ok=True)

os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if use_bf16 else torch.float32

print("Loading model")
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    device_map="auto",
    low_cpu_mem_usage=True
)
base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.use_cache = False
if tokenizer.pad_token_id is not None and base_model.get_input_embeddings().num_embeddings != len(tokenizer):
    base_model.resize_token_embeddings(len(tokenizer))

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj']
)
model = get_peft_model(base_model, peft_config)
model.train()

def read_two_col_csv(path):
    try:
        df = pd.read_csv(path, engine="python", sep=None, dtype=str, keep_default_na=False)
        if {"prompt","output"}.issubset(df.columns): return df[["prompt","output"]].astype(str)
    except Exception:
        pass
    try:
        df = pd.read_csv(path, engine="python", sep=";", dtype=str, keep_default_na=False)
        if {"prompt","output"}.issubset(df.columns): return df[["prompt","output"]].astype(str)
    except Exception:
        pass
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        _ = f.readline()
        for line in f:
            line = line.rstrip("\n\r")
            if not line: continue
            if "\t" in line:
                parts = line.split("\t"); prompt = "\t".join(parts[:-1]); out = parts[-1]
            elif ";" in line:
                parts = line.split(";"); prompt = ";".join(parts[:-1]); out = parts[-1]
            else:
                idx = line.rfind(",")
                prompt, out = (line, "") if idx == -1 else (line[:idx], line[idx+1:])
            rows.append({"prompt": prompt, "output": out})
    return pd.DataFrame(rows)

df_train = read_two_col_csv(train_file)
df_val   = read_two_col_csv(val_file)
df_test  = read_two_col_csv(test_file)

for _df in (df_train, df_val, df_test):
    _df["prompt"] = _df["prompt"].astype(str)
    _df["output"] = _df["output"].astype(str)

before = len(df_train)
df_train = df_train.drop_duplicates(subset=["prompt"]).copy()
df_train = df_train[df_train["output"].astype(str).str.strip()!=""].copy()
print(f"Sizes train/val/test: {len(df_train)}/{len(df_val)}/{len(df_test)} (dropped {before-len(df_train)})")

def _escape_json_val(v):
    return str(v).replace('\\','\\\\').replace('"','\\\"')

def _extract_value(txt):
    try:
        j = json.loads(txt); v = j.get("treatment", "")
        return "" if v is None else str(v)
    except Exception:
        m = re.search(r'"treatment"\s*:\s*"([^"]*)"', str(txt))
        return m.group(1) if m else str(txt)

UNK_TOKS = {"unknown","unk","n/a","na","not specified","not stated","no treatment","none","no intervention","no therapy","untreated"}

for _df in (df_train, df_val, df_test):
    _df["output_json"] = _df["output"].apply(lambda v: f'{{"treatment": "{_escape_json_val(v)}"}}')
    _df["norm_val"] = _df["output"].str.strip().str.lower()

raw_hf = {
    "train": Dataset.from_pandas(df_train[["prompt","output_json","norm_val"]]),
    "validation": Dataset.from_pandas(df_val[["prompt","output_json","norm_val"]]),
    "test": Dataset.from_pandas(df_test[["prompt","output_json","norm_val"]]),
}

def tokenize_fn(example):
    prompt   = example["prompt"].strip()
    out_json = example["output_json"].strip()
    m = re.search(r'"treatment"\s*:\s*"([^"]*)"', out_json)
    value_str  = m.group(1) if m else ""
    prefix_str = '{"treatment": "'
    suffix_str = '"}'
    max_in, max_out, max_total = 3000, 96, 4096
    in_enc   = tokenizer(prompt, truncation=True, padding=False, max_length=max_in, add_special_tokens=False)
    pref_ids = tokenizer(prefix_str, add_special_tokens=False)["input_ids"]
    val_ids  = tokenizer(value_str, truncation=True, padding=False, max_length=max_out, add_special_tokens=False)["input_ids"]
    suff_ids = tokenizer(suffix_str, add_special_tokens=False)["input_ids"]
    ids  = in_enc["input_ids"] + pref_ids + val_ids + suff_ids
    attn = [1]*len(ids)
    teach_value = (value_str.strip().lower() not in UNK_TOKS) and (len(value_str.strip()) > 0)
    labels = [-100]*len(in_enc["input_ids"])
    labels += [-100]*len(pref_ids)
    if teach_value:
        labels += val_ids
        if len(suff_ids) > 0:
            labels += [suff_ids[0]] + [-100]*(len(suff_ids)-1)
    else:
        labels += [-100]*len(val_ids)
        labels += [-100]*len(suff_ids)
    if len(ids) > max_total:
        ids  = ids[:max_total]; attn = attn[:max_total]; labels = labels[:max_total]
    return {"input_ids": ids, "attention_mask": attn, "labels": labels}

tok = {k: v.map(tokenize_fn, remove_columns=v.column_names) for k, v in raw_hf.items()}

class CLMPadCollator:
    def __init__(self, tokenizer): self.padder = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    def __call__(self, features):
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        for f in features: f.pop("labels", None)
        batch = self.padder(features)
        max_len = batch["input_ids"].size(1)
        padded = []
        for l in labels:
            if l.numel() < max_len:
                pad = torch.full((max_len - l.numel(),), -100, dtype=torch.long)
                l = torch.cat([l, pad], dim=0)
            else:
                l = l[:max_len]
            padded.append(l)
        batch["labels"] = torch.stack(padded, dim=0)
        return batch

collator = CLMPadCollator(tokenizer)

sem_model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
SIM_THRESH = 0.4

def _cos(a, b):
    an = a / max(np.linalg.norm(a), 1e-12); bn = b / max(np.linalg.norm(b), 1e-12)
    return float(np.dot(an, bn))

@torch.no_grad()
def predict_values(model_for_eval, ds, max_new_tokens=64):
    model_for_eval.eval()
    dev = next(model_for_eval.parameters()).device
    preds = []
    rows = ds.to_pandas()
    for _, r in rows.iterrows():
        prompt = r["prompt"]
        prefix = prompt + '{"treatment": "'
        inp = tokenizer(prefix, return_tensors="pt").to(dev)
        out_ids = model_for_eval.generate(
            **inp,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        cont = tokenizer.decode(out_ids[0][inp["input_ids"].size(1):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        pred = re.split(r'["\n,}]', cont)[0].strip()
        preds.append(pred)
    model_for_eval.train()
    return preds

def eval_soft_acc(preds, refs):
    ref_embs = sem_model.encode(refs, convert_to_numpy=True, normalize_embeddings=True)
    acc, n = 0, len(refs)
    for i, p in enumerate(preds):
        sim = _cos(sem_model.encode(p, convert_to_numpy=True, normalize_embeddings=True), ref_embs[i])
        acc += int(sim >= SIM_THRESH)
    return acc / max(1, n)

def eval_exact_match(preds, refs):
    norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
    acc = sum(1 for p, r in zip(preds, refs) if norm(p) == norm(r))
    return acc / max(1, len(refs))

def majority_baseline(train_vals, n):
    vc = pd.Series(train_vals).value_counts()
    top = next((v for v in vc.index if v not in UNK_TOKS), "unknown")
    return [top]*n, top

def permutation_test(preds, refs):
    refs_shuf = refs.copy()
    rng = np.random.default_rng(42)
    rng.shuffle(refs_shuf)
    return eval_soft_acc(preds, refs_shuf)

def seen_unseen_split(train_vals_set, eval_refs):
    seen_mask = [ (r in train_vals_set) for r in eval_refs ]
    idx_seen  = [i for i,b in enumerate(seen_mask) if b]
    idx_unseen= [i for i,b in enumerate(seen_mask) if not b]
    return idx_seen, idx_unseen

print("Computing pre-train baseline on val")
init_val_preds_base = predict_values(base_model, raw_hf["validation"])
val_refs = [ _extract_value(x) for x in raw_hf["validation"]["output_json"] ]
init_val_soft = eval_soft_acc(init_val_preds_base, val_refs)
init_val_exact = eval_exact_match(init_val_preds_base, val_refs)
print(f"Pre-train val soft acc: {init_val_soft:.4f}, exact: {init_val_exact:.4f}")

training_args = TrainingArguments(
    output_dir=os.path.join(base_path, "chk_treatment_simple"),
    num_train_epochs=2,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    weight_decay=0.05,
    warmup_ratio=0.06,
    lr_scheduler_type="cosine",
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="no",
    bf16=use_bf16,
    fp16=False,
    max_grad_norm=1.0,
    remove_unused_columns=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tok["train"],
    eval_dataset=tok["validation"],
    tokenizer=tokenizer,
    data_collator=collator,
)

print("Training")
trainer.train()

print("Evaluating")
val_preds = predict_values(trainer.model, raw_hf["validation"])
val_soft = eval_soft_acc(val_preds, val_refs)
val_exact = eval_exact_match(val_preds, val_refs)

train_preds = predict_values(trainer.model, raw_hf["train"], max_new_tokens=64)
train_refs = [ _extract_value(x) for x in raw_hf["train"]["output_json"] ]
train_soft = eval_soft_acc(train_preds, train_refs)

test_preds = predict_values(trainer.model, raw_hf["test"])
test_refs = [ _extract_value(x) for x in raw_hf["test"]["output_json"] ]
test_soft = eval_soft_acc(test_preds, test_refs)
test_exact = eval_exact_match(test_preds, test_refs)

train_vals_norm = df_train["norm_val"].tolist()
val_major_preds, top_val = majority_baseline(train_vals_norm, len(val_refs))
maj_soft = eval_soft_acc(val_major_preds, val_refs)

perm_soft = permutation_test(val_preds, val_refs)

train_vals_set = set(train_vals_norm)
val_refs_norm = [ s.strip().lower() for s in val_refs ]
idx_seen, idx_unseen = seen_unseen_split(train_vals_set, val_refs_norm)
def _sub(lst, idx): return [lst[i] for i in idx]
val_soft_seen   = eval_soft_acc(_sub(val_preds, idx_seen), _sub(val_refs, idx_seen)) if idx_seen else None
val_soft_unseen = eval_soft_acc(_sub(val_preds, idx_unseen), _sub(val_refs, idx_unseen)) if idx_unseen else None

improvement = val_soft - init_val_soft
overfit_gap = train_soft - val_soft
flags = []
if improvement < 0.03:
    flags.append(f"Low progress (+{improvement:.3f})")
if overfit_gap > 0.15:
    flags.append(f"Possible overfit (gap={overfit_gap:.3f})")
if val_soft < maj_soft:
    flags.append("Under majority baseline")

report = {
    "pretrain_val_soft": round(init_val_soft, 4),
    "posttrain_train_soft": round(train_soft, 4),
    "posttrain_val_soft": round(val_soft, 4),
    "posttrain_val_exact": round(val_exact, 4),
    "posttrain_test_soft": round(test_soft, 4),
    "posttrain_test_exact": round(test_exact, 4),
    "val_majority_soft": round(maj_soft, 4),
    "val_permutation_soft": round(perm_soft, 4),
    "improvement_val_soft": round(improvement, 4),
    "overfit_gap_train_minus_val": round(overfit_gap, 4),
    "val_seen_soft": (None if val_soft_seen is None else round(val_soft_seen, 4)),
    "val_unseen_soft": (None if val_soft_unseen is None else round(val_soft_unseen, 4)),
    "val_seen_count": len(idx_seen),
    "val_unseen_count": len(idx_unseen),
    "majority_label": top_val,
    "flags": flags
}
print("Report:")
print(json.dumps(report, indent=2))
with open(os.path.join(base_path, "overfit_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)

trainer.model.save_pretrained(adapter_out_dir)
tokenizer.save_pretrained(adapter_out_dir)
print("Saved adapter")
