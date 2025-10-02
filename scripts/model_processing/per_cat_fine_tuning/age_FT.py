import os, re, json, math, argparse, random
import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorWithPadding, EarlyStoppingCallback, TrainerCallback
)
from peft import LoraConfig, get_peft_model

parser = argparse.ArgumentParser(description="Fine-tune age (multitask), eval ponctuation-insensitive, case-sensitive")
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--cls_loss_weight", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
base_path = args.base_path

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

train_file = os.path.join(base_path, "finetune_data_train.csv")
val_file   = os.path.join(base_path, "finetune_data_val.csv")
test_file  = os.path.join(base_path, "finetune_data_test.csv")
model_name = os.path.join(base_path, "Mistral-7B-Instruct-v0.3")

tb_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/logs2/tensorboard_age"
adapter_out_dir = os.path.join(base_path, "cat_age")
os.makedirs(tb_dir, exist_ok=True); os.makedirs(adapter_out_dir, exist_ok=True)

os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if use_bf16 else torch.float16

print("Load base model", flush=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=dtype,
    device_map="auto",
    low_cpu_mem_usage=True
)
base_model.config.pad_token_id = tokenizer.pad_token_id
if base_model.get_input_embeddings().num_embeddings != len(tokenizer):
    base_model.resize_token_embeddings(len(tokenizer))

print("Configure LoRA", flush=True)
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.3,
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
        v = j.get("age","")
        return "" if v is None else str(v)
    except Exception:
        m = re.search(r'"age"\s*:\s*"([^"]*)"', str(txt))
        return m.group(1) if m else str(txt)

_CANON = re.compile(r'(?i)\b(\d{1,3})\s*(y|yr|yrs|yo|year|years|ans|años|anni|лет|岁)\b|\bage(?:d)?\s*[:=]?\s*(\d{1,3})\b|\b(\d{1,3})\s*(?:years?\s*old)\b')
def _canon_age(s):
    s = str(s).strip()
    if s.lower() in {"", "na", "n/a", "none", "null", "unknown", "unk"}:
        return "unknown"
    m = _CANON.search(s)
    if not m:
        return "unknown"
    n = None
    for g in (1,3,4):
        if m.group(g):
            n = int(m.group(g))
            break
    if n is None or n < 0 or n > 120:
        return "unknown"
    return f"{n} years"

def _strip_punct_keep_case(s):
    return re.sub(r'[^A-Za-z0-9]+', '', str(s))

def _extract_age_candidates(text):
    text = str(text)
    pats = [
        r'(?i)\bage(?:d)?\s*[:=]?\s*(\d{1,3})\b',
        r'(?i)\b(\d{1,3})\s*(?:years?|y|yr|yo)\b',
        r'(?i)\b(\d{1,3})\s*(?:years?\s*old)\b'
    ]
    nums = []
    for p in pats:
        for m in re.finditer(p, text):
            try:
                n = int(m.group(1))
                if 0 <= n <= 120:
                    nums.append(n)
            except:
                pass
    return sorted(set(nums))

def _prompt_has_age(prompt):
    return len(_extract_age_candidates(prompt)) > 0

def _score_continuation(model, tokenizer, prefix, continuation, device):
    with torch.no_grad():
        ids_pref = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        ids_cont = tokenizer(continuation, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
        inp = torch.cat([ids_pref, ids_cont], dim=1)
        out = model(input_ids=inp)
        logits = out.logits[:, :-1, :]
        tgt = inp[:, 1:]
        start = ids_pref.size(1) - 1
        logits = logits[:, start:start+ids_cont.size(1), :]
        tgt = tgt[:, start:start+ids_cont.size(1)]
        logprobs = torch.log_softmax(logits, dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        return float(logprobs.sum().item())

def _argmax_over_candidates(model, tokenizer, prompt, device):
    prefix = prompt + '{"age": "'
    cands_num = _extract_age_candidates(prompt)
    cand_strs = [f"{n} years" for n in cands_num]
    cand_strs.append("unknown")
    scores = []
    for c in cand_strs:
        s = _score_continuation(model, tokenizer, prefix, c + '"}', device)
        scores.append((s, c))
    scores.sort(reverse=True, key=lambda x: x[0])
    return scores[0][1]

def _cmp_ignore_punct_keep_case(a, b):
    return _strip_punct_keep_case(a) == _strip_punct_keep_case(b)

def gen_eval_accuracy(model, tokenizer, hf_dataset, max_examples=None, tag="eval"):
    model.eval()
    dev = next(model.parameters()).device
    n_ok, n_all = 0, 0
    rows = hf_dataset.to_pandas()
    if max_examples is not None:
        rows = rows.sample(min(max_examples, len(rows)), random_state=42)
    for i, r in rows.iterrows():
        prompt   = r["prompt"]
        expected = r["output_json"]
        ref_val  = _canon_age(_extract_value(expected))
        pred_val = _argmax_over_candidates(model, tokenizer, prompt, dev) if _prompt_has_age(prompt) else "unknown"
        ok = _cmp_ignore_punct_keep_case(pred_val, ref_val)
        n_ok += int(ok); n_all += 1
        print(f'{tag}/pair {n_all}: pred="{pred_val}" ref="{ref_val}" pred_norm="{_strip_punct_keep_case(pred_val)}" ref_norm="{_strip_punct_keep_case(ref_val)}" match={ok}', flush=True)
    model.train()
    return (n_ok / max(1, n_all))

for _df in (df_train, df_val, df_test):
    _df["output_raw"]  = _df["output"].astype(str)
    _df["output_raw"]  = _df["output_raw"].map(_canon_age)
    _df["output_json"] = _df["output_raw"].apply(lambda v: f'{{"age": "{_escape_json_val(v)}"}}')

datasets = {
    "train": Dataset.from_pandas(df_train[["prompt","output_json"]]),
    "validation": Dataset.from_pandas(df_val[["prompt","output_json"]]),
    "test": Dataset.from_pandas(df_test[["prompt","output_json"]])
}
print({k: len(v) for k,v in datasets.items()}, flush=True)

validation_raw = datasets["validation"]

def tokenize_fn(example):
    prompt   = example["prompt"].strip()
    out_json = example["output_json"].strip()
    m = re.search(r'"age"\s*:\s*"([^"]*)"', out_json)
    value_str  = m.group(1) if m else ""
    prefix_str = '{"age": "'
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
    return {"input_ids": ids, "attention_mask": attn, "labels": labels}

tokenized = {k: v.map(tokenize_fn, remove_columns=v.column_names) for k, v in datasets.items()}

class CausalLMPadCollator:
    def __init__(self, tokenizer):
        self._padder = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    def __call__(self, features):
        labels = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        for f in features:
            f.pop("labels", None)
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
        batch["labels"] = torch.stack(padded, dim=0)
        return batch

data_collator = CausalLMPadCollator(tokenizer)

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

class GenEvalCallback(TrainerCallback):
    def __init__(self, eval_ds, max_examples=128):
        self.eval_ds = eval_ds; self.max_examples = max_examples
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        model     = kwargs.get("model", None)
        tokenizer = kwargs.get("tokenizer", None)
        if model is None or tokenizer is None: return
        acc = gen_eval_accuracy(model, tokenizer, self.eval_ds, max_examples=self.max_examples, tag="eval")
        print(f"eval/accuracy_age={acc:.4f} at step={state.global_step}", flush=True)
        writer.add_scalar("eval/accuracy_age", acc, state.global_step)
        if isinstance(metrics, dict):
            metrics["eval_accuracy_age"] = float(acc)

print("Initial evaluation on validation with BASE model (punctuation-insensitive, case-sensitive)", flush=True)
base_model.eval()
init_val_acc = gen_eval_accuracy(base_model, tokenizer, datasets["validation"], max_examples=None, tag="initial")
print(f"initial/validation_accuracy_age={init_val_acc:.4f}", flush=True)
writer.add_scalar("initial/validation_accuracy_age", init_val_acc, 0)
writer.flush()

training_args = TrainingArguments(
    output_dir=os.path.join(base_path, "checkpoints_age"),
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=250,
    save_steps=250,
    save_total_limit=1,
    learning_rate=1e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=1,
    weight_decay=0.1,
    logging_strategy="steps",
    logging_steps=50,
    fp16=(not use_bf16),
    bf16=use_bf16,
    gradient_accumulation_steps=8,
    report_to=["tensorboard"],
    logging_dir=tb_dir,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy_age",
    greater_is_better=True,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
)

class MyTrainer(Trainer):
    def __init__(self, *args, eval_raw=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_raw = eval_raw
    def evaluate(self, eval_dataset=None, **kwargs):
        metrics = super().evaluate(eval_dataset=eval_dataset, **kwargs)
        raw = self.eval_raw if self.eval_raw is not None else eval_dataset
        try:
            if hasattr(raw, "column_names") and {"prompt","output_json"}.issubset(set(raw.column_names)):
                acc = gen_eval_accuracy(self.model, self.tokenizer, raw, max_examples=128, tag="eval")
                metrics["eval_accuracy_age"] = float(acc)
            else:
                print("warn: eval_raw missing 'prompt'/'output_json'; skipping age accuracy", flush=True)
        except Exception as e:
            print(f"warn: custom eval failed: {e}", flush=True)
        return metrics

trainer = MyTrainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=1),
        TBCallback(),
        PrintProgressCallback(),
        GradNormLogger(every=50),
        GenEvalCallback(validation_raw, max_examples=128)
    ],
    eval_raw=validation_raw,
)

print("Begin training", flush=True)
trainer.train()
writer.flush()

print("Save adapter only", flush=True)
trainer.model.save_pretrained(adapter_out_dir)
tokenizer.save_pretrained(adapter_out_dir)

print("Evaluate on validation with generation (punctuation-insensitive, case-sensitive)", flush=True)
val_acc = gen_eval_accuracy(trainer.model, tokenizer, datasets["validation"], max_examples=None, tag="final_val")
print(f"final/validation_accuracy_age={val_acc:.4f}", flush=True)
writer.add_scalar("final/validation_accuracy_age", val_acc, trainer.state.global_step)
writer.flush()

print("Generate on test set (punctuation-insensitive, case-sensitive)", flush=True)
dev = next(trainer.model.parameters()).device
rows = datasets["test"].to_pandas()
n_ok, n_all = 0, 0
for i, r in rows.iterrows():
    prompt   = r["prompt"]
    expected = r["output_json"]
    ref_val  = _canon_age(_extract_value(expected))
    pred_val = _argmax_over_candidates(trainer.model, tokenizer, prompt, dev) if _prompt_has_age(prompt) else "unknown"
    pred_json = f'{{"age": "{_escape_json_val(pred_val)}"}}'
    ok = _cmp_ignore_punct_keep_case(pred_val, ref_val)
    print(f"--- Predicted output {i+1}: {pred_json}", flush=True)
    print(f"--- Expected output  {i+1}: {expected}", flush=True)
    print(f"--- pred_norm=\"{_strip_punct_keep_case(pred_val)}\" ref_norm=\"{_strip_punct_keep_case(ref_val)}\" match={ok}", flush=True)
    print("-"*50, flush=True)
    n_ok += int(ok); n_all += 1

test_acc = (n_ok / max(1, n_all))
print(f"final/test_accuracy_age={test_acc:.4f}", flush=True)
writer.add_scalar("final/test_accuracy_age", test_acc, trainer.state.global_step)
writer.close()
