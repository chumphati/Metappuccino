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

parser = argparse.ArgumentParser(description="Fine-tune treatment (multitask) with soft semantic accuracy")
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

tb_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/logs11/tensorboard_treatment"
adapter_out_dir = os.path.join(base_path, "cat_treatment")
os.makedirs(tb_dir, exist_ok=True); os.makedirs(adapter_out_dir, exist_ok=True)

os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if use_bf16 else torch.float16

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    device_map="auto",
    low_cpu_mem_usage=True
)
base_model.config.pad_token_id = tokenizer.pad_token_id
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
peft_model = get_peft_model(base_model, peft_config)
peft_model.train()

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

df_train = df_train.drop_duplicates(subset=["prompt","output"]).copy()
df_train = df_train[df_train["output"].astype(str).str.strip()!=""].copy()

INSTR = (
  "You are extracting the definitive clinical treatment (drug or procedure).\n"
  "Do not return research tools, preclinical compounds (e.g., JQ1, siRNA), anesthetics, or vague phrases.\n"
  "Return only the exact treatment phrase.\n\n"
)

def _escape_json_val(v):
    return str(v).replace('\\','\\\\').replace('"','\\"')

def _extract_value(txt):
    try:
        j = json.loads(txt)
        v = j.get("treatment","")
        return "" if v is None else str(v)
    except Exception:
        m = re.search(r'"treatment"\s*:\s*"([^"]*)"', str(txt))
        return m.group(1) if m else str(txt)

for _df in (df_train, df_val, df_test):
    _df["output_raw"]  = _df["output"].astype(str)
    _df["output_json"] = _df["output_raw"].apply(lambda v: f'{{"treatment": "{_escape_json_val(v)}"}}')

labels_train = sorted(set(df_train["output_raw"].tolist()))
label2id = {lbl:i for i,lbl in enumerate(labels_train)}
id2label = {i:lbl for lbl,i in label2id.items()}

for _df in (df_train, df_val, df_test):
    _df["cls_label"] = _df["output_raw"].map(lambda x: label2id.get(x, -1))

candidate_bank_train = sorted(set(df_train["output_raw"].tolist()))
candidate_bank_eval = sorted(set(df_train["output_raw"].tolist()) | set(df_val["output_raw"].tolist()))

datasets = {
    "train": Dataset.from_pandas(df_train[["prompt","output_json","cls_label"]]),
    "validation": Dataset.from_pandas(df_val[["prompt","output_json","cls_label"]]),
    "test": Dataset.from_pandas(df_test[["prompt","output_json","cls_label"]])
}

def tokenize_fn(example):
    prompt   = INSTR + example["prompt"].strip()
    out_json = example["output_json"].strip()
    m = re.search(r'"treatment"\s*:\s*"([^"]*)"', out_json)
    value_str  = m.group(1) if m else ""
    prefix_str = '{"treatment": "'
    suffix_str = '"}'
    max_in, max_out, max_total = 3500, 350, 3850
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
    return {
        "input_ids": ids,
        "attention_mask": attn,
        "labels": labels,
        "split_idx": len(in_enc["input_ids"]),
        "cls_label": int(example.get("cls_label", -1)),
    }

tokenized = {k: v.map(tokenize_fn, remove_columns=v.column_names) for k, v in datasets.items()}

class CausalLMPadCollator:
    def __init__(self, tokenizer):
        self._padder = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    def __call__(self, features):
        labels    = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        split_idx = torch.tensor([int(f.get("split_idx", 0)) for f in features], dtype=torch.long)
        cls_label = torch.tensor([int(f.get("cls_label", -1)) for f in features], dtype=torch.long)
        for f in features:
            f.pop("labels", None); f.pop("split_idx", None); f.pop("cls_label", None)
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
        return batch

data_collator = CausalLMPadCollator(tokenizer)

sem_model_name = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
model_sem = SentenceTransformer(sem_model_name)
SIM_THRESH = 0.30

SYN = {
  "radioactive iodine": "radioiodine ablation",
  "radioiodine": "radioiodine ablation",
  "raia": "radioiodine ablation",
  "parathyroid resection": "parathyroidectomy",
  "thiamazole": "methimazole",
  "thyroxine": "levothyroxine",
  "thymus resection": "thymectomy",
  "magnesium": "magnesium sulfate"
}

def _norm_syn(s):
    s = str(s).strip().lower()
    return SYN.get(s, s)

def _cos_sim(a, b):
    an = a / max(np.linalg.norm(a), 1e-12)
    bn = b / max(np.linalg.norm(b), 1e-12)
    return float(np.dot(an, bn))

def _sum_logprob_for_suffix(model, tokenizer, prefix, suffix):
    dev = next(model.parameters()).device
    pref_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    suff_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
    full_ids = pref_ids + suff_ids
    inp = torch.tensor([full_ids], dtype=torch.long, device=dev)
    att = torch.ones_like(inp)
    with torch.no_grad():
        logits = model(input_ids=inp, attention_mask=att).logits
        tgt = inp[:, 1:]
        logp = torch.log_softmax(logits[:, :-1, :], dim=-1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    start = max(len(pref_ids) - 1, 0)
    end = start + len(suff_ids)
    return float(logp[0, start:end].sum().item())

def pick_by_logprob(model, tokenizer, prompt, candidates):
    prefix = prompt + '{"treatment": "'
    best_cand, best_score = None, -1e30
    for c in candidates:
        score = _sum_logprob_for_suffix(model, tokenizer, prefix, c + '"}')
        if score > best_score:
            best_score, best_cand = score, c
    return best_cand if best_cand is not None else ""

def build_prompt(p):
    return INSTR + p

def gen_eval_soft_accuracy(model, tokenizer, hf_dataset, max_examples=None, tag="eval"):
    model.eval()
    dev = next(model.parameters()).device
    n_ok, n_all = 0, 0
    rows = hf_dataset.to_pandas()
    if max_examples is not None:
        rows = rows.sample(min(max_examples, len(rows)), random_state=42)
    ref_texts = [_norm_syn(_extract_value(r["output_json"])) for _, r in rows.iterrows()]
    ref_embs = model_sem.encode(ref_texts, convert_to_numpy=True, normalize_embeddings=True)
    for idx_row, r in enumerate(rows.itertuples(index=False)):
        prompt   = build_prompt(getattr(r, "prompt"))
        ref_val  = ref_texts[idx_row]
        pred_val = pick_by_logprob(model, tokenizer, prompt, candidate_bank_eval)
        pred_val_norm = _norm_syn(pred_val)
        pred_emb = model_sem.encode(pred_val_norm, convert_to_numpy=True, normalize_embeddings=True)
        sim = _cos_sim(pred_emb, ref_embs[idx_row])
        ok = sim >= SIM_THRESH
        n_ok += int(ok); n_all += 1
        print(f'{tag}/pair {n_all}: pred="{pred_val}" ref="{ref_val}" cos_sim={sim:.4f} match={ok}', flush=True)
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

class GenEvalCallback(TrainerCallback):
    def __init__(self, eval_ds, max_examples=128):
        self.eval_ds = eval_ds; self.max_examples = max_examples
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        model     = kwargs.get("model", None)
        tokenizer = kwargs.get("tokenizer", None)
        if model is None or tokenizer is None: return
        acc = gen_eval_soft_accuracy(model, tokenizer, self.eval_ds, max_examples=self.max_examples, tag="eval")
        print(f"eval/accuracy_treatment={acc:.4f} at step={state.global_step}", flush=True)
        writer.add_scalar("eval/accuracy_treatment", acc, state.global_step)
        if isinstance(metrics, dict):
            metrics["eval_accuracy_treatment"] = float(acc)

print("Initial evaluation on validation with BASE model (semantic soft accuracy)", flush=True)
base_model.eval()
init_val_acc = gen_eval_soft_accuracy(base_model, tokenizer, datasets["validation"], max_examples=None, tag="initial")
print(f"initial/validation_accuracy_treatment={init_val_acc:.4f}", flush=True)
writer.add_scalar("initial/validation_accuracy_treatment", init_val_acc, 0)
writer.flush()

hidden_size = getattr(peft_model.config, "hidden_size", None) or getattr(peft_model.config, "n_embd", None)
num_classes = len(label2id)
cls_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(hidden_size, num_classes)
)
cls_head.to(next(peft_model.parameters()).device)

class MultiTaskTrainer(Trainer):
    def __init__(self, cls_head: nn.Module, cls_weight: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.cls_head = cls_head
        self.cls_weight = float(cls_weight)
        self.ce = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)
    def create_optimizer(self):
        if self.optimizer is None:
            model_params = [p for p in self.model.parameters() if p.requires_grad]
            head_params  = [p for p in self.cls_head.parameters() if p.requires_grad]
            optimizer_grouped_parameters = [
                {"params": model_params, "weight_decay": self.args.weight_decay},
                {"params": head_params,  "weight_decay": 0.01},
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        return self.optimizer
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        try:
            ds = datasets.get("validation", None)
            acc = gen_eval_soft_accuracy(self.model, self.tokenizer, ds if ds is not None else datasets["validation"], max_examples=128, tag=metric_key_prefix)
            metrics[f"{metric_key_prefix}_accuracy_treatment"] = float(acc)
            step = getattr(self.state, "global_step", -1)
            print(f"{metric_key_prefix}/accuracy_treatment={acc:.4f} at step={step}", flush=True)
        except Exception as e:
            print(f"warn/eval_callback_exception: {e}", flush=True)
        return metrics
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        split_idx = inputs.pop("split_idx")
        cls_label = inputs.pop("cls_label")
        outputs   = model(**inputs, output_hidden_states=True)
        lm_loss   = outputs.loss
        hidden    = outputs.hidden_states[-1]
        B = hidden.size(0)
        idx = torch.clamp(split_idx - 1, min=0)
        gather = hidden[torch.arange(B, device=hidden.device), idx, :]
        if next(self.cls_head.parameters()).device != gather.device:
            self.cls_head.to(gather.device)
        logits = self.cls_head(gather)
        mask = (cls_label >= 0)
        if mask.any():
            cls_loss = self.ce(logits[mask], cls_label[mask])
        else:
            cls_loss = torch.tensor(0.0, device=hidden.device)
        loss = lm_loss + self.cls_weight * cls_loss
        if return_outputs:
            return loss, outputs
        return loss

training_args = TrainingArguments(
    output_dir=os.path.join(base_path, "checkpoints_treatment"),
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    save_total_limit=3,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=2,
    weight_decay=0.01,
    logging_strategy="steps",
    logging_steps=50,
    fp16=(not use_bf16),
    bf16=use_bf16,
    gradient_accumulation_steps=16,
    report_to=["tensorboard"],
    logging_dir=tb_dir,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy_treatment",
    greater_is_better=True,
    warmup_ratio=0.05,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
)

trainer = MultiTaskTrainer(
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
        GenEvalCallback(datasets["validation"], max_examples=128)
    ],
    cls_head=cls_head,
    cls_weight=args.cls_loss_weight
)

print("Begin training", flush=True)
trainer.train()
writer.flush()

print("Save adapter only", flush=True)
trainer.model.save_pretrained(adapter_out_dir)
tokenizer.save_pretrained(adapter_out_dir)

print("Evaluate on validation with candidate re-ranking (semantic soft accuracy)", flush=True)
val_acc = gen_eval_soft_accuracy(trainer.model, tokenizer, datasets["validation"], max_examples=None, tag="final_val")
print(f"final/validation_accuracy_treatment={val_acc:.4f}", flush=True)
writer.add_scalar("final/validation_accuracy_treatment", val_acc, trainer.state.global_step)
writer.flush()

print("Generate on test set with candidate re-ranking (semantic soft accuracy)", flush=True)
rows = datasets["test"].to_pandas()
n_ok, n_all = 0, 0
ref_texts = [_norm_syn(_extract_value(r["output_json"])) for _, r in rows.iterrows()]
ref_embs = model_sem.encode(ref_texts, convert_to_numpy=True, normalize_embeddings=True)
for i, r in enumerate(rows.itertuples(index=False)):
    prompt   = build_prompt(getattr(r, "prompt"))
    expected = getattr(r, "output_json")
    ref_val  = ref_texts[i]
    pred_val = pick_by_logprob(trainer.model, tokenizer, prompt, candidate_bank_eval)
    pred_json = f'{{"treatment": "{_escape_json_val(pred_val)}"}}'
    pred_val_norm = _norm_syn(pred_val)
    pred_emb = model_sem.encode(pred_val_norm, convert_to_numpy=True, normalize_embeddings=True)
    sim = _cos_sim(pred_emb, ref_embs[i])
    ok = sim >= SIM_THRESH
    print(f"--- Predicted output {i+1}: {pred_json}", flush=True)
    print(f"--- Expected output  {i+1}: {expected}", flush=True)
    print(f"--- cos_sim={sim:.4f} match={ok}", flush=True)
    print("-"*50, flush=True)
    n_ok += int(ok); n_all += 1

test_acc = (n_ok / max(1, n_all))
print(f"final/test_accuracy_treatment={test_acc:.4f}", flush=True)
writer.add_scalar("final/test_accuracy_treatment", test_acc, trainer.state.global_step)
writer.close()

import numpy as np

def _normalize_text(s):
    x = str(s).strip().lower()
    return " ".join(x.split())

def _metrics_soft_exact(model, tokenizer, hf_dataset):
    model.eval()
    rows = hf_dataset.to_pandas()
    ref_texts = [_norm_syn(_extract_value(r["output_json"])) for _, r in rows.iterrows()]
    ref_embs = model_sem.encode(ref_texts, convert_to_numpy=True, normalize_embeddings=True)
    soft_hits = []
    exact_hits = []
    for i, r in enumerate(rows.itertuples(index=False)):
        prompt = build_prompt(getattr(r, "prompt"))
        pred_val = pick_by_logprob(model, tokenizer, prompt, candidate_bank_eval)
        pred_val_norm = _norm_syn(pred_val)
        pred_emb = model_sem.encode(pred_val_norm, convert_to_numpy=True, normalize_embeddings=True)
        soft_hits.append(float(np.dot(pred_emb, ref_embs[i])) >= SIM_THRESH)
        exact_hits.append(_normalize_text(pred_val_norm) == _normalize_text(ref_texts[i]))
    model.train()
    soft = float(np.mean(soft_hits)) if len(soft_hits)>0 else 0.0
    exact = float(np.mean(exact_hits)) if len(exact_hits)>0 else 0.0
    return soft, exact, soft_hits, exact_hits, ref_texts

pre_val_soft = float(init_val_acc)
train_soft, train_exact, _, _, _ = _metrics_soft_exact(trainer.model, tokenizer, datasets["train"])
val_soft2, val_exact, val_soft_hits, _, val_refs = _metrics_soft_exact(trainer.model, tokenizer, datasets["validation"])
test_soft, test_exact, _, _, _ = _metrics_soft_exact(trainer.model, tokenizer, datasets["test"]) if len(datasets["test"])>0 else (0.0, 0.0, [], [], [])

majority_label = str(df_train["output_raw"].value_counts().idxmax()) if len(df_train)>0 else ""
val_ref_embs = model_sem.encode(val_refs, convert_to_numpy=True, normalize_embeddings=True) if len(val_refs)>0 else np.zeros((0,384))
maj_emb = model_sem.encode(majority_label, convert_to_numpy=True, normalize_embeddings=True) if majority_label!="" else np.zeros((384,))
val_majority_soft = float(np.mean([float(np.dot(maj_emb, val_ref_embs[i])) >= SIM_THRESH for i in range(len(val_refs))])) if len(val_refs)>0 and majority_label!="" else 0.0
rng = np.random.RandomState(args.seed)
perm = rng.permutation(len(val_refs)) if len(val_refs)>0 else []
perm_embs = val_ref_embs[perm] if len(val_refs)>0 else np.zeros((0,384))
val_permutation_soft = float(np.mean([float(np.dot(perm_embs[i], val_ref_embs[i])) >= SIM_THRESH for i in range(len(val_refs))])) if len(val_refs)>0 else 0.0

seen_mask = [(v in label2id) for v in val_refs]
val_seen_count = int(sum(seen_mask))
val_unseen_count = int(len(val_refs) - val_seen_count)
val_seen_soft = float(np.mean([val_soft_hits[i] for i in range(len(val_refs)) if seen_mask[i]])) if val_seen_count>0 else 0.0
val_unseen_soft = float(np.mean([val_soft_hits[i] for i in range(len(val_refs)) if not seen_mask[i]])) if val_unseen_count>0 else 0.0

improvement = float(val_soft2 - pre_val_soft)
gap = float(train_soft - val_soft2)
flags = []
if improvement < 0.05:
    flags.append(f"Low progress ({improvement:+.3f})")
if gap >= 0.30:
    flags.append(f"Possible overfit (gap={gap:.3f})")

def _r(x):
    try:
        return float(np.round(float(x), 4))
    except Exception:
        return 0.0

report = {
  "pretrain_val_soft": _r(pre_val_soft),
  "posttrain_train_soft": _r(train_soft),
  "posttrain_val_soft": _r(val_soft2),
  "posttrain_val_exact": _r(val_exact),
  "posttrain_test_soft": _r(test_soft),
  "posttrain_test_exact": _r(test_exact),
  "val_majority_soft": _r(val_majority_soft),
  "val_permutation_soft": _r(val_permutation_soft),
  "improvement_val_soft": _r(improvement),
  "overfit_gap_train_minus_val": _r(gap),
  "val_seen_soft": _r(val_seen_soft),
  "val_unseen_soft": _r(val_unseen_soft),
  "val_seen_count": int(val_seen_count),
  "val_unseen_count": int(val_unseen_count),
  "majority_label": majority_label,
  "flags": flags
}

print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
