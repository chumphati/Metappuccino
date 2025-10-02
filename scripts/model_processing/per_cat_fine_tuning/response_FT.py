import os, re, json, math, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback, TrainerCallback
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description="Fine-tune response (multitask) with fixed categories + soft semantic accuracy")
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--cls_loss_weight", type=float, default=2.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--sim_thresh", type=float, default=0.25)
parser.add_argument("--head_lr", type=float, default=1e-3)
parser.add_argument("--head_warmup_steps", type=int, default=300)
args = parser.parse_args()
base_path = args.base_path

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

train_file = os.path.join(base_path, "finetune_data_train.csv")
val_file   = os.path.join(base_path, "finetune_data_val.csv")
test_file  = os.path.join(base_path, "finetune_data_test.csv")
model_name = os.path.join(base_path, "Mistral-7B-Instruct-v0.3")

ALLOWED_CATS = ["no treatment", "unknown", "stable", "progressive", "success"]

SYN2CANON = {
    "no treatment": "no treatment", "no tx": "no treatment", "none": "no treatment", "untreated": "no treatment",
    "treatment naive": "no treatment", "no therapy": "no treatment", "baseline": "no treatment",
    "unknown": "unknown", "not applicable": "unknown", "n/a": "unknown", "na": "unknown", "none specified": "unknown",
    "not reported": "unknown", "unavailable": "unknown", "missing": "unknown", "dose-limiting toxicity": "unknown",
    "stable": "stable", "stable disease": "stable", "sd": "stable", "no change": "stable", "unchanged": "stable",
    "progressive": "progressive", "progression": "progressive", "progressive disease": "progressive", "pd": "progressive",
    "worsened": "progressive", "worsening": "progressive", "relapse": "progressive", "refractory": "progressive",
    "success": "success", "responder": "success", "responded": "success", "response": "success",
    "effective": "success", "efficacy": "success", "benefit": "success", "improved": "success", "improvement": "success",
    "remission": "success", "partial response": "success", "complete response": "success", "cr": "success", "pr": "success",
    "symptom resolution": "success",
    "no response": "unknown"
}

def _canonize_label(x: str) -> str:
    s = str(x).strip().lower()
    s = re.sub(r"\s+", " ", s)
    if s in SYN2CANON:
        return SYN2CANON[s]
    for k, v in SYN2CANON.items():
        if k in s:
            return v
    return "unknown"

def _escape_json_val(v):
    return str(v).replace('\\','\\\\').replace('"','\\"')

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

for _df in (df_train, df_val, df_test):
    _df["output_raw"]   = _df["output"].astype(str)
    _df["response_cat"] = _df["output_raw"].apply(_canonize_label)
    _df["output_json"]  = _df["response_cat"].apply(lambda v: f'{{"response": "{_escape_json_val(v)}"}}')

before = len(df_train)
df_train = df_train[df_train["response_cat"].astype(str).str.strip() != ""].copy()
print(f"filter/train_dropped={before - len(df_train)}", flush=True)

from collections import Counter
for name, _df in ("train", df_train), ("val", df_val), ("test", df_test):
    counts = Counter(_df["response_cat"].tolist())
    total = sum(counts.values()) or 1
    top = ", ".join([f"{k}={counts.get(k,0)} ({counts.get(k,0)*100/total:.1f}%)" for k in ALLOWED_CATS])
    print(f"dist/{name}: {top}", flush=True)

labels = ALLOWED_CATS
label2id = {lbl:i for i,lbl in enumerate(labels)}
id2label = {i:lbl for lbl,i in label2id.items()}
print(f"Label space fixed = {len(label2id)} -> {list(label2id.keys())}", flush=True)

for _df in (df_train, df_val, df_test):
    _df["cls_label"] = _df["response_cat"].map(lambda x: label2id.get(x, label2id["unknown"]))

datasets = {
    "train": Dataset.from_pandas(df_train[["prompt","output_json","cls_label"]]),
    "validation": Dataset.from_pandas(df_val[["prompt","output_json","cls_label"]]),
    "test": Dataset.from_pandas(df_test[["prompt","output_json","cls_label"]])
}
print({k: len(v) for k,v in datasets.items()}, flush=True)

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

print("Tokenize", flush=True)

def tokenize_fn(example):
    prompt   = example["prompt"].strip()
    out_json = example["output_json"].strip()
    m = re.search(r'"response"\s*:\s*"([^"]*)"', out_json)
    value_str  = m.group(1) if m else ""
    prefix_str = '{"response": "'
    suffix_str = '"}'
    max_in, max_out, max_total = 3500, 16, 3520
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
SIM_THRESH = float(args.sim_thresh)

def _cos_sim(a, b):
    an = a / max(np.linalg.norm(a), 1e-12)
    bn = b / max(np.linalg.norm(b), 1e-12)
    return float(np.dot(an, bn))

def _choose_forced_category(model, tokenizer, prompt: str, cats=ALLOWED_CATS):
    dev = next(model.parameters()).device
    prefix = prompt + '{"response": "'
    pref_ids = tokenizer(prefix, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
    def avg_logprob(candidate: str):
        cand = candidate + '"}'
        cand_ids = tokenizer(cand, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
        inp = torch.cat([pref_ids, cand_ids[:, :-1]], dim=1)
        with torch.no_grad():
            out = model(input_ids=inp, attention_mask=torch.ones_like(inp))
            logits = out.logits[:, -cand_ids.size(1):, :]
            logp = torch.log_softmax(logits, dim=-1)
            token_logp = logp.gather(2, cand_ids.unsqueeze(-1)).squeeze(-1)
            return float(token_logp.mean().item())
    scores = [(c, avg_logprob(c)) for c in cats]
    scores.sort(key=lambda x: x[1], reverse=True)
    print("debug/forced_scores=" + ", ".join([f"{c}:{s:.3f}" for c,s in scores]), flush=True)
    return scores[0][0]

def _predict_with_cls(model, tokenizer, cls_head, prompt: str):
    dev = next(model.parameters()).device
    was_training = model.training
    model.eval()
    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(dev)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[-1]
        feat = hidden.mean(dim=1)
        logits = cls_head.to(dev)(feat)
        pred_id = int(logits.argmax(dim=-1).item())
    if was_training: model.train()
    return id2label[pred_id]

def gen_eval_soft_accuracy(model, tokenizer, hf_dataset, max_examples=None, tag="eval", use_classifier=False, cls_head=None):
    model.eval()
    rows = hf_dataset.to_pandas()
    if max_examples is not None:
        rows = rows.sample(min(max_examples, len(rows)), random_state=42)
    refs = []
    for _, r in rows.iterrows():
        try:
            j = json.loads(r["output_json"]); refs.append(str(j.get("response","")))
        except Exception:
            m = re.search(r'"response"\s*:\s*"([^"]*)"', str(r["output_json"]))
            refs.append(m.group(1) if m else "")
    preds = []
    for r in rows.itertuples(index=False):
        prompt = getattr(r, "prompt")
        if use_classifier and cls_head is not None:
            pred = _predict_with_cls(model, tokenizer, cls_head, prompt)
        else:
            pred = _choose_forced_category(model, tokenizer, prompt, ALLOWED_CATS)
        preds.append(pred)
    n_all = len(refs)
    n_ok  = sum(int(p==g) for p,g in zip(preds, refs))
    acc   = n_ok / max(1, n_all)
    from collections import Counter, defaultdict
    cm = defaultdict(Counter)
    for p,g in zip(preds, refs):
        cm[g][p]+=1
    per_cls = {c: (cm[c][c] / max(1, sum(cm[c].values()))) for c in ALLOWED_CATS}
    macro = sum(per_cls.values())/len(ALLOWED_CATS)
    print(f'{tag}/accuracy_exact={acc:.4f} macro={macro:.4f} | ' + ", ".join([f'{c}={per_cls.get(c,0):.3f}' for c in ALLOWED_CATS]), flush=True)
    model.train()
    return acc

writer = SummaryWriter(os.path.join(base_path, "tb_response_fixed"))

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
    def __init__(self, eval_ds, max_examples=128, cls_head=None):
        self.eval_ds = eval_ds; self.max_examples = max_examples; self.cls_head = cls_head
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        model     = kwargs.get("model", None)
        tokenizer = kwargs.get("tokenizer", None)
        if model is None or tokenizer is None: return
        acc = gen_eval_soft_accuracy(model, tokenizer, self.eval_ds, max_examples=self.max_examples, tag="eval", use_classifier=True, cls_head=self.cls_head)
        print(f"eval/accuracy_response={acc:.4f} at step={state.global_step}", flush=True)
        writer.add_scalar("eval/accuracy_response", acc, state.global_step)
        if isinstance(metrics, dict):
            metrics["eval_accuracy_response"] = float(acc)

class HeadWarmupCallback(TrainerCallback):
    def __init__(self, model_ref, head_ref, warmup_steps):
        self.model_ref = model_ref
        self.head_ref = head_ref
        self.warmup_steps = warmup_steps
        self.frozen = False
        self.unfroze = False
    def on_train_begin(self, args, state, control, **kwargs):
        for p in self.model_ref.parameters(): p.requires_grad = False
        for p in self.head_ref.parameters(): p.requires_grad = True
        self.frozen = True
        print(f"warmup/freeze_backbone=1 steps={self.warmup_steps}", flush=True)
    def on_step_end(self, args, state, control, **kwargs):
        if self.frozen and not self.unfroze and state.global_step >= self.warmup_steps:
            for p in self.model_ref.parameters(): p.requires_grad = True
            self.unfroze = True
            print("warmup/unfreeze_backbone=1", flush=True)

print("Initial evaluation on validation with BASE model (forced categories + semantic)", flush=True)
base_model.eval()
init_val_acc = gen_eval_soft_accuracy(base_model, tokenizer, datasets["validation"], max_examples=None, tag="initial", use_classifier=False, cls_head=None)
print(f"initial/validation_accuracy_response={init_val_acc:.4f}", flush=True)
writer.add_scalar("initial/validation_accuracy_response", init_val_acc, 0)
writer.flush()

hidden_size = getattr(peft_model.config, "hidden_size", None) or getattr(peft_model.config, "n_embd", None) or getattr(peft_model.config, "d_model", None)
num_classes = len(label2id)
cls_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_size, num_classes)
)
cls_head.to(next(peft_model.parameters()).device)

def _weights_from_counts(df):
    counts = np.array([(df["cls_label"]==i).sum() for i in range(num_classes)], dtype=np.float32)
    counts = np.maximum(counts, 1.0)
    w = counts.sum() / counts
    w = w / w.mean()
    w = np.clip(w, 0.3, 3.0)
    return torch.tensor(w, dtype=torch.float32)

CE_WEIGHTS = _weights_from_counts(df_train)
print("debug/class_weights=" + ", ".join([f"{id2label[i]}:{CE_WEIGHTS[i].item():.2f}" for i in range(num_classes)]), flush=True)

class MultiTaskTrainer(Trainer):
    def __init__(self, cls_head: nn.Module, cls_weight: float = 0.6, head_lr: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.cls_head = cls_head
        self.cls_weight = float(cls_weight)
        self.head_lr = float(head_lr)
        self.ce = nn.CrossEntropyLoss(reduction='mean', weight=CE_WEIGHTS.to(next(self.model.parameters()).device))
    def create_optimizer(self):
        if self.optimizer is None:
            model_params = [p for p in self.model.parameters() if p.requires_grad]
            head_params  = [p for p in self.cls_head.parameters() if p.requires_grad]
            param_groups = [
                {"params": model_params, "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate},
                {"params": head_params,  "weight_decay": 0.0,                   "lr": self.head_lr},
            ]
            self.optimizer = AdamW(param_groups)
        return self.optimizer
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        try:
            ds = datasets.get("validation", None)
            acc = gen_eval_soft_accuracy(self.model, self.tokenizer, ds if ds is not None else datasets["validation"], max_examples=128, tag=metric_key_prefix, use_classifier=True, cls_head=self.cls_head)
            metrics[f"{metric_key_prefix}_accuracy_response"] = float(acc)
            step = getattr(self.state, "global_step", -1)
            print(f"{metric_key_prefix}/accuracy_response={acc:.4f} at step={step}", flush=True)
        except Exception as e:
            print(f"warn/eval_callback_exception: {e}", flush=True)
        return metrics
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        split_idx = inputs.pop("split_idx")
        cls_label = inputs.pop("cls_label")
        outputs   = model(**inputs, output_hidden_states=True)
        lm_loss   = outputs.loss
        hidden    = outputs.hidden_states[-1]
        B, T, H = hidden.size()
        arange = torch.arange(T, device=hidden.device).unsqueeze(0).expand(B, T)
        prompt_mask = (arange < split_idx.unsqueeze(1)).float()
        den = prompt_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        gather = (hidden * prompt_mask.unsqueeze(-1)).sum(dim=1) / den
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
    output_dir=os.path.join(base_path, "checkpoints_response_fixed"),
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=250,
    save_steps=250,
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
    logging_dir=os.path.join(base_path, "tb_response_fixed"),
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy_response",
    greater_is_better=True,
    warmup_ratio=0.03,
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
        EarlyStoppingCallback(early_stopping_patience=2),
        TBCallback(),
        PrintProgressCallback(),
        GradNormLogger(every=50),
        GenEvalCallback(datasets["validation"], max_examples=128, cls_head=cls_head),
        HeadWarmupCallback(peft_model, cls_head, warmup_steps=args.head_warmup_steps)
    ],
    cls_head=cls_head,
    cls_weight=args.cls_loss_weight,
    head_lr=args.head_lr
)

print("Begin training", flush=True)
trainer.train()
writer.flush()

print("Save adapter only", flush=True)
trainer.model.save_pretrained(os.path.join(base_path, "cat_response"))
tokenizer.save_pretrained(os.path.join(base_path, "cat_response"))

print("Evaluate on validation (classifier exact-match)", flush=True)
val_acc = gen_eval_soft_accuracy(trainer.model, tokenizer, datasets["validation"], max_examples=None, tag="final_val", use_classifier=True, cls_head=cls_head)
print(f"final/validation_accuracy_response={val_acc:.4f}", flush=True)
writer.add_scalar("final/validation_accuracy_response", val_acc, trainer.state.global_step)
writer.flush()

print("Generate on test set (classifier exact-match + cosine info)", flush=True)
rows = datasets["test"].to_pandas()
refs = []
for _, rr in rows.iterrows():
    try:
        refs.append(json.loads(rr["output_json"]).get("response",""))
    except Exception:
        m = re.search(r'"response"\s*:\s*"([^"]*)"', str(rr["output_json"]))
        refs.append(m.group(1) if m else "")
preds = []
for r in rows.itertuples(index=False):
    prompt   = getattr(r, "prompt")
    pred_val = _predict_with_cls(trainer.model, tokenizer, cls_head, prompt)
    preds.append(pred_val)
ref_embs = model_sem.encode(refs, convert_to_numpy=True, normalize_embeddings=True)
pred_embs = model_sem.encode(preds, convert_to_numpy=True, normalize_embeddings=True)
from collections import Counter, defaultdict
n_all = len(refs)
n_ok = sum(int(p==g) for p,g in zip(preds, refs))
acc_exact = n_ok / max(1, n_all)
cm = defaultdict(Counter)
for p,g in zip(preds, refs):
    cm[g][p]+=1
per_cls = {c: (cm[c][c] / max(1, sum(cm[c].values()))) for c in ALLOWED_CATS}
macro = sum(per_cls.values())/len(ALLOWED_CATS)
for i, (p,g) in enumerate(zip(preds, refs)):
    sim = _cos_sim(pred_embs[i], ref_embs[i])
    print(f"--- Predicted output {i+1}: {{\"response\": \"{_escape_json_val(p)}\"}}", flush=True)
    print(f"--- Expected output  {i+1}: {{\"response\": \"{_escape_json_val(g)}\"}}", flush=True)
    print(f"--- cos_sim={sim:.4f} match_exact={p==g}", flush=True)
    print("-"*50, flush=True)
print(f"final/test_accuracy_exact={acc_exact:.4f} macro={macro:.4f} | " + ", ".join([f'{c}={per_cls.get(c,0):.3f}' for c in ALLOWED_CATS]), flush=True)
print("Top confusions:", flush=True)
for gold in ALLOWED_CATS:
    row = cm[gold]
    if sum(row.values())>0:
        worst = sorted([(k,v) for k,v in row.items() if k!=gold], key=lambda x:-x[1])[:2]
        print(f"  {gold} -> {worst}", flush=True)
writer.add_scalar("final/test_accuracy_response_exact", acc_exact, trainer.state.global_step)
writer.close()
