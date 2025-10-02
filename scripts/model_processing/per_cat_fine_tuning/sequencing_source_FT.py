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

parser = argparse.ArgumentParser(description="Fine-tune sequencing_source (multitask) — corrigé: éviter la dérive vers 'unknown', garder les compétences de base, et respecter la déduction contextuelle")
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--cls_loss_weight", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--sim_thresh", type=float, default=0.25)
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--target_modules", type=str, default="q_proj,v_proj")
parser.add_argument("--learning_rate", type=float, default=2e-5)
parser.add_argument("--unknown_penalty", type=float, default=8.0)
parser.add_argument("--length_normalize", action="store_true", default=True)
parser.add_argument("--kl_weight", type=float, default=0.03)
parser.add_argument("--kl_only_if_not_unknown", action="store_true", default=True)
parser.add_argument("--proto_weight", type=float, default=0.1)
args = parser.parse_args()
base_path = args.base_path

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

train_file = os.path.join(base_path, "finetune_data_train.csv")
val_file   = os.path.join(base_path, "finetune_data_val.csv")
test_file  = os.path.join(base_path, "finetune_data_test.csv")
model_name = os.path.join(base_path, "Mistral-7B-Instruct-v0.3")

aLL = ["bulk", "single cell", "spatial", "unknown"]
ALLOWED_CATS = aLL

SYN2CANON = {
    "unknown": "unknown", "not applicable": "unknown", "n/a": "unknown", "na": "unknown", "none specified": "unknown", "not reported": "unknown", "unavailable": "unknown", "missing": "unknown",
    "spatial transcriptomics": "spatial", "spatial": "spatial", "10x visium": "spatial", "visium": "spatial", "xenium": "spatial", "geomx": "spatial", "cosmx": "spatial", "slide-seq": "spatial", "slideseq": "spatial", "hdst": "spatial", "st ": "spatial", " spatial-rna": "spatial", "spatial-seq": "spatial", "merfish": "spatial", "seqfish": "spatial", "stereo-seq": "spatial",
    "single cell": "single cell", "single-cell": "single cell", "sc ": "single cell", "scrna": "single cell", "sc rna": "single cell", "scrna-seq": "single cell", "snrna": "single cell", "sn rna": "single cell", "snrna-seq": "single cell", "single nuclei": "single cell", "nuclei": "single cell", "smart-seq2": "single cell", "smart-seq": "single cell", "smartseq2": "single cell", "smartseq": "single cell", "ss2": "single cell", "drop-seq": "single cell", "dropseq": "single cell", "droplet": "single cell", "10x": "single cell", "cellranger": "single cell",
    "bulk": "bulk", "bulk rna": "bulk", "bulk rna-seq": "bulk", "total rna": "bulk", "total rna-seq": "bulk", "ribodepletion": "bulk", "ribo-depletion": "bulk", "rrna depletion": "bulk", "ribominus": "bulk", "polya": "bulk", "poly a": "bulk", "polya selection": "bulk", "whole tissue": "bulk", "tissue bulk": "bulk", "population": "bulk", "bulk-tissue": "bulk"
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
    _df["sequencing_source_cat"] = _df["output_raw"].apply(_canonize_label)
    _df["output_json"]  = _df["sequencing_source_cat"].apply(lambda v: f'{{"sequencing_source": "{_escape_json_val(v)}"}}')

before = len(df_train)
df_train = df_train[df_train["sequencing_source_cat"].astype(str).str.strip() != ""].copy()
print(f"filter/train_dropped={before - len(df_train)}", flush=True)

from collections import Counter
for name, _df in ("train", df_train), ("val", df_val), ("test", df_test):
    counts = Counter(_df["sequencing_source_cat"].tolist())
    total = sum(counts.values()) or 1
    top = ", ".join([f"{k}={counts.get(k,0)} ({counts.get(k,0)*100/total:.1f}%)" for k in ALLOWED_CATS])
    print(f"dist/{name}: {top}", flush=True)

labels = ALLOWED_CATS
label2id = {lbl:i for i,lbl in enumerate(labels)}
id2label = {i:lbl for lbl,i in label2id.items()}
print(f"Label space fixed = {len(label2id)} -> {list(label2id.keys())}", flush=True)

for _df in (df_train, df_val, df_test):
    _df["cls_label"] = _df["sequencing_source_cat"].map(lambda x: label2id.get(x, label2id["unknown"]))

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

print("Load base model (student)", flush=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    device_map="auto",
    low_cpu_mem_usage=True
)
base_model.config.pad_token_id = tokenizer.pad_token_id
if tokenizer.pad_token_id is not None and base_model.get_input_embeddings().num_embeddings != len(tokenizer):
    base_model.resize_token_embeddings(len(tokenizer))

print("Configure lightweight LoRA", flush=True)
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=int(args.lora_r),
    lora_alpha=int(args.lora_alpha),
    lora_dropout=float(args.lora_dropout),
    target_modules=[m.strip() for m in args.target_modules.split(",") if m.strip()]
)
peft_model = get_peft_model(base_model, peft_config)
peft_model.train()

print("Load teacher (frozen base) for gentle retention", flush=True)
teacher = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    device_map="auto",
    low_cpu_mem_usage=True
)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)

def tokenize_fn(example):
    prompt   = example["prompt"].strip()
    out_json = example["output_json"].strip()
    m = re.search(r'"sequencing_source"\s*:\s*"([^"]*)"', out_json)
    value_str  = m.group(1) if m else ""
    prefix_str = '{"sequencing_source": "'
    suffix_str = '"}'
    max_in, max_out, max_total = 2200, 8, 2210
    in_enc  = tokenizer(prompt, truncation=True, padding=False, max_length=max_in, add_special_tokens=False)
    pref_ids = tokenizer(prefix_str, add_special_tokens=False)["input_ids"]
    val_ids  = tokenizer(value_str, truncation=True, padding=False, max_length=max_out, add_special_tokens=False)["input_ids"]
    suff_ids = tokenizer(suffix_str, add_special_tokens=False)["input_ids"]
    out_ids = pref_ids + val_ids + suff_ids
    ids  = in_enc["input_ids"] + out_ids
    attn = [1]*len(ids)
    labels = [-100]*len(in_enc["input_ids"]) + [-100]*len(pref_ids) + val_ids + suff_ids
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

ANCHORS = {
    "bulk": ["bulk RNA-seq", "whole tissue", "population average", "total RNA"],
    "single cell": ["scRNA-seq", "single-cell", "single nuclei", "droplet 10x"],
    "spatial": ["spatial transcriptomics", "Visium", "GeoMx", "MERFISH"],
    "unknown": ["unknown", "not reported", "n/a"]
}

cat_proto = {}
for cat, phrases in ANCHORS.items():
    emb = model_sem.encode(phrases, convert_to_numpy=True, normalize_embeddings=True)
    cat_proto[cat] = torch.tensor(emb.mean(axis=0), dtype=torch.float32)

proto_dim = cat_proto["bulk"].numel()

hidden_size = getattr(peft_model.config, "hidden_size", None) or getattr(peft_model.config, "n_embd", None) or getattr(peft_model.config, "d_model", None)
num_classes = len(ALLOWED_CATS)
cls_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(hidden_size, num_classes)
)
proj = nn.Linear(hidden_size, proto_dim)
for m in (cls_head, proj):
    m.to(next(peft_model.parameters()).device)

def _cos_sim(a, b):
    an = a / (a.norm(p=2, dim=-1, keepdim=True) + 1e-12)
    bn = b / (b.norm(p=2, dim=-1, keepdim=True) + 1e-12)
    return (an * bn).sum(dim=-1)

cat_proto_mat = torch.stack([cat_proto[c] for c in ALLOWED_CATS], dim=0).to(next(peft_model.parameters()).device)

def _choose_forced_category(model, tokenizer, prompt: str, cats=ALLOWED_CATS, unknown_penalty=0.0, length_normalize=False):
    dev = next(model.parameters()).device
    prefix = prompt + '{"sequencing_source": "'
    pref = tokenizer(prefix, add_special_tokens=False, truncation=True, max_length=2200, return_tensors="pt")["input_ids"].to(dev)
    def score(candidate: str):
        cand = candidate + '"}'
        cand_ids = tokenizer(cand, add_special_tokens=False, return_tensors="pt")["input_ids"].to(dev)
        inp = torch.cat([pref, cand_ids[:, :-1]], dim=1)
        with torch.no_grad():
            out = model(input_ids=inp, attention_mask=torch.ones_like(inp))
            logits = out.logits[:, -cand_ids.size(1):, :]
            logp = torch.log_softmax(logits, dim=-1)
            token_logp = logp.gather(2, cand_ids.unsqueeze(-1)).squeeze(-1)
            s = token_logp.mean() if length_normalize else token_logp.sum()
            if candidate == "unknown":
                s = s - unknown_penalty
            return float(s.item())
    scores = [(c, score(c)) for c in cats]
    scores.sort(key=lambda x: x[1], reverse=True)
    print("debug/forced_scores=" + ", ".join([f"{c}:{s:.2f}" for c,s in scores]), flush=True)
    return scores[0][0]

def gen_eval_closedset_accuracy(model, tokenizer, hf_dataset, max_examples=None, tag="eval", unknown_penalty=0.0, length_normalize=False):
    model.eval()
    rows = hf_dataset.to_pandas()
    if max_examples is not None:
        rows = rows.sample(min(max_examples, len(rows)), random_state=42)
    preds, refs = [], []
    for _, r in rows.iterrows():
        try:
            j = json.loads(r["output_json"]); ref = str(j.get("sequencing_source", "unknown"))
        except Exception:
            m = re.search(r'"sequencing_source"\s*:\s*"([^"]*)"', str(r["output_json"]))
            ref = m.group(1) if m else "unknown"
        pred = _choose_forced_category(model, tokenizer, r["prompt"], ALLOWED_CATS, unknown_penalty, length_normalize)
        preds.append(pred); refs.append(ref)
        print(f'{tag}/pair pred="{pred}" ref="{ref}" match={int(pred==ref)}', flush=True)
    acc = sum(int(p==r) for p,r in zip(preds,refs)) / max(1,len(refs))
    model.train()
    return acc

writer = SummaryWriter(os.path.join(base_path, "tb_sequencing_source"))

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
    def __init__(self, every=100): self.every = every
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
        acc = gen_eval_closedset_accuracy(model, tokenizer, self.eval_ds, max_examples=self.max_examples, tag="eval", unknown_penalty=float(args._unknown_penalty), length_normalize=bool(args._length_normalize))
        print(f"eval/accuracy_sequencing_source={acc:.4f} at step={state.global_step}", flush=True)
        writer.add_scalar("eval/accuracy_sequencing_source", acc, state.global_step)
        if isinstance(metrics, dict):
            metrics["eval_accuracy_sequencing_source"] = float(acc)

TrainingArguments._unknown_penalty = float(args.unknown_penalty)
TrainingArguments._length_normalize = bool(args.length_normalize)

print("Initial evaluation on validation with BASE model (forced cats, length-norm & unknown penalty)", flush=True)
base_model.eval()
init_val_acc = gen_eval_closedset_accuracy(base_model, tokenizer, datasets["validation"], max_examples=None, tag="initial", unknown_penalty=float(args.unknown_penalty), length_normalize=bool(args.length_normalize))
print(f"initial/validation_accuracy_sequencing_source={init_val_acc:.4f}", flush=True)
writer.add_scalar("initial/validation_accuracy_sequencing_source", init_val_acc, 0)
writer.flush()

CE_WEIGHTS = None

class MultiTaskTrainer(Trainer):
    def __init__(self, cls_head: nn.Module, proj: nn.Module, cls_weight: float = 0.5, kl_weight: float = 0.03, proto_weight: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.cls_head = cls_head
        self.proj = proj
        self.cls_weight = float(cls_weight)
        self.kl_weight = float(kl_weight)
        self.proto_weight = float(proto_weight)
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.ce_teacher = nn.CrossEntropyLoss(reduction='mean')

    def create_optimizer(self):
        if self.optimizer is None:
            model_params = [p for p in self.model.parameters() if p.requires_grad]
            extra_params = list(self.cls_head.parameters()) + list(self.proj.parameters())
            optimizer_grouped_parameters = [
                {"params": model_params, "weight_decay": self.args.weight_decay},
                {"params": extra_params,  "weight_decay": 0.05},
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        return self.optimizer

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        try:
            ds = datasets.get("validation", None)
            acc = gen_eval_closedset_accuracy(self.model, self.tokenizer, ds if ds is not None else datasets["validation"], max_examples=128, tag=metric_key_prefix, unknown_penalty=float(self.args._unknown_penalty), length_normalize=bool(self.args._length_normalize))
            metrics[f"{metric_key_prefix}_accuracy_sequencing_source"] = float(acc)
            step = getattr(self.state, "global_step", -1)
            print(f"{metric_key_prefix}/accuracy_sequencing_source={acc:.4f} at step={step}", flush=True)
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

        logits_cls = self.cls_head(gather)
        mask = (cls_label >= 0)
        cls_loss = self.ce(logits_cls[mask], cls_label[mask]) if mask.any() else torch.tensor(0.0, device=hidden.device)

        with torch.no_grad():
            teacher_targets = []
            for b in range(B):
                in_ids = inputs["input_ids"][b, :split_idx[b]].detach().cpu().tolist()
                prompt = self.tokenizer.decode(in_ids, skip_special_tokens=False)
                t_pred = _choose_forced_category(teacher, self.tokenizer, prompt, ALLOWED_CATS, unknown_penalty=float(self.args._unknown_penalty), length_normalize=bool(self.args._length_normalize))
                teacher_targets.append(label2id.get(t_pred, label2id["unknown"]))
        teacher_targets = torch.tensor(teacher_targets, device=hidden.device, dtype=torch.long)
        if bool(args.kl_only_if_not_unknown):
            mask_t = teacher_targets != label2id["unknown"]
        else:
            mask_t = torch.ones_like(teacher_targets, dtype=torch.bool)
        distill_loss = self.ce_teacher(logits_cls[mask_t], teacher_targets[mask_t]) if mask_t.any() else torch.tensor(0.0, device=hidden.device)

        z = self.proj(gather)
        z = nn.functional.normalize(z, dim=-1)
        proto = nn.functional.normalize(cat_proto_mat.to(z.device), dim=-1)
        pos = proto[cls_label.clamp(min=0)] if mask.any() else proto[0:1].expand(B, -1)
        cos_pos = _cos_sim(z[mask], pos[mask]) if mask.any() else torch.tensor(0.0, device=z.device)
        cos_unk = _cos_sim(z, proto[label2id["unknown"]].unsqueeze(0).expand_as(z))
        proto_loss = (1.0 - cos_pos.mean()) if mask.any() else torch.tensor(0.0, device=z.device)
        proto_loss = proto_loss + 0.5 * nn.functional.relu(cos_unk.mean())

        loss = lm_loss + self.cls_weight * cls_loss + self.kl_weight * distill_loss + self.proto_weight * proto_loss
        if return_outputs:
            return loss, outputs
        return loss

training_args = TrainingArguments(
    output_dir=os.path.join(base_path, "checkpoints_sequencing_source"),
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=250,
    save_steps=250,
    save_total_limit=2,
    learning_rate=float(args.learning_rate),
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    weight_decay=0.05,
    logging_strategy="steps",
    logging_steps=250,
    fp16=(not use_bf16),
    bf16=use_bf16,
    gradient_accumulation_steps=8,
    report_to=["tensorboard"],
    logging_dir=os.path.join(base_path, "tb_sequencing_source"),
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy_sequencing_source",
    greater_is_better=True,
    warmup_ratio=0.04,
    lr_scheduler_type="cosine",
    max_grad_norm=0.8,
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
        GradNormLogger(every=100),
        GenEvalCallback(datasets["validation"], max_examples=128)
    ],
    cls_head=cls_head,
    proj=proj,
    cls_weight=args.cls_loss_weight,
    kl_weight=args.kl_weight,
    proto_weight=args.proto_weight
)

print("Begin training", flush=True)
trainer.train()
writer.flush()

print("Save adapter only", flush=True)
trainer.model.save_pretrained(os.path.join(base_path, "cat_sequencing_source"))
tokenizer.save_pretrained(os.path.join(base_path, "cat_sequencing_source"))

print("Evaluate on validation (forced categories, exact-match)", flush=True)
val_acc = gen_eval_closedset_accuracy(trainer.model, tokenizer, datasets["validation"], max_examples=None, tag="final_val", unknown_penalty=float(args.unknown_penalty), length_normalize=bool(args.length_normalize))
print(f"final/validation_accuracy_sequencing_source={val_acc:.4f}", flush=True)
writer.add_scalar("final/validation_accuracy_sequencing_source", val_acc, trainer.state.global_step)
writer.flush()

print("Generate on test set (forced categories, exact-match + cos-sim debug)", flush=True)
rows = datasets["test"].to_pandas()
n_ok, n_all = 0, 0
ref_texts = []
for _, rr in rows.iterrows():
    try:
        j = json.loads(rr["output_json"]) ; ref_texts.append(str(j.get("sequencing_source","")))
    except Exception:
        m = re.search(r'"sequencing_source"\s*:\s*"([^"]*)"', str(rr["output_json"]))
        ref_texts.append(m.group(1) if m else "")
ref_embs = model_sem.encode(ref_texts, convert_to_numpy=True, normalize_embeddings=True)
for i, r in enumerate(rows.itertuples(index=False)):
    prompt   = getattr(r, "prompt")
    ref_val  = ref_texts[i]
    pred_val = _choose_forced_category(trainer.model, tokenizer, prompt, ALLOWED_CATS, unknown_penalty=float(args.unknown_penalty), length_normalize=bool(args.length_normalize))
    pred_json = f'{{"sequencing_source": "{_escape_json_val(pred_val)}"}}'
    pred_emb = model_sem.encode(pred_val, convert_to_numpy=True, normalize_embeddings=True)
    sim = float(np.dot(pred_emb, ref_embs[i]))
    ok = (pred_val == ref_val)
    print(f"--- Predicted output {i+1}: {pred_json}", flush=True)
    print(f"--- Expected output  {i+1}: {{\"sequencing_source\": \"{_escape_json_val(ref_val)}\"}}", flush=True)
    print(f"--- cos_sim={sim:.4f} exact_match={ok}", flush=True)
    print("-"*50, flush=True)
    n_ok += int(ok); n_all += 1

test_acc = (n_ok / max(1, n_all))
print(f"final/test_accuracy_sequencing_source={test_acc:.4f}", flush=True)
writer.add_scalar("final/test_accuracy_sequencing_source", test_acc, trainer.state.global_step)
writer.close()
