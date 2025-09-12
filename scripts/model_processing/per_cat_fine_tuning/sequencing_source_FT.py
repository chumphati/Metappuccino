import os, re, json, math, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorWithPadding, EarlyStoppingCallback, TrainerCallback
)
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter

parser = argparse.ArgumentParser(description="Fine-tune sequencing_source (multitask) exact 3-class classification with closed-set scoring")
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--cls_loss_weight", type=float, default=0.7)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
base_path = args.base_path

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

train_file = os.path.join(base_path, "finetune_data_train.csv")
val_file   = os.path.join(base_path, "finetune_data_val.csv")
test_file  = os.path.join(base_path, "finetune_data_test.csv")
model_name = os.path.join(base_path, "Mistral-7B-Instruct-v0.3")

tb_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/logs13/tensorboard_sequencing_source"
adapter_out_dir = os.path.join(base_path, "cat_sequencing_source")
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

# NOTE: consigne neutre (pas de "otherwise choose bulk")
RULES = "\nTask: Determine sequencing_source among ['spatial','bulk','single cell'] from transcriptomics context. Choose 'single cell' for single-cell or single-nucleus RNA-seq (scRNA-seq, snRNA-seq, 10x Chromium, SMART-Seq, Drop-seq, inDrops, SPLiT-seq). Choose 'spatial' for spatial transcriptomics (10x Visium, Xenium, CosMx, GeoMx, Slide-seq, MERFISH, seqFISH, Stereo-seq, HDST). Choose 'bulk' when the context explicitly indicates bulk RNA-seq or lacks single-cell/spatial cues. Return strictly JSON.\n"

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

ALLOWED = ["spatial","bulk","single cell"]
label2id = {k:i for i,k in enumerate(ALLOWED)}
id2label = {i:k for k,i in label2id.items()}

def _escape_json_val(v):
    return str(v).replace('\\','\\\\').replace('"','\\"')

def _canon_src(s):
    t = re.sub(r'\s+', ' ', str(s).strip().lower())
    if ("visium" in t or "xenium" in t or "cosmx" in t or "geomx" in t or "slide-seq" in t or "slideseq" in t or
        "merfish" in t or "seqfish" in t or "stereo-seq" in t or "stereoseq" in t or "hdst" in t or "spatial" in t):
        return "spatial"
    if (re.search(r'\bsingle[\s-]?cell\b', t) or "singlecell" in t or "single nucleus" in t or "single-nucleus" in t or
        "snrna" in t or "snrna-seq" in t or "sn rna" in t or "scrna" in t or "sc rna" in t or "scrna-seq" in t or
        "10x chromium" in t or "chromium" in t or "drop-seq" in t or "indrops" in t or "split-seq" in t or "smart-seq" in t or "smartseq" in t):
        return "single cell"
    return "bulk"

def _extract_value(txt):
    try:
        j = json.loads(txt)
        v = j.get("sequencing_source","")
        return _canon_src(v)
    except Exception:
        m = re.search(r'"sequencing_source"\s*:\s*"([^"]*)"', str(txt))
        return _canon_src(m.group(1) if m else str(txt))

for _df in (df_train, df_val, df_test):
    _df["output_raw"]  = _df["output"].astype(str).map(_canon_src)
    _df["output_json"] = _df["output_raw"].apply(lambda v: f'{{"sequencing_source": "{_escape_json_val(v)}"}}')
    _df["cls_label"]   = _df["output_raw"].map(lambda x: label2id.get(x, label2id["bulk"]))

# distribution + poids (Class-Balanced Loss)
class_counts = Counter(df_train["output_raw"].tolist())
print("Train distribution:", dict(class_counts), flush=True)

beta = 0.999
cb_weights = []
for i in range(len(ALLOWED)):
    n = max(1, class_counts.get(id2label[i], 1))
    w = (1.0 - beta) / (1.0 - (beta ** n))
    cb_weights.append(1.0 / w)
cb_weights = torch.tensor(cb_weights, dtype=torch.float32)
cb_weights = cb_weights / cb_weights.mean()
print("CE class weights (CBLoss):", cb_weights.tolist(), flush=True)

# sampler équilibré
inv_freq = {k: 1.0 / max(1, class_counts.get(k, 1)) for k in ALLOWED}
train_sample_weights = np.array([inv_freq[row.output_raw] for row in df_train.itertuples(index=False)], dtype=np.float64)

# priors pour calibration du mode génératif fermé
total_n = sum(class_counts.values())
train_priors = {k: class_counts.get(k, 0)/max(1,total_n) for k in ALLOWED}
print("Train priors:", train_priors, flush=True)

datasets = {
    "train": Dataset.from_pandas(df_train[["prompt","output_json","cls_label"]]),
    "validation": Dataset.from_pandas(df_val[["prompt","output_json","cls_label"]]),
    "test": Dataset.from_pandas(df_test[["prompt","output_json","cls_label"]])
}
print({k: len(v) for k,v in datasets.items()}, flush=True)

def tokenize_fn(example):
    prompt   = example["prompt"].strip() + RULES
    out_json = example["output_json"].strip()
    m = re.search(r'"sequencing_source"\s*:\s*"([^"]*)"', out_json)
    value_str  = m.group(1) if m else ""
    prefix_str = '{"sequencing_source": "'
    suffix_str = '"}'
    max_in, max_out, max_total = 3500, 8, 3512

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

def _score_candidate_logprob(model, context_ids, candidate_ids):
    dev = context_ids.device
    ids = context_ids
    total = 0.0
    with torch.no_grad():
        for tid in candidate_ids:
            out = model(input_ids=ids, attention_mask=torch.ones_like(ids, device=dev), use_cache=False)
            logits = out.logits[:, -1, :]
            logp = F.log_softmax(logits, dim=-1)
            total += float(logp[0, tid].item())
            ids = torch.cat([ids, torch.tensor([[tid]], device=dev)], dim=1)
    return total / max(1, len(candidate_ids))

def _predict_closed_set(model, prompt, prior=None, tau=1.0):
    dev = next(model.parameters()).device
    prefix = prompt + RULES + '{"sequencing_source": "'
    ids = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=3500, add_special_tokens=False).to(dev)["input_ids"]
    scores = {}
    for lab in ALLOWED:
        cand_ids = tokenizer(lab, add_special_tokens=False)["input_ids"]
        s = _score_candidate_logprob(model, ids, cand_ids)
        if prior is not None and lab in prior:
            s = s - tau * math.log(max(prior[lab], 1e-8))
        scores[lab] = s
    best = max(scores.items(), key=lambda kv: kv[1])[0]
    print("debug/gen_scores:", scores, flush=True)
    return best

def _predict_head(model, cls_head, prompt):
    dev = next(model.parameters()).device
    text = prompt + RULES
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=3500, add_special_tokens=False).to(dev)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
        hidden = out.hidden_states[-1]  # [1, T, H]
        vec = hidden[:, -1, :]          # last token of input (no labels appended)
        if next(cls_head.parameters()).device != vec.device:
            cls_head.to(vec.device)
        logits = cls_head(vec)
        probs = torch.softmax(logits.float(), dim=-1)[0].tolist()
    pred_id = int(np.argmax(probs))
    pred = id2label[pred_id]
    print("debug/head_probs:", {id2label[i]: float(probs[i]) for i in range(len(ALLOWED))}, flush=True)
    return pred

def gen_eval_accuracy(model, tokenizer, hf_dataset, max_examples=None, tag="eval", mode="head", cls_head=None, prior=None):
    model.eval()
    n_ok, n_all = 0, 0
    per_class = {k: {"tp":0, "tot":0} for k in ALLOWED}
    rows = hf_dataset.to_pandas()
    if max_examples is not None:
        rows = rows.sample(min(max_examples, len(rows)), random_state=42)
    for _, r in rows.iterrows():
        prompt   = r["prompt"].strip()
        expected = r["output_json"]
        ref_val  = _extract_value(expected)
        if mode == "gen":
            pred_val = _predict_closed_set(model, prompt, prior=prior, tau=1.0)
        else:
            pred_val = _predict_head(model, cls_head, prompt)
        ok = (pred_val == ref_val)
        n_ok += int(ok); n_all += 1
        per_class[ref_val]["tp"] += int(ok); per_class[ref_val]["tot"] += 1
        print(f'{tag}/pair {n_all}: pred="{pred_val}" ref="{ref_val}" match={ok}', flush=True)
    model.train()
    acc = (n_ok / max(1, n_all))
    recalls = []
    for k in ALLOWED:
        tot = max(1, per_class[k]["tot"])
        rec = per_class[k]["tp"] / tot
        recalls.append(rec)
    bal_acc = float(np.mean(recalls))
    print(f"{tag}/accuracy={acc:.4f} balanced_accuracy={bal_acc:.4f}", flush=True)
    return acc, bal_acc, {k: per_class[k]["tp"]/max(1,per_class[k]["tot"]) for k in ALLOWED}

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
    def __init__(self, eval_ds, max_examples=128, cls_head=None, prior=None):
        self.eval_ds = eval_ds; self.max_examples = max_examples; self.cls_head = cls_head; self.prior = prior
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        model     = kwargs.get("model", None)
        tokenizer = kwargs.get("tokenizer", None)
        if model is None or tokenizer is None: return
        acc, bal_acc, _ = gen_eval_accuracy(model, tokenizer, self.eval_ds, max_examples=self.max_examples, tag="eval", mode="head", cls_head=self.cls_head, prior=self.prior)
        print(f"eval/accuracy_sequencing_source={acc:.4f} eval/balanced_accuracy={bal_acc:.4f} at step={state.global_step}", flush=True)
        writer.add_scalar("eval/accuracy_sequencing_source", acc, state.global_step)
        writer.add_scalar("eval/balanced_accuracy", bal_acc, state.global_step)
        if isinstance(metrics, dict):
            metrics["eval_accuracy_sequencing_source"] = float(acc)
            metrics["eval_balanced_accuracy"] = float(bal_acc)

print("Initial evaluation on validation with BASE model (closed-set scoring, prior-calibrated)", flush=True)
base_model.eval()
init_val_acc_gen, init_val_bal_gen, _ = gen_eval_accuracy(base_model, tokenizer, datasets["validation"], max_examples=None, tag="initial_gen", mode="gen", prior=train_priors)
print(f"initial/validation_gen_accuracy={init_val_acc_gen:.4f} initial/validation_gen_balanced_accuracy={init_val_bal_gen:.4f}", flush=True)
writer.add_scalar("initial/validation_gen_accuracy", init_val_acc_gen, 0)
writer.add_scalar("initial/validation_gen_balanced_accuracy", init_val_bal_gen, 0)
writer.flush()

hidden_size = getattr(peft_model.config, "hidden_size", None) or getattr(peft_model.config, "n_embd", None)
num_classes = len(ALLOWED)
cls_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_size, num_classes)
)
cls_head.to(next(peft_model.parameters()).device)

class MultiTaskTrainer(Trainer):
    def __init__(self, cls_head: nn.Module, cls_weight: float = 0.7, class_weights=None, train_sample_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.cls_head = cls_head
        self.cls_weight = float(cls_weight)
        self.label_smoothing = 0.05
        self.class_weights = (class_weights.detach().clone().float().cpu() if class_weights is not None else torch.ones(num_classes, dtype=torch.float32))
        self.train_sample_weights = torch.tensor(train_sample_weights, dtype=torch.double) if train_sample_weights is not None else None
        print("debug/class_weights(CB):", self.class_weights.tolist(), flush=True)

    def create_optimizer(self):
        if self.optimizer is None:
            model_params = [p for p in self.model.parameters() if p.requires_grad]
            head_params  = [p for p in self.cls_head.parameters() if p.requires_grad]
            optimizer_grouped_parameters = [
                {"params": model_params, "weight_decay": self.args.weight_decay},
                {"params": head_params,  "weight_decay": 0.05},
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        return self.optimizer

    def get_train_dataloader(self):
        if self.train_dataset is None:
            return None
        if self.train_sample_weights is not None:
            sampler = WeightedRandomSampler(weights=self.train_sample_weights, num_samples=len(self.train_sample_weights), replacement=True)
            return DataLoader(self.train_dataset, batch_size=self.args.train_batch_size, sampler=sampler, collate_fn=self.data_collator, drop_last=self.args.dataloader_drop_last, num_workers=self.args.dataloader_num_workers, pin_memory=self.args.dataloader_pin_memory)
        return super().get_train_dataloader()

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        try:
            ds = datasets.get("validation", None)
            acc, bal_acc, per_cls = gen_eval_accuracy(self.model, self.tokenizer, ds if ds is not None else datasets["validation"], max_examples=256, tag=metric_key_prefix, mode="head", cls_head=self.cls_head, prior=train_priors)
            metrics[f"{metric_key_prefix}_accuracy_sequencing_source"] = float(acc)
            metrics[f"{metric_key_prefix}_balanced_accuracy"] = float(bal_acc)
            step = getattr(self.state, "global_step", -1)
            print(f"{metric_key_prefix}/accuracy_sequencing_source={acc:.4f} {metric_key_prefix}/balanced_accuracy={bal_acc:.4f} at step={step}", flush=True)
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
            ce = nn.CrossEntropyLoss(
                reduction='mean',
                weight=self.class_weights.to(logits.device, dtype=torch.float32),
                label_smoothing=self.label_smoothing
            )
            cls_loss = ce(logits.float()[mask], cls_label[mask])
        else:
            cls_loss = torch.tensor(0.0, device=hidden.device, dtype=logits.dtype)
        loss = lm_loss + self.cls_weight * cls_loss
        if return_outputs:
            return loss, outputs
        return loss

training_args = TrainingArguments(
    output_dir=os.path.join(base_path, "checkpoints_sequencing_source_fix"),
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    save_total_limit=3,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.05,
    logging_strategy="steps",
    logging_steps=50,
    fp16=(not use_bf16),
    bf16=use_bf16,
    gradient_accumulation_steps=16,
    report_to=["tensorboard"],
    logging_dir=tb_dir,
    load_best_model_at_end=True,
    metric_for_best_model="eval_balanced_accuracy",
    greater_is_better=True,
    warmup_ratio=0.06,
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
        GenEvalCallback(datasets["validation"], max_examples=256, cls_head=cls_head, prior=train_priors)
    ],
    cls_head=cls_head,
    cls_weight=args.cls_loss_weight,
    class_weights=cb_weights,
    train_sample_weights=train_sample_weights
)

print("Begin training", flush=True)
trainer.train()
writer.flush()

print("Save adapter only", flush=True)
trainer.model.save_pretrained(adapter_out_dir)
tokenizer.save_pretrained(adapter_out_dir)

print("Evaluate on validation with HEAD (primary) + GEN(calibrated)", flush=True)
val_acc_head, val_bal_head, per_cls_head = gen_eval_accuracy(trainer.model, tokenizer, datasets["validation"], max_examples=None, tag="final_val_head", mode="head", cls_head=cls_head, prior=train_priors)
val_acc_gen,  val_bal_gen,  _           = gen_eval_accuracy(trainer.model, tokenizer, datasets["validation"], max_examples=None, tag="final_val_gen",  mode="gen",  cls_head=None,      prior=train_priors)
writer.add_scalar("final/validation_accuracy_head", val_acc_head, trainer.state.global_step)
writer.add_scalar("final/validation_balanced_accuracy_head", val_bal_head, trainer.state.global_step)
writer.add_scalar("final/validation_accuracy_gen", val_acc_gen, trainer.state.global_step)
writer.add_scalar("final/validation_balanced_accuracy_gen", val_bal_gen, trainer.state.global_step)
writer.flush()

print("Generate on test set (HEAD; primary inference)", flush=True)
rows = datasets["test"].to_pandas()
n_ok, n_all = 0, 0
for i, r in rows.iterrows():
    prompt   = r["prompt"].strip()
    expected = r["output_json"]
    ref_val  = _extract_value(expected)
    pred_val = _predict_head(trainer.model, cls_head, prompt)
    pred_json = f'{{"sequencing_source": "{_escape_json_val(pred_val)}"}}'
    ok = (pred_val == ref_val)
    print(f"--- Predicted output {i+1}: {pred_json}", flush=True)
    print(f"--- Expected output  {i+1}: {expected}", flush=True)
    print(f"--- match={ok}", flush=True)
    print("-"*50, flush=True)
    n_ok += int(ok); n_all += 1

test_acc = (n_ok / max(1, n_all))
print(f"final/test_accuracy_sequencing_source(head)={test_acc:.4f}", flush=True)
writer.add_scalar("final/test_accuracy_sequencing_source_head", test_acc, trainer.state.global_step)

print("Generate on test set (GEN; calibrated, diagnostic only)", flush=True)
n_ok, n_all = 0, 0
for i, r in rows.iterrows():
    prompt   = r["prompt"].strip()
    expected = r["output_json"]
    ref_val  = _extract_value(expected)
    pred_val = _predict_closed_set(trainer.model, prompt, prior=train_priors, tau=1.0)
    ok = (pred_val == ref_val)
    n_ok += int(ok); n_all += 1
gen_test_acc = (n_ok / max(1, n_all))
print(f"final/test_accuracy_sequencing_source(gen_calibrated)={gen_test_acc:.4f}", flush=True)
writer.add_scalar("final/test_accuracy_sequencing_source_gen_calibrated", gen_test_acc, trainer.state.global_step)

writer.close()
