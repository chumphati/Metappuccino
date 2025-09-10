##########################################################################################
# IMPORTS
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

##########################################################################################
# ARGS / PATHS
parser = argparse.ArgumentParser(description="Fine-tune cell_line (multitask), eval ponctuation-insensitive, case-sensitive")
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--aug_per_sample", type=int, default=1)
parser.add_argument("--cls_loss_weight", type=float, default=0.4)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
base_path = args.base_path

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

train_file = os.path.join(base_path, "finetune_data_train.csv")
val_file   = os.path.join(base_path, "finetune_data_val.csv")
test_file  = os.path.join(base_path, "finetune_data_test.csv")
model_name = os.path.join(base_path, "Mistral-7B-Instruct-v0.3")

tb_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/logs/tensorboard_cell_line"
adapter_out_dir = os.path.join(base_path, "cat_cell_line")
os.makedirs(tb_dir, exist_ok=True); os.makedirs(adapter_out_dir, exist_ok=True)

##########################################################################################
# MODEL / TOKENIZER
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

##########################################################################################
# DATA LOADING
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

##########################################################################################
# UTILS
def _escape_json_val(v):
    return str(v).replace('\\','\\\\').replace('"','\\"')

def _extract_value(txt):
    try:
        j = json.loads(txt)
        v = j.get("cell_line","")
        return "" if v is None else str(v)
    except Exception:
        m = re.search(r'"cell_line"\s*:\s*"([^"]*)"', str(txt))
        return m.group(1) if m else str(txt)

def _strip_punct_keep_case(s):
    return re.sub(r'[^A-Za-z0-9]+', '', str(s))

##########################################################################################
# TARGETS RAW
for _df in (df_train, df_val, df_test):
    _df["output_raw"]  = _df["output"].astype(str)
    _df["output_json"] = _df["output_raw"].apply(lambda v: f'{{"cell_line": "{_escape_json_val(v)}"}}')

labels_train = sorted(set(df_train["output_raw"].tolist()))
label2id = {lbl:i for i,lbl in enumerate(labels_train)}
id2label = {i:lbl for lbl,i in label2id.items()}
print(f"Label space (train) size = {len(label2id)}", flush=True)

for _df in (df_train, df_val, df_test):
    _df["cls_label"] = _df["output_raw"].map(lambda x: label2id.get(x, -1))

##########################################################################################
# PROMPT AUGMENTATION
_subs = [("0","o"),("o","0"),("1","i"),("i","1"),("5","s"),("s","5"),("t","7"),("7","t")]

def _noisy_alias(canon: str) -> str:
    s = canon
    if len(s) >= 4 and random.random() < 0.7:
        k = random.randint(1, min(3, len(s)//2))
        idxs = sorted(random.sample(range(1,len(s)), k))
        chars = list(s)
        for j, pos in enumerate(idxs):
            chars.insert(pos+j, random.choice(['-',' ']))
        s = ''.join(chars)
    if random.random() < 0.5:
        s = s.upper() if random.random()<0.5 else s.capitalize()
    if random.random() < 0.5:
        s = f"{s} (cell line)" if random.random()<0.5 else f"cell line {s}"
    if random.random() < 0.6 and len(s) > 2:
        nswap = random.randint(1,2)
        for _ in range(nswap):
            a,b = random.choice(_subs)
            s = re.sub(a, b, s)
    return s

AUG_PER_SAMPLE = max(0, int(args.aug_per_sample))
if AUG_PER_SAMPLE > 0:
    rows = []
    for _, r in df_train.iterrows():
        rows.append({"prompt": r["prompt"], "output_json": r["output_json"], "cls_label": r["cls_label"]})
        canon_for_noise = re.sub(r'[^a-z0-9]+','', r["output_raw"].lower())
        for _ in range(AUG_PER_SAMPLE):
            noisy = _noisy_alias(canon_for_noise)
            aug_prompt = r["prompt"] + f"\nNoisy mention (may include typos): '{noisy}'. Return strictly JSON with the exact string."
            rows.append({"prompt": aug_prompt, "output_json": r["output_json"], "cls_label": r["cls_label"]})
    df_train_aug = pd.DataFrame(rows)
else:
    df_train_aug = df_train[["prompt","output_json","cls_label"]].copy()

datasets = {
    "train": Dataset.from_pandas(df_train_aug[["prompt","output_json","cls_label"]]),
    "validation": Dataset.from_pandas(df_val[["prompt","output_json","cls_label"]]),
    "test": Dataset.from_pandas(df_test[["prompt","output_json","cls_label"]])
}
print({k: len(v) for k,v in datasets.items()}, flush=True)

##########################################################################################
# TOKENIZATION
def tokenize_fn(example):
    prompt   = example["prompt"].strip()
    out_json = example["output_json"].strip()
    m = re.search(r'"cell_line"\s*:\s*"([^"]*)"', out_json)
    value_str  = m.group(1) if m else ""
    prefix_str = '{"cell_line": "'
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
    value_mask = [0]*len(in_enc["input_ids"]) + [0]*len(pref_ids) + [1]*len(val_ids) + [0]*len(suff_ids)

    if len(ids) > max_total:
        ids = ids[:max_total]; attn = attn[:max_total]; labels = labels[:max_total]; value_mask = value_mask[:max_total]

    return {
        "input_ids": ids,
        "attention_mask": attn,
        "labels": labels,
        "split_idx": len(in_enc["input_ids"]),
        "cls_label": int(example.get("cls_label", -1)),
        "value_mask": value_mask,  # <<< ajouté
    }

tokenized = {
    k: v.map(
        tokenize_fn,
        remove_columns=v.column_names,
        load_from_cache_file=False
    )
    for k, v in datasets.items()
}

class CausalLMPadCollator:
    def __init__(self, tokenizer):
        self._padder = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    def __call__(self, features):
        labels     = []
        split_idx  = []
        cls_label  = []
        value_masks = []
        # récupère d'abord ce qu'il faut, en prévoyant un fallback pour value_mask
        for f in features:
            lab = torch.tensor(f["labels"], dtype=torch.long)
            labels.append(lab)
            split_idx.append(int(f.get("split_idx", 0)))
            cls_label.append(int(f.get("cls_label", -1)))
            vm = f.get("value_mask", None)
            if vm is None:
                # fallback : pondère toutes les positions supervisées (labels != -100)
                vm = [1 if t != -100 else 0 for t in f["labels"]]
            value_masks.append(torch.tensor(vm, dtype=torch.long))
            # on nettoie pour laisser DataCollatorWithPadding bosser
            f.pop("labels", None); f.pop("split_idx", None); f.pop("cls_label", None); f.pop("value_mask", None)

        batch = self._padder(features)
        max_len = batch["input_ids"].size(1)

        def _pad_seq(seq, pad_token, dtype):
            if seq.numel() < max_len:
                pad = torch.full((max_len - seq.numel(),), pad_token, dtype=dtype)
                return torch.cat([seq, pad], dim=0)
            return seq[:max_len]

        padded_labels = [_pad_seq(l, -100, torch.long) for l in labels]
        padded_vmask  = [_pad_seq(vm, 0, torch.long)   for vm in value_masks]

        batch["labels"]     = torch.stack(padded_labels, dim=0)
        batch["split_idx"]  = torch.tensor(split_idx, dtype=torch.long)
        batch["cls_label"]  = torch.tensor(cls_label, dtype=torch.long)
        batch["value_mask"] = torch.stack(padded_vmask, dim=0)
        return batch

data_collator = CausalLMPadCollator(tokenizer)

##########################################################################################
# EVAL
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
        ref_val  = _extract_value(expected)
        prefix = prompt + '{"cell_line": "'
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
        pred_val = re.split(r'["\n,}]', cont)[0]
        ok = _cmp_ignore_punct_keep_case(pred_val, ref_val)
        n_ok += int(ok); n_all += 1
        print(f'{tag}/pair {n_all}: pred="{pred_val}" ref="{ref_val}" pred_norm="{_strip_punct_keep_case(pred_val)}" ref_norm="{_strip_punct_keep_case(ref_val)}" match={ok}', flush=True)
    model.train()
    return (n_ok / max(1, n_all))

##########################################################################################
# CALLBACKS / LOGS
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
        print(f"eval/accuracy_cell_line={acc:.4f} at step={state.global_step}", flush=True)
        writer.add_scalar("eval/accuracy_cell_line", acc, state.global_step)
        if isinstance(metrics, dict):
            metrics["eval_accuracy_cell_line"] = float(acc)

##########################################################################################
# INITIAL EVAL = BASE MODEL (aucune aide)
print("Initial evaluation on validation with BASE model (punctuation-insensitive, case-sensitive)", flush=True)
base_model.eval()
init_val_acc = gen_eval_accuracy(base_model, tokenizer, datasets["validation"], max_examples=None, tag="initial")
print(f"initial/validation_accuracy_cell_line={init_val_acc:.4f}", flush=True)
writer.add_scalar("initial/validation_accuracy_cell_line", init_val_acc, 0)
writer.flush()

##########################################################################################
# CLASSIFICATION HEAD
hidden_size = getattr(peft_model.config, "hidden_size", None) or getattr(peft_model.config, "n_embd", None)
num_classes = len(label2id)
cls_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_size, num_classes)
)
cls_head.to(next(peft_model.parameters()).device)

##########################################################################################
# TRAINER
class MultiTaskTrainer(Trainer):
    def __init__(self, cls_head: nn.Module, cls_weight: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self.cls_head = cls_head
        self.cls_weight = float(cls_weight)
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.value_token_weight = 3.0
        self.rdrop_alpha = 0.5

    def create_optimizer(self):
        if self.optimizer is None:
            named = list(self.model.named_parameters())
            lora_params = [p for n,p in named if p.requires_grad and "lora_" in n]
            other_params = [p for n,p in named if p.requires_grad and "lora_" not in n]
            head_params  = [p for p in self.cls_head.parameters() if p.requires_grad]

            optimizer_grouped_parameters = []
            if lora_params:
                optimizer_grouped_parameters.append({"params": lora_params, "lr": 2e-5, "weight_decay": self.args.weight_decay})
            if other_params:
                optimizer_grouped_parameters.append({"params": other_params, "lr": 2e-5, "weight_decay": self.args.weight_decay})
            optimizer_grouped_parameters.append({"params": head_params,  "lr": 1e-4, "weight_decay": 0.0})

            self.optimizer = AdamW(optimizer_grouped_parameters)
        return self.optimizer

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        try:
            ds = datasets.get("validation", None)
            acc = gen_eval_accuracy(self.model, self.tokenizer, ds if ds is not None else datasets["validation"], max_examples=128, tag=metric_key_prefix)
            metrics[f"{metric_key_prefix}_accuracy_cell_line"] = float(acc)
            step = getattr(self.state, "global_step", -1)
            print(f"{metric_key_prefix}/accuracy_cell_line={acc:.4f} at step={step}", flush=True)
        except Exception as e:
            print(f"warn/eval_callback_exception: {e}", flush=True)
        return metrics

    def _weighted_lm_loss(self, logits, labels, value_mask, weight_value):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_value  = value_mask[:, 1:].contiguous().to(dtype=shift_logits.dtype)

        valid = (shift_labels != -100)
        if not valid.any():
            return shift_logits.new_tensor(0.0)

        w = torch.ones_like(shift_labels, dtype=shift_logits.dtype, device=shift_logits.device)
        w = torch.where(shift_value > 0, w * weight_value, w)

        ce = F.cross_entropy(
            shift_logits[valid],
            shift_labels[valid],
            reduction='none'
        )
        w_flat = w[valid]
        loss = (ce * w_flat).sum() / w_flat.sum()
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        split_idx  = inputs.pop("split_idx")
        cls_label  = inputs.pop("cls_label")
        value_mask = inputs.pop("value_mask").to(next(model.parameters()).device)
        outputs1   = model(**inputs, output_hidden_states=True)
        lm_loss1   = self._weighted_lm_loss(outputs1.logits, inputs["labels"], value_mask, self.value_token_weight)

        if model.training:
            outputs2 = model(**inputs, output_hidden_states=False)
            lm_loss2 = self._weighted_lm_loss(outputs2.logits, inputs["labels"], value_mask, self.value_token_weight)
            s1 = F.log_softmax(outputs1.logits[:, :-1, :], dim=-1)
            s2 = F.log_softmax(outputs2.logits[:, :-1, :], dim=-1)
            valid = (inputs["labels"][:, 1:] != -100)
            p1 = s1[valid]; p2 = s2[valid]
            kl = 0.5 * (F.kl_div(p1, p2.exp(), reduction='batchmean') + F.kl_div(p2, p1.exp(), reduction='batchmean'))
            lm_loss = 0.5 * (lm_loss1 + lm_loss2) + self.rdrop_alpha * kl
        else:
            lm_loss = lm_loss1

        hidden    = outputs1.hidden_states[-1]
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
            return loss, outputs1
        return loss

##########################################################################################
# TRAIN
training_args = TrainingArguments(
    output_dir=os.path.join(base_path, "checkpoints_cell_line"),
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
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
    metric_for_best_model="eval_accuracy_cell_line",
    greater_is_better=True,
    warmup_ratio=0.10,
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
        EarlyStoppingCallback(early_stopping_patience=2),  # <<< patience ↑
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

##########################################################################################
# SAVE
print("Save adapter only", flush=True)
trainer.model.save_pretrained(adapter_out_dir)
tokenizer.save_pretrained(adapter_out_dir)

##########################################################################################
# EVAL VAL
print("Evaluate on validation with generation (punctuation-insensitive, case-sensitive)", flush=True)
val_acc = gen_eval_accuracy(trainer.model, tokenizer, datasets["validation"], max_examples=None, tag="final_val")
print(f"final/validation_accuracy_cell_line={val_acc:.4f}", flush=True)
writer.add_scalar("final/validation_accuracy_cell_line", val_acc, trainer.state.global_step)
writer.flush()

##########################################################################################
# EVAL TEST (même comparateur)
print("Generate on test set (punctuation-insensitive, case-sensitive)", flush=True)
dev = next(trainer.model.parameters()).device
rows = datasets["test"].to_pandas()
n_ok, n_all = 0, 0
for i, r in rows.iterrows():
    prompt   = r["prompt"]
    expected = r["output_json"]
    ref_val  = _extract_value(expected)
    prefix = prompt + '{"cell_line": "'
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
    pred_val = re.split(r'["\n,}]', cont)[0]
    pred_json = f'{{"cell_line": "{_escape_json_val(pred_val)}"}}'
    ok = _cmp_ignore_punct_keep_case(pred_val, ref_val)
    print(f"--- Predicted output {i+1}: {pred_json}", flush=True)
    print(f"--- Expected output  {i+1}: {expected}", flush=True)
    print(f"--- pred_norm=\"{_strip_punct_keep_case(pred_val)}\" ref_norm=\"{_strip_punct_keep_case(ref_val)}\" match={ok}", flush=True)
    print("-"*50, flush=True)
    n_ok += int(ok); n_all += 1

test_acc = (n_ok / max(1, n_all))
print(f"final/test_accuracy_cell_line={test_acc:.4f}", flush=True)
writer.add_scalar("final/test_accuracy_cell_line", test_acc, trainer.state.global_step)
writer.close()
