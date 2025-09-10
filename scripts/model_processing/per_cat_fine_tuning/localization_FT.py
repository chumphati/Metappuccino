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
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description="Fine-tune localization (multitask) with soft semantic accuracy and span extraction")
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--span_loss_weight", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
base_path = args.base_path

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

train_file = os.path.join(base_path, "finetune_data_train.csv")
val_file   = os.path.join(base_path, "finetune_data_val.csv")
test_file  = os.path.join(base_path, "finetune_data_test.csv")
model_name = os.path.join(base_path, "Mistral-7B-Instruct-v0.3")

tb_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/logs10/tensorboard_localization_v2"
adapter_out_dir = os.path.join(base_path, "cat_localization")
os.makedirs(tb_dir, exist_ok=True); os.makedirs(adapter_out_dir, exist_ok=True)

os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
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

RULES = (
    "\nTask: Extract the localization (place/site) from the context. "
    "Answer with the shortest noun phrase that names the localization only. "
    "If no localization is given, answer 'not applicable'. Return strictly JSON.\n"
)

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
    return str(v).replace('\\\\','\\\\\\\\').replace('"','\\"')

def _extract_value(txt):
    try:
        j = json.loads(txt)
        v = j.get("localization","")
        return "" if v is None else str(v)
    except Exception:
        m = re.search(r'"localization"\s*:\s*"([^"]*)"', str(txt))
        return m.group(1) if m else str(txt)

for _df in (df_train, df_val, df_test):
    _df["output_raw"]  = _df["output"].astype(str)
    _df["output_json"] = _df["output_raw"].apply(lambda v: f'{{"localization": "{_escape_json_val(v)}"}}')

datasets = {
    "train": Dataset.from_pandas(df_train[["prompt","output_json"]]),
    "validation": Dataset.from_pandas(df_val[["prompt","output_json"]]),
    "test": Dataset.from_pandas(df_test[["prompt","output_json"]])
}
print({k: len(v) for k,v in datasets.items()}, flush=True)

def _find_span_offsets(text, value):
    t = str(text)
    v = str(value)
    i = t.lower().find(v.lower())
    if i == -1:
        return (-1, -1)
    return (i, i + len(v))

def tokenize_fn(example):
    prompt   = example["prompt"].strip()
    out_json = example["output_json"].strip()
    m = re.search(r'"localization"\s*:\s*"([^"]*)"', out_json)
    value_str  = m.group(1) if m else ""
    prefix_str = '{"localization": "'
    suffix_str = '"}'
    max_in, max_out, max_total = 3500, 64, 3568

    in_enc  = tokenizer(prompt + RULES, truncation=True, padding=False, max_length=max_in, add_special_tokens=False, return_offsets_mapping=True)
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

    split_idx = len(in_enc["input_ids"])
    span_start_tok = -1
    span_end_tok = -1
    raw_prompt = prompt + RULES
    c_start, c_end = _find_span_offsets(raw_prompt, value_str)
    if c_start != -1:
        offsets = in_enc["offset_mapping"]
        for ti, (a,b) in enumerate(offsets):
            if a <= c_start < b and span_start_tok == -1:
                span_start_tok = ti
            if a < c_end <= b and span_end_tok == -1:
                span_end_tok = ti
        if span_start_tok != -1 and span_end_tok == -1:
            for ti in range(span_start_tok, len(offsets)):
                a,b = offsets[ti]
                if b >= c_end:
                    span_end_tok = ti
                    break
        if span_start_tok != -1 and span_end_tok == -1:
            span_end_tok = min(len(offsets)-1, span_start_tok)

    return {
        "input_ids": ids,
        "attention_mask": attn,
        "labels": labels,
        "split_idx": split_idx,
        "span_start": int(span_start_tok) if (0 <= span_start_tok < split_idx) else -1,
        "span_end": int(span_end_tok) if (0 <= span_end_tok < split_idx) else -1,
    }

tokenized = {
    k: v.map(tokenize_fn, remove_columns=v.column_names, load_from_cache_file=False, desc=f"Tokenize {k}")
    for k, v in datasets.items()
}

class CausalLMPadCollator:
    def __init__(self, tokenizer):
        self._padder = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    def __call__(self, features):
        labels     = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        split_idx  = torch.tensor([int(f.get("split_idx", 0)) for f in features], dtype=torch.long)
        span_start = torch.tensor([int(f.get("span_start", -1)) for f in features], dtype=torch.long)
        span_end   = torch.tensor([int(f.get("span_end", -1)) for f in features], dtype=torch.long)
        for f in features:
            f.pop("labels", None); f.pop("split_idx", None); f.pop("span_start", None); f.pop("span_end", None)
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
        batch["labels"]     = torch.stack(padded, dim=0)
        batch["split_idx"]  = split_idx
        batch["span_start"] = span_start
        batch["span_end"]   = span_end
        return batch

data_collator = CausalLMPadCollator(tokenizer)

sem_model_name = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
model_sem = SentenceTransformer(sem_model_name)
SIM_THRESH = 0.45

@torch.no_grad()
def constrained_generate_value(model, prompt_text):
    dev = next(model.parameters()).device
    prefix = prompt_text + RULES + '{"localization": "'
    enc = tokenizer(prefix, return_tensors="pt", truncation=True, max_length=3500, add_special_tokens=False)
    prompt_ids = enc["input_ids"][0].tolist()
    inp = {k: v.to(dev) for k,v in enc.items()}
    ids = inp["input_ids"]
    gen = []
    max_tokens = 32
    min_content_tokens = 1
    allowed_re = re.compile(r"[A-Za-z0-9 \-_/(),.'’]")
    prompt_id_set = set(prompt_ids)
    for step in range(max_tokens):
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids, device=dev), use_cache=False)
        logits = out.logits[:, -1, :]
        boost = 0.6
        logits[:, list(prompt_id_set)] += boost
        if gen:
            logits[:, gen[-1]] -= 1.0
        if tokenizer.eos_token_id is not None:
            logits[:, tokenizer.eos_token_id] = -float("inf")
        next_id = torch.argmax(logits, dim=-1, keepdim=True)
        tid = int(next_id.item())
        piece = tokenizer.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if ('"' in piece) or any(c in piece for c in ['}', '\n', '`']):
            if len([t for t in gen if tokenizer.decode([t]).strip()]) >= min_content_tokens:
                break
            else:
                logits[:, tid] = -float("inf")
                next_id = torch.argmax(logits, dim=-1, keepdim=True)
                tid = int(next_id.item())
                piece = tokenizer.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        if not all(allowed_re.match(ch) for ch in piece):
            logits[:, tid] = -float("inf")
            next_id = torch.argmax(logits, dim=-1, keepdim=True)
            tid = int(next_id.item())
            piece = tokenizer.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        ids = torch.cat([ids, next_id], dim=1)
        gen.append(tid)
    text = tokenizer.decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    text = re.split(r'[";\n`,}:()\[\]]', text)[0]
    text = re.sub(r'\s+', ' ', text).strip(' ,;.-').strip()
    if not text:
        text = "not applicable"
    return text

def _cos_sim(a, b):
    an = a / max(np.linalg.norm(a), 1e-12)
    bn = b / max(np.linalg.norm(b), 1e-12)
    return float(np.dot(an, bn))

def gen_eval_soft_accuracy(model, tokenizer, hf_dataset, max_examples=None, tag="eval"):
    model.eval()
    dev = next(model.parameters()).device
    n_ok, n_all = 0, 0
    rows = hf_dataset.to_pandas()
    if max_examples is not None:
        rows = rows.sample(min(max_examples, len(rows)), random_state=42)
    ref_texts = [ _extract_value(r["output_json"]) for _, r in rows.iterrows() ]
    ref_embs = model_sem.encode(ref_texts, convert_to_numpy=True, normalize_embeddings=True)
    for idx_row, r in enumerate(rows.itertuples(index=False)):
        prompt   = getattr(r, "prompt")
        expected = getattr(r, "output_json")
        ref_val  = ref_texts[idx_row]
        pred_val = constrained_generate_value(model, prompt)
        pred_emb = model_sem.encode(pred_val, convert_to_numpy=True, normalize_embeddings=True)
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
        print(f"eval/accuracy_localization={acc:.4f} at step={state.global_step}", flush=True)
        writer.add_scalar("eval/accuracy_localization", acc, state.global_step)
        if isinstance(metrics, dict):
            metrics["eval_accuracy_localization"] = float(acc)

print("Initial evaluation on validation with BASE model (semantic soft accuracy + copy-biased decoding)", flush=True)
base_model.eval()
init_val_acc = gen_eval_soft_accuracy(base_model, tokenizer, datasets["validation"], max_examples=None, tag="initial")
print(f"initial/validation_accuracy_localization={init_val_acc:.4f}", flush=True)
writer.add_scalar("initial/validation_accuracy_localization", init_val_acc, 0)
writer.flush()

hidden_size = getattr(peft_model.config, "hidden_size", None) or getattr(peft_model.config, "n_embd", None)

class SpanHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.start = nn.Linear(hidden, 1)
        self.end   = nn.Linear(hidden, 1)
    def forward(self, h):
        start_logits = self.start(h).squeeze(-1)
        end_logits   = self.end(h).squeeze(-1)
        return start_logits, end_logits

span_head = SpanHead(hidden_size).to(next(peft_model.parameters()).device)

class MultiTaskTrainer(Trainer):
    def __init__(self, span_head: nn.Module, span_weight: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.span_head = span_head
        self.span_weight = float(span_weight)

    def create_optimizer(self):
        if self.optimizer is None:
            model_params = [p for p in self.model.parameters() if p.requires_grad]
            head_params  = [p for p in self.span_head.parameters() if p.requires_grad]
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
            metrics[f"{metric_key_prefix}_accuracy_localization"] = float(acc)
            step = getattr(self.state, "global_step", -1)
            print(f"{metric_key_prefix}/accuracy_localization={acc:.4f} at step={step}", flush=True)
        except Exception as e:
            print(f"warn/eval_callback_exception: {e}", flush=True)
        return metrics

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        split_idx  = inputs.pop("split_idx")
        span_start = inputs.pop("span_start")
        span_end   = inputs.pop("span_end")
        outputs    = model(**inputs, output_hidden_states=True)
        lm_loss    = outputs.loss
        hidden     = outputs.hidden_states[-1]
        B, T, H = hidden.size()
        start_logits, end_logits = self.span_head(hidden)
        ar = torch.arange(T, device=hidden.device).unsqueeze(0).expand(B, T)
        mask = ar < split_idx.unsqueeze(1)
        start_logits = start_logits.masked_fill(~mask, -1e9)
        end_logits   = end_logits.masked_fill(~mask, -1e9)
        if (span_start >= 0).any() and (span_end >= 0).any():
            span_loss = F.cross_entropy(start_logits, span_start, ignore_index=-1) + \
                        F.cross_entropy(end_logits,   span_end,   ignore_index=-1)
        else:
            span_loss = torch.tensor(0.0, device=hidden.device, dtype=hidden.dtype)
        loss = lm_loss + self.span_weight * span_loss
        if return_outputs:
            return loss, outputs
        return loss

training_args = TrainingArguments(
    output_dir=os.path.join(base_path, "checkpoints_localization_v2"),
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
    metric_for_best_model="eval_accuracy_localization",
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
        GenEvalCallback(datasets["validation"], max_examples=128)
    ],
    span_head=span_head,
    span_weight=args.span_loss_weight,
)

print("Begin training", flush=True)
trainer.train()
writer.flush()

print("Save adapter only", flush=True)
trainer.model.save_pretrained(adapter_out_dir)
tokenizer.save_pretrained(adapter_out_dir)

print("Evaluate on validation with generation (semantic soft accuracy + copy-biased decoding)", flush=True)
val_acc = gen_eval_soft_accuracy(trainer.model, tokenizer, datasets["validation"], max_examples=None, tag="final_val")
print(f"final/validation_accuracy_localization={val_acc:.4f}", flush=True)
writer.add_scalar("final/validation_accuracy_localization", val_acc, trainer.state.global_step)
writer.flush()

print("Generate on test set (semantic soft accuracy + copy-biased decoding)", flush=True)
dev = next(trainer.model.parameters()).device
rows = datasets["test"].to_pandas()
n_ok, n_all = 0, 0
ref_texts = [ _extract_value(r["output_json"]) for _, r in rows.iterrows() ]
ref_embs = model_sem.encode(ref_texts, convert_to_numpy=True, normalize_embeddings=True)
for i, r in enumerate(rows.itertuples(index=False)):
    prompt   = getattr(r, "prompt")
    expected = getattr(r, "output_json")
    ref_val  = ref_texts[i]
    pred_val = constrained_generate_value(trainer.model, prompt)
    pred_json = f'{{"localization": "{_escape_json_val(pred_val)}"}}'
    pred_emb = model_sem.encode(pred_val, convert_to_numpy=True, normalize_embeddings=True)
    sim = _cos_sim(pred_emb, ref_embs[i])
    ok = sim >= SIM_THRESH
    print(f"--- Predicted output {i+1}: {pred_json}", flush=True)
    print(f"--- Expected output  {i+1}: {expected}", flush=True)
    print(f"--- cos_sim={sim:.4f} match={ok}", flush=True)
    print("-"*50, flush=True)
    n_ok += int(ok); n_all += 1

test_acc = (n_ok / max(1, n_all))
print(f"final/test_accuracy_localization={test_acc:.4f}", flush=True)
writer.add_scalar("final/test_accuracy_localization", test_acc, trainer.state.global_step)
writer.close()
