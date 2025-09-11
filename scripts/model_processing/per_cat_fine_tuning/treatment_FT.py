import os, re, json, math, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from datasets import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer,
    DataCollatorWithPadding, EarlyStoppingCallback, TrainerCallback
)
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from sentence_transformers import SentenceTransformer

parser = argparse.ArgumentParser(description="Fine-tune treatment/intervention with semantic InfoNCE + copy alignment + R-Drop + balanced sampling")
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--sem_loss_weight", type=float, default=0.6)
parser.add_argument("--copy_loss_weight", type=float, default=0.3)
parser.add_argument("--rdrop_alpha", type=float, default=0.3)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
base_path = args.base_path

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

train_file = os.path.join(base_path, "finetune_data_train.csv")
val_file   = os.path.join(base_path, "finetune_data_val.csv")
test_file  = os.path.join(base_path, "finetune_data_test.csv")
model_name = os.path.join(base_path, "Mistral-7B-Instruct-v0.3")

tb_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/logs15/tensorboard_treatment"
adapter_out_dir = os.path.join(base_path, "cat_treatment")
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
    r=32,
    lora_alpha=64,
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
print(f"Train/Val/Test sizes: {len(df_train)}/{len(df_val)}/{len(df_test)} (dropped {before-len(df_train)} dup/empty)", flush=True)

SIM_THRESH = 0.4

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
    _df["output_raw"]  = _df["output"].astype(str)
    _df["output_json"] = _df["output_raw"].apply(lambda v: f'{{"treatment": "{_escape_json_val(v)}"}}')

sem_model_name = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
model_sem = SentenceTransformer(sem_model_name)

def _sem_vecs(series):
    vals = [ _extract_value(s) for s in series.tolist() ]
    vecs = model_sem.encode(vals, convert_to_numpy=True, normalize_embeddings=True)
    mask = [ 0 if str(v).strip().lower() in UNK_TOKS or len(str(v).strip())<2 else 1 for v in vals ]
    return vecs.astype(np.float32), np.array(mask, dtype=np.int64)

vecs_train, mask_train = _sem_vecs(df_train["output_json"])
vecs_val,   mask_val   = _sem_vecs(df_val["output_json"])
vecs_test,  mask_test  = _sem_vecs(df_test["output_json"])

def _mk_ds(df, vecs, mask):
    df = df.copy()
    vecs = np.asarray(vecs, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.int64)
    vecs = vecs.copy()
    for i in range(len(vecs)):
        if mask[i] == 0: vecs[i] = 0.0
    df["sem_target"] = [v.tolist() for v in vecs]
    df["sem_valid"]  = mask.tolist()
    return Dataset.from_pandas(df[["prompt","output_json","sem_target","sem_valid"]])

datasets = {
    "train": _mk_ds(df_train, vecs_train, mask_train),
    "validation": _mk_ds(df_val, vecs_val, mask_val),
    "test": _mk_ds(df_test, vecs_test, mask_test)
}
print({k: len(v) for k,v in datasets.items()}, flush=True)

def _find_subseq(hay, needle):
    if len(needle)==0 or len(hay)==0 or len(needle)<2: return -1, -1
    H, N = len(hay), len(needle)
    for i in range(H - N + 1):
        if hay[i:i+N] == needle: return i, i+N
    return -1, -1

def tokenize_fn(example):
    prompt   = example["prompt"].strip()
    out_json = example["output_json"].strip()
    m = re.search(r'"treatment"\s*:\s*"([^"]*)"', out_json)
    value_str  = m.group(1) if m else ""
    prefix_str = '{"treatment": "'
    suffix_str = '"}'
    max_in, max_out, max_total = 3500, 96, 4096

    in_enc  = tokenizer(prompt, truncation=True, padding=False, max_length=max_in, add_special_tokens=False)
    pref_ids = tokenizer(prefix_str, add_special_tokens=False)["input_ids"]
    val_ids  = tokenizer(value_str, truncation=True, padding=False, max_length=max_out, add_special_tokens=False)["input_ids"]
    suff_ids = tokenizer(suffix_str, add_special_tokens=False)["input_ids"]

    out_ids = pref_ids + val_ids + suff_ids
    ids  = in_enc["input_ids"] + out_ids
    attn = [1]*len(ids)

    sem_valid_flag = int(example["sem_valid"]) if "sem_valid" in example else 1
    if sem_valid_flag == 1:
        labels_out_vals = val_ids
        labels_suf0 = suff_ids[0] if len(suff_ids)>0 else -100
    else:
        labels_out_vals = [-100]*len(val_ids)
        labels_suf0 = -100

    labels_out = [-100]*len(pref_ids) + labels_out_vals
    if len(suff_ids) > 0:
        labels_out += [labels_suf0] + [-100]*(len(suff_ids)-1)
    labels = [-100]*len(in_enc["input_ids"]) + labels_out

    if len(ids) > max_total:
        ids = ids[:max_total]; attn = attn[:max_total]; labels = labels[:max_total]

    s_copy, e_copy = _find_subseq(in_enc["input_ids"], val_ids)

    return {
        "input_ids": ids,
        "attention_mask": attn,
        "labels": labels,
        "split_idx": len(in_enc["input_ids"]),
        "pref_len": len(pref_ids),
        "val_len": len(val_ids),
        "sem_target": example["sem_target"],
        "sem_valid": sem_valid_flag,
        "copy_start": s_copy,
        "copy_end": e_copy,
    }

class CausalLMPadCollator:
    def __init__(self, tokenizer):
        self._padder = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    def __call__(self, features):
        labels    = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        split_idx = torch.tensor([int(f.get("split_idx", 0)) for f in features], dtype=torch.long)
        pref_len  = torch.tensor([int(f.get("pref_len", 0)) for f in features], dtype=torch.long)
        val_len   = torch.tensor([int(f.get("val_len", 0)) for f in features], dtype=torch.long)
        sem_valid = torch.tensor([int(f.get("sem_valid", 0)) for f in features], dtype=torch.long)
        copy_s    = torch.tensor([int(f.get("copy_start", -1)) for f in features], dtype=torch.long)
        copy_e    = torch.tensor([int(f.get("copy_end", -1)) for f in features], dtype=torch.long)
        raw_sem = [f.get("sem_target") for f in features]
        dim = None
        for v in raw_sem:
            if v is not None:
                dim = len(v) if hasattr(v, "__len__") else 768
                break
        dim = 768 if dim is None else dim
        fixed = []
        for v in raw_sem:
            if v is None: fixed.append([0.0]*dim)
            else:         fixed.append(v.tolist() if hasattr(v, "tolist") else list(v))
        sem_tgts  = torch.tensor(fixed, dtype=torch.float32)
        for f in features:
            f.pop("labels", None); [f.pop(k, None) for k in ("split_idx","pref_len","val_len","sem_target","sem_valid","copy_start","copy_end")]
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
        batch["pref_len"]   = pref_len
        batch["val_len"]    = val_len
        batch["sem_target"] = sem_tgts
        batch["sem_valid"]  = sem_valid
        batch["copy_start"] = copy_s
        batch["copy_end"]   = copy_e
        return batch

tokenized = {k: v.map(tokenize_fn, remove_columns=v.column_names) for k, v in datasets.items()}
data_collator = CausalLMPadCollator(tokenizer)

print("Initial evaluation on validation with BASE model (semantic soft accuracy)", flush=True)

def _cos_sim(a, b):
    an = a / max(np.linalg.norm(a), 1e-12)
    bn = b / max(np.linalg.norm(b), 1e-12)
    return float(np.dot(an, bn))

@torch.no_grad()
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
        ref_val  = ref_texts[idx_row]
        prefix = prompt + '{"treatment": "'
        inp = tokenizer(prefix, return_tensors="pt").to(dev)
        out_ids = model.generate(
            **inp,
            max_new_tokens=96,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
        cont = tokenizer.decode(out_ids[0][inp["input_ids"].size(1):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        pred_val = re.split(r'["\n,}]', cont)[0]
        pred_emb = model_sem.encode(pred_val, convert_to_numpy=True, normalize_embeddings=True)
        sim = _cos_sim(pred_emb, ref_embs[idx_row])
        ok = sim >= SIM_THRESH
        n_ok += int(ok); n_all += 1
        print(f'{tag}/pair {n_all}: pred="{pred_val}" ref="{ref_val}" cos_sim={sim:.4f} match={ok}', flush=True)
    model.train()
    return (n_ok / max(1, n_all))

init_val_acc = gen_eval_soft_accuracy(base_model, tokenizer, datasets["validation"], max_examples=None, tag="initial")
print(f"initial/validation_accuracy_treatment={init_val_acc:.4f}", flush=True)

writer = SummaryWriter(tb_dir)
writer.add_scalar("initial/validation_accuracy_treatment", init_val_acc, 0)
writer.flush()

hidden_size = getattr(peft_model.config, "hidden_size", None) or getattr(peft_model.config, "n_embd", None)
sem_dim = 768
sem_head = nn.Linear(hidden_size, sem_dim).to(next(peft_model.parameters()).device)

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

train_vals = datasets["train"].to_pandas()["output_json"].apply(_extract_value).str.strip().str.lower()
freq = train_vals.value_counts().to_dict()
def _weight_for_val(v):
    f = max(1, freq.get(v, 1))
    base = 1.0 / math.sqrt(f)
    if v in UNK_TOKS: base *= 0.5
    return base
train_sample_weights = torch.tensor([_weight_for_val(v) for v in train_vals], dtype=torch.float)

class SemMultiTaskTrainer(Trainer):
    def __init__(self, sem_head: nn.Module, sem_weight: float = 0.6, copy_weight: float = 0.3, rdrop_alpha: float = 0.3, **kwargs):
        super().__init__(**kwargs)
        self.sem_head = sem_head
        self.sem_weight = float(sem_weight)
        self.copy_weight = float(copy_weight)
        self.infoNCE_temp = 0.07
        self.rdrop_alpha = float(rdrop_alpha)
    def create_optimizer(self):
        if self.optimizer is None:
            model_params = [p for p in self.model.parameters() if p.requires_grad]
            head_params  = [p for p in self.sem_head.parameters() if p.requires_grad]
            optimizer_grouped_parameters = [
                {"params": model_params, "weight_decay": self.args.weight_decay},
                {"params": head_params,  "weight_decay": 0.05},
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        return self.optimizer
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return super().get_train_dataloader()
        sampler = WeightedRandomSampler(weights=train_sample_weights, num_samples=len(self.train_dataset), replacement=True)
        return DataLoader(self.train_dataset, batch_size=self.args.per_device_train_batch_size, sampler=sampler, collate_fn=self.data_collator, drop_last=self.args.dataloader_drop_last)
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
    def _lm_ce_with_smoothing(self, logits, labels, smoothing=0.1):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        V = shift_logits.size(-1)
        loss = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            ignore_index=-100,
            label_smoothing=smoothing
        )
        return loss
    def _pool_out_span(self, hidden, split_idx, pref_len, val_len):
        B, T, H = hidden.size()
        start = (split_idx + pref_len).clamp(min=0, max=T-1)
        end   = (start + val_len).clamp(min=0, max=T)
        pooled = []
        for b in range(B):
            s = int(start[b].item()); e = int(end[b].item())
            if e <= s:
                pooled.append(hidden[b, int(max(0, s-1)), :])
            else:
                pooled.append(hidden[b, s:e, :].mean(dim=0))
        return torch.stack(pooled, dim=0), start, end
    def _sem_loss(self, pooled, sem_target, sem_valid):
        pred_vec = F.normalize(self.sem_head(pooled), dim=-1)
        tgt_vec  = F.normalize(sem_target.to(pred_vec.dtype), dim=-1)
        mask = (sem_valid > 0)
        if mask.sum() >= 2:
            z = pred_vec[mask]; y = tgt_vec[mask]
            logits = (z @ y.T) / self.infoNCE_temp
            labels = torch.arange(z.size(0), device=z.device)
            return F.cross_entropy(logits, labels)
        elif mask.any():
            return (1.0 - (pred_vec[mask] * tgt_vec[mask]).sum(dim=-1)).mean()
        else:
            return torch.tensor(0.0, device=pooled.device)
    def _copy_align_loss(self, hidden, pooled_out, copy_start, copy_end):
        B, T, H = hidden.size()
        has_copy = (copy_start >= 0) & (copy_end > copy_start)
        if not has_copy.any():
            return torch.tensor(0.0, device=hidden.device)
        reps_in, reps_out = [], []
        for b in range(B):
            if bool(has_copy[b].item()):
                s = int(copy_start[b].item()); e = int(copy_end[b].item())
                s = max(0, min(s, T-1)); e = max(s+1, min(e, T))
                reps_in.append(hidden[b, s:e, :].mean(dim=0))
                reps_out.append(pooled_out[b])
        IN  = F.normalize(torch.stack(reps_in, dim=0), dim=-1)
        OUT = F.normalize(torch.stack(reps_out, dim=0), dim=-1)
        return (1.0 - (IN * OUT).sum(dim=-1)).mean()
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        split_idx  = inputs.pop("split_idx")
        pref_len   = inputs.pop("pref_len")
        val_len    = inputs.pop("val_len")
        sem_target = inputs.pop("sem_target")
        sem_valid  = inputs.pop("sem_valid")
        copy_start = inputs.pop("copy_start")
        copy_end   = inputs.pop("copy_end")
        labels     = inputs.get("labels")

        out1 = model(**inputs, output_hidden_states=True)
        out2 = model(**inputs, output_hidden_states=True)

        lm1 = self._lm_ce_with_smoothing(out1.logits, labels, smoothing=0.1)
        lm2 = self._lm_ce_with_smoothing(out2.logits, labels, smoothing=0.1)

        pooled1, start, end = self._pool_out_span(out1.hidden_states[-1], split_idx, pref_len, val_len)
        pooled2, _,   _    = self._pool_out_span(out2.hidden_states[-1], split_idx, pref_len, val_len)

        sem1 = self._sem_loss(pooled1, sem_target, sem_valid)
        sem2 = self._sem_loss(pooled2, sem_target, sem_valid)

        copy1 = self._copy_align_loss(out1.hidden_states[-1], pooled1, copy_start, copy_end)
        copy2 = self._copy_align_loss(out2.hidden_states[-1], pooled2, copy_start, copy_end)

        def _gather_value_logits(logits):
            B, T, V = logits.size()
            rows = []
            for b in range(B):
                s = int(start[b]); e = int(end[b])
                if e > s:
                    rows.append(logits[b, s:e, :])
            if len(rows)==0:
                return logits.new_zeros((0, logits.size(-1)))
            return torch.cat([r.reshape(-1, V) for r in rows], dim=0)

        L1 = _gather_value_logits(out1.logits)
        L2 = _gather_value_logits(out2.logits)
        if L1.numel() > 0 and L2.numel() > 0:
            p1 = F.log_softmax(L1, dim=-1); q1 = F.softmax(L2, dim=-1)
            p2 = F.log_softmax(L2, dim=-1); q2 = F.softmax(L1, dim=-1)
            kl = 0.5*(F.kl_div(p1, q1, reduction="batchmean") + F.kl_div(p2, q2, reduction="batchmean"))
        else:
            kl = torch.tensor(0.0, device=labels.device)

        loss = 0.5*(lm1+lm2) + self.sem_weight*0.5*(sem1+sem2) + self.copy_weight*0.5*(copy1+copy2) + self.rdrop_alpha*kl
        if return_outputs:
            return loss, out1
        return loss

training_args = TrainingArguments(
    output_dir=os.path.join(base_path, "checkpoints_treatment_semhead"),
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    save_total_limit=3,
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=8,
    weight_decay=0.1,
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
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    max_grad_norm=1.0,
    remove_unused_columns=False, 
)

trainer = SemMultiTaskTrainer(
    model=peft_model,
    args=training_args,
    train_dataset= tokenized["train"],
    eval_dataset= tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),
        TBCallback(),
        PrintProgressCallback(),
        GradNormLogger(every=50),
        GenEvalCallback(datasets["validation"], max_examples=128)
    ],
    sem_head=sem_head,
    sem_weight=args.sem_loss_weight,
    copy_weight=args.copy_loss_weight,
    rdrop_alpha=args.rdrop_alpha
)

print("Begin training", flush=True)
trainer.train()
writer.flush()

print("Save adapter only", flush=True)
trainer.model.save_pretrained(adapter_out_dir)
tokenizer.save_pretrained(adapter_out_dir)

print("Evaluate on validation with generation (semantic soft accuracy)", flush=True)
val_acc = gen_eval_soft_accuracy(trainer.model, tokenizer, datasets["validation"], max_examples=None, tag="final_val")
print(f"final/validation_accuracy_treatment={val_acc:.4f}", flush=True)
writer.add_scalar("final/validation_accuracy_treatment", val_acc, trainer.state.global_step)
writer.flush()

print("Generate on test set (semantic soft accuracy)", flush=True)
dev = next(trainer.model.parameters()).device
rows = datasets["test"].to_pandas()
n_ok, n_all = 0, 0
ref_texts = [ _extract_value(r["output_json"]) for _, r in rows.iterrows() ]
ref_embs = model_sem.encode(ref_texts, convert_to_numpy=True, normalize_embeddings=True)
for i, r in enumerate(rows.itertuples(index=False)):
    prompt   = getattr(r, "prompt")
    expected = getattr(r, "output_json")
    ref_val  = ref_texts[i]
    prefix = prompt + '{"treatment": "'
    inp = tokenizer(prefix, return_tensors="pt").to(dev)
    with torch.no_grad():
        out_ids = trainer.model.generate(
            **inp,
            max_new_tokens=96,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    cont = tokenizer.decode(out_ids[0][inp["input_ids"].size(1):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    pred_val = re.split(r'["\n,}]', cont)[0]
    pred_json = f'{{"treatment": "{_escape_json_val(pred_val)}"}}'
    pred_emb = model_sem.encode(pred_val, convert_to_numpy=True, normalize_embeddings=True)
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
