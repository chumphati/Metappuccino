import os, re, json, argparse, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModelForCausalLM
from collections import Counter

parser = argparse.ArgumentParser(description="Simple 2-epoch baseline for sequencing_source (robust, no comments)")
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--lr", type=float, default=5e-4)
parser.add_argument("--max_len", type=int, default=2048)
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_path = args.base_path
train_file = os.path.join(base_path, "finetune_data_train.csv")
val_file   = os.path.join(base_path, "finetune_data_val.csv")
test_file  = os.path.join(base_path, "finetune_data_test.csv")
model_name = os.path.join(base_path, "Mistral-7B-Instruct-v0.3")

print(f"device={device}")

ALLOWED = ["spatial","bulk","single cell"]
label2id = {k:i for i,k in enumerate(ALLOWED)}
id2label = {i:k for k,i in label2id.items()}

def _canon_src(s):
    t = re.sub(r"\s+", " ", str(s).strip().lower())
    if ("visium" in t or "xenium" in t or "cosmx" in t or "geomx" in t or "slide-seq" in t or "slideseq" in t or "merfish" in t or "seqfish" in t or "stereo-seq" in t or "stereoseq" in t or "hdst" in t or "spatial" in t):
        return "spatial"
    if (re.search(r"\bsingle[\s-]?cell\b", t) or "singlecell" in t or "single nucleus" in t or "single-nucleus" in t or "snrna" in t or "snrna-seq" in t or "sn rna" in t or "scrna" in t or "sc rna" in t or "scrna-seq" in t or "10x chromium" in t or "chromium" in t or "drop-seq" in t or "indrops" in t or "split-seq" in t or "smart-seq" in t or "smartseq" in t):
        return "single cell"
    return "bulk"

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
    _df["label_txt"] = _df["output"].map(_canon_src)
    _df["label_id"]  = _df["label_txt"].map(lambda x: label2id.get(x, label2id["bulk"]))

print(f"sizes train/val/test: {len(df_train)}/{len(df_val)}/{len(df_test)}")
print("class distribution train:", dict(Counter(df_train["label_txt"])) )
print("class distribution val:", dict(Counter(df_val["label_txt"])) )
print("class distribution test:", dict(Counter(df_test["label_txt"])) )

missing = [k for k in ALLOWED if k not in set(df_train["label_txt"]) ]
if missing:
    print("warning: missing classes in train:", missing)

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
try:
    tokenizer.padding_side = "right"
except Exception:
    pass

class TxtClsDataset(TorchDataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df.reset_index(drop=True)
        self.tok = tokenizer
        self.max_len = max_len
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        enc = self.tok(str(row.prompt), truncation=True, padding=False, max_length=self.max_len, add_special_tokens=False)
        return {"input_ids": torch.tensor(enc["input_ids"], dtype=torch.long), "attention_mask": torch.tensor([1]*len(enc["input_ids"]), dtype=torch.long), "label": int(row.label_id)}

def pad_collate(batch):
    maxlen = max(len(x["input_ids"]) for x in batch)
    ids = []
    mask = []
    labels = []
    for x in batch:
        t = x["input_ids"]
        m = x["attention_mask"]
        if len(t) < maxlen:
            pad = torch.full((maxlen - len(t),), tokenizer.pad_token_id, dtype=torch.long)
            t = torch.cat([t, pad], dim=0)
            m = torch.cat([m, torch.zeros((pad.numel(),), dtype=torch.long)], dim=0)
        ids.append(t.unsqueeze(0))
        mask.append(m.unsqueeze(0))
        labels.append(x["label"])
    return {"input_ids": torch.cat(ids, dim=0), "attention_mask": torch.cat(mask, dim=0), "labels": torch.tensor(labels, dtype=torch.long)}

train_ds = TxtClsDataset(df_train[["prompt","label_id"]], tokenizer, args.max_len)
val_ds   = TxtClsDataset(df_val[["prompt","label_id"]], tokenizer, args.max_len)
test_ds  = TxtClsDataset(df_test[["prompt","label_id"]], tokenizer, args.max_len)

counts = Counter(df_train["label_id"].tolist())
inv_freq = [1.0 / counts.get(i, 1) for i in df_train["label_id"].tolist()]
train_sampler = WeightedRandomSampler(weights=torch.tensor(inv_freq, dtype=torch.double), num_samples=len(inv_freq), replacement=True)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler, collate_fn=pad_collate, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=pad_collate, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, collate_fn=pad_collate, pin_memory=True)

use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
dtype = torch.bfloat16 if use_bf16 else torch.float16

print(f"loading base model: {model_name}")
base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto", low_cpu_mem_usage=True)
if tokenizer.pad_token_id is not None and base_model.get_input_embeddings().num_embeddings != len(tokenizer):
    base_model.resize_token_embeddings(len(tokenizer))
base_model.eval()
for p in base_model.parameters():
    p.requires_grad = False

hidden_size = getattr(base_model.config, "hidden_size", None) or getattr(base_model.config, "n_embd", None)
head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.2), nn.Linear(hidden_size, len(ALLOWED)))
head.to(device)

class_counts = torch.tensor([counts.get(i, 0) for i in range(len(ALLOWED))], dtype=torch.float32)
class_weights = (1.0 / torch.clamp(class_counts, min=1.0))
class_weights = class_weights * (len(ALLOWED) / torch.sum(class_weights))
print("class weights (inverse frequency, normalized):", class_weights.tolist())

criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)


def forward_encode(batch):
    with torch.no_grad():
        out = base_model(input_ids=batch["input_ids"].to(base_model.device), attention_mask=batch["attention_mask"].to(base_model.device), output_hidden_states=True, use_cache=False)
        h = out.hidden_states[-1].to(torch.float32)
        m = batch["attention_mask"].to(base_model.device).unsqueeze(-1).to(h.dtype)
        s = (h * m).sum(dim=1)
        d = m.sum(dim=1).clamp_min(1.0)
        pooled = (s / d).to(device)
    return pooled


def evaluate(dloader, tag):
    head.eval()
    n_ok, n_all = 0, 0
    per_cls = {i: {"tp":0, "tot":0} for i in range(len(ALLOWED))}
    for batch in dloader:
        pooled = forward_encode(batch)
        logits = head(pooled)
        pred = torch.argmax(logits, dim=-1).cpu().tolist()
        refs = batch["labels"].cpu().tolist()
        for p, r in zip(pred, refs):
            ok = int(p == r)
            n_ok += ok; n_all += 1
            per_cls[r]["tp"] += ok; per_cls[r]["tot"] += 1
    acc = n_ok / max(1, n_all)
    recalls = []
    for i in range(len(ALLOWED)):
        tot = max(1, per_cls[i]["tot"])
        rec = per_cls[i]["tp"] / tot
        recalls.append(rec)
    bal_acc = float(np.mean(recalls))
    per_str = ", ".join([f"{id2label[i]}:{per_cls[i]['tp']}/{max(1, per_cls[i]['tot'])}" for i in range(len(ALLOWED))])
    print(f"{tag} accuracy={acc:.4f} balanced_accuracy={bal_acc:.4f} per_class_recall={{ {per_str} }}")
    head.train()
    return acc, bal_acc

print("initial evaluation on validation")
evaluate(val_loader, "val_initial")

epochs = 2
step = 0
for epoch in range(1, epochs+1):
    head.train()
    running_loss = 0.0
    for i, batch in enumerate(train_loader, start=1):
        pooled = forward_encode(batch)
        logits = head(pooled)
        loss = criterion(logits, batch["labels"].to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += float(loss.item())
        step += 1
        if i % 50 == 0:
            print(f"epoch={epoch} step={step} minibatch={i} loss={running_loss/50:.6f}")
            running_loss = 0.0
    if (i % 50) != 0:
        print(f"epoch={epoch} step={step} minibatch={i} loss={running_loss/max(1,(i%50)):.6f}")
    print(f"evaluation after epoch {epoch}")
    evaluate(val_loader, f"val_epoch{epoch}")

print("final evaluation on validation and test")
evaluate(val_loader, "val_final")
evaluate(test_loader, "test_final")

save_dir = os.path.join(base_path, "simple_head_sequencing_source")
os.makedirs(save_dir, exist_ok=True)
try:
    torch.save(head.state_dict(), os.path.join(save_dir, "cls_head.pt"))
    with open(os.path.join(save_dir, "labels.json"), "w") as f:
        json.dump({"ALLOWED": ALLOWED, "label2id": label2id}, f)
    print("saved classifier head and labels to:", save_dir)
except Exception as e:
    print("warning: save failed:", e)

print("sample predictions on first 10 test items")
head.eval()
count = 0
for batch in test_loader:
    pooled = forward_encode(batch)
    logits = head(pooled)
    pred = torch.argmax(logits, dim=-1).cpu().item()
    ref = batch["labels"].cpu().item()
    print(f"pred={id2label[pred]} ref={id2label[ref]} match={pred==ref}")
    count += 1
    if count >= 10:
        break
