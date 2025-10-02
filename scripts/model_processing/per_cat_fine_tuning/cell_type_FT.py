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

parser = argparse.ArgumentParser(description="Fine-tune cell_type (multitask) with semantic alignment + constrained decoding")
parser.add_argument("--base_path", type=str, required=True)
parser.add_argument("--cls_loss_weight", type=float, default=0.4)
parser.add_argument("--sem_loss_weight", type=float, default=0.15)   # poids de la perte sémantique
parser.add_argument("--aug_ratio", type=float, default=0.35)         # part de lignes dupliquées avec bruit contextuel
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
base_path = args.base_path

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

train_file = os.path.join(base_path, "finetune_data_train.csv")
val_file   = os.path.join(base_path, "finetune_data_val.csv")
test_file  = os.path.join(base_path, "finetune_data_test.csv")
model_name = os.path.join(base_path, "Mistral-7B-Instruct-v0.3")

tb_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/logs4/tensorboard_cell_type"
adapter_out_dir = os.path.join(base_path, "cat_cell_type")
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
    lora_dropout=0.35,
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

# Nettoyage léger: dédupes et supprime sorties vides (on ne modifie PAS le contenu "positive/negative")
df_train = df_train.drop_duplicates(subset=["prompt"]).copy()
df_train = df_train[df_train["output"].astype(str).strip()!=""].copy()

print(f"Train/Val/Test sizes (raw): {len(df_train)}/{len(df_val)}/{len(df_test)}", flush=True)

def _escape_json_val(v):
    return str(v).replace('\\','\\\\').replace('"','\\"')

def _extract_value(txt):
    try:
        j = json.loads(txt)
        v = j.get("cell_type","")
        return "" if v is None else str(v)
    except Exception:
        m = re.search(r'"cell_type"\s*:\s*"([^"]*)"', str(txt))
        return m.group(1) if m else str(txt)

# Ajoute colonnes JSON (on garde le label tel quel, y compris “positive/negative”)
for _df in (df_train, df_val, df_test):
    _df["output_raw"]  = _df["output"].astype(str)
    _df["output_json"] = _df["output_raw"].apply(lambda v: f'{{"cell_type": "{_escape_json_val(v)}"}}')

# ======= AUGMENTATION GÉNÉRIQUE (optionnelle) =======
# Duplique ~aug_ratio des lignes d'entraînement en injectant du bruit contextuel neutre
def augment_prompts(df, ratio=0.35, seed=42):
    if ratio <= 0 or len(df) == 0:
        return df
    rnd = np.random.default_rng(seed)
    n_aug = int(len(df) * ratio)
    idxs = rnd.choice(len(df), size=n_aug, replace=False)
    noise_phrases = [
        "Note: marker status (positive/negative) may be reported.",
        "Caution: mentions like 'negative for malignancy' do not define cell type.",
        "Context may include cancer-related words not tied to the cell identity.",
        "Marker intensity or positivity/negativity is informational, not the class label.",
        "Beware of clinical terms like primary/metastasis; output only the cell type."
    ]
    cancer_noise = [
        "The report mentions cancer screening.",
        "A sentence says: no evidence of malignancy.",
        "A sentence says: malignancy cannot be ruled out.",
        "Cancer is discussed in a general context.",
        "Non-malignant features also described."
    ]
    marker_noise = [
        "Marker ITGB4: negative.",
        "Marker HER2: positive.",
        "Ki-67: high index.",
        "p53: negative.",
        "EGFR: positive."
    ]
    rows = []
    for i in idxs:
        r = df.iloc[i].copy()
        tail = "\n".join([
            random.choice(noise_phrases),
            random.choice(cancer_noise),
            random.choice(marker_noise),
        ])
        r["prompt"] = r["prompt"] + "\n" + tail
        rows.append(r)
    df_aug = pd.DataFrame(rows) if rows else df.iloc[0:0].copy()
    return pd.concat([df, df_aug], ignore_index=True)

df_train = augment_prompts(df_train, ratio=args.aug_ratio, seed=args.seed)

# ======= LABEL SPACE =======
labels_train = sorted(set(df_train["output_raw"].tolist()))
label2id = {lbl:i for i,lbl in enumerate(labels_train)}
id2label = {i:lbl for lbl,i in label2id.items()}
print(f"Label space (train) size = {len(label2id)}", flush=True)

for _df in (df_train, df_val, df_test):
    _df["cls_label"] = _df["output_raw"].map(lambda x: label2id.get(x, -1))

datasets = {
    "train": Dataset.from_pandas(df_train[["prompt","output_json","cls_label"]]),
    "validation": Dataset.from_pandas(df_val[["prompt","output_json","cls_label"]]),
    "test": Dataset.from_pandas(df_test[["prompt","output_json","cls_label"]])
}
print({k: len(v) for k,v in datasets.items()}, flush=True)

# ======= SentenceTransformer pour métriques + perte sémantique =======
sem_model_name = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
model_sem = SentenceTransformer(sem_model_name)
SIM_THRESH = 0.4
with torch.no_grad():
    _tmp = model_sem.encode(["test"], convert_to_numpy=True)
SEM_DIM = int(_tmp.shape[1])

# ======= Tokenization + ajout de l'embedding cible pour la perte sem =======
def tokenize_fn(example):
    prompt   = example["prompt"].strip()
    out_json = example["output_json"].strip()
    m = re.search(r'"cell_type"\s*:\s*"([^"]*)"', out_json)
    value_str  = m.group(1) if m else ""
    prefix_str = '{"cell_type": "'
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

    # Embedding sémantique de la valeur cible (normalisé)
    sem_emb = model_sem.encode(value_str, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32).tolist()

    return {
        "input_ids": ids,
        "attention_mask": attn,
        "labels": labels,
        "split_idx": len(in_enc["input_ids"]),
        "cls_label": int(example.get("cls_label", -1)),
        "sem_emb": sem_emb,
    }

tokenized = {k: v.map(tokenize_fn, remove_columns=v.column_names) for k, v in datasets.items()}

class CausalLMPadCollator:
    def __init__(self, tokenizer):
        self._padder = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    def __call__(self, features):
        labels    = [torch.tensor(f["labels"], dtype=torch.long) for f in features]
        split_idx = torch.tensor([int(f.get("split_idx", 0)) for f in features], dtype=torch.long)
        cls_label = torch.tensor([int(f.get("cls_label", -1)) for f in features], dtype=torch.long)
        sem_embs  = [torch.tensor(f["sem_emb"], dtype=torch.float32) for f in features]
        for f in features:
            f.pop("labels", None); f.pop("split_idx", None); f.pop("cls_label", None); f.pop("sem_emb", None)
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
        batch["sem_emb"]   = torch.stack(sem_embs, dim=0)
        return batch

data_collator = CausalLMPadCollator(tokenizer)

# ======= Évaluation par génération (avec contrainte) =======
def _cos_sim(a, b):
    an = a / max(np.linalg.norm(a), 1e-12)
    bn = b / max(np.linalg.norm(b), 1e-12)
    return float(np.dot(an, bn))

# Trie des labels pour contraindre le décodage
CANDIDATES = labels_train
CAND_TOKEN_IDS = [tokenizer(c, add_special_tokens=False)["input_ids"] for c in CANDIDATES]
# construit un trie simple (dictionnaire imbriqué)
TRIE = [{}]  # liste de noeuds; chaque noeud: dict id->next_idx, et clé "_end" pour fin
def add_to_trie(seq):
    node = 0
    for tid in seq:
        if tid not in TRIE[node]:
            TRIE[node][tid] = len(TRIE)
            TRIE.append({})
        node = TRIE[node][tid]
    TRIE[node]["_end"] = True
for seq in CAND_TOKEN_IDS:
    if len(seq) == 0:  # ignorer labels vides
        continue
    add_to_trie(seq)

# Variables globales pour la contrainte, mises à jour avant chaque .generate()
_PREFIX_TOKENS = tokenizer('{"cell_type": "', add_special_tokens=False)["input_ids"]
_START_POS_BY_BATCH = {}

def _find_start_pos(input_ids_list, prefix_tokens):
    # cherche la dernière occurrence de prefix_tokens dans input_ids_list
    n = len(input_ids_list); m = len(prefix_tokens)
    if m == 0 or n < m: return None
    # on parcourt à rebours pour capter le dernier préfixe
    for start in range(n - m, -1, -1):
        if input_ids_list[start:start+m] == prefix_tokens:
            return start + m
    return None

def prefix_allowed_tokens_fn(batch_id, input_ids):
    seq = input_ids.tolist()
    start_pos = _START_POS_BY_BATCH.get(int(batch_id), None)
    if start_pos is None:
        # essaie de le détecter dynamiquement
        start_pos = _find_start_pos(seq, _PREFIX_TOKENS)
        if start_pos is None:
            # pas dans la zone contrainte (on laisse tout)
            return list(range(tokenizer.vocab_size))
    # on marche dans le trie à partir de la séquence générée après start_pos
    node = 0
    for tid in seq[start_pos:]:
        if tid in TRIE[node]:
            node = TRIE[node][tid]
        else:
            # chemin invalide => aucune continuation autorisée (on retombe au début des labels)
            node = 0
            break
    # autorisés = clefs sortantes du noeud courant
    allowed = [tid for tid in TRIE[node].keys() if tid != "_end"]
    # si on est sur un noeud terminal, autoriser le guillemet de fermeture pour clore la valeur
    if TRIE[node].get("_end", False):
        quote_id = tokenizer.convert_tokens_to_ids('"')
        if quote_id is not None:
            allowed = allowed + [quote_id]
    # garde un fallback minimal pour éviter blocage (rare)
    return allowed if len(allowed) > 0 else list(range(tokenizer.vocab_size))

def generate_constrained(model, tokenizer, prompt, dev):
    prefix = prompt + '{"cell_type": "'
    inp = tokenizer(prefix, return_tensors="pt").to(dev)
    # setup positions pour la fonction de contrainte
    global _START_POS_BY_BATCH
    _START_POS_BY_BATCH = {0: inp["input_ids"].size(1)}  # batch=1
    with torch.no_grad():
        out_ids = model.generate(
            **inp,
            max_new_tokens=64,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn
        )
    cont = tokenizer.decode(out_ids[0][inp["input_ids"].size(1):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    pred_val = re.split(r'["\n,}]', cont)[0]
    return pred_val

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
        pred_val = generate_constrained(model, tokenizer, prompt, dev)
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
        print(f"eval/accuracy_cell_type={acc:.4f} at step={state.global_step}", flush=True)
        writer.add_scalar("eval/accuracy_cell_type", acc, state.global_step)
        if isinstance(metrics, dict):
            metrics["eval_accuracy_cell_type"] = float(acc)

print("Initial evaluation on validation with BASE model (semantic soft accuracy)", flush=True)
base_model.eval()
init_val_acc = gen_eval_soft_accuracy(base_model, tokenizer, datasets["validation"], max_examples=None, tag="initial")
print(f"initial/validation_accuracy_cell_type={init_val_acc:.4f}", flush=True)
writer.add_scalar("initial/validation_accuracy_cell_type", init_val_acc, 0)
writer.flush()

hidden_size = getattr(peft_model.config, "hidden_size", None) or getattr(peft_model.config, "n_embd", None)
num_classes = len(label2id)

# Tête de classification
cls_head = nn.Sequential(
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Dropout(0.35),
    nn.Linear(hidden_size, num_classes)
).to(next(peft_model.parameters()).device)

# Proj pour perte sémantique (projette l'embedding STS vers l'espace hidden)
sem_proj = nn.Linear(SEM_DIM, hidden_size, bias=False).to(next(peft_model.parameters()).device)

class MultiTaskTrainer(Trainer):
    def __init__(self, cls_head: nn.Module, sem_proj: nn.Module, cls_weight: float = 0.7, sem_weight: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.cls_head = cls_head
        self.sem_proj = sem_proj
        self.cls_weight = float(cls_weight)
        self.sem_weight = float(sem_weight)
        self.ce = nn.CrossEntropyLoss(reduction='mean')
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-8)

    def create_optimizer(self):
        if self.optimizer is None:
            model_params = [p for p in self.model.parameters() if p.requires_grad]
            head_params  = [p for p in self.cls_head.parameters() if p.requires_grad]
            sem_params   = [p for p in self.sem_proj.parameters() if p.requires_grad]
            optimizer_grouped_parameters = [
                {"params": model_params, "weight_decay": self.args.weight_decay},
                {"params": head_params,  "weight_decay": 0.05},
                {"params": sem_params,   "weight_decay": 0.0},
            ]
            self.optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate)
        return self.optimizer

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix)
        try:
            ds = datasets.get("validation", None)
            acc = gen_eval_soft_accuracy(self.model, self.tokenizer, ds if ds is not None else datasets["validation"], max_examples=128, tag=metric_key_prefix)
            metrics[f"{metric_key_prefix}_accuracy_cell_type"] = float(acc)
            step = getattr(self.state, "global_step", -1)
            print(f"{metric_key_prefix}/accuracy_cell_type={acc:.4f} at step={step}", flush=True)
        except Exception as e:
            print(f"warn/eval_callback_exception: {e}", flush=True)
        return metrics

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        split_idx = inputs.pop("split_idx")
        cls_label = inputs.pop("cls_label")
        sem_emb   = inputs.pop("sem_emb")  # [B, SEM_DIM] tensor float32
        outputs   = model(**inputs, output_hidden_states=True)
        lm_loss   = outputs.loss
        hidden    = outputs.hidden_states[-1]            # [B, T, H]
        B = hidden.size(0)
        idx = torch.clamp(split_idx - 1, min=0)
        gather = hidden[torch.arange(B, device=hidden.device), idx, :]  # [B, H]

        # classification
        if next(self.cls_head.parameters()).device != gather.device:
            self.cls_head.to(gather.device)
        logits = self.cls_head(gather)
        mask = (cls_label >= 0)
        if mask.any():
            cls_loss = self.ce(logits[mask], cls_label[mask])
        else:
            cls_loss = torch.tensor(0.0, device=hidden.device)

        # perte sémantique (1 - cos(proj(sem_emb), gather))
        if next(self.sem_proj.parameters()).device != gather.device:
            self.sem_proj.to(gather.device)
        sem_emb = sem_emb.to(gather.device)
        proj = self.sem_proj(sem_emb)                    # [B, H]
        proj = nn.functional.normalize(proj, p=2, dim=-1)
        gath = nn.functional.normalize(gather, p=2, dim=-1)
        cos_sim = self.cos(proj, gath)                   # [B]
        sem_loss = (1.0 - cos_sim).mean()

        loss = lm_loss + self.cls_weight * cls_loss + self.sem_weight * sem_loss
        if return_outputs:
            return loss, outputs
        return loss

training_args = TrainingArguments(
    output_dir=os.path.join(base_path, "checkpoints_cell_type"),
    eval_strategy="steps",
    save_strategy="steps",
    eval_steps=250,
    save_steps=250,
    save_total_limit=3,
    learning_rate=3e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
    weight_decay=0.05,
    logging_strategy="steps",
    logging_steps=50,
    fp16=(not use_bf16),
    bf16=use_bf16,
    gradient_accumulation_steps=8,
    report_to=["tensorboard"],
    logging_dir=tb_dir,
    load_best_model_at_end=True,
    metric_for_best_model="eval_accuracy_cell_type",
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
    sem_proj=sem_proj,
    cls_weight=args.cls_loss_weight,
    sem_weight=args.sem_loss_weight
)

print("Begin training", flush=True)
trainer.train()
writer.flush()

print("Save adapter only", flush=True)
trainer.model.save_pretrained(adapter_out_dir)
tokenizer.save_pretrained(adapter_out_dir)

print("Evaluate on validation with generation (semantic soft accuracy, constrained decoding)", flush=True)
val_acc = gen_eval_soft_accuracy(trainer.model, tokenizer, datasets["validation"], max_examples=None, tag="final_val")
print(f"final/validation_accuracy_cell_type={val_acc:.4f}", flush=True)
writer.add_scalar("final/validation_accuracy_cell_type", val_acc, trainer.state.global_step)
writer.flush()

print("Generate on test set (semantic soft accuracy, constrained decoding)", flush=True)
dev = next(trainer.model.parameters()).device
rows = datasets["test"].to_pandas()
n_ok, n_all = 0, 0
ref_texts = [ _extract_value(r["output_json"]) for _, r in rows.iterrows() ]
ref_embs = model_sem.encode(ref_texts, convert_to_numpy=True, normalize_embeddings=True)
for i, r in enumerate(rows.itertuples(index=False)):
    prompt   = getattr(r, "prompt")
    expected = getattr(r, "output_json")
    ref_val  = ref_texts[i]
    pred_val = generate_constrained(trainer.model, tokenizer, prompt, dev)
    pred_json = f'{{"cell_type": "{_escape_json_val(pred_val)}"}}'
    pred_emb = model_sem.encode(pred_val, convert_to_numpy=True, normalize_embeddings=True)
    sim = _cos_sim(pred_emb, ref_embs[i])
    ok = sim >= SIM_THRESH
    print(f"--- Predicted output {i+1}: {pred_json}", flush=True)
    print(f"--- Expected output  {i+1}: {expected}", flush=True)
    print(f"--- cos_sim={sim:.4f} match={ok}", flush=True)
    print("-"*50, flush=True)
    n_ok += int(ok); n_all += 1

test_acc = (n_ok / max(1, n_all))
print(f"final/test_accuracy_cell_type={test_acc:.4f}", flush=True)
writer.add_scalar("final/test_accuracy_cell_type", test_acc, trainer.state.global_step)
writer.close()
