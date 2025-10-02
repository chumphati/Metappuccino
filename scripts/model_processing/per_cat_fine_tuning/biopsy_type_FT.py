import os, re, json, argparse, random, unicodedata
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import WeightedRandomSampler, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

SAVE_ROOT = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/logs5"
ALLOWED = ["primary", "metastasis", "blood", "unknown"]
label2id = {k: i for i, k in enumerate(ALLOWED)}
id2label = {i: k for k, i in label2id.items()}

SYNONYMS = {
    "primary": [
        "primary","primaire","primary tumor","tumor of origin","primary site",
        "localized","de novo","primary lesion","site primitif","tumeur primitive"
    ],
    "metastasis": [
        "metastasis","metastases","metastatic","métastase","metastatique",
        "secondary","mets","met","distant","metastatic lesion"
    ],
    "blood": [
        "blood","sang","plasma","serum","sérum","pbmc","buffy coat",
        "whole blood","wb","leukapheresis"
    ],
    "unknown": [
        "unknown","unk","n/a","na","not available","unspecified",
        "indeterminate","undetermined","inconnu","non précisé"
    ],
}
SYN_PATTERNS = {k: re.compile(r"\b(?:" + "|".join(map(re.escape, v)) + r")\b", flags=re.IGNORECASE) for k, v in SYNONYMS.items()}

def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", str(text)) if unicodedata.category(c) != "Mn").lower()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def canon_biopsy_type(s: str) -> str:
    t = str(s).strip().lower()
    tt = _strip_accents(t)
    if t in ALLOWED:
        return t
    for label, pat in SYN_PATTERNS.items():
        if pat.search(tt):
            return label
    if re.search(r"\bmet(astasis|astatic|s|astatic lesions)?\b", tt):
        return "metastasis"
    if re.search(r"\b(pbmc|blood|plasma|serum|buffy|hemat|haemat|sang|serum)\b", tt):
        return "blood"
    if re.search(r"\bprimary|primaire|primary site|tumeur primitive\b", tt):
        return "primary"
    return "unknown"

def read_two_col_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, engine="python", sep=None, dtype=str, keep_default_na=False)
    except Exception:
        df = pd.read_csv(path, engine="python", sep=";", dtype=str, keep_default_na=False)
    cols = [c.lower() for c in df.columns]
    df.columns = cols
    assert {"prompt", "output"}.issubset(df.columns), f"Colonnes manquantes dans {path}"
    return df[["prompt", "output"]].astype(str)

@dataclass
class BiopsyExample:
    input_ids: list
    attention_mask: list
    label: int

class Collator(DataCollatorWithPadding):
    def __call__(self, features):
        labels = torch.tensor([f.get("labels") for f in features], dtype=torch.long)
        for f in features:
            f.pop("labels", None)
            for k in list(f.keys()):
                if k not in ("input_ids", "attention_mask", "token_type_ids"):
                    f.pop(k, None)
        batch = super().__call__(features)
        batch["labels"] = labels
        return batch

class Head(nn.Module):
    def __init__(self, hidden_size: int, num_classes: int = 4, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_size, num_classes),
        )
    def forward(self, x):
        return self.net(x)

class SimpleClsTrainer(Trainer):
    def __init__(self, cls_head: nn.Module, cls_weight: float, class_weights: torch.Tensor, **kwargs):
        super().__init__(**kwargs)
        self.cls_head = cls_head
        self.cls_weight = float(cls_weight)
        self.class_weights = class_weights
    def create_optimizer(self):
        if self.optimizer is None:
            lora_params = [p for n, p in self.model.named_parameters() if p.requires_grad and "lora_" in n]
            head_params = [p for p in self.cls_head.parameters() if p.requires_grad]
            optim_groups = [
                {"params": lora_params, "weight_decay": self.args.weight_decay, "lr": self.args.learning_rate},
                {"params": head_params,  "weight_decay": 0.0,                   "lr": self.args.learning_rate * 5.0},
            ]
            self.optimizer = torch.optim.AdamW(optim_groups)
        return self.optimizer
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"), output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        attn = inputs["attention_mask"].bool()
        last_index = attn.long().sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_index, :]
        logits = self.cls_head(pooled)
        labels = inputs["labels"].to(logits.device)
        ce = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = ce(logits, labels)
        if return_outputs:
            return loss, {"logits": logits}
        return loss
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **kwargs):
        with torch.no_grad():
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask"), output_hidden_states=True)
            hidden = outputs.hidden_states[-1]
            attn = inputs["attention_mask"].bool()
            last_index = attn.long().sum(dim=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_index, :]
            logits = self.cls_head(pooled)
            if prediction_loss_only:
                labels = inputs.get("labels")
                loss = nn.functional.cross_entropy(logits, labels.to(logits.device), reduction="mean") if labels is not None else None
                return (loss, None, None)
            return (None, logits.detach().cpu(), inputs.get("labels").detach().cpu())

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_path", type=str, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--dropout", type=float, default=0.1)
    args = p.parse_args()

    set_seed(args.seed)

    os.makedirs(SAVE_ROOT, exist_ok=True)
    ADAPTER_DIR = os.path.join(SAVE_ROOT, "cat_biopsy_type")
    CHECKPOINT_DIR = os.path.join(SAVE_ROOT, "checkpoints_biopsy_type_simple")
    TB_DIR = os.path.join(SAVE_ROOT, "tensorboard_biopsy_type")
    MERGED_DIR = os.path.join(SAVE_ROOT, "merged_full_model")
    os.makedirs(ADAPTER_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(TB_DIR, exist_ok=True)

    train_file = os.path.join(args.base_path, "finetune_data_train.csv")
    val_file   = os.path.join(args.base_path, "finetune_data_val.csv")
    test_file  = os.path.join(args.base_path, "finetune_data_test.csv")
    model_name = os.path.join(args.base_path, "Mistral-7B-Instruct-v0.3")

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "</s>"

    df_tr = read_two_col_csv(train_file)
    df_va = read_two_col_csv(val_file)
    df_te = read_two_col_csv(test_file)
    for _df in (df_tr, df_va, df_te):
        _df["label_txt"] = _df["output"].map(canon_biopsy_type)
        _df["label"] = _df["label_txt"].map(label2id)

    counts = df_tr["label"].value_counts().reindex(range(len(ALLOWED)), fill_value=0).astype(float)
    inv = 1.0 / np.maximum(counts.values, 1.0)
    class_weights = torch.tensor(inv / inv.sum() * len(inv), dtype=torch.float)
    sample_weights = df_tr["label"].map({i: inv[i] for i in range(len(ALLOWED))}).values

    def tok(batch):
        enc = tokenizer(batch["prompt"], truncation=True, padding=False, max_length=args.max_len)
        enc["labels"] = batch["label"]
        return enc

    from datasets import Dataset
    d_tr = Dataset.from_pandas(df_tr[["prompt", "label"]]).map(tok, batched=True, remove_columns=["prompt", "label"])
    d_va = Dataset.from_pandas(df_va[["prompt", "label"]]).map(tok, batched=True, remove_columns=["prompt", "label"])
    d_te = Dataset.from_pandas(df_te[["prompt", "label"]]).map(tok, batched=True, remove_columns=["prompt", "label"])

    collator = Collator(tokenizer=tokenizer, return_tensors="pt")

    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    base = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto", low_cpu_mem_usage=True, trust_remote_code=True)
    if base.get_input_embeddings().num_embeddings != len(tokenizer):
        base.resize_token_embeddings(len(tokenizer))
    base.config.use_cache = False

    peft_cfg = LoraConfig(task_type="CAUSAL_LM", inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
    model = get_peft_model(base, peft_cfg)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.gradient_checkpointing_enable()

    hidden_size = getattr(model.config, "hidden_size", None) or getattr(model.config, "n_embd", None)
    cls_head = Head(hidden_size, num_classes=len(ALLOWED), p_drop=args.dropout).to(model.device)

    sampler = WeightedRandomSampler(weights=torch.tensor(sample_weights, dtype=torch.double), num_samples=len(sample_weights), replacement=True)

    targs = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.05,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=50,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        fp16=(dtype == torch.float16),
        bf16=(dtype == torch.bfloat16),
        max_grad_norm=1.0,
        report_to=["tensorboard"],
        logging_dir=TB_DIR,
    )

    class Metrics:
        def __call__(self, eval_pred):
            logits, labels = eval_pred
            y_pred = logits.argmax(-1)
            acc = accuracy_score(labels, y_pred)
            f1m = f1_score(labels, y_pred, average="macro", zero_division=0)
            return {"accuracy": acc, "macro_f1": f1m}

    trainer = SimpleClsTrainer(
        model=model,
        args=targs,
        train_dataset=d_tr,
        eval_dataset=d_va,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        cls_head=cls_head,
        cls_weight=1.0,
        class_weights=class_weights,
        compute_metrics=Metrics(),
    )

    def train_dl():
        return DataLoader(d_tr, batch_size=targs.per_device_train_batch_size, sampler=sampler, collate_fn=collator)
    trainer.get_train_dataloader = train_dl

    print("debug/fit", flush=True)
    trainer.train()

    print("debug/eval", flush=True)
    out = trainer.predict(d_va)
    y_true = out.label_ids
    y_pred = out.predictions.argmax(-1)
    print(classification_report(y_true, y_pred, labels=list(range(len(ALLOWED))), target_names=ALLOWED, digits=4, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred):\n", confusion_matrix(y_true, y_pred, labels=list(range(len(ALLOWED)))))

    print("debug/test", flush=True)
    out_te = trainer.predict(d_te)
    y_true_te = out_te.label_ids
    y_pred_te = out_te.predictions.argmax(-1)
    print(classification_report(y_true_te, y_pred_te, labels=list(range(len(ALLOWED))), target_names=ALLOWED, digits=4, zero_division=0))
    test_acc = accuracy_score(y_true_te, y_pred_te)
    test_f1m = f1_score(y_true_te, y_pred_te, average="macro", zero_division=0)
    print(f"final/test_accuracy={test_acc:.4f}")
    print(f"final/test_macro_f1={test_f1m:.4f}")

    print("debug/save_adapter_start", flush=True)
    try:
        trainer.model.save_pretrained(ADAPTER_DIR, safe_serialization=True)
        print(f"save/adapter_ok path={ADAPTER_DIR}", flush=True)
    except Exception as e:
        print(f"error/save_adapter: {e}", flush=True)

    print("debug/save_tokenizer_start", flush=True)
    try:
        tokenizer.save_pretrained(ADAPTER_DIR)
        print("save/tokenizer_ok", flush=True)
    except Exception as e:
        print(f"error/save_tokenizer: {e}", flush=True)

    print("debug/save_head_start", flush=True)
    try:
        cls_head_cpu = Head(hidden_size, num_classes=len(ALLOWED))
        cls_head_cpu.load_state_dict(cls_head.state_dict())
        torch.save(cls_head_cpu.state_dict(), os.path.join(ADAPTER_DIR, "cls_head.pt"))
        with open(os.path.join(ADAPTER_DIR, "labels.json"), "w", encoding="utf-8") as f:
            json.dump({"id2label": {int(k): v for k, v in id2label.items()}, "label2id": label2id}, f)
        print("save/head_ok", flush=True)
    except Exception as e:
        print(f"error/save_head: {e}", flush=True)

    print("debug/merge_and_save_full_model_start", flush=True)
    try:
        os.makedirs(MERGED_DIR, exist_ok=True)
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(MERGED_DIR, safe_serialization=True)
        print(f"save/merged_ok path={MERGED_DIR}", flush=True)
    except Exception as e:
        print(f"warn/merge_save_failed: {e}", flush=True)

if __name__ == "__main__":
    main()
