import os, re, json, math, argparse, random, torch
from dataclasses import dataclass
from typing import Dict, List, Any
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, TrainerCallback, EarlyStoppingCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.tensorboard import SummaryWriter

def extract_value(s):
    try:
        o=json.loads(s)
        if isinstance(o,dict) and "cell_line" in o and isinstance(o["cell_line"],str):
            return o["cell_line"]
    except Exception:
        pass
    m=re.search(r'"cell_line"\s*:\s*"([^"]+)"',s)
    return m.group(1) if m else ""
def norm_keep_case_no_punct(s):
    return re.sub(r"[^A-Za-z0-9]+","",s)
def cmp_ignore_punct_keep_case(a,b):
    return norm_keep_case_no_punct(a)==norm_keep_case_no_punct(b)
def escape_json_val(s):
    return s.replace('\\','\\\\').replace('"','\\"')
def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

@dataclass
class SFTExample:
    prompt: str
    output_json: str

def build_label_space(ds: Dataset):
    labels=[]
    for r in ds:
        v=extract_value(r["output_json"])
        if v and v not in labels:
            labels.append(v)
    labels=sorted(labels)
    label2id={v:i for i,v in enumerate(labels)}
    id2label={i:v for v,i in label2id.items()}
    return label2id,id2label

def tokenize_prompts(tokenizer, ds: Dataset, max_len: int):
    def _tok(x):
        s=x["prompt"]
        y=tokenizer(s, truncation=True, max_length=max_len)
        return y
    return ds.map(_tok, batched=False, remove_columns=[c for c in ds.column_names if c not in ["attention_mask","input_ids","prompt","output_json"]])

def last_token_indices(input_ids):
    return [max(0,len(x)-1) for x in input_ids]

class CLSHead(torch.nn.Module):
    def __init__(self, hidden_size, num_labels):
        super().__init__()
        self.linear=torch.nn.Linear(hidden_size, num_labels)
    def forward(self, x):
        return self.linear(x)

def collect_features(model, tokenizer, ds: Dataset, device, max_len: int, batch_size: int):
    model.eval()
    feats=[]; y=[]
    dl_idx=list(range(0, len(ds), batch_size))
    with torch.no_grad():
        for i in dl_idx:
            batch=ds.select(range(i, min(i+batch_size, len(ds))))
            enc=tokenizer(batch["prompt"], return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
            out=model(**enc, output_hidden_states=True)
            h=out.hidden_states[-1]
            idxs=last_token_indices(enc["input_ids"].tolist())
            idxs=torch.tensor(idxs, device=device)
            b_idx=torch.arange(h.size(0), device=device)
            vec=h[b_idx, idxs, :]
            feats.append(vec.detach().cpu())
            y.extend([extract_value(s) for s in batch["output_json"]])
    return torch.cat(feats, dim=0), y

def fit_cls_head(model, tokenizer, cls_head, train_ds, label2id, device, max_len: int, batch_size: int, epochs: int=3, lr: float=1e-3, weight_decay: float=0.0):
    X, y_txt = collect_features(model, tokenizer, train_ds, device, max_len, batch_size)
    y=torch.tensor([label2id[t] for t in y_txt], dtype=torch.long)
    cls_head.train().to(device)
    X=X.to(device); y=y.to(device)
    opt=torch.optim.AdamW(cls_head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn=torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        perm=torch.randperm(X.size(0), device=device)
        for i in range(0, X.size(0), batch_size):
            idx=perm[i:i+batch_size]
            logits=cls_head(X[idx])
            loss=loss_fn(logits, y[idx])
            opt.zero_grad(); loss.backward(); opt.step()

def predict_one_cls(model, cls_head, tokenizer, prompt, device, max_len: int, id2label: Dict[int,str]):
    model.eval()
    with torch.no_grad():
        enc=tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_len).to(device)
        out=model(**enc, output_hidden_states=True)
        h=out.hidden_states[-1]
        idx=max(0, enc["input_ids"].size(1)-1)
        vec=h[0, idx, :].unsqueeze(0)
        logits=cls_head.to(vec.device)(vec)
        pid=int(logits.argmax(dim=-1).item())
        return id2label[pid]

def eval_accuracy_cls(model, cls_head, tokenizer, ds, device, id2label, max_len: int, tag: str, writer=None, global_step: int=None, verbose_pairs: int=0):
    ok=0; tot=0
    for i,r in enumerate(ds):
        prompt=r["prompt"]
        ref=extract_value(r["output_json"])
        pred=predict_one_cls(model, cls_head, tokenizer, prompt+'{"cell_line": "', device, max_len, id2label)
        match=cmp_ignore_punct_keep_case(pred, ref)
        if i<verbose_pairs:
            print(f'{tag}/pair {i+1}: pred="{pred}" ref="{ref}" pred_norm="{norm_keep_case_no_punct(pred)}" ref_norm="{norm_keep_case_no_punct(ref)}" match={match}', flush=True)
        ok+=int(match); tot+=1
    acc=ok/max(1,tot)
    print(f"{tag}/accuracy_cell_line={acc:.4f}", flush=True)
    if writer is not None and global_step is not None:
        writer.add_scalar(f"{tag}/accuracy_cell_line", acc, global_step)
    return acc

class GenEvalCallback(TrainerCallback):
    def __init__(self, eval_ds, tokenizer, cls_head, label2id, id2label, device, max_len, head_bs, head_epochs, writer):
        self.eval_ds=eval_ds
        self.tok=tokenizer
        self.cls_head=cls_head
        self.label2id=label2id
        self.id2label=id2label
        self.device=device
        self.max_len=max_len
        self.bs=head_bs
        self.epochs=head_epochs
        self.writer=writer
    def on_evaluate(self, args, state, control, **kwargs):
        m=kwargs.get("model")
        if m is None: return
        fit_cls_head(m, self.tok, self.cls_head, self.eval_ds, self.label2id, self.device, self.max_len, self.bs, epochs=self.epochs, lr=1e-3)
        acc=eval_accuracy_cls(m, self.cls_head, self.tok, self.eval_ds, self.device, self.id2label, self.max_len, "eval_cls", self.writer, state.global_step, verbose_pairs=0)
        state.log_history.append({"eval_accuracy_cell_line": float(acc)})
        if acc>getattr(state, "_best_acc", -1):
            state._best_acc=acc
            p=os.path.join(args.output_dir, "best_cls_head.pt")
            torch.save(self.cls_head.state_dict(), p)
    def on_train_end(self, args, state, control, **kwargs):
        m=kwargs.get("model")
        if m is None: return
        fit_cls_head(m, self.tok, self.cls_head, self.eval_ds, self.label2id, self.device, self.max_len, self.bs, epochs=self.epochs, lr=1e-3)
        torch.save(self.cls_head.state_dict(), os.path.join(args.output_dir, "cls_head.pt"))

def build_sft_text(example: Dict[str,Any]):
    return example["prompt"]+example["output_json"]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, required=True)
    parser.add_argument("--test_json", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_steps", type=int, default=250)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--bits", type=int, default=16)
    parser.add_argument("--head_bs", type=int, default=64)
    parser.add_argument("--head_epochs", type=int, default=3)
    args=parser.parse_args()

    seed_all(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    writer=SummaryWriter(log_dir=os.path.join(args.output_dir, "tb"))

    data=load_dataset("json", data_files={"train": args.train_json, "validation": args.val_json, "test": args.test_json})
    label2id,id2label=build_label_space(data["train"])
    print(f"Train/Val/Test sizes: {len(data['train'])}/{len(data['validation'])}/{len(data['test'])}", flush=True)
    print(f"Label space (train) size = {len(label2id)}", flush=True)
    print(str({k: len(data[k]) for k in data.keys()}), flush=True)

    tokenizer=AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token

    def map_sft(ds):
        return ds.map(lambda ex: {"text": build_sft_text(ex)}, remove_columns=ds.column_names)
    sft_train=map_sft(data["train"])
    sft_val=map_sft(data["validation"])
    collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    load_in_8bit=False; load_in_4bit=False; torch_dtype=torch.float16 if args.bf16 else (torch.bfloat16 if args.bits==16 and args.bf16 else torch.float16)
    model=AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch_dtype, load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit, device_map="auto")
    try:
        model=prepare_model_for_kbit_training(model)
    except Exception:
        pass
    lconf=LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, bias="none", task_type="CAUSAL_LM")
    model=get_peft_model(model, lconf)
    model.print_trainable_parameters()

    hidden_size=model.config.hidden_size if hasattr(model.config,"hidden_size") else model.model.model.embed_tokens.embedding_dim
    cls_head=CLSHead(hidden_size, len(label2id))

    class SFTDataset(torch.utils.data.Dataset):
        def __init__(self, ds, tok, max_len):
            self.ds=ds
            self.tok=tok
            self.max_len=max_len
        def __len__(self): return len(self.ds)
        def __getitem__(self, idx):
            t=self.ds[idx]["text"]
            x=self.tok(t, truncation=True, max_length=self.max_len)
            return {"input_ids": torch.tensor(x["input_ids"], dtype=torch.long), "attention_mask": torch.tensor(x["attention_mask"], dtype=torch.long), "labels": torch.tensor(x["input_ids"], dtype=torch.long)}
    train_dataset=SFTDataset(sft_train, tokenizer, args.max_length)
    val_dataset=SFTDataset(sft_val, tokenizer, args.max_length)

    class TBCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is None: return
            for k,v in logs.items():
                if isinstance(v,(int,float)): writer.add_scalar(k, v, state.global_step)
        def on_step_end(self, args, state, control, **kwargs):
            return

    training_args=TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        num_train_epochs=args.epochs,
        learning_rate=2e-5,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=50,
        bf16=args.bf16,
        fp16=not args.bf16,
        report_to=[],
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy_cell_line",
        greater_is_better=True
    )

    device=next(model.parameters()).device
    head_cb=GenEvalCallback(data["validation"], tokenizer, cls_head, label2id, id2label, device, args.max_length, args.head_bs, args.head_epochs, writer)
    trainer=Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[TBCallback(), EarlyStoppingCallback(early_stopping_patience=2), head_cb]
    )

    print("Initial evaluation on validation with BASE model (CLS head)", flush=True)
    fit_cls_head(model, tokenizer, cls_head, data["train"], label2id, device, args.max_length, args.head_bs, epochs=args.head_epochs, lr=1e-3)
    _=eval_accuracy_cls(model, cls_head, tokenizer, data["validation"], device, id2label, args.max_length, "initial_cls", writer, 0, verbose_pairs=10)

    print("Begin training", flush=True)
    trainer.train()
    trainer.save_model(args.output_dir)

    torch.save(cls_head.state_dict(), os.path.join(args.output_dir, "cls_head.pt"))

    print("Evaluate on validation with CLS head", flush=True)
    val_acc=eval_accuracy_cls(trainer.model, cls_head, tokenizer, data["validation"], device, id2label, args.max_length, "final_val_cls", writer, trainer.state.global_step, verbose_pairs=20)
    print(f"final/validation_accuracy_cell_line_cls={val_acc:.4f}", flush=True)

    print("Evaluate on test with CLS head", flush=True)
    ok=0; tot=0
    for i,r in enumerate(data["test"]):
        pred=predict_one_cls(trainer.model, cls_head, tokenizer, r["prompt"]+'{"cell_line": "', device, args.max_length, id2label)
        ref=extract_value(r["output_json"])
        print(f'--- Predicted output {i+1}: {{"cell_line": "{escape_json_val(pred)}"}}', flush=True)
        print(f'--- Expected output  {i+1}: {r["output_json"]}', flush=True)
        m=cmp_ignore_punct_keep_case(pred, ref)
        ok+=int(m); tot+=1
    test_acc=ok/max(1,tot)
    print(f"final/test_accuracy_cell_line_cls={test_acc:.4f}", flush=True)

    writer.flush(); writer.close()

if __name__=="__main__":
    main()
