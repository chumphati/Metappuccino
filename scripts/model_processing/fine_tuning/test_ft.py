import argparse
import math
import os
import random
import re

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold, train_test_split
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    EvalPrediction,
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# PATHS
parser = argparse.ArgumentParser(description="Fine-tune model")
parser.add_argument("--base_path", type=str, required=True, help="Base path to MetaMap")
parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds")
args = parser.parse_args()

base_path = args.base_path
n_splits = args.n_splits

prompt_file = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/test_finetune_data.csv"
model_name = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/Mistral-7B-Instruct-v0.3"
tensorboard_log_dir = "tensorboard"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model_base = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map="auto"
).to("cuda")


# tokenize input
def tokenize_function(example):
    prompt = example["prompt"].strip()
    output = example["output"].strip()
    prompt_ids = tokenizer(prompt, truncation=True, max_length=2000)["input_ids"]
    output_ids = tokenizer(output, truncation=True, max_length=200)["input_ids"]

    input_ids = prompt_ids + output_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + output_ids

    max_length = 2000
    padding_length = max_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    attention_mask += [0] * padding_length
    labels += [-100] * padding_length

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# deduplicate prediction categories
def deduplicate_categories(pred_text):
    allowed = {
        'cell_type', 'tissue_type', 'cell_line', 'organ', 'disease',
        'host_phenotype', 'library_selection', 'library_source',
        'treatment', 'treatment_time', 'response', 'donor_information'
    }
    seen = set()
    final_output = []
    for line in pred_text.splitlines():
        if ':' not in line:
            continue
        key, val = line.split(':', 1)
        key = key.strip()
        key_lower = key.lower()
        if key_lower not in allowed:
            continue
        if key_lower in seen:
            continue
        final_output.append(f"{key_lower}: {val.strip()}")
        seen.add(key_lower)
    return '\n'.join(final_output)


# clean output before tokenization
def clean_output_text(example):
    example["output"] = deduplicate_categories(example["output"])
    return example


# parse raw generation into deduplicated block
def parse_pred_block(raw_pred):
    split_output = raw_pred.split("Here is the output:")
    after = split_output[1].strip() if len(split_output) > 1 else raw_pred
    return deduplicate_categories(after)

def compute_categorical_metrics(pred_texts, ref_texts, categories):
    metrics = {}
    for cat in categories:
        accs = []
        for pred, ref in zip(pred_texts, ref_texts):
            print("--------------------")
            print("raw pred: ", pred, flush=True)
            print("raw ref: ", ref, flush=True)

            p_block = parse_pred_block(pred)
            p_dict = {}
            for line in p_block.splitlines():
                if ":" not in line:
                    continue
                raw_cat, val = line.split(":", 1)
                cat_clean = re.sub(r'^[^A-Za-z0-9]+', '', raw_cat.strip())
                p_dict[cat_clean] = val.strip()

            r_dict = {l.split(":", 1)[0].strip(): l.split(":", 1)[1].strip()
                      for l in ref.splitlines() if ":" in l}
            print("--------------------")
            print("p_dict: ", p_dict, flush=True)
            print("r_dict: ", r_dict, flush=True)

            p_val = ""
            for key, val in p_dict.items():
                if cat in key:
                    p_val = val.strip()
                    break
            r_val = ""
            for key, val in r_dict.items():
                if cat in key:
                    r_val = val.strip()
                    break

            print("--------------------")
            print("category: ", cat, flush=True)
            print("pred val: ", p_val, flush=True)
            print("ref val: ", r_val, flush=True)
            print("--------------------")
            # nan or empty references
            #if ref is nan
            if r_val.lower() == 'nan':
                #acc=T if only pred is nan to != empty
                acc = (p_val.lower() == 'nan')
                accs.append(acc)
                continue
            #if ref empty
            if not r_val:
                continue

            #if ref is not empty sur pred empty = false
            if not p_val or p_val.lower() == 'nan':
                acc = False
            else:
                emb_ref = model_sem.encode([r_val], convert_to_tensor=True)
                emb_pred = model_sem.encode([p_val], convert_to_tensor=True)
                cos = cosine_similarity(emb_ref.cpu().numpy(), emb_pred.cpu().numpy())[0][0]
                acc = cos > threshold
            accs.append(acc)
        metrics[f"accuracy_{cat.lower()}"] = sum(accs) / len(accs) if accs else 0.0
    metrics["accuracy_overall"] = sum(metrics[f"accuracy_{cat.lower()}"] for cat in categories) / len(categories) if categories else 0.0
    return metrics


def compute_metrics(eval_preds: EvalPrediction):
    gen_ids, label_ids = eval_preds.predictions, eval_preds.label_ids
    decoded_preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    cat_metrics = compute_categorical_metrics(decoded_preds, decoded_labels, model_base.config.categories)
    return {f"eval_{k}": v for k,v in cat_metrics.items()}

df = pd.read_csv(prompt_file)

all_cats = set()
for out in df["output"].fillna("").tolist():
    for line in out.splitlines():
        if ":" in line:
            all_cats.add(line.split(":", 1)[0].strip())
categories = sorted(all_cats)
model_base.config.categories = categories

ds = Dataset.from_pandas(df)
preds, refs = [], []
for example in ds:
    prompt = example['prompt']
    expected = example['output']
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(model_base.device)
    with torch.no_grad():
        out_ids = model_base.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=inputs["input_ids"].shape[-1] + 200,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.2,
        )
    raw = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    preds.append(raw)
    refs.append(expected)

cat_metrics = compute_categorical_metrics(preds, refs, model_base.config.categories)
