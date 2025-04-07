##########################################################################################
# IMPORT
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
import torch
from datasets import Dataset
import argparse
import os
import pandas as pd
from peft import PeftModel
from torch.utils.tensorboard import SummaryWriter
from transformers import EarlyStoppingCallback, TrainerCallback
import re
import numpy as np

##########################################################################################
# PATHS
parser = argparse.ArgumentParser(description="Fine-tune model")
parser.add_argument("--base_path", type=str, required=True, help="Base path to MetaMap")
args = parser.parse_args()
base_path = args.base_path

prompt_file = os.path.join(base_path, "finetune_data.csv")
train_model = os.path.join(base_path, "mistral7B_train")
output_model = os.path.join(base_path, "mistral7B_fine_tuned")
merged_model_path = os.path.join(base_path, "mistral7B_full_finetuned")
model_name = os.path.join(base_path, "Mistral-7B-Instruct-v0.3")
tensorboard_log_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/tensorboard"

##########################################################################################
# MODEL
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Load model in FP16", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto'
)

print("Config LoRA", flush=True)
peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=['q_proj', 'v_proj']
)

model = get_peft_model(model, peft_config)

##########################################################################################
# FUNCTIONS


def tokenize_function(example):
    prompt = example["prompt"].strip()
    output = example["output"].strip()
    prompt_ids = tokenizer(prompt, truncation=True, padding=False, max_length=512)["input_ids"]
    output_ids = tokenizer(output, truncation=True, padding=False, max_length=128)["input_ids"]

    input_ids = prompt_ids + output_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + output_ids

    max_length = 640
    padding_length = max_length - len(input_ids)
    if padding_length > 0:
        input_ids += [tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        labels += [-100] * padding_length
    else:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def extract_clean_response(text):
    split_output = text.split("Here is the output:")
    after_output = split_output[1].strip() if len(split_output) > 1 else text
    match = re.match(r'^(.*?)([\-\[\(\n,\"\']|$)', after_output)
    return match.group(1).strip().lower() if match else after_output.strip().lower()


def semantic_match(pred, ref):
    pred = extract_clean_response(pred)
    ref = extract_clean_response(ref)
    return ref in pred or pred in ref

##########################################################################################
# MAIN


print("Load dataset with prompts", flush=True)
df = pd.read_csv(prompt_file)
dataset_full = Dataset.from_pandas(df)
dataset_split = dataset_full.train_test_split(test_size=0.2, seed=42)
train_val_split = dataset_split['train'].train_test_split(test_size=0.1, seed=42)
dataset_split = {
    'train': train_val_split['train'],
    'validation': train_val_split['test'],
    'test': dataset_split['test']
}
print(dataset_split, flush=True)
tokenized_datasets = {k: v.map(tokenize_function) for k, v in dataset_split.items()}

print("Content of test set:\n", flush=True)
test_dataset = dataset_split["test"].to_pandas()
run_accessions_list = []
for i, row in test_dataset.iterrows():
    prompt = row["prompt"].strip()
    first_line = prompt.splitlines()[0]
    match = re.search(r"Run accession:\s*(\S+)", first_line)
    run_accession = match.group(1) if match else "N/A"
    run_accessions_list.append(run_accession)
    print(f"Test example {i+1} → Run accession: {run_accession}", flush=True)
print(run_accessions_list, flush=True)

print("Config training args", flush=True)
training_args = TrainingArguments(
    output_dir=train_model,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    num_train_epochs=20,
    weight_decay=0.01,
    save_strategy='epoch',
    logging_strategy='steps',
    logging_steps=10,
    fp16=True,
    bf16=False,
    gradient_accumulation_steps=4,
    dataloader_num_workers=30,
    report_to=["tensorboard"],
    logging_dir=tensorboard_log_dir,
    prediction_loss_only=False,
    eval_accumulation_steps=10,
    dataloader_pin_memory=False,
    load_best_model_at_end=True,
    eval_steps=None,
    metric_for_best_model="eval_loss"
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt", padding=True)

writer = SummaryWriter(tensorboard_log_dir)

print("Begin training", flush=True)


class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if "loss" in logs:
                writer.add_scalar("train_loss", logs["loss"], state.global_step)
            if "eval_loss" in logs:
                writer.add_scalar("val_loss", logs["eval_loss"], state.global_step)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3), CustomLoggingCallback()]
)

trainer.train()

print("Save new model", flush=True)
trainer.save_model(output_model)
print("Merge and save full model", flush=True)
model = model.merge_and_unload()
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)
print("Evaluating on validation set", flush=True)
eval_df = dataset_split["validation"].to_pandas()
predictions, references = [], []

for i, row in eval_df.iterrows():
    print("BEFORE", flush=True)
    prompt = row["prompt"].strip()
    print(prompt, flush=True)
    expected = row["output"].strip()
    print(expected, flush=True)
    input_encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**input_encoded, max_new_tokens=20, do_sample=False, early_stopping=True)
    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    print("AFTER", flush=True)
    print(prediction, flush=True)
    predictions.append(prediction)
    references.append(expected)
    print(expected, flush=True)

acc = sum(semantic_match(p, r) for p, r in zip(predictions, references)) / len(predictions)
print(f"Semantic Match Accuracy on validation set: {acc:.4f}", flush=True)
writer.add_scalar("semantic_match", acc, trainer.state.global_step)

##########################################################################################
# PREDICTIONS POST-TRAINING
print("Generating predictions on test set", flush=True)
eval_df = dataset_split["test"].to_pandas()
for i, row in eval_df.iterrows():
    prompt = row["prompt"].strip()
    expected = row["output"].strip()
    input_encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(**input_encoded, max_new_tokens=20, do_sample=False, early_stopping=True)
    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    split_output = prediction.split("Here is the output:")
    after_output = split_output[1].strip() if len(split_output) > 1 else prediction
    match = re.match(r'^(.*?)([\-\[\(\n,\"\']|$)', after_output)
    clean_prediction = match.group(1).strip() if match else after_output
    print(f"--- Predicted output {i+1}: {clean_prediction}", flush=True)
    print(f"--- Expected output {i+1}: {expected}", flush=True)
    print("-" * 50, flush=True)
    writer.add_text(f"Prediction/Example_{i+1}", f"Pred: {clean_prediction} | Label: {expected}", global_step=trainer.state.global_step)

writer.close()
