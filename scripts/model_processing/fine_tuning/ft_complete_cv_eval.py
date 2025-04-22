##########################################################################################
# IMPORT
import random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import argparse
import os
import pandas as pd
from peft import PeftModel
from torch.utils.tensorboard import SummaryWriter
from transformers import EarlyStoppingCallback, TrainerCallback
import re
from sklearn.model_selection import KFold, train_test_split
import optuna

##########################################################################################
# SEEDS
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

##########################################################################################
# PATHS
parser = argparse.ArgumentParser(description="Fine-tune model (with optional cross-validation)")
parser.add_argument("--base_path", type=str, required=True, help="Base path to MetaMap")
parser.add_argument("--n_splits", type=int, default=5, help="Number of CV folds")
args = parser.parse_args()

base_path = args.base_path
n_splits = args.n_splits

prompt_file = os.path.join(base_path, "finetune_data.csv")
train_model = os.path.join(base_path, "mistral7B_train")
output_model = os.path.join(base_path, "mistral7B_fine_tuned")
merged_model_path = os.path.join(base_path, "mistral7B_full_finetuned")
model_name = os.path.join(base_path, "Mistral-7B-Instruct-v0.3")
tensorboard_log_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results_train/FINE_TUNING_COMPLETE/tensorboard"

##########################################################################################
# MODEL
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

print("Load model in FP16", flush=True)
model_base = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto'
)

print("Config LoRA", flush=True)

##########################################################################################
# FUNCTIONS

# tokenize input
def tokenize_function(example):
    prompt = example["prompt"].strip()
    output = example["output"].strip()
    prompt_ids = tokenizer(prompt, truncation=True, max_length=512)["input_ids"]
    output_ids = tokenizer(output, truncation=True, max_length=128)["input_ids"]

    input_ids = prompt_ids + output_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + output_ids

    max_length = 640
    padding_length = max_length - len(input_ids)
    input_ids += [tokenizer.pad_token_id] * padding_length
    attention_mask += [0] * padding_length
    labels += [-100] * padding_length

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# extract output LLM answer
def extract_clean_response(text):
    split_output = text.split("Here is the output:")
    after_output = split_output[1].strip() if len(split_output) > 1 else text
    match = re.match(r'^(.*?)([\-\[\("\',]|$)', after_output)
    return match.group(1).strip().lower() if match else after_output.strip().lower()

# see if match between prediction and reference
def semantic_match(pred, ref):
    pred = extract_clean_response(pred)
    ref = extract_clean_response(ref)
    return ref in pred or pred in ref

# deduplicate prediction categories
def deduplicate_categories(pred_text):
    seen = set()
    final_output = []
    for line in pred_text.splitlines():
        if ':' in line:
            cat, val = line.split(':', 1)
            cat = cat.strip()
            if cat not in seen:
                final_output.append(f"{cat}: {val.strip()}")
                seen.add(cat)
    return '\n'.join(final_output)

# clean output before tokenization
def clean_output_text(example):
    example["output"] = deduplicate_categories(example["output"])
    return example

##########################################################################################
# MAIN

print("Load dataset with prompts", flush=True)
df = pd.read_csv(prompt_file)

# independant test
df_trainval, df_test = train_test_split(df, test_size=0.1, random_state=SEED)
df_trainval = df_trainval.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print(f"Size of train+val set: {len(df_trainval)}, size of test set: {len(df_test)}", flush=True)

print("Content of test set:\n", flush=True)
run_accessions_list = []
for i, row in df_test.iterrows():
    prompt = row["prompt"].strip()
    first_line = prompt.splitlines()[0]
    match = re.search(r"Run accession:\s*(\S+)", first_line)
    run_accession = match.group(1) if match else "N/A"
    run_accessions_list.append(run_accession)
    print(f"Test example {i + 1} → Run accession: {run_accession}", flush=True)
print(run_accessions_list, flush=True)

##########################################################################################
# OPTUNA HYPERPARAM SEARCH

subsample_train_frac = 0.2  #20% train+val for optimisation
subsample_val_frac = 0.1    #10% du train+val for eval

writer_folds = SummaryWriter(os.path.join(tensorboard_log_dir, "folds"))

def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 8)
    r = trial.suggest_categorical("r", [8, 16, 32])
    lora_alpha = trial.suggest_int("lora_alpha", 16, 64, step=16)
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.3)

    #LoRA config
    peft_config_current = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=['q_proj', 'v_proj']
    )

    model = get_peft_model(
        model_base.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto'
        ),
        peft_config_current
    )

    #sub sample
    df_sub = df_trainval.sample(frac=subsample_train_frac, random_state=SEED).reset_index(drop=True)
    df_sub_train, df_sub_val = train_test_split(df_sub, test_size=subsample_val_frac, random_state=SEED)
    df_sub_train = df_sub_train.reset_index(drop=True)
    df_sub_val = df_sub_val.reset_index(drop=True)

    train_dataset_fold = Dataset.from_pandas(df_sub_train)
    val_dataset_fold = Dataset.from_pandas(df_sub_val)

    train_tokenized = train_dataset_fold.map(clean_output_text).map(tokenize_function)
    val_tokenized = val_dataset_fold.map(clean_output_text).map(tokenize_function)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt", padding=True)

    training_args = TrainingArguments(
        output_dir=f"./tmp_model_trial_{trial.number}",
        logging_dir=os.path.join(tensorboard_log_dir, f"fold_{trial.number}"),
        run_name=f"fold_{trial.number}",
        evaluation_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        fp16=True,
        gradient_accumulation_steps=2,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks
    )

    trainer.train()

    eval_results = trainer.evaluate()
    eval_loss = eval_results.get("eval_loss")

    writer_folds.add_scalar("fold/train_size", len(df_sub_train), trial.number)
    writer_folds.add_scalar("fold/val_size", len(df_sub_val), trial.number)
    writer_folds.add_scalar("fold/eval_loss", eval_loss, trial.number)

    trainer.save_model(training_args.output_dir)
    del model
    torch.cuda.empty_cache()

    return eval_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

best_params = study.best_trial.params
print("\n" + "-" * 80, flush=True)
print(f"Best hyperparams after Optuna search: {best_params}", flush=True)
print("-" * 80 + "\n", flush=True)

##########################################################################################
# FINAL TRAINING ON ALL DATA (TRAIN+VAL)

#new cut
df_train_final, df_val_final = train_test_split(df_trainval, test_size=0.1, random_state=SEED)
df_train_final = df_train_final.reset_index(drop=True)
df_val_final = df_val_final.reset_index(drop=True)
print(f"Final training set size: {len(df_train_final)}, Final validation set size: {len(df_val_final)}", flush=True)

final_writer_training = SummaryWriter(os.path.join(tensorboard_log_dir, "final_training"))
final_writer_training.add_scalar("final/train_size", len(df_train_final), 0)
final_writer_training.add_scalar("final/val_size", len(df_val_final), 0)
final_writer_training.close()

train_dataset_final = Dataset.from_pandas(df_train_final)
val_dataset_final = Dataset.from_pandas(df_val_final)
tokenized_train_final = train_dataset_final.map(clean_output_text).map(tokenize_function)
tokenized_val_final = val_dataset_final.map(clean_output_text).map(tokenize_function)

final_peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=best_params["r"],
    lora_alpha=best_params["lora_alpha"],
    lora_dropout=best_params["lora_dropout"],
    target_modules=['q_proj', 'v_proj']
)

model_final = get_peft_model(
    model_base.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto'
    ),
    final_peft_config
)

training_args_final = TrainingArguments(
    output_dir=train_model + "_final",
    evaluation_strategy="epoch",
    learning_rate=best_params["learning_rate"],
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=best_params["num_train_epochs"],
    weight_decay=0.01,
    save_strategy='epoch',
    logging_strategy='steps',
    logging_steps=10,
    fp16=True,
    gradient_accumulation_steps=2,
    report_to=["tensorboard"],
    logging_dir=os.path.join(tensorboard_log_dir, "final_training"),
    load_best_model_at_end=True,
    eval_steps=None,
    metric_for_best_model="eval_loss"
)

trainer_final = Trainer(
    model=model_final,
    args=training_args_final,
    train_dataset=tokenized_train_final,
    eval_dataset=tokenized_val_final,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt", padding=True),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

trainer_final.train()

print("Save new model", flush=True)
trainer_final.save_model(output_model)

print("Merge and save full model", flush=True)
model_final = model_final.merge_and_unload()
model_final.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

##########################################################################################
# EVAL ON TEST

print("\n" + "-" * 80, flush=True)
print("Evaluating on validation set", flush=True)
eval_df = df_val_final
predictions, references = [], []

for i, row in eval_df.iterrows():
    print("-" * 50, flush=True)
    print("BEFORE", flush=True)
    print("PROMPT", flush=True)
    prompt = row["prompt"].strip()
    print(prompt, flush=True)
    print("EXPECTED OUTPUT", flush=True)
    expected = row["output"].strip()
    print(expected, flush=True)

    input_encoded = tokenizer(prompt, return_tensors="pt").to(model_final.device)

    with torch.no_grad():
        output_ids = model_final.generate(**input_encoded, max_new_tokens=100, do_sample=False, early_stopping=True)

    raw_pred = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    split_output = raw_pred.split("Here is the output:")
    after_output = split_output[1].strip() if len(split_output) > 1 else raw_pred
    prediction = deduplicate_categories(after_output)

    print("AFTER", flush=True)
    print("PREDICTION", flush=True)
    print(prediction, flush=True)
    print("-" * 50, flush=True)

    predictions.append(prediction)
    references.append(expected)

    print("EXPECTED OUTPUT", flush=True)
    print(expected, flush=True)

print("\n" + "-" * 80, flush=True)

acc_val = sum(semantic_match(p, r) for p, r in zip(predictions, references)) / len(predictions)
print(f"\nSemantic Match Accuracy on validation set: {acc_val:.4f}", flush=True)
print("\n" + "-" * 80, flush=True)

##########################################################################################
# EVAL ON TEST

print("\nGenerating predictions on final test set", flush=True)
test_df = df_test

final_writer = SummaryWriter(os.path.join(tensorboard_log_dir, "final_predictions"))

for i, row in test_df.iterrows():
    prompt = row["prompt"].strip()
    print(prompt, flush=True)
    expected = row["output"].strip()
    input_encoded = tokenizer(prompt, return_tensors="pt").to(model_final.device)
    with torch.no_grad():
        output_ids = model_final.generate(**input_encoded, max_new_tokens=100, do_sample=False, early_stopping=True)
    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    split_output = prediction.split("Here is the output:")
    after_output = split_output[1].strip() if len(split_output) > 1 else prediction
    clean_prediction = deduplicate_categories(after_output)

    print(f"--- Predicted output {i+1}: {clean_prediction}", flush=True)
    print(f"--- Expected output {i+1}: {expected}", flush=True)
    print("-" * 50, flush=True)

    final_writer.add_text(
        f"Prediction/Test_Example_{i+1}",
        f"Prompt: {prompt}\nPred: {clean_prediction} | Label: {expected}",
        global_step=i
    )

final_writer.close()
