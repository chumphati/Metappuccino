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
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

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
tensorboard_log_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/tensorboard"

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
df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
dataset_for_cv = df_train.reset_index(drop=True)

print(f"Performing {n_splits}-fold cross-validation on the entire dataset of size {len(dataset_for_cv)}", flush=True)

##########################################################################################
#HYPERPARAM GRID TEST
hyperparam_grid = [
    {
        "learning_rate": 2e-5,
        "num_train_epochs": 5,
        "grad_accum_steps": 4,
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05
    },
    {
        "learning_rate": 2e-5,
        "num_train_epochs": 8, #test epoch
        "grad_accum_steps": 4,
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05
    },
    {
        "learning_rate": 3e-5, #test lr
        "num_train_epochs": 5,
        "grad_accum_steps": 4,
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05
    },
    {
        "learning_rate": 2e-5,
        "num_train_epochs": 5,
        "grad_accum_steps": 4,
        "r": 8, #test r
        "lora_alpha": 32,
        "lora_dropout": 0.05
    },
    {
        "learning_rate": 2e-5,
        "num_train_epochs": 5,
        "grad_accum_steps": 4,
        "r": 16,
        "lora_alpha": 16, #test la
        "lora_dropout": 0.05
    },
    {
        "learning_rate": 2e-5,
        "num_train_epochs": 5,
        "grad_accum_steps": 4,
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.2 #test dropout
    },
]

best_params = None
best_cv_score = -1.0

##########################################################################################
#CROSS VALIDATION = TUNE HYPERPARAMS & PERFS
for param_idx, params in enumerate(hyperparam_grid, start=1):
    print("=" * 80)
    print(f"Testing hyperparam set {param_idx}/{len(hyperparam_grid)}: {params}", flush=True)
    print("=" * 80)

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    fold_number = 1
    fold_model_paths = []

    for train_index, val_index in kfold.split(dataset_for_cv):
        print("-" * 80)
        print(f"Starting fold {fold_number}", flush=True)
        print("-" * 80)

        #LoRAConfig adapted to the grid
        peft_config_current = LoraConfig(
            task_type="CAUSAL_LM",
            inference_mode=False,
            r=params["r"],
            lora_alpha=params["lora_alpha"],
            lora_dropout=params["lora_dropout"],
            target_modules=['q_proj', 'v_proj']
        )

        #create new LoRA wrapped model from the base each fold
        model = get_peft_model(model_base.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto'
        ), peft_config_current)

        #build train/val subsets
        train_df_fold = dataset_for_cv.iloc[train_index].copy()
        val_df_fold = dataset_for_cv.iloc[val_index].copy()

        #convert to HuggingFace Datasets
        train_dataset_fold = Dataset.from_pandas(train_df_fold)
        val_dataset_fold = Dataset.from_pandas(val_df_fold)

        #tokenize
        train_tokenized = train_dataset_fold.map(tokenize_function)
        val_tokenized = val_dataset_fold.map(tokenize_function)
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt", padding=True)

        #fold tensorboard
        fold_tensorboard_dir = os.path.join(tensorboard_log_dir, f"params_{param_idx}", f"fold_{fold_number}")
        fold_writer = SummaryWriter(fold_tensorboard_dir)

        #training arguments en prenant les params depuis la grille
        training_args = TrainingArguments(
            output_dir=train_model + f"_param{param_idx}_fold{fold_number}",
            evaluation_strategy="epoch",
            learning_rate=params["learning_rate"],
            per_device_train_batch_size=4,
            per_device_eval_batch_size=1,
            num_train_epochs=params["num_train_epochs"],
            weight_decay=0.01,
            save_strategy='epoch',
            logging_strategy='steps',
            logging_steps=10,
            fp16=True,
            bf16=False,
            gradient_accumulation_steps=params["grad_accum_steps"],
            dataloader_num_workers=2,
            report_to=["tensorboard"],
            logging_dir=fold_tensorboard_dir,
            prediction_loss_only=False,
            eval_accumulation_steps=10,
            dataloader_pin_memory=False,
            load_best_model_at_end=True,
            eval_steps=None,
            metric_for_best_model="eval_loss"
        )

        # custom callback for logging
        class CustomLoggingCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs is not None:
                    if "loss" in logs:
                        fold_writer.add_scalar(f"Fold{fold_number}_train_loss", logs["loss"], state.global_step)
                    if "eval_loss" in logs:
                        fold_writer.add_scalar(f"Fold{fold_number}_val_loss", logs["eval_loss"], state.global_step)


        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_tokenized,
            eval_dataset=val_tokenized,
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3),
                       CustomLoggingCallback()]
        )

        #train
        trainer.train()

        #save best model from this fold
        fold_output_dir = output_model + f"_param{param_idx}_fold{fold_number}"
        print("Save fold model", flush=True)
        trainer.save_model(fold_output_dir)
        fold_model_paths.append(fold_output_dir)
        print("Merge and unload fold model", flush=True)
        model = model.merge_and_unload()

        #evaluate
        print(f"Evaluating on fold {fold_number} validation set", flush=True)
        val_preds = []
        val_refs = []

        for i, row in val_df_fold.iterrows():
            prompt = row["prompt"].strip()
            expected = row["output"].strip()
            input_encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(**input_encoded, max_new_tokens=20, do_sample=False, early_stopping=True)
            prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
            val_preds.append(prediction)
            val_refs.append(expected)

        #semantic match accuracy
        acc = sum(semantic_match(p, r) for p, r in zip(val_preds, val_refs)) / len(val_preds)
        cv_scores.append(acc)
        print(f"Fold {fold_number} - Semantic Match Accuracy: {acc:.4f}", flush=True)
        fold_writer.add_scalar(f"Fold{fold_number}_semantic_match_accuracy", acc, trainer.state.global_step)
        fold_writer.close()

        fold_number += 1

    #end cv
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)

    print("-" * 80, flush=True)
    print(f"Hyperparam set {param_idx} - Cross-Validation {n_splits}-fold Results:", flush=True)
    for i, score in enumerate(cv_scores, 1):
        print(f"Fold {i} accuracy: {score:.4f}", flush=True)
    print(f"Mean CV accuracy: {mean_score:.4f} ± {std_score:.4f}", flush=True)
    print("-" * 80, flush=True)

    #find best set
    if mean_score > best_cv_score:
        best_cv_score = mean_score
        best_params = params

print("\n\n", flush=True)
print("-" * 80, flush=True)
print(f"Best hyperparams after cross-validation: {best_params}, CV mean accuracy={best_cv_score:.4f}", flush=True)
print("-" * 80, flush=True)
print("\n\n")

##########################################################################################
# FINAL TRAINING ON BEST HYPERPARAMS

print("Retraining a final model on the entire dataset (optional) ...", flush=True)
full_dataset = Dataset.from_pandas(df)
tokenized_full = full_dataset.map(tokenize_function)

#best config
final_peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=best_params["r"],
    lora_alpha=best_params["lora_alpha"],
    lora_dropout=best_params["lora_dropout"],
    target_modules=['q_proj', 'v_proj']
)

model_final = get_peft_model(model_base.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto'
), final_peft_config)

training_args_final = TrainingArguments(
    output_dir=train_model + "_final",
    evaluation_strategy="epoch",
    learning_rate=best_params["learning_rate"],
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    num_train_epochs=best_params["num_train_epochs"],
    weight_decay=0.01,
    save_strategy='epoch',
    logging_strategy='steps',
    logging_steps=10,
    fp16=True,
    gradient_accumulation_steps=best_params["grad_accum_steps"],
    report_to=["tensorboard"],
    logging_dir=os.path.join(tensorboard_log_dir, "final_training"),
    load_best_model_at_end=False
)

trainer_final = Trainer(
    model=model_final,
    args=training_args_final,
    train_dataset=tokenized_full,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt", padding=True),
)

trainer_final.train()

print("Save final model", flush=True)
trainer_final.save_model(output_model + "_final")
model_final = model_final.merge_and_unload()
model_final.save_pretrained(merged_model_path + "_final")
tokenizer.save_pretrained(merged_model_path + "_final")
