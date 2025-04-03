from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding
from peft import LoraConfig, get_peft_model
import torch
from datasets import load_dataset
import argparse
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from peft import PeftModel
from torch.utils.tensorboard import SummaryWriter
from transformers import EarlyStoppingCallback
import re

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
    # print(output, flush=True)

    prompt_ids = tokenizer(prompt, truncation=True, padding=False, max_length=512)["input_ids"]
    output_ids = tokenizer(output, truncation=True, padding=False, max_length=128)["input_ids"]
    # print(output_ids, flush=True)

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


##########################################################################################
# MAIN
print("Load dataset with prompts", flush=True)
df = pd.read_csv(prompt_file)
# print(df, flush=True)
dataset_full = Dataset.from_pandas(df)
dataset_split = dataset_full.train_test_split(test_size=0.1, seed=42)
# print(dataset_split, flush=True)
tokenized_datasets = dataset_split.map(tokenize_function)

print("Config training args", flush=True)
training_args = TrainingArguments(
    output_dir=train_model,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=1,
    num_train_epochs=2,
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
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt", padding=True)

writer = SummaryWriter(tensorboard_log_dir)

print("Begin training", flush=True)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['test'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

trainer.train()

print("Save new model", flush=True)
trainer.save_model(output_model)
print("Merge and save full model", flush=True)
model = model.merge_and_unload()
model.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

##########################################################################################
# PREDICTIONS POST-TRAINING
print("Generating predictions on validation set", flush=True)
eval_df = dataset_split["test"].to_pandas()
for i, row in eval_df.iterrows():
    prompt = row["prompt"].strip()
    print("Prompt: ", prompt, flush=True)
    expected = row["output"].strip()
    print("Expected: ", expected, flush=True)

    input_encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        # output_ids = model.generate(**input_encoded, max_new_tokens=50)
        output_ids = model.generate(**input_encoded, max_new_tokens=20, do_sample=False, early_stopping=True)

    prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    split_output = prediction.split("Here is the output:")
    if len(split_output) > 1:
        after_output = split_output[1].strip()
    else:
        after_output = prediction

    match = re.match(r"^(.*?)([,\-\[\(\n]|$)", after_output)
    clean_prediction = match.group(1).strip() if match else after_output

    print(f"--- Predicted output {i+1}: {clean_prediction}", flush=True)
    print(f"--- Expected output {i+1}: {expected}", flush=True)
    print("-" * 50, flush=True)
    writer.add_text(f"Prediction/Example_{i+1}", f"Pred: {clean_prediction} | Label: {expected}", global_step=trainer.state.global_step)

writer.close()
