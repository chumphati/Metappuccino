##########################################################################################
# IMPORT
import random
import math
import torch
import re
import os
import optuna
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding,  TrainerCallback, TrainerControl, TrainerState
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import argparse
from peft import PeftModel
from torch.utils.tensorboard import SummaryWriter
from transformers import EarlyStoppingCallback, TrainerCallback
import torch.nn.functional as F
from sklearn.model_selection import KFold, train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
parser = argparse.ArgumentParser(description="Fine-tune model")
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

# parameters for semantic matching
sem_model_name = 'sentence-transformers/all-mpnet-base-v2'
model_sem = SentenceTransformer(sem_model_name)
# value to consider prediction true
threshold = 0.45

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


# compute per-category metrics using semantic similarity
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

##########################################################################################
#CUSTOMED EARLY STOP IF NO CATEGORY IMPROVE
class MultiMetricEarlyStoppingCallback(TrainerCallback):
    def __init__(self, metrics: list, patience: int = 3, verbose: bool = True):
        self.metrics = metrics
        self.patience = patience
        self.verbose = verbose
        self.best = {m: -math.inf for m in metrics}
        self.num_bad_epochs = 0

    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        logs = logs or {}
        improved = False
        for m in self.metrics:
            current = logs.get(m)
            if current is not None and current > self.best[m]:
                if self.verbose:
                    print(f"Improvement {m}: {self.best[m]:.4f} → {current:.4f}")
                self.best[m] = current
                improved = True

        if improved:
            self.num_bad_epochs = 0
            control.should_save = True
        else:
            self.num_bad_epochs += 1
            if self.verbose:
                print(f"Epoch without improvement = {self.num_bad_epochs}")

        if self.num_bad_epochs >= self.patience:
            if self.verbose:
                print(f"Patience off ({self.patience}), stop")
            control.should_early_stop = True
            control.should_save = True

        return control


##########################################################################################
#LOG TRAIN
class LossLoggerCallback(TrainerCallback):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(os.path.join(log_dir, "train_losses"))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        if "loss" in logs:
            self.writer.add_scalar("train/loss_total", logs["loss"], step)
        if "train_loss_ce" in logs:
            self.writer.add_scalar("train/loss_ce", logs["train_loss_ce"], step)
        if "train_loss_sem" in logs:
            self.writer.add_scalar("train/loss_sem", logs["train_loss_sem"], step)

##########################################################################################
#CUSTOMED TRAINER WITH SEMANTIC LOSS
class MyTrainer(Trainer):
    def __init__(self, *args, sem_model=None, sem_loss_weight: float = 1.0, tokenizer=None, **kwargs):
        super().__init__(*args, **kwargs)
        #freeze semantic model
        self.sem_model = sem_model.eval()
        for p in self.sem_model.parameters():
            p.requires_grad = False
        self.sem_loss_weight = sem_loss_weight
        self.tokenizer = tokenizer

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        #cross-entropy
        outputs = model(**inputs)
        loss_ce = outputs.loss
        #prepare text
        logits = outputs.logits.detach()
        pred_ids = torch.argmax(logits, dim=-1)
        pred_texts, ref_texts = [], []
        for ids, lab in zip(pred_ids, inputs['labels']):
            mask = lab != -100
            p = self.tokenizer.decode(ids[mask], skip_special_tokens=True).strip()
            r = self.tokenizer.decode(lab[mask], skip_special_tokens=True).strip()
            if r and r.lower() != 'nan':
                pred_texts.append(p)
                ref_texts.append(r)

        #semantic loss
        if pred_texts:
            emb_p = self.sem_model.encode(pred_texts, convert_to_tensor=True).to(loss_ce.device)
            emb_r = self.sem_model.encode(ref_texts, convert_to_tensor=True).to(loss_ce.device)
            cos = F.cosine_similarity(emb_p, emb_r, dim=-1)
            loss_sem = (1.0 - cos).mean()
        else:
            loss_sem = torch.tensor(0.0, device=loss_ce.device)

        loss = loss_ce + self.sem_loss_weight * loss_sem

        if self.state.is_world_process_zero and self.model.training:
            self.log({
                "train_loss_ce": loss_ce.detach().item(),
                "train_loss_sem": loss_sem.detach().item()
            })

        #return for loss eval
        if return_outputs:
            outputs.loss_ce = loss_ce
            outputs.loss_sem = loss_sem
            outputs.loss_total = loss
            return loss, outputs
        else:
            return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kwargs):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        total_ce, total_sem, total_loss, count = 0.0, 0.0, 0.0, 0

        for inputs in eval_dataloader:
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                loss, outputs = self.compute_loss(self.model, inputs, return_outputs=True)
            total_ce += outputs.loss_ce.item()
            total_sem += outputs.loss_sem.item()
            total_loss += outputs.loss_total.item()
            count += 1

        avg_ce = total_ce / count if count else 0.0
        avg_sem = total_sem / count if count else 0.0
        avg_total = total_loss / count if count else 0.0

        results = {
            f"{metric_key_prefix}_loss_ce": avg_ce,
            f"{metric_key_prefix}_loss_sem": avg_sem,
            f"{metric_key_prefix}_loss": avg_total
        }
        self.log(results)

        #calculate metrics per category
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        preds, refs = [], []
        for example in ds:
            prompt = example['prompt']
            expected = example['output']
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000).to(self.model.device)
            with torch.no_grad():
                out_ids = self.model.generate(**inputs, max_new_tokens=100, do_sample=False, early_stopping=True)
            raw = self.tokenizer.decode(out_ids[0], skip_special_tokens=True)
            preds.append(raw)
            refs.append(expected)

        cat_metrics = compute_categorical_metrics(preds, refs, self.model.config.categories)
        for k, v in cat_metrics.items():
            key = f"{metric_key_prefix}_{k}"
            self.log({key: v})
            results[key] = v

        return results


##########################################################################################
# MAIN
print("Load dataset with prompts", flush=True)
df = pd.read_csv(prompt_file)

all_cats = set()
for out in df["output"].fillna("").tolist():
    for line in out.splitlines():
        if ":" in line:
            all_cats.add(line.split(":", 1)[0].strip())
categories = sorted(all_cats)
model_base.config.categories = categories

metric_names = [f"eval_accuracy_{c.lower()}" for c in categories]

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
    # print(f"Test example {i + 1} → Run accession: {run_accession}", flush=True)
print(run_accessions_list, flush=True)

##########################################################################################
# OPTUNA HYPERPARAM SEARCH

# subsample to find hp quicker
# subsample_train_frac = 0.025 #~400 sample train
subsample_train_frac = 1
subsample_val_frac = 0.2 #~75 sample val

writer_folds = SummaryWriter(os.path.join(tensorboard_log_dir, "folds"))


def objective(trial):
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    num_train_epochs = trial.suggest_int("num_train_epochs", 3, 8)
    r = trial.suggest_categorical("r", [8, 16, 32])
    lora_alpha = trial.suggest_int("lora_alpha", 16, 64, step=16)
    lora_dropout = trial.suggest_float("lora_dropout", 0.2, 0.5)

    # LoRA config
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
    model.config.categories = categories

    # sub sample
    df_sub = df_trainval.sample(frac=subsample_train_frac, random_state=SEED).reset_index(drop=True)
    df_sub_train, df_sub_val = train_test_split(df_sub, test_size=subsample_val_frac, random_state=SEED)
    df_sub_train = df_sub_train.reset_index(drop=True)
    df_sub_val = df_sub_val.reset_index(drop=True)

    train_dataset_fold = Dataset.from_pandas(df_sub_train)
    val_dataset_fold = Dataset.from_pandas(df_sub_val)

    train_tokenized = train_dataset_fold.map(clean_output_text).map(tokenize_function)
    val_tokenized = val_dataset_fold.map(clean_output_text).map(tokenize_function)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt", padding=True)

    print("FOLD", trial.number, flush=True)
    print("Subsampled run accessions in train:", flush=True)
    print(df_sub_train["prompt"].str.extract(r"Run accession:\s*(\S+)")[0].value_counts(), flush=True)
    print("Subsampled run accessions in val:", flush=True)
    print(df_sub_val["prompt"].str.extract(r"Run accession:\s*(\S+)")[0].value_counts(), flush=True)

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
        load_best_model_at_end=False,
        metric_for_best_model="eval_accuracy_overall",
        greater_is_better=True
    )

    multi_es = MultiMetricEarlyStoppingCallback(
        metrics=[f"eval_accuracy_{c.lower()}" for c in categories],
        patience=2,
        verbose=True
    )

    trainer = MyTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[
            multi_es,
            LossLoggerCallback(tensorboard_log_dir)
        ],
        sem_model=model_sem,
        sem_loss_weight=0.45
    )

    trainer.train()

    eval_results = trainer.evaluate()
    eval_loss = eval_results.get("eval_loss")

    writer_folds.add_scalar("fold/train_size", len(df_sub_train), trial.number)
    writer_folds.add_scalar("fold/val_size", len(df_sub_val), trial.number)
    writer_folds.add_scalar("fold/eval_loss", eval_loss, trial.number)

    for cat in categories:
        key = f"eval_accuracy_{cat.lower()}"
        writer_folds.add_scalar(f"fold_{trial.number}/{cat}", eval_results[key], trial.number)
    writer_folds.add_scalar(f"fold_{trial.number}/overall", eval_results["eval_accuracy_overall"], trial.number)

    trainer.save_model(training_args.output_dir)
    del model
    torch.cuda.empty_cache()

    return eval_loss


study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5)

best_params = study.best_trial.params
print("\n" + "-" * 80, flush=True)
print(f"Best hyperparams after Optuna search: {best_params}", flush=True)
print("-" * 80 + "\n", flush=True)

##########################################################################################
# FINAL TRAINING ON ALL DATA (TRAIN+VAL)

# new cut
# df_sub = df_trainval.sample(frac=subsample_train_frac, random_state=SEED).reset_index(drop=True)
df_train_final, df_val_final = train_test_split(df_trainval, test_size=0.2, random_state=SEED)
df_train_final = df_train_final.reset_index(drop=True)
df_val_final = df_val_final.reset_index(drop=True)
print(f"Final training set size: {len(df_train_final)}, Final validation set size: {len(df_val_final)}", flush=True)

final_writer_training = SummaryWriter(os.path.join(tensorboard_log_dir, "final_training"))
final_writer_training.add_scalar("final/train_size", len(df_train_final), 0)
final_writer_training.add_scalar("final/val_size", len(df_val_final), 0)

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
model_final.config.categories = categories

training_args_final = TrainingArguments(
    output_dir=train_model + "_final",
    evaluation_strategy="epoch",
    learning_rate=best_params["learning_rate"],
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=best_params["num_train_epochs"],
    weight_decay=0.05,
    save_strategy='epoch',
    logging_strategy='steps',
    logging_steps=100,
    fp16=True,
    gradient_accumulation_steps=2,
    report_to=["tensorboard"],
    logging_dir=os.path.join(tensorboard_log_dir, "final_training"),
    load_best_model_at_end=False,
    eval_steps=None,
    metric_for_best_model="eval_accuracy_overall",
    greater_is_better=True
)

multi_es = MultiMetricEarlyStoppingCallback(
    metrics=[f"eval_accuracy_{c.lower()}" for c in categories],
    patience=3,
    verbose=True
)

trainer_final = MyTrainer(
    model=model_final,
    args=training_args_final,
    train_dataset=tokenized_train_final,
    eval_dataset=tokenized_val_final,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt", padding=True),
    callbacks=[
        multi_es,
        LossLoggerCallback(tensorboard_log_dir)
    ],
    sem_model=model_sem,
    sem_loss_weight=0.45
)

init_val = trainer_final.evaluate(eval_dataset=tokenized_val_final, metric_key_prefix="eval")
print("Initial val metrics - original model (step=0) →", init_val)

final_writer_training.add_scalar("eval_loss", init_val["eval_loss"], 0)
for cat in categories:
    final_writer_training.add_scalar(f"eval_accuracy_{cat.lower()}", init_val[f"eval_accuracy_{cat.lower()}"], 0)

trainer_final.train()
final_writer_training.close()

print("Save new model", flush=True)
trainer_final.save_model(output_model)

print("Merge and save full model", flush=True)
model_final = model_final.merge_and_unload()
model_final.save_pretrained(merged_model_path)
tokenizer.save_pretrained(merged_model_path)

##########################################################################################
# EVAL ON VALIDATION SET

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

# acc_val = sum(semantic_match(p, r) for p, r in zip(predictions, references)) / len(predictions)
# print(f"\nSemantic Match Accuracy on validation set: {acc_val:.4f}", flush=True)

metrics_val = compute_categorical_metrics(predictions, references, categories)
print("\nCategory SMA - validation set:", flush=True)
for cat in categories:
    print(f"{cat}: {metrics_val[f'accuracy_{cat.lower()}']:.4f}", flush=True)

print("\n" + "-" * 80, flush=True)

final_val_writer = SummaryWriter(os.path.join(tensorboard_log_dir, "final_validation"))
for cat in categories:
    final_val_writer.add_scalar(f"validation/{cat}", metrics_val[f"accuracy_{cat.lower()}"], 0)
final_val_writer.close()

##########################################################################################
# EVAL ON TEST
print("\nGenerating predictions on final test set", flush=True)
test_df = df_test
final_writer = SummaryWriter(os.path.join(tensorboard_log_dir, "final_predictions"))
# store preds and refs
test_preds, test_refs = [], []

for i, row in test_df.iterrows():
    prompt = row["prompt"].strip()
    print(prompt, flush=True)
    expected = row["output"].strip()
    input_encoded = tokenizer(prompt, return_tensors="pt").to(model_final.device)

    with torch.no_grad():
        output_ids = model_final.generate(
            **input_encoded,
            max_new_tokens=100,
            do_sample=False,
            early_stopping=True
        )
    raw = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
    split_output = raw.split("Here is the output:")
    after_output = split_output[1].strip() if len(split_output) > 1 else raw
    clean_prediction = deduplicate_categories(after_output)

    print(f"--- Predicted output {i + 1}: {clean_prediction}", flush=True)
    print(f"--- Expected output  {i + 1}: {expected}", flush=True)
    print("-" * 50, flush=True)

    test_preds.append(clean_prediction)
    test_refs.append(expected)

    final_writer.add_text(
        f"Prediction/Test_Example_{i + 1}",
        f"Prompt: {prompt}\nPred: {clean_prediction} | Label: {expected}",
        global_step=i
    )

final_writer.close()

metrics_test = compute_categorical_metrics(test_preds, test_refs, categories)
print("\nCategory SMA - test set:", flush=True)
for cat in categories:
    print(f"{cat}: {metrics_test[f'accuracy_{cat.lower()}']:.4f}", flush=True)

final_test_writer = SummaryWriter(os.path.join(tensorboard_log_dir, "final_test"))
for cat in categories:
    final_test_writer.add_scalar(f"test/{cat}", metrics_test[f"accuracy_{cat.lower()}"], 0)
final_test_writer.close()