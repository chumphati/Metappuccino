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
from transformers import AutoModelForCausalLM, StoppingCriteria, AutoTokenizer, TrainingArguments, Trainer, \
    DataCollatorWithPadding, TrainerCallback, TrainerControl, TrainerState, EvalPrediction, StoppingCriteriaList, \
    MaxLengthCriteria
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
from torch.nn import CrossEntropyLoss

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

prompt_train_file = os.path.join(base_path, "finetune_data_train_corrected.csv")
prompt_val_file = os.path.join(base_path, "finetune_data_val_corrected.csv")
train_model = os.path.join(base_path, "mistral7B_train")
output_model = os.path.join(base_path, "mistral7B_fine_tuned")
merged_model_path = os.path.join(base_path, "mistral7B_full_finetuned")
model_name = os.path.join(base_path, "Mistral-7B-Instruct-v0.3")
tensorboard_log_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/tensorboard"

# parameters for semantic matching
sem_model_name = 'pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb'
model_sem = SentenceTransformer(sem_model_name)
# value to consider prediction true
threshold = 0.45

##########################################################################################
# MODEL
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '<pad>'})
tokenizer.pad_token = '<pad>'

print("Load model in FP16", flush=True)
model_base = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map='auto'
)
model_base.resize_token_embeddings(len(tokenizer))

print("Config LoRA", flush=True)

##########################################################################################
# FUNCTIONS

# tokenize input
def tokenize_function(example):
    prompt = example["prompt"].strip()
    output = example["output"].strip()
    prompt_ids = tokenizer(prompt, truncation=True, max_length=4000)["input_ids"]
    output_ids = tokenizer(output, truncation=True, max_length=200)["input_ids"]

    input_ids = prompt_ids + output_ids
    attention_mask = [1] * len(input_ids)
    labels = [-100] * len(prompt_ids) + output_ids

    max_length = 4000
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


normal_accuracy_categories = {
    'cell_line', 'phenotype', 'library_selection', 'library_source', 'treatment_time'
}


def normalize(x):
    return re.sub(r'[-_]', ' ', str(x)).strip().lower()


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
            elif cat in normal_accuracy_categories:
                acc = normalize(p_val) == normalize(r_val)
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


##########################################################################################
#CUSTOMED CALLBACKS
class GenerationEarlyStoppingCallback(TrainerCallback):
    def __init__(self, metric_name: str, patience: int=3, verbose: bool=True):
        self.metric_name = metric_name
        self.patience = patience
        self.verbose = verbose
        self.best = -math.inf
        self.num_bad=0
    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        current = logs.get(self.metric_name)
        if current and current>self.best:
            self.best=current; self.num_bad=0
            control.should_save=True
        else:
            self.num_bad+=1
            if self.num_bad>=self.patience:
                control.should_early_stop=True; control.should_save=True
        return control

##########################################################################################
#CUSTOMED TRAINER WITH SEMANTIC LOSS
class MyTrainer(Trainer):
    def __init__(self, *args, sem_model=None, sem_loss_weight=0.05, label_smoothing=0.2, **kwargs):
        super().__init__(*args, **kwargs)
        #freeze semantic model
        self.sem_model = sem_model.eval()
        for p in self.sem_model.parameters():
            p.requires_grad = False
        self.sem_loss_weight = sem_loss_weight
        self.label_smoothing = label_smoothing
        self.loss_fct = CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=-100)
        self._tot_sum = 0.0
        self._count = 0
        self.hidden_to_sem = torch.nn.Linear(
            self.model.config.hidden_size,
            self.sem_model.get_sentence_embedding_dimension()
        ).to(self.model.device)

    def log(self, logs):
        logs = {k: v for k, v in logs.items() if k != "loss"}
        return super().log(logs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        inputs_ss = inputs.copy()
        #cross-entropy
        outputs = model(**inputs_ss, output_hidden_states=True)
        logits = outputs.logits
        labels = inputs['labels']
        logits_shift = logits[:, :-1, :].contiguous()
        labels_shift = labels[:, 1:].contiguous()
        #CE with label smoothing
        loss_ce = self.loss_fct(logits_shift.view(-1, logits_shift.size(-1)), labels_shift.view(-1))

        loss = loss_ce

        self._tot_sum += loss.detach().cpu().item()
        self._count += 1

        if self.state.global_step % self.args.logging_steps == 0 and self.state.global_step > 0:
            total = self._tot_sum / self._count

            self.log({
                "train/loss": total
            })

            self._tot_sum = 0.0
            self._count = 0

        #return for loss eval
        if return_outputs:
            outputs.loss_total = loss
            return loss, outputs
        else:
            return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kwargs):
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        total_loss, count = 0.0, 0

        for inputs in eval_dataloader:
            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                loss, outputs = self.compute_loss(self.model, inputs, return_outputs=True)
            total_loss += outputs.loss_total.item()
            count += 1

        avg_total = total_loss / count if count else 0.0

        results = {
            f"{metric_key_prefix}_loss": avg_total
        }
        self.log(results)

        #calculate metrics per category
        ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        preds, refs = [], []
        for example in ds:
            prompt = example['prompt']
            expected = example['output']
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4000).to(self.model.device)

            with torch.no_grad():
                out_ids = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=150,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    early_stopping=True,
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    repetition_penalty=1.2,
                )

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
df_train = pd.read_csv(prompt_train_file)
df_val = pd.read_csv(prompt_val_file)

# Select 5% of train and 20% of val for test
df_train_test, df_train_final = train_test_split(df_train, test_size=0.95, random_state=SEED)
df_val_test, df_val_final = train_test_split(df_val, test_size=0.80, random_state=SEED)
df_test = pd.concat([df_train_test, df_val_test]).reset_index(drop=True)

print(f"Training set size: {len(df_train_final)}, Validation set size: {len(df_val_final)}, Test set size: {len(df_test)}", flush=True)

all_cats = set()
for out in df_train_final["output"].fillna("").tolist():
    for line in out.splitlines():
        if ":" in line:
            all_cats.add(line.split(":", 1)[0].strip())
categories = sorted(all_cats)
model_base.config.categories = categories

metric_names = [f"eval_accuracy_{c.lower()}" for c in categories]

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
# FINAL TRAINING ON ALL DATA (TRAIN+VAL)

# new cut
train_dataset_final = Dataset.from_pandas(df_train_final)
val_dataset_final = Dataset.from_pandas(df_val_final)
tokenized_train_final = train_dataset_final.map(clean_output_text).map(tokenize_function)
tokenized_val_final = val_dataset_final.map(clean_output_text).map(tokenize_function)

# final_writer_training = SummaryWriter(os.path.join(tensorboard_log_dir, "final_training"))
# final_writer_training.add_scalar("final/train_size", len(df_train_final), 0)
# final_writer_training.add_scalar("final/val_size", len(df_val_final), 0)

final_peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=4,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=['q_proj', 'v_proj']
)

model_final = get_peft_model(model_base, final_peft_config)
model_final.resize_token_embeddings(len(tokenizer))
model_final.config.categories = categories

training_args_final = TrainingArguments(
    output_dir=train_model + "_final",
    evaluation_strategy="steps",
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy='steps',
    logging_strategy='steps',
    logging_steps=250,
    fp16=True,
    gradient_accumulation_steps=1,
    report_to=["tensorboard"],
    logging_dir=os.path.join(tensorboard_log_dir, "final_training"),
    load_best_model_at_end=True,
    eval_steps=None,
    metric_for_best_model="eval_accuracy_overall",
    greater_is_better=True
)

trainer_final = MyTrainer(
    model=model_final,
    args=training_args_final,
    train_dataset=tokenized_train_final,
    eval_dataset=tokenized_val_final,
    label_smoothing=0.2,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt", padding=True),
    callbacks=[
            GenerationEarlyStoppingCallback("eval_accuracy_overall", patience=2),
        ],
    sem_model=model_sem,
    compute_metrics=compute_metrics
)

final_writer_training = SummaryWriter(os.path.join(tensorboard_log_dir, "final_training"))
final_writer_training.add_scalar("train/size", len(df_train_final), 0)
final_writer_training.add_scalar("val/size",   len(df_val_final),   0)

init_val = trainer_final.evaluate(eval_dataset=tokenized_val_final, metric_key_prefix="eval")
print("Initial val metrics - original model (step=0) →", init_val)

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
            input_ids=input_encoded["input_ids"],
            attention_mask=input_encoded["attention_mask"],
            max_new_tokens=150,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            early_stopping=True,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.2,
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