##########################################################################################
# IMPORT
import os
import psutil
from llama_cpp import Llama
import torch
import sys
import argparse
import math
import numpy as np
import re

##########################################################################################
# PATHS
parser = argparse.ArgumentParser(description="Process metadata with LLM")
parser.add_argument("--base_path", type=str, required=True, help="Base path to MetaMap")
parser.add_argument("--input_metadata_path", type=str, required=True, help="Path to input metadata file")
parser.add_argument("--error_file_path", type=str, required=True, help="Path to error log file")
parser.add_argument("--context_file_path", type=str, required=True, help="Path to context log file")
parser.add_argument("--log_file_path", type=str, required=True, help="Path to log file")
parser.add_argument("--flag_file", type=str, required=True, help="Path to flag file for process completion")
parser.add_argument("--initial_n_ctx", type=int, default=1200, help="Initial context size for Llama model")

args = parser.parse_args()

base_path = args.base_path
input_metadata_path = args.input_metadata_path
error_file_path = args.error_file_path
context_file_path = args.context_file_path
log_file_path = args.log_file_path
FLAG_FILE = args.flag_file
initial_n_ctx = args.initial_n_ctx

raw_final_info_path = os.path.join(base_path, "initial_raw_metadata.txt")
output_dir = os.path.join(base_path, "INFO_BIO_LLM")
model_path = os.path.join(base_path, "Mistral-7B-Instruct-v0.3-original.gguf")
error_file_header = "run_accession\tsample_title\tsample_description\tdescription\tstudy_title"

##########################################################################################
# FUNCTIONS
sys.stdout = open(log_file_path, "a")
sys.stderr = sys.stdout


# check memory
def print_memory_usage(process):
    memory_info = process.memory_info()
    vm = psutil.virtual_memory()
    print(f"rss memory usage: {memory_info.rss / 1024 ** 2:.2f} mb")
    print(f"peak virtual memory usage: {vm.used / 1024 ** 2:.2f} mb")


# check if gpu available
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Number of GPUs: {torch.cuda.device_count()}")
use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count() if use_gpu else 0
if use_gpu:
    print(f"{gpu_count} gpus available. using: {[torch.cuda.get_device_name(i) for i in range(gpu_count)]}")
else:
    print("no gpu detected. using cpu.")


# load llama
def get_llama_model(model_path, n_ctx):
    # tensor_split = [1, 1, 1, 1, 1, 1, 1, 1]
    # return Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=-1, use_mmap=True, n_threads=8, logits_all=True)
    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=-1,
        use_mmap=True,
        n_threads=4,
        logits_all=True,
        flash_attn=True,
    )


def logits_to_probabilities(logits):
    max_logit = max(logits)
    exp_logits = [math.exp(logit - max_logit) for logit in logits]
    sum_exp_logits = sum(exp_logits)
    probabilities = [exp_logit / sum_exp_logits for exp_logit in exp_logits]
    return probabilities


# write prompt that is out of context
def write_reload_file(filepath, header, line):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    line_str = ";".join(map(str, line))
    line_exists = False
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            existing_lines = set(l.strip() for l in file)
            line_exists = line_str in existing_lines
    if not os.path.exists(filepath):
        with open(filepath, 'w') as file:
            file.write(header + '\n')
    #append the problematic line to the file
    if not line_exists:
        with open(filepath, 'a') as file:
            file.write(line_str + '\n')


def is_run_accession_logged(run_accession, error_file_path):
    if os.path.exists(error_file_path):
        with open(error_file_path, 'r') as file:
            return any(line.startswith(run_accession + ";") for line in file)
    return False


def calculate_entropy_optimized(token_logprobs):
    logprobs_array = np.array(token_logprobs)
    max_logprob = np.max(logprobs_array)
    exp_logprobs = np.exp(logprobs_array - max_logprob)
    probabilities = exp_logprobs / np.sum(exp_logprobs)
    return -np.sum(probabilities * np.log(probabilities + 1e-10))


def clean_duplicate_answers(response_lines):
    unique_answers = {}
    cleaned_lines = []

    for line in response_lines.split("\n"):
        line = line.strip()
        match = re.match(r"^(.*?):\s*(.*)$", line)
        if match:
            category, value = match.groups()
            category = category.strip()
            value = value.strip()

            if category in unique_answers:
                if value.lower() not in unique_answers[category].lower():
                    unique_answers[category] += f", {value}"
            else:
                unique_answers[category] = value

    # return "\n".join([f"{key}: {value}" for key, value in unique_answers.items()])
    cleaned_text = "\n".join([f"{key}: {value}" for key, value in unique_answers.items()])
    return cleaned_text, unique_answers


total_processed = 0
skipped_entries = []


# prompt to llm metadata
def process_metadata_llm(metadata_lines, llm):
    global total_processed
    for idx, line in enumerate(metadata_lines):
        clean_metadata = line.strip().split(";")
        run_accession = clean_metadata[0]
        total_processed += 1
        print(f"🔄 Process on going {total_processed}/{len(metadata_lines)}: {run_accession}", flush=True)

        if run_accession in raw_data:
            print(run_accession)
            raw_info = raw_data[run_accession]
            na_columns = [raw_headers[i] for i, value in enumerate(raw_info) if value == "NA"]

            if na_columns:
                instructions = []
                if "Tissue type" in na_columns:
                    instructions.append(
                        "Tissue type – The tissue type from which the sample originates (e.g., liver, lung, brain). If not specified, deduce from context in the two last columns.")
                if "Cell line" in na_columns:
                    instructions.append(
                        "Cell line – Specify the cell line, or state 'Primary tissue' if the sample is from a primary tissue and not a cell line.")
                if "Cell type" in na_columns:
                    instructions.append(
                        "Cell type – The type of cell in the sample (e.g., neuron, fibroblast, CD8 T cell, CD4 T cell, monocyte NK cell, mast cell, melanocyte, dendritic cell, etc...). If not provided, deduce based on the tissue type and the rest of the context and state the inference. Use thee Cell Ontology terms terminology.")
                if "UBERON organ and code" in na_columns:
                    instructions.append(
                        "UBERON organ and code – Provide me the organ(s) concerned by this study, in the UBERON GTEX terminology for the tissue type (e.g., UBERON:000XXXX + name of the organ). If not specified, deduce from context, or search one related to the tissue.")
                if "Disease Ontology Term" in na_columns:
                    instructions.append(
                        "Disease Ontology Term – Return the Disease Ontology term corresponding to the disease associated with the sample in the format DOID:XXXXX + Disease Name. If the sample is explicitly described as 'normal' or 'healthy', or something similar do not infer any disease. In this case, do not search for disease-related information in the context. If the sample is not explicitly labeled as 'normal' or 'healthy' or 'no disease' etc, infer the disease from the context only if it is directly related to the sample (e.g., sample title, description, or metadata fields directly describing the sample). In case of cancer, something adjacent means that it's healthy. Non-disease conditions (e.g., pregnancy, aging, lifestyle factors) should be placed in the Donor information output column instead of the Disease Ontology Term field. DO NOT JUST STATE 'DISEASE' without inferring the type of disease. If nothing says there is a disease or any problem, state 'normal'.")
                if "Treatment" in na_columns:
                    instructions.append(
                        "Treatment - Determine from the context the treatment that could be used for the pathology identified. It can be the name of a medicamentation (eg: Doliprane, Nivolumab, Ipilimumab, vemurafenib, etc...) or a biological treatment technique (eg: gene therapy, siRNA, etc...). If no pathology identified, return 'no treatment'. If you don't find the treatment from context, don't try to inferrate ot and return 'no treatment'. If there is PBS or Phosphate Buffered Saline written, it means 'no treatment'."
                    )
                if "Treatment Time" in na_columns:
                    instructions.append(
                        "Treatment Time - Based on the given context, determine the treatment time category by searching in which state the treatment is on the given sample. Only two answer are possible: Assign 'Pre-treatment' if the context indicates that the sample or data was collected before the start of treatment. Or assign 'On-treatment' if the context suggests that the sample or data was collected while the patient was undergoing treatment. If no pathology identified, return 'no treatment'. If no clear indication is found or if the treatment is unknown, return 'nan'."
                    )
                if "Response" in na_columns:
                    instructions.append(
                        "Response - Search on the context, on protocols if any kind of resistance to the disease or the reverse is notified. Answer within those categories: 'Progressive Disease', 'Stable Disease', 'Recist criteria'. If no such information founded or can't be deducted from context, answer nan."
                    )
                if "Phenotype" in na_columns:
                    instructions.append(
                        "Phenotype - Based on the given context, determine if the phenotype classification is 'parental' (Refers to the original, untreated cell line or population, which has not been exposed to selective pressure (such as drug treatment). Typically represents the baseline phenotype.) or 'persistant' (Refers to cells or populations that have survived treatment and exhibit drug persistence or resistance, often through adaptive mechanisms rather than genetic mutations.). Choose between those two possibilities, use your knowledge if the answer is not clear."
                    )
                if "Library strategy" in na_columns:
                    instructions.append(
                        "Library strategy - Get the sequencing strategy."
                    )
                if "Library selection fixed" in na_columns:
                    instructions.append(
                        "Library selection fixed - Based on the given context, determine the library selection fixed category by searching for specific keywords or synonyms that match one of the five strict categories: 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or 'other'. Assign 'polyA' if the context contains any of the following terms OR SIMIILAR MEANING THAT CAN BE INFERRED: 'PolyA', 'poly.A', 'oligo.dT', 'oligodT', 'truseq.mrna', 'truseq.stranded.mrna', 'truseq.standard.mrna', 'smarter.mRNA', 'stran ded.mRNA'. Assign 'inverse rRNA' if the context mentions depletion of ribosomal RNA with any of these terms OR SIMIILAR MEANING THAT CAN BE INFERRED: 'ribominus', 'ribodep', 'ribozero', 'ribo.zero', 'riboerase', 'ribogone', 'ribocop', 'ribo-dep', 'ribo-mi', 'ribo minus', 'depleted ribosom', 'remove ribosom', 'TruSeq.Stranded.Total', 'TruSeq.Total', 'SMARTer.Stranded.Total', 'SMARTer.Total'. Assign 'hybrid selection' if the context refers to hybrid capture or exon selection using any of these terms OR SIMIILAR MEANING THAT CAN BE INFERRED: 'Hybrid.Selection', 'Exon.capture', 'Exome.capture', 'RNA.Exome', 'geoMX'. Assign 'small RNA' if the context refers to small RNA isolation with keywords such as 'TruSeq.Small', 'size.fraction' OR SIMIILAR MEANING THAT CAN BE INFERRED. Assign 'other' if none of the above terms are found. Return only the exact category name: 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or 'other', with no additional text."
                    )
                if "Library source" in na_columns:
                    instructions.append(
                        "Library source - Based on the given context, determine the library source category by searching for specific keywords that match one of the two strict categories: 'single-cell' or 'bulk'. Assign 'single-cell' if the context contains any of the following terms: 'TRANSCRIPTOMIC SINGLE CELL', 'chromium', '10x', 'single.cell' OR SIMIILAR MEANING THAT CAN BE INFERRED. Assign 'bulk' if none of the above terms are found. Return only the exact category name: 'single-cell' or 'bulk', with no additional text."
                    )
                if "Donor information" in na_columns:
                    instructions.append(
                        "Donor information - All information on the host that can be deduce of the context (eg., age, sex, blood analysis, any personnal information). It can also be protocol summary information,methods or information about how the sample has been obtained (eg. Patient-Derived Xenograf, control cells, etc...). If no information founded, just specify 'nan', no other sentence.")

                prompt = f"""
                Run accession: {run_accession}
                Metadata to analyze: {clean_metadata}

                For each row in the metadata line (the first line contains the column names), extract and format the following information concisely. For each missing category, provide a single answer without redundancy. Each category **MUST** have one distinct and explicit answer, even if inferred. **Do not leave any category empty.** Do not repeat information already provided in previous categories. Remove redundant text.
                {chr(10).join(instructions)}
                
                If any information is missing in the metadat can't be inferred for previous instruction, specify 'nan'. Don't double the answer. I want only one answer per category.
                Strict output format (no additional text or special characters, no duplicated answers), ONLY print the answer. Do not elaborate.:
                Output in this form: Organ: [single unique answer]
    
                Respond with exactly one line. Do not elaborate. Only one word (or 3 max) is allowed after the "Category:".
                """ + chr(10).join(
                    [f"{col}: [single unique answer]" for col in na_columns]) + " RETURN ALL CATEGORIES. Here is the strict output: "

                print("PROMPT:", flush=True)
                print(prompt, flush=True)

                # ram before answer
                print_memory_usage(process)

                try:
                    print("BEGIN:", flush=True)
                    response = llm(prompt, max_tokens=180, logprobs=True)
                    print(response)
                    print("ANSWER:", flush=True)
                    print(response["choices"][0]["text"])
                    logprobs = response["choices"][0].get("logprobs", None)
                    token_logprobs = logprobs["token_logprobs"]

                    # split answer to get each instruction
                    response_text = response["choices"][0]["text"].strip()
                    response_text = re.sub(r'(?<!^)(\d+\.\s*)', r'\n\1', response_text)
                    response_text = re.sub(r'^(["\'])(.*?)(["\'])$', r'\2', response_text, flags=re.MULTILINE)
                    response_lines = response_text.split("\n")
                    response_lines = [re.sub(r'^\d+[\.\)\-]\s*', '', line) for line in response_lines]
                    response_lines = [line.replace("*", "") for line in response_lines]
                    print("response line", response_lines)

                    entropy_dict = {}

                    # entropy calculation for each instruction
                    for instruction in na_columns:
                        found = False
                        for idx, line in enumerate(response_lines):
                            stripped_line = line.strip()
                            if stripped_line and stripped_line.startswith(instruction):
                                token_index = sum(len(l.split()) for l in response_lines[:idx])
                                num_tokens = len(stripped_line.split())
                                logprobs_segment = token_logprobs[token_index: token_index + num_tokens]
                                entropy = calculate_entropy_optimized(logprobs_segment)
                                print(f"Entropy for {instruction}: {entropy}")
                                entropy_dict[instruction] = entropy
                                found = True
                                break
                            else:
                                print(
                                    f"Warning: Skipping entropy calculation for {instruction} because the line does not start with the expected category.")
                        if not found and not is_run_accession_logged(run_accession, error_file_path):
                            print(f"Error: No response line available for {instruction}. Skipping entropy calculation.")
                            write_reload_file(error_file_path, error_file_header, clean_metadata)

                    cleaned_text, unique_answers = clean_duplicate_answers(response_text)
                    num_filled_categories = sum(1 for value in unique_answers.values() if value.strip() != "")
                    total_categories = len(na_columns)
                    filled_percentage = (num_filled_categories / total_categories) * 100
                    if filled_percentage < 30:
                        write_reload_file(error_file_path, error_file_header, clean_metadata)
                        print(f"Warning: Only {filled_percentage}% of categories filled for {run_accession}. Data written to reload file.")

                    output_file = os.path.join(output_dir, f"{run_accession}_bio.txt")
                    # print(output_file, flush=True)
                    with open(output_file, "w") as f:
                        f.write(cleaned_text)
                        f.write(f"\n")
                        for key, value in entropy_dict.items():
                            f.write(f"\n{key} Entropy: {value}")

                except ValueError as e:
                    if "Requested tokens" in str(e):
                        print("Warning: context size too large, analysis postponed to wait for other model.")
                        write_reload_file(context_file_path, error_file_header, clean_metadata)

                except MemoryError:
                    print(f"Memory error: line {idx}")
                    break

                except Exception as excep:
                    print(f"Error in {run_accession} process: {excep}", flush=True)
                    skipped_entries.append(run_accession)
                    continue

                # ram after answer
                print_memory_usage(process)

                print(f"✅ Process over: {total_processed}/{len(metadata_lines)} runs inferred.")
                print(f"❌ Ignored runs : {len(skipped_entries)}")


##########################################################################################
# MAIN


# params ram / gpu
process = psutil.Process(os.getpid())
# number max gpu = 2 (total noeud 49/51)
gpu_to_use = min(gpu_count, 2)

# model
if use_gpu and gpu_count > 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))
    print(f"Using GPU(s): {os.environ['CUDA_VISIBLE_DEVICES']}")

llm = get_llama_model(model_path, initial_n_ctx)
print(f"Model loaded with {gpu_to_use} GPU layers.")

# cuda profiling
# torch.backends.cudnn.benchmark = True

# read metadata
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
with open(input_metadata_path, "r") as metadata_file:
    metadata_lines = metadata_file.readlines()[1:]

with open(raw_final_info_path, "r") as raw_file:
    raw_lines = raw_file.readlines()
    raw_headers = raw_lines[0].strip().split("|")
    raw_data = {line.split("|")[0]: line.strip().split("|") for line in raw_lines[1:]}

# process metadata sequentially
process_metadata_llm(metadata_lines, llm)

sys.stdout.close()

# create flag end process before cleaning
open(FLAG_FILE, 'w').close()

if llm is not None:
    try:
        llm.close()
    except Exception as e:
        print(f"Error closing model: {e}")
del llm