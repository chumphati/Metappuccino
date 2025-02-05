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

##########################################################################################
# PATHS
parser = argparse.ArgumentParser(description="Process metadata with LLM")
parser.add_argument("--base_path", type=str, required=True, help="Base path to MetaMap")
args = parser.parse_args()

base_path = args.base_path
input_metadata_path = os.path.join(base_path, "sample_info.txt")
raw_final_info_path = os.path.join(base_path, "initial_raw_metadata.txt")
output_dir = os.path.join(base_path, "INFO_BIO_LLM")
model_path = os.path.join(base_path, "Llama-3.1-Nemotron-70B-Instruct-HF-Q4_K_M.gguf")
log_file_path = os.path.join(base_path, "llm_log_SB.txt")
error_file_path = os.path.join(base_path, "reload_model_bio_info.txt")
error_file_header = "run_accession\tsample_title\tsample_description\tdescription\tstudy_title"
FLAG_FILE = os.path.join(base_path, "STEP2_3.flag")

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
    return Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=-1, use_mmap=True, n_threads=30, logits_all=True)

def logits_to_probabilities(logits):
    max_logit = max(logits)
    exp_logits = [math.exp(logit - max_logit) for logit in logits]
    sum_exp_logits = sum(exp_logits)
    probabilities = [exp_logit / sum_exp_logits for exp_logit in exp_logits]
    return probabilities


def calculate_entropy(probabilities):
    return -sum(p * math.log(p) for p in probabilities if p > 0)


# write prompt that is out of context
def write_reload_file(filepath, header, line):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not os.path.exists(filepath):
        with open(filepath, 'w') as file:
            file.write(header + '\n')

    #append the problematic line to the file
    with open(filepath, 'a') as file:
        file.write("\t".join(map(str, line)) + '\n')


def calculate_entropy_optimized(token_logprobs):
    logprobs_array = np.array(token_logprobs)
    max_logprob = np.max(logprobs_array)
    exp_logprobs = np.exp(logprobs_array - max_logprob)
    probabilities = exp_logprobs / np.sum(exp_logprobs)

    return -np.sum(probabilities * np.log(probabilities + 1e-10))


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
                    instructions.append("Tissue type – The tissue type from which the sample originates (e.g., liver, lung, brain). If not specified, deduce from context in the two last columns.")
                if "Cell line" in na_columns:
                    instructions.append("Cell line – Specify the cell line, or state 'Primary tissue' if the sample is from a primary tissue and not a cell line.")
                if "Cell type" in na_columns:
                    instructions.append("Cell type – The type of cell in the sample (e.g., neuron). If not provided, deduce based on the tissue type and state the inference.")
                if "UBERON organ and code" in na_columns:
                    instructions.append("UBERON organ code – The UBERON code and organ for the tissue type (e.g., UBERON:000XXXX + name of the organ). If not specified, deduce from context, or search one related to the tissue.")
                if "Disease Ontology Term" in na_columns:
                    instructions.append("Disease Ontology Term – The Disease Ontology Term for the disease type + the name (e.g., DOID:XXXXX + term related to the code). Deduce from context if not specified.")

                prompt = f"""
                Run accession: {run_accession}
                Metadata to analyze: {clean_metadata}

                Attached is the metadata of a run from the NCBI SRA. The first line contains the column names. For each row in the metadata table, I would like the following concise information as a list:

                {chr(10).join(instructions)}
                If any information is missing in the metadata:

                Provide an informed estimate when possible (e.g., based on general knowledge or known standards of the platform).
                Return the result in a plain text format with one entry per row as follows (please specify all the tag fields below).
                Write short senteces, no additionnal sentences, no useless words (very important), like this):
                """ + chr(10).join(
                    [f"{col}: [value]" for col in na_columns if col != "Donor information"]) + " Here is the askep output :"

                print("PROMPT:", flush=True)
                print(prompt, flush=True)

                # ram before answer
                print_memory_usage(process)

                try:
                    response = llm(prompt, max_tokens=180, logprobs=True)

                    print("ANSWER:", flush=True)
                    print(response["choices"][0]["text"])
                    logprobs = response["choices"][0].get("logprobs", None)
                    token_logprobs = logprobs["token_logprobs"]

                    #split answer to get each instruction
                    response_lines = response["choices"][0]["text"].strip().split("\n")
                    entropy_dict = {}
                    token_index = 0

                    #entropie calculation for each instruction
                    for i, instruction in enumerate(na_columns):
                        if i < len(response_lines):
                            line = response_lines[i]
                            num_tokens = len(line.split())

                            #extract log-probabilités per instruction
                            logprobs_segment = token_logprobs[token_index: token_index + num_tokens]

                            #entropy
                            entropy = calculate_entropy_optimized(logprobs_segment)
                            entropy_dict[instruction] = entropy
                            token_index += num_tokens

                    output_file = os.path.join(output_dir, f"{run_accession}_bio.txt")
                    print(output_file, flush=True)
                    with open(output_file, "w") as f:
                        f.write(response["choices"][0]["text"])
                        f.write(f"\n")
                        for key, value in entropy_dict.items():
                            f.write(f"\n{key} Entropy: {value}")

                except ValueError as e:
                    if "Requested tokens" in str(e):
                        print("Warning: context size too large, analysis postponed to wait for other model.")
                        write_reload_file(error_file_path, error_file_header, clean_metadata)

                except MemoryError:
                    print(f"Memory error: line {idx}")
                    break

                except Exception as e:
                    print(f"Error in {run_accession} process: {e}", flush=True)
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
initial_n_ctx = 1200
llm = get_llama_model(model_path, initial_n_ctx)
print(f"Model loaded with {gpu_to_use} GPU layers.")

# cuda profiling
torch.backends.cudnn.benchmark = True

# read metadata
os.makedirs(output_dir, exist_ok=True)
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