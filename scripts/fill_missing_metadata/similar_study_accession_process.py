##########################################################################################
# IMPORT
import os
import psutil
from llama_cpp import Llama
import torch
import sys
import argparse
import math

##########################################################################################
# PATHS
parser = argparse.ArgumentParser(description="Process metadata with LLM")
parser.add_argument("--base_path", type=str, required=True, help="Base path to MetaMap")
args = parser.parse_args()

base_path = args.base_path
input_metadata_path = os.path.join(base_path, "study_info.txt")
raw_final_info_path = os.path.join(base_path, "final_llm_sample_analysis.csv")
output_dir = os.path.join(base_path, "INFO_STUDY_LLM")
model_path = os.path.join(base_path, "Llama-3.1-Nemotron-70B-Instruct-HF-Q4_K_M.gguf")
log_file_path = os.path.join(base_path, "llm_log_study.txt")
error_file_path = os.path.join(base_path, "reload_model_study_info.txt")
error_file_header = "study_accession\trun_accession_list\tlibrary_construction_protocol\tstudy_metadata_ncbi"
FLAG_FILE = os.path.join(base_path, "STEP2_4.flag")

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


# prompt to llm metadata
def process_metadata_llm(metadata_lines, llm):
    for idx, line in enumerate(metadata_lines):
        clean_metadata = line.strip().split(";")
        study_accession = clean_metadata[0]

        if study_accession in raw_data:
            print(study_accession)
            raw_info = raw_data[study_accession]
            na_columns = [raw_headers[i] for i, value in enumerate(raw_info) if value == "NA"] #if entropy <2.5

            if na_columns:
                instructions = []
                if "Tissue type" in na_columns:
                    instructions.append("Tissue type – The tissue type from which the sample originates (e.g., liver, lung, brain).")
                if "Cell line" in na_columns:
                    instructions.append("Cell line – Specify the cell line, or state 'Primary tissue' if the sample is from a primary tissue and not a cell line.")
                if "Cell type" in na_columns:
                    instructions.append("Cell type – The type of cell in the sample (e.g., neuron). If not provided, deduce based on the tissue type and state the inference.")
                if "UBERON term" in na_columns:
                    instructions.append("UBERON organ code – The UBERON code and organ for the tissue type (e.g., UBERON:000XXXX + name of the organ). If not specified, deduce from context, or search one related to the tissue.")
                if "DOT term" in na_columns:
                    instructions.append("Disease Ontology Term – The Disease Ontology Term for the disease type + the name (e.g., DOID:XXXXX + term related to the code) with validation status (e.g., 'Validated' or 'Estimated'). Deduce from context if not specified.")

                prompt = f"""
                Run accession: {study_accession}
                Metadata to analyze: {clean_metadata}

                Attached is the metadata of a run from the NCBI SRA. The first line contains the column names. For each row in the metadata table, I would like the following concise information as a list:

                {chr(10).join(instructions)}
                If any information is missing in the metadata:

                Clearly state that it is 'Not specified.'
                Provide an informed estimate when possible (e.g., based on general knowledge or known standards of the platform).
                Specify when a detail requires further validation (e.g., from external sources like GTEx for UBERON codes).
                Return the result in a plain text format with one entry per row as follows (please specify all the tag fields below) Provide this format directly in the response for any metadata table shared in the future, in a txt file. Use LLM inference not python (write only the table as ouput, no additionnal sentences, one run only provided here):
                """ + chr(10).join(
                    [f"{col}: [value]" for col in na_columns if col != "Donor information"]) + " Here is the askep output :"

                print("PROMPT:", flush=True)
                print(prompt, flush=True)

                # ram before answer
                print_memory_usage(process)

                try:
                    response = llm(prompt, max_tokens=180, logprobs=True)

                    print("ANSWER:", flush=True)
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
                            words = line.split()
                            num_tokens = len(words)

                            #extract log-probabilités per instruction
                            logprobs_segment = token_logprobs[token_index: token_index + num_tokens]
                            probabilities_segment = [math.exp(lp) for lp in logprobs_segment]
                            sum_prob_segment = sum(probabilities_segment)
                            normalized_prob_segment = [p / sum_prob_segment for p in probabilities_segment]

                            #entropy
                            entropy = -sum(p * math.log(p) for p in normalized_prob_segment if p > 0)
                            entropy_dict[instruction] = entropy
                            token_index += num_tokens

                    output_file = os.path.join(output_dir, f"{study_accession}_study.txt")
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

                # ram after answer
                print_memory_usage(process)

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
    raw_headers = raw_lines[0].strip().split("\t")
    raw_data = {line.split("\t")[0]: line.strip().split("\t") for line in raw_lines[1:]}

# process metadata sequentially
process_metadata_llm(metadata_lines, llm)

sys.stdout.close()

# create flag end process before cleaning
open(FLAG_FILE, 'w').close()