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
input_metadata_path = os.path.join(base_path, "study_info.txt")
raw_final_info_path = os.path.join(base_path, "final_llm_sample_analysis.csv")
output_dir = os.path.join(base_path, "INFO_STUDY_LLM")
model_path = os.path.join(base_path, "Llama-3.1-Nemotron-70B-Instruct-HF-Q4_K_M.gguf")
log_file_path = os.path.join(base_path, "llm_log_study.txt")
error_file_path = os.path.join(base_path, "reload_model_study_info.txt")
error_file_header = "study_accession\trun_accession_list\tlibrary_construction_protocol\tstudy_metadata_ncbi"
FLAG_FILE = os.path.join(base_path, "STEP4_1.flag")

NA_COLUMNS = ["Tissue type", "Cell line", "Cell type", "UBERON code", "UBERON term", "DOT code", "DOT term"]

##########################################################################################
# FUNCTIONS
sys.stdout = open(log_file_path, "a")
sys.stderr = sys.stdout


def load_final_info(raw_final_info_path):
    """Charge final_llm_sample_analysis.csv and get runs with NA values."""
    with open(raw_final_info_path, "r") as raw_file:
        raw_lines = raw_file.readlines()
        headers = raw_lines[0].strip().split("\t")
        na_indices = [i for i, col in enumerate(headers) if col in NA_COLUMNS]
        raw_data = {}

        for line in raw_lines[1:]:
            values = line.strip().split("\t")
            run_accession = values[0]
            na_columns = [headers[i] for i in na_indices if values[i] == "NA"]
            if na_columns:
                raw_data[run_accession] = na_columns

    return raw_data


def load_study_info(input_metadata_path):
    """Charge study_info.txt and map the study_accessions with their run_accessions."""
    study_map = {}
    study_metadata = {}
    with open(input_metadata_path, "r") as metadata_file:
        for line in metadata_file.readlines():
            clean_metadata = line.strip().split(";")
            study_accession = clean_metadata[0]
            run_accessions = clean_metadata[1].split(",")
            study_map[study_accession] = run_accessions
            study_metadata[study_accession] = line.strip()

    return study_map, study_metadata


def filter_studies_for_llm(study_map, raw_data):
    """Filter study_accessions that have at least one run_accession with a NA column."""
    return {study: runs for study, runs in study_map.items() if any(run in raw_data for run in runs)}


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


def calculate_entropy_optimized(token_logprobs):
    logprobs_array = np.array(token_logprobs)
    max_logprob = np.max(logprobs_array)
    exp_logprobs = np.exp(logprobs_array - max_logprob)
    probabilities = exp_logprobs / np.sum(exp_logprobs)

    return -np.sum(probabilities * np.log(probabilities + 1e-10))


# prompt to llm metadata
def process_metadata_llm(filtered_studies, raw_data, study_metadata):
    for study_accession, run_accessions in filtered_studies.items():
        print(f"Processing Study Accession: {study_accession}", flush=True)

        na_columns = set()
        for run in run_accessions:
            if run in raw_data:
                na_columns.update(raw_data[run])

        if na_columns:
            instructions = []
            if "Tissue type" in na_columns:
                instructions.append("Tissue type – The tissue type from which the sample originates (e.g., liver, lung, brain).")
            if "Cell line" in na_columns:
                instructions.append("Cell line – Specify the cell line, or state 'Primary tissue' if the sample is from a primary tissue and not a cell line.")
            if "Cell type" in na_columns:
                instructions.append("Cell type – The list of type of cells in the study (e.g., neuron). If not provided, deduce based on the tissue type and state the inference.")
            if "UBERON term" in na_columns:
                instructions.append("UBERON organ code – The list of UBERON code and organ for the tissue types (e.g., UBERON:000XXXX + name of the organ). If not specified, deduce from context, or search one related to the tissue.")
            if "DOT term" in na_columns:
                instructions.append("Disease Ontology Term – The list of possible Disease Ontology Term for all the study (e.g., DOID:XXXXX + term related to the code) with validation status (e.g., 'Validated' or 'Estimated'). Deduce from context if not specified.")

            prompt = f"""
            Study accession: {study_accession}
            Metadata to analyze: {study_metadata[study_accession]}

            This metadata corresponds to a study from the NCBI SRA. The study-level information provides essential context for understanding the biological and experimental conditions. Based on the details available, I would like to determine the missing information globally for this study, even if specific values for individual runs are unavailable.
            Please infer the following key attributes based on the study metadata and general biological knowledge:

            {chr(10).join(instructions)}

            If any information is missing:
            - Provide an informed estimate when possible, based on common knowledge, study context, or platform-specific standards.
            - Indicate if further validation is required (e.g., using external databases such as GTEx for UBERON codes).

            Return the results in a structured plain text format with one entry per row as follows. Do not add any other sentence. Keep answers short. Ensure that all fields are explicitly stated, even if inferred from general study details:
            """ + chr(10).join(
                [f"{col}: [value]" for col in na_columns if col != "Donor information" and col != "UBERON code" and col != "DOT code"]) + " Here is the askep output :"

            print("PROMPT:", flush=True)
            print(prompt, flush=True)

            # ram before answer
            print_memory_usage(process)

            try:
                response = llm(prompt, max_tokens=180, logprobs=True)

                print("ANSWER:", flush=True)
                print(response["choices"][0]["text"], flush=True)
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

                output_file = os.path.join(output_dir, f"{study_accession}_study.txt")
                with open(output_file, "w") as f:
                    f.write(response["choices"][0]["text"])
                    f.write(f"\n")
                    for key, value in entropy_dict.items():
                        f.write(f"\n{key} Entropy: {value}")

            except ValueError as e:
                if "Requested tokens" in str(e):
                    print("Warning: context size too large, analysis postponed to wait for other model.")
                    write_reload_file(error_file_path, error_file_header, [study_metadata[study_accession]])

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
initial_n_ctx = 10000
llm = get_llama_model(model_path, initial_n_ctx)
print(f"Model loaded with {gpu_to_use} GPU layers.")

# cuda profiling
torch.backends.cudnn.benchmark = True

# read metadata
os.makedirs(output_dir, exist_ok=True)

raw_data = load_final_info(raw_final_info_path)
study_map, study_metadata = load_study_info(input_metadata_path)
filtered_studies = filter_studies_for_llm(study_map, raw_data)

# process metadata sequentially
process_metadata_llm(filtered_studies, raw_data, study_metadata)

sys.stdout.close()

# create flag end process before cleaning
open(FLAG_FILE, 'w').close()

if llm is not None:
    try:
        llm.close()
    except Exception as e:
        print(f"Error closing model: {e}")
del llm