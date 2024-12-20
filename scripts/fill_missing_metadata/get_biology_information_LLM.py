##########################################################################################
#IMPORT
import os
import psutil
from llama_cpp import Llama
import torch
import sys

##########################################################################################
#PATHS
base_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results"
input_metadata_path = os.path.join(base_path, "LLM_METADATA_READY/sample_info.txt")
raw_final_info_path = os.path.join(base_path, "RAW_FINAL_INFO.txt")
output_dir = os.path.join(base_path, "INFO_BIO_LLM")
model_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/models/Llama-3.1-Nemotron-70B-Instruct-HF-Q4_K_M.gguf"
log_file_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/llm_log_SB.txt"

##########################################################################################
#FUNCTIONS
sys.stdout = open(log_file_path, "a")
sys.stderr = sys.stdout
#check memory
def print_memory_usage(process):
    memory_info = process.memory_info()
    vm = psutil.virtual_memory()
    print(f"rss memory usage: {memory_info.rss / 1024 ** 2:.2f} mb")
    print(f"peak virtual memory usage: {vm.used / 1024 ** 2:.2f} mb")

#check if gpu available
use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count() if use_gpu else 0
if use_gpu:
    print(f"{gpu_count} gpus available. using: {[torch.cuda.get_device_name(i) for i in range(gpu_count)]}")
else:
    print("no gpu detected. using cpu.")

#load llama
def get_llama_model(model_path, gpu_to_use):
    return Llama(model_path=model_path, n_ctx=780, use_mmap=True, n_gpu_layers=gpu_to_use)

##########################################################################################
#MAIN
#ram
process = psutil.Process(os.getpid())
#number max gpu = 2 (total noeud 49/51)
gpu_to_use = min(gpu_count, 80)
#model
llm = get_llama_model(model_path, gpu_to_use)
print(f"Model loaded with {gpu_to_use} GPU layers.")

#read metadata
os.makedirs(output_dir, exist_ok=True)
with open(input_metadata_path, "r") as metadata_file:
    metadata_lines = metadata_file.readlines()[1:]

with open(raw_final_info_path, "r") as raw_file:
    raw_lines = raw_file.readlines()
    raw_headers = raw_lines[0].strip().split(",")
    raw_data = {line.split(",")[0]: line.strip().split(",") for line in raw_lines[1:]}


#process metadata
for idx, line in enumerate(metadata_lines, start=1):
    clean_metadata = line.strip().split("\t")
    run_accession = clean_metadata[0]

    if run_accession in raw_data:
        raw_info = raw_data[run_accession]
        na_columns = [raw_headers[i] for i, value in enumerate(raw_info) if value == "NA"]

        if na_columns:
            instructions = []
            if "Tissue type" in na_columns:
                instructions.append("Tissue type: [value: The tissue type from which the sample originates (e.g., liver, lung, brain). If not specified, deduce from context.]")
            if "Cell line" in na_columns:
                instructions.append("Cell line: [value: Specify the cell line, or state 'Primary tissue' if the sample is from a primary tissue and not a cell line.]")
            if "Cell type" in na_columns:
                instructions.append("Cell type: [value: The type of cell in the sample (e.g., neuron). If not provided, deduce based on the tissue type and state the inference.]")
            if "UBERON organ and code" in na_columns:
                instructions.append("UBERON organ: [value: The UBERON code and organ for the tissue type (e.g., UBERON:000XXXX + name of the organ). If not specified, deduce from context in the two last columns, or search one related to the tissue.]")
            if "Disease Ontology Term" in na_columns:
                instructions.append("Disease Ontology Term: [value: normal or deduce disease type from context.]")

            prompt = f"""
            Run accession: {run_accession}
            Metadata to analyze: {clean_metadata}

            If any information is missing in the metadata for the following task: Clearly state that it is 'NA'.
            Return the result in a plain text format with one entry per row as follows (please specify all the tag fields below).
            Deduce missing values based on the metadata and general knowledge and format the output as:

            {chr(10).join(instructions)}

            Format the output as:
            """ + chr(10).join([f"{col}: [value]" for col in na_columns if col != "Donor information"])

            #ram before answer
            print_memory_usage(process)

            try:
                response = llm(prompt, max_tokens=100)
                output_file = os.path.join(output_dir, f"{run_accession}_llm.txt")
                with open(output_file, "w") as f:
                    f.write(response["choices"][0]["text"])

            except MemoryError:
                print(f"memory error: line {idx}")
                break

            #ram after answer
            print_memory_usage(process)


sys.stdout.close()
