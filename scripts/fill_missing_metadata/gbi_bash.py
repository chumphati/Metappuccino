##########################################################################################
# IMPORT
import os
import psutil
import subprocess
import sys
import torch

##########################################################################################
# PATHS
base_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results"
input_metadata_path = os.path.join(base_path, "LLM_METADATA_READY/test.txt")
raw_final_info_path = os.path.join(base_path, "RAW_FINAL_INFO.txt")
output_dir = os.path.join(base_path, "TEST_LLM_GPU")
model_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/models/Llama-3.1-Nemotron-70B-Instruct-HF-Q4_K_M.gguf"
log_file_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/llm_log_SB.txt"

##########################################################################################
# FUNCTIONS
sys.stdout = open(log_file_path, "a", buffering=1)
sys.stderr = sys.stdout

# check memory
def print_memory_usage(process):
    memory_info = process.memory_info()
    vm = psutil.virtual_memory()
    print(f"rss memory usage: {memory_info.rss / 1024 ** 2:.2f} mb", flush=True)
    print(f"peak virtual memory usage: {vm.used / 1024 ** 2:.2f} mb", flush=True)

# check if gpu available
use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count() if use_gpu else 0
if use_gpu:
    print(f"{gpu_count} gpus available. using: {[torch.cuda.get_device_name(i) for i in range(gpu_count)]}", flush=True)
else:
    print("no gpu detected. using cpu.", flush=True)

# llama.cpp inference
def run_llama_cpp(prompt, model_path, n_tokens):
    command = [
        "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/llama.cpp/llama-cli",
        "--model", model_path,
        "--tensor-split", "1,1", #each gpu charge equal
        "--split-mode", "layer", #divide layers model on each gpu
        "-n", str(n_tokens),
        "--prompt", prompt,
        "--threads", "30",
        "--log-file", log_file_path,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error while running llama.cpp: {e.stderr}", flush=True)
        return None

##########################################################################################
# MAIN
# ram
process = psutil.Process(os.getpid())

# read metadata
os.makedirs(output_dir, exist_ok=True)
with open(input_metadata_path, "r") as metadata_file:
    metadata_lines = metadata_file.readlines()[1:]

with open(raw_final_info_path, "r") as raw_file:
    raw_lines = raw_file.readlines()
    raw_headers = raw_lines[0].strip().split(",")
    raw_data = {line.split(",")[0]: line.strip().split(",") for line in raw_lines[1:]}

# process metadata
n_ctx_increment = 400
max_n_ctx = 5000
current_n_ctx = 700
for idx, line in enumerate(metadata_lines, start=62):
    clean_metadata = line.strip().split("\t")
    run_accession = clean_metadata[0]

    if run_accession in raw_data:
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
                    "Cell type – The type of cell in the sample (e.g., neuron). If not provided, deduce based on the tissue type and state the inference.")
            if "UBERON organ and code" in na_columns:
                instructions.append(
                    "UBERON organ code – The UBERON code and organ for the tissue type (e.g., UBERON:000XXXX + name of the organ). If not specified, deduce from context, or search one related to the tissue.")
            if "Disease Ontology Term" in na_columns:
                instructions.append(
                    "Disease Ontology Term – The Disease Ontology Term for the disease type + the name (e.g., DOID:XXXXX + term related to the code) with validation status (e.g., 'Validated' or 'Estimated'). Deduce from context if not specified.")

            prompt = f"""
            Run accession: {run_accession}
            Metadata to analyze: {clean_metadata}

            Attached is the metadata of a run from the NCBI SRA. The first line contains the column names. For each row in the metadata table, I would like the following concise information as a list:

            {chr(10).join(instructions)}
            If any information is missing in the metadata:

            Clearly state that it is 'Not specified.'
            Provide an informed estimate when possible (e.g., based on general knowledge or known standards of the platform).
            Specify when a detail requires further validation (e.g., from external sources like GTEx for UBERON codes).
            Return the result in a plain text format with one entry per row as follows (please specify all the tag fields below) Provide this format directly in the response for any metadata table shared in the future, in a txt file. Use LLM inference not python (write only the table as ouput, no additionnal sentences, one run only provided here):
            """ + chr(10).join([f"{col}: [value]" for col in na_columns if col != "Donor information"]) + " Here is the askep output :"

            print("PROMMPPPPTTT", flush=True)
            print(prompt, flush=True)

            # ram before answer
            print_memory_usage(process)

            try:
                response = run_llama_cpp(prompt, model_path, 180)
                print("ANNSSWWEERRR", flush=True)
                print(response, flush=True)
                output_file = os.path.join(output_dir, f"{run_accession}_bio.txt")
                with open(output_file, "w") as f:
                    f.write(response)

            except MemoryError:
                print(f"Memory error: line {idx}", flush=True)
                break

            # ram after answer
            print_memory_usage(process)

sys.stdout.close()
