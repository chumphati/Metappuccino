##########################################################################################
# IMPORT
import os
import psutil
from llama_cpp import Llama
import torch

##########################################################################################
# PATHS
base_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results"
error_file_path = os.path.join(base_path, "tmp/reload_model_bio_info.txt")
output_dir = os.path.join(base_path, "TEST_LLM_GPU")
model_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/models/Llama-3.1-Nemotron-70B-Instruct-HF-Q4_K_M.gguf"
log_file_path = os.path.join(base_path, "logs/reload_llm_log_SB.txt")

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


# load llama
def get_llama_model(model_path, n_ctx):
    """Load Llama model with the specified context size."""
    return Llama(model_path=model_path, n_ctx=n_ctx, use_mmap=True, tensor_split=[1, 1], n_threads=30)


# prompt llm
def process_metadata_line(line, llm, process, output_dir):
    """Process a single line of metadata."""
    clean_metadata = line.strip().split("\t")
    run_accession = clean_metadata[0]

    # Create prompt based on metadata
    prompt = f"""
    Run accession: {run_accession}
    Metadata to analyze: {clean_metadata}

    Attached is the metadata of a run from the NCBI SRA. The first line contains the column names. For each row in the metadata table, I would like the following concise information as a list:

    Clearly state that it is 'Not specified.'
    Provide an informed estimate when possible (e.g., based on general knowledge or known standards of the platform).
    Specify when a detail requires further validation (e.g., from external sources like GTEx for UBERON codes).
    Return the result in a plain text format with one entry per row. Use LLM inference not Python. Here is the output format:
    Tissue type: [value]
    Cell line: [value]
    Cell type: [value]
    UBERON organ and code: [value]
    Disease Ontology Term: [value]
    """

    print("PROMPT:", flush=True)
    print(prompt, flush=True)

    try:
        response = llm(prompt, max_tokens=180)
        print("ANSWER:", flush=True)
        print(response["choices"][0]["text"], flush=True)

        output_file = os.path.join(output_dir, f"{run_accession}_bio.txt")
        with open(output_file, "w") as f:
            f.write(response["choices"][0]["text"])

    except Exception as e:
        print(f"Error {run_accession}: {e}", flush=True)


##########################################################################################
# MAIN
process = psutil.Process(os.getpid())

# set initial context and maximum context
initial_n_ctx = 1200
max_n_ctx = 5000
n_ctx_increment = 500
current_n_ctx = initial_n_ctx

# Load model with initial context
llm = get_llama_model(model_path, current_n_ctx)

# read the temporary error file
if not os.path.exists(error_file_path):
    print("No error file found. Exiting.")
    sys.exit()

with open(error_file_path, "r") as error_file:
    lines = error_file.readlines()[1:]  #skip header line

# process each line in the error file
for line in lines:
    print_memory_usage(process)

    try:
        process_metadata_line(line, llm, process, output_dir)
    except ValueError as e:
        if "Requested tokens" in str(e):
            # increment context and reload model
            if current_n_ctx + n_ctx_increment <= max_n_ctx:
                current_n_ctx += n_ctx_increment
                llm = get_llama_model(model_path, current_n_ctx)
                print(f"Reloaded model with n_ctx={current_n_ctx}", flush=True)
            else:
                print(f"Maximum n_ctx reached. Skipping line.", flush=True)
                continue

sys.stdout.close()
