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
args = parser.parse_args()

base_path = args.base_path
input_metadata_path = os.path.join(base_path, "study_info.txt")
raw_final_info_path = os.path.join(base_path, "final_llm_sample_analysis.csv")
output_dir = os.path.join(base_path, "INFO_STUDY_LLM")
model_path = os.path.join(base_path, "Mistral-7B-Instruct-v0.3-f16.gguf")
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
            na_columns = [headers[i] for i in na_indices if values[i] == "nan"]
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


def pre_filter_studies(filtered_studies):
    studies_to_analyze = {}
    studies_to_eliminate = {}
    info_bio_llm_dir = os.path.join(base_path, "INFO_BIO_LLM")

    for study, run_accessions in filtered_studies.items():
        all_run_entropies = []
        for run in run_accessions:
            run_file = os.path.join(info_bio_llm_dir, f"{run}_bio.txt")
            # print(run_file, flush=True)
            if os.path.exists(run_file):
                with open(run_file, "r") as rf:
                    for line in rf:
                        if "Entropy:" in line:
                            try:
                                entropy_value = float(line.split("Entropy:")[1].strip())
                                all_run_entropies.append(entropy_value)
                            except ValueError:
                                pass
            else:
                print(f"Error: file not found: {run_file}", flush=True)

        if all_run_entropies and all(ent < 1.5 for ent in all_run_entropies):
            studies_to_eliminate[study] = run_accessions
        else:
            studies_to_analyze[study] = run_accessions

    return studies_to_analyze, studies_to_eliminate


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
    # return Llama(model_path=model_path, n_ctx=n_ctx, n_gpu_layers=-1, use_mmap=True, n_threads=8, logits_all=True)
    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_gpu_layers=-1,
        use_mmap=True,
        n_threads=4,
        logits_all=True,
        n_batch=2000,
        n_ubatch=2000,
        offload_kqv=True,
        flash_attn=True,
    )


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

    # append the problematic line to the file
    with open(filepath, 'a') as file:
        file.write("\t".join(map(str, line)) + '\n')


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

    return "\n".join([f"{key}: {value}" for key, value in unique_answers.items()])


# prompt to llm metadata
def process_metadata_llm(filtered_studies, raw_data, study_metadata):
    for study_accession, run_accessions in filtered_studies.items():
        print(f"Processing Study Accession: {study_accession}", flush=True)
        na_columns = set()
        for run in run_accessions:
            if run in raw_data:
                na_columns.update(raw_data[run])

        if na_columns:
            fields_for_prompt = [col for col in na_columns if
                                 col not in ["Donor information", "UBERON code", "DOT code"]]

            instructions = []
            if "Tissue type" in fields_for_prompt:
                instructions.append(
                    "Tissue type – The tissue type from which the sample originates (e.g., liver, lung, brain). If not specified, deduce from context in the two last columns.")
            if "Cell line" in fields_for_prompt:
                instructions.append(
                    "Cell line – Specify the cell line, or state 'Primary tissue' if the sample is from a primary tissue and not a cell line.")
            if "Cell type" in fields_for_prompt:
                instructions.append(
                    "Cell type – The type of cell in the sample (e.g., neuron, fibroblast, CD8 T cell, CD4 T cell, monocyte NK cell, mast cell, melanocyte, dendritic cell, etc...). If not provided, deduce based on the tissue type and state the inference. Use thee Cell Ontology terms terminology.")
            if "UBERON term" in fields_for_prompt:
                instructions.append(
                    "UBERON term – Provide me the organ(s) concerned by this study, in the UBERON GTEX terminology for the tissue type (e.g., UBERON:000XXXX + name of the organ). If not specified, deduce from context, or search one related to the tissue.")
            if "DOT term" in fields_for_prompt:
                instructions.append(
                    "DOT term – Return the Disease Ontology term corresponding to the disease associated with the sample in the format DOID:XXXXX + Disease Name. If the sample is explicitly described as 'normal' or 'healthy', or something similar do not infer any disease. In this case, do not search for disease-related information in the context. If the sample is not explicitly labeled as 'normal' or 'healthy' or 'no disease' etc, infer the disease from the context only if it is directly related to the sample (e.g., sample title, description, or metadata fields directly describing the sample). In case of cancer, something adjacent means that it's healthy. Non-disease conditions (e.g., pregnancy, aging, lifestyle factors) should be placed in the Donor information output column instead of the Disease Ontology Term field. DO NOT JUST STATE 'DISEASE' without inferring the type of disease. If nothing says there is a disease or any problem, state 'normal'.")
            if "Treatment" in na_columns:
                instructions.append(
                    "Treatment - Determine from the context and the desease estimated with treatment could be possible for the pathology (eg: Nivolumab, Ipilimumab, vemurafenib, etc...). If no treatment avaible, try to find with your knowledge a path to create a new treatment or a gene to target for example."
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
                    "Phenotype - Based on the given context, determine if the phenotype classification is 'parental' (Refers to the original, untreated cell line or population, which has not been exposed to selective pressure (such as drug treatment). Typically represents the baseline phenotype.) or 'persistant' (Refers to cells or populations that have survived treatment and exhibit drug persistence or resistance, often through adaptive mechanisms rather than genetic mutations.)."
                )
            if "Library strategy" in na_columns:
                instructions.append(
                    "Library strategy - Get the sequencing strategy."
                )
            if "Library selection fixed" in na_columns:
                instructions.append(
                    "Library selection fixed - Based on the given context, determine the library selection fixed category by searching for specific keywords or synonyms that match one of the five strict categories: 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or 'other'. Assign 'polyA' if the context contains any of the following terms or similar meaning: 'PolyA', 'poly.A', 'oligo.dT', 'oligodT', 'truseq.mrna', 'truseq.stranded.mrna', 'truseq.standard.mrna', 'smarter.mRNA', 'stranded.mRNA'. Assign 'inverse rRNA' if the context mentions depletion of ribosomal RNA with any of these terms or similar meaning: 'ribominus', 'ribodep', 'ribozero', 'ribo.zero', 'riboerase', 'ribogone', 'ribocop', 'ribo-dep', 'ribo-mi', 'ribo minus', 'depleted ribosom', 'remove ribosom', 'TruSeq.Stranded.Total', 'TruSeq.Total', 'SMARTer.Stranded.Total', 'SMARTer.Total'. Assign 'hybrid selection' if the context refers to hybrid capture or exon selection using any of these terms or similar meaning: 'Hybrid.Selection', 'Exon.capture', 'Exome.capture', 'RNA.Exome', 'geoMX'. Assign 'small RNA' if the context refers to small RNA isolation with keywords such as 'TruSeq.Small', 'size.fraction' or similar meaning. Assign 'other' if none of the above terms are found. Return only the exact category name: 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or 'other', with no additional text."
                )
            if "Library source" in na_columns:
                instructions.append(
                    "Library source - Based on the given context, determine the library source category by searching for specific keywords that match one of the two strict categories: 'single-cell' or 'bulk'. Assign 'single-cell' if the context contains any of the following terms: 'TRANSCRIPTOMIC SINGLE CELL', 'chromium', '10x', 'single.cell' or similar meaning. Assign 'bulk' if none of the above terms are found. Return only the exact category name: 'single-cell' or 'bulk', with no additional text."
                )

            prompt = f"""
            Study accession: {study_accession}
            Metadata to analyze: {study_metadata[study_accession]}

            For each row in the metadata line (the first line contains the column names), extract and format the following information concisely. For each missing category, provide a single answer without redundancy. Each category **MUST** have one distinct and explicit answer, even if inferred. **Do not leave any category empty.** Do not repeat information already provided in previous categories. Remove redundant text.
            {chr(10).join(instructions)}

            If any information is missing in the metadat can't be inferred for previous instruction, specify 'nan'. Don't double the answer. I want only one answer per category.
            Strict output format (no additional text or special characters, no duplicated answers), ONLY print the answer. Do not elaborate.:
            Output in this form: Organ: [single unique answer]

            Respond with exactly one line. Do not elaborate. Only one word (or 3 max) is allowed after the "Category:".
            """ + chr(10).join([f"{col}: [single unique answer]" for col in fields_for_prompt]) + " RETURN ALL CATEGORIES. Here is the strict output: "

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

                # split answer to get each instruction
                response_text = response["choices"][0]["text"].strip()
                response_text = re.sub(r'(?<!^)(\d+\.\s*)', r'\n\1', response_text)
                response_text = re.sub(r'^(["\'])(.*?)(["\'])$', r'\2', response_text, flags=re.MULTILINE)
                response_lines = response_text.split("\n")
                response_lines = [re.sub(r'^\d+[\.\)\-]\s*', '', line) for line in response_lines]
                response_lines = [line.replace("*", "") for line in response_lines]
                print("response line", response_lines)
                entropy_dict = {}

                # entropie calculation for each instruction (using fields_for_prompt)
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
                    if not found and not is_run_accession_logged(study_accession, error_file_path):
                        print(f"No response line available for {instruction}. Skipping entropy calculation.")

                output_file = os.path.join(output_dir, f"{study_accession}_study.txt")
                print(output_file, flush=True)
                with open(output_file, "w") as f:
                    f.write(clean_duplicate_answers(response_text))
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
initial_n_ctx = 15000

if use_gpu and gpu_count > 0:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(gpu_count))
    print(f"Using GPU(s): {os.environ['CUDA_VISIBLE_DEVICES']}")

llm = get_llama_model(model_path, initial_n_ctx)
print(f"Model loaded with {gpu_to_use} GPU layers.")

# cuda profiling
# torch.backends.cudnn.benchmark = True

# read metadata
os.makedirs(output_dir, exist_ok=True)

raw_data = load_final_info(raw_final_info_path)
study_map, study_metadata = load_study_info(input_metadata_path)
filtered_studies = filter_studies_for_llm(study_map, raw_data)
excluded_studies = set(study_map.keys()) - set(filtered_studies.keys())
if len(excluded_studies) > 0:
    print("Warning: the following studies are excluded because study_accession already complete:", flush=True)
    for study in excluded_studies:
        print(study, flush=True)

studies_to_analyze, studies_to_eliminate = pre_filter_studies(filtered_studies)

print("Study accessions to analyze:", flush=True)
for study in studies_to_analyze.keys():
    print(study, flush=True)
print("Study accessions to eliminate (all run-level entropies < 1.5):", flush=True)
for study in studies_to_eliminate.keys():
    print(study, flush=True)

process_metadata_llm(studies_to_analyze, raw_data, study_metadata)

sys.stdout.close()

open(FLAG_FILE, 'w').close()

if llm is not None:
    try:
        llm.close()
    except Exception as e:
        print(f"Error closing model: {e}")
del llm
