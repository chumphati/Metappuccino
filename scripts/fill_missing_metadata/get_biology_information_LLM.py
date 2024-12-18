########################################################################################################################
# IMPORT LIB
import os
import psutil
from llama_cpp import Llama

########################################################################################################################
#PATHS
BASE_PATH = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results"
input_metadata_path = os.path.join(BASE_PATH, "CLEAN_METADATA_SRA.txt")
output_dir = os.path.join(BASE_PATH, "")
model_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/llama.cpp/models/Llama-3.3-70B-Instruct-Q5_K_M.gguf"

########################################################################################################################
# MONITOR MEMORY USAGE FUNCTION
def print_memory_usage(process):
    memory_info = process.memory_info()
    print(f"Memory usage: {memory_info.rss / 1024 ** 2:.2f} MB")
    print(f"Peak memory usage: {psutil.Process(os.getpid()).memory_info().peak_wset / 1024 ** 2:.2f} MB")

########################################################################################################################
#PROCESS LLM
#monitor ram
process = psutil.Process(os.getpid())
#charge model with number of token accepted
llm = Llama(model_path=model_path, n_ctx=7000)
#create outdir
os.makedirs(output_dir, exist_ok=True)
#full metadata
with open(input_metadata_path, "r") as metadata_file:
    lines = metadata_file.readlines()[1:]

#pour rach run (line), find missing information
for idx, line in enumerate(lines, start=1):
    prompt = f"""Header: run_accession    first_public    study_title    project_name    study_accession    sample_accession    sample_title    sample_description    library_name    library_selection    library_source    library_strategy    library_construction_protocol    library_layout    rna_integrity_num    instrument_platform    rt_prep_protocol    cell_line    cell_type    tissue_lib    tissue_type    host_phenotype    isolate    age    host_body_site    sampling_site    base_count    description    sample_metadata_ncbi    study_metadata_ncbi
            Metadata to analyse: {line.strip()}
            
            Attached is the metadata of a run from the NCBI SRA. The first line contains the column names. For each row in the metadata table, I would like the following concise information as a list:
            
            Run accession number - first column.
            Number of base pairs – Provide the number of the column base_count.
            Tissue type – The tissue type from which the sample originates (e.g., liver, lung, brain). If not specified, deduce from context in the two last columns.
            Cell line – Specify the cell line, or state 'Primary tissue' if the sample is from a primary tissue and not a cell line.
            Cell type – The type of cell in the sample (e.g., neuron). If not provided, deduce based on the tissue type and state the inference.
            UBERON organ code – The UBERON code and organ for the tissue type (e.g., UBERON:000XXXX + name of the organ). If not specified, deduce from context in the two last columns, or search one related to the tissue.
            Disease Ontology Term – The Disease Ontology Term for the disease type + the name (e.g., DOID:XXXXX + term related to the code) with validation status (e.g., 'Validated' or 'Estimated'). Deduce from context in the two last columns if not specified.
            Library strategy – Provide the library strategy used (e.g., polyA, ribodepleted).
            Instrument platform – Specify the sequencing platform (e.g., Illumina, ONT, PacBio).
            Donor information – All information of the donor, if available (e.g., age, gender, height, etc...).
            If any information is missing in the metadata:
            
            Clearly state that it is 'Not specified.'
            Provide an informed estimate when possible (e.g., based on general knowledge or known standards of the platform).
            Specify when a detail requires further validation (e.g., from external sources like GTEx for UBERON codes).
            Return the result in a plain text format with one entry per row as follows (please specify all the tag fields below) Provide this format directly in the response for any metadata table shared in the future, in a txt file. Use LLM inference not python (write only the table as output, no additional sentences, one run only provided here):
            
            Run accession number: [value]
            Number of base pairs: [value based on base_count]
            Tissue type: [value or 'Estimated: X based on context']
            Cell line: [value or 'Primary tissue']
            Cell type: [value or 'Estimate: X based on context']
            UBERON organ and code: [value for organ and code or 'Estimate: X based on context']
            Disease Ontology Term: [value]
            Library strategy: [value]
            Instrument platform: [value]
            Donor information: [value or 'Not specified']
            
            Here is the askep output:
            """

    #memory before inference
    print_memory_usage(process)

    try:
        response = llm(prompt, max_tokens=200)
        output_file = os.path.join(output_dir, f"resultat_{idx}.txt")
        with open(output_file, "w") as f:
            f.write(response["choices"][0]["text"])

    except MemoryError:
        print(f"Memory error: line {idx}")
        break

    #memory after inference
    print_memory_usage(process)

    #answer
    response = llm(prompt, max_tokens=200)
    output_file = os.path.join(output_dir, f"resultat_{idx}.txt")
    with open(output_file, "w") as f:
        f.write(response["choices"][0]["text"])
