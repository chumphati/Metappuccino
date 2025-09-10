import os
import sys
import csv
import re

# definitions = {
#     "library_selection": "one of: 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or extract other rare value (exclude cDNA or similar that are previous steps before real library selection). IF not in those categories, state 'other'",
#     "sequencing_source": "one of: 'spatial', 'bulk', 'single cell'. search for transcriptomics information in context",
#     "biopsy_site": "organ, body part or fluid WHERE TISSUE WAS SAMPLED",
#     "biopsy_type": "state 'metastasis' IF CANCER AND METASTASIS MENTIONNED, OR 'blood' if no metastasis and blood related information mentionned, OTHERWISE state 'primary'. CAN ONLY STATE THOSE THREE INFORMATION, YOU SHOULD ALWAYS BE CAPABLE TO DETERMINE ONE OF THE 3 VALUES",
#     "cell_line": "exact cell line, ie standardized names of cultured/immortalized laboratory cell populations. Get the cell line code anywhere in the text",
#     "cell_type": "extract cell type: if known, specify it (e.g., 'T cell', 'fibroblast', etc). state specific cell type; otherwise, write 'primary tissue'. If the cell type is not directly available, TRY to deduce it from the organ before answering 'primary tissue'.",
#     "organ": "organ studied or affected (not where the sample is from, very different from biopsy_site)",
#     "disease": "report associated disease (BE SPECIFIC) or 'healthy' status (be careful to specific vocabulary that could indicate that the sample is healthy, for eg. adjacent is something next to the disease, or normal, etc...)",
#     "treatment": "treatment applied treatment applied = may be molecules or drug treatments or surgical operations, or something implanted, or events altering the state of the organism, etc. if not explicitly stated, infer from the context. DON'T STATE the disease, get info just from treatment",
#     "treatment_time": "time or phase relative to treatment (qualitative or quantitative information, but favour quantitative data). BE CAREFUL TO GET THE TIME RELATED TO THE DEDUCED TREATMENT(S)",
#     "response": "treatment response, state of the cell after treatment, without mention again the treatment any kind of event after treatment if applicable. if no clear statement, try to infer from context the stage of the disease after treatment if possible",
#     "age": "sample donor age. Can be quantitative (range or exact age) or qualitative (eg: child, teenage, adult, senior, ETC)",
#     "sex": "sample donor sex",
#     "ethnicity": "sample donor ethnicity (origins, genetics)",
#     "localization": "all geographical information available, if several list them all",
#     "is_cancer": "return 'True' if the disease is cancer related, 'False' otherwise"
# }

STATIC_PROMPT = """Run accession: {run_accession}
Summary: {context}

Categories and definitions:
- treatment_time: time or phase relative to treatment (qualitative or quantitative information, but favour quantitative data). BE CAREFUL TO GET THE TIME RELATED TO THE DEDUCED TREATMENT(S)

For each category below:
- Extract information from the summary if possible
- If one value is impossible to extract, even by deducing it, return "unknown"

BE CAREFUL: Sometimes the information concerns several samples from the same study. It is important to distinguish between them and semantically extract what applies to the current run, so everything must be consistent.
FOR EACH CATEGORY SEVERAL ANSWERS CAN BE POSSIBLE, CITE THEM ALL WITH A ',' OR ';' SEPARATOR

Respond strictly in a few words with valid JSON (double quotes around keys and values), no extra keys. ONLY THE CATEGORIES CITED UNDER 'Categories and definitions'

Here is the output:
"""

RUNS_TSV = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/DATA_PER_CAT/treatment_time/known_treatment_time.tsv"

def read_cellline_map(path):
    m = {}
    with open(path, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        cols = next(r, [])
        col_idx = {name: i for i, name in enumerate(cols)}
        ra_i = col_idx.get("run_accession", 0)
        cl_i = col_idx.get("value", len(cols) - 1)
        for row in r:
            if len(row) <= max(ra_i, cl_i):
                continue
            ra = row[ra_i].strip()
            cl = row[cl_i].strip()
            if cl.lower().startswith("cellline."):
                cl = cl[len("cellline."):]
            m[ra] = cl
    return m

def safe_prompt(s):
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", s)

def process_input_file(in_path, cell_map):
    base = os.path.splitext(os.path.basename(in_path))[0]
    out_path = os.path.join(os.path.dirname(in_path), base + "_prompt_output.tsv")
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8", newline="") as fout:
        r = csv.reader(fin, delimiter="\t")
        w = csv.writer(fout, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        cols = next(r, [])
        col_idx = {name: i for i, name in enumerate(cols)}
        ra_i = col_idx.get("run_accession", 0)
        su_i = col_idx.get("summary", 1)
        w.writerow(["prompt", "output"])
        for row in r:
            if not row or len(row) <= max(ra_i, su_i):
                continue
            run_accession = row[ra_i].strip()
            summary = "\t".join(row[su_i:]).strip()
            prompt = STATIC_PROMPT.format(run_accession=run_accession, context=summary)
            prompt = safe_prompt(prompt)
            output = cell_map.get(run_accession, "")
            w.writerow([prompt, output])

def main():
    inputs = sys.argv[1:]
    if not inputs:
        inputs = ["/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/DATA_PER_CAT/treatment_time/metadata_sra_summarized.txt"]
    cell_map = read_cellline_map(RUNS_TSV)
    for p in inputs:
        if os.path.exists(p):
            process_input_file(p, cell_map)

if __name__ == "__main__":
    main()
