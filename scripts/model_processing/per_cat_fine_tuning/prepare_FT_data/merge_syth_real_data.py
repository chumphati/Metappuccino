import csv
import json

KEY_TO_KEEP = "cell_line"

IN_PATH = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/archive_models/test7/finetune_data_train_corrected.csv"
OUT_PATH = f"/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/DATA_PER_CAT/{KEY_TO_KEEP}/finetune_data_train_corrected__{KEY_TO_KEEP}.csv"

# IN_PATH = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/archive_models/test7/finetune_data_val_corrected.csv"
# OUT_PATH = f"/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/DATA_PER_CAT/{KEY_TO_KEEP}/finetune_data_val_corrected__{KEY_TO_KEEP}.csv"

# IN_PATH = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/archive_models/test7/finetune_data_test_oov_corrected.csv"
# OUT_PATH = f"/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/DATA_PER_CAT/{KEY_TO_KEEP}/finetune_data_test_corrected__{KEY_TO_KEEP}.csv"

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

STATIC_SUFFIX = """Categories and definitions:
- cell_line: exact cell line, ie standardized names of cultured/immortalized laboratory cell populations. Get the cell line code anywhere in the text

For each category below:
- Extract information from the summary if possible
- If one value is impossible to extract, even by deducing it, return "unknown"

BE CAREFUL: Sometimes the information concerns several samples from the same study. It is important to distinguish between them and semantically extract what applies to the current run, so everything must be consistent.
FOR EACH CATEGORY SEVERAL ANSWERS CAN BE POSSIBLE, CITE THEM ALL WITH A ',' OR ';' SEPARATOR

Respond strictly in a few words with valid JSON (double quotes around keys and values), no extra keys. ONLY THE CATEGORIES CITED UNDER 'Categories and definitions'

Here is the output:
"""

def extract_run_summary(text):
    lines = text.splitlines()
    run_line = ""
    summary_line = ""
    for l in lines:
        s = l.strip()
        if not run_line and s.lower().startswith("run accession:"):
            run_line = s
        if not summary_line and s.lower().startswith("summary:"):
            summary_line = s
        if run_line and summary_line:
            break
    if not run_line and lines:
        run_line = lines[0].strip()
    if not summary_line and len(lines) > 1:
        summary_line = lines[1].strip()
    return run_line, summary_line

def parse_json_field(s):
    s = s.strip()
    try:
        return json.loads(s)
    except:
        if s.startswith('"') and s.endswith('"'):
            s2 = s[1:-1].replace('""', '"')
            try:
                return json.loads(s2)
            except:
                return {}
        s2 = s.replace('""', '"')
        try:
            return json.loads(s2)
        except:
            return {}

with open(IN_PATH, "r", encoding="utf-8", newline="") as fin, open(OUT_PATH, "w", encoding="utf-8", newline="") as fout:
    r = csv.DictReader(fin)
    w = csv.DictWriter(fout, fieldnames=["prompt", "output"])
    w.writeheader()
    for row in r:
        original_prompt = row.get("prompt", "")
        original_output = row.get("output", "")
        run_line, summary_line = extract_run_summary(original_prompt)
        new_prompt = f"{run_line}\n{summary_line}\n\n{STATIC_SUFFIX}"
        data = parse_json_field(original_output)
        value = data.get(KEY_TO_KEEP, "")
        if isinstance(value, (dict, list)):
            value = ""
        w.writerow({"prompt": new_prompt, "output": value})
