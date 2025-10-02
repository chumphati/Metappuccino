import os
import pandas as pd

TRAIN_CSV = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/results/model_replace_final/out/train_metadata_replaced_table.completed.csv"
VAL_CSV   = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/results/model_replace_final/out/val_metadata_replaced_table.completed.csv"
OUT_DIR   = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/results/model_replace_final/data_ft"

CATEGORIES = [
    "library_selection","sequencing_source","biopsy_site","biopsy_type",
    "cell_line","cell_type","organ","disease","treatment","treatment_time",
    "response","age","sex","ethnicity","is_cancer"
]

DEFINITIONS = {
    "library_selection": "based on the given context, determine the library selection fixed category by searching for specific keywords or synonyms that match one of the five strict categories: 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or 'other'. Assign 'polyA' if the context contains any of the following terms OR SIMIILAR MEANING THAT CAN BE INFERRED: 'PolyA', 'poly.A', 'oligo.dT', 'oligodT', 'truseq.mrna', 'truseq.stranded.mrna', 'truseq.standard.mrna', 'smarter.mRNA', 'stran ded.mRNA'. Assign 'inverse rRNA' if the context mentions depletion of ribosomal RNA with any of these terms OR SIMIILAR MEANING THAT CAN BE INFERRED: 'ribominus', 'ribodep', 'ribozero', 'ribo.zero', 'riboerase', 'ribogone', 'ribocop', 'ribo-dep', 'ribo-mi', 'ribo minus', 'depleted ribosom', 'remove ribosom', 'TruSeq.Stranded.Total', 'TruSeq.Total', 'SMARTer.Stranded.Total', 'SMARTer.Total'. Assign 'hybrid selection' if the context refers to hybrid capture or exon selection using any of these terms OR SIMIILAR MEANING THAT CAN BE INFERRED: 'Hybrid.Selection', 'Exon.capture', 'Exome.capture', 'RNA.Exome', 'geoMX'. Assign 'small RNA' if the context refers to small RNA isolation with keywords such as 'TruSeq.Small', 'size.fraction' OR SIMIILAR MEANING THAT CAN BE INFERRED. Assign 'other' if none of the above terms are found. Return only the exact category name: 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or 'other', with no additional text.",
    "sequencing_source": "one of: 'spatial', 'bulk', 'single cell'. Spatial refers to transcriptomics that preserves in-tissue localization of molecules, mapping expression directly onto the tissue architecture. Bulk means sequencing a mixture of cells together, producing an aggregate average signal across the population. Single cell captures RNA from individual cells, yielding per-cell expression profiles and cellular heterogeneity.",
    "biopsy_site": "organ, body part or fluid WHERE TISSUE WAS SAMPLED. same as organ if not cancer. If it is a xenograft mention it here. Must not be related to the disease, just the tissue sampled. If possible DEDUCE IT FROM WHERE THE CELL LINE COMES FROM",
    "biopsy_type": "state 'metastasis' IF CANCER AND METASTASIS MENTIONNED, OR 'blood' if no metastasis and blood related information mentionned, OTHERWISE state 'primary'. DO NOT STATE METASTASIS OR BLOOD IF NOT EXPLICITELY IN THE CONTEXT. CAN ONLY STATE THOSE THREE INFORMATION, YOU SHOULD ALWAYS BE CAPABLE TO DETERMINE ONE OF THE 3 VALUES",
    "cell_line": "exact cell line, ie standardized names of cultured/immortalized laboratory cell populations. Specify the cell line, or state 'Primary tissue' if the sample is from a primary tissue and not a cell line.",
    "cell_type": "The type of cell in the sample (e.g., neuron, fibroblast, CD8 T cell, CD4 T cell, monocyte NK cell, mast cell, melanocyte, dendritic cell, etc...). If not provided, deduce based on context. otherwise, state 'primary tissue'",
    "organ": "organ studied or affected (not where the sample is from, very different from biopsy_site). Must not be related to the disease, just the tissue concerned.",
    "disease": "report associated disease (BE SPECIFIC) or 'healthy' status (be careful to specific vocabulary that could indicate that the sample is healthy, for eg. adjacent is something next to the disease, or normal, etc...)",
    "treatment": "treatment applied treatment applied = may be molecules or drug treatments or surgical operations, or something implanted, or events altering the state of the organism, etc. if not explicitly stated, infer from the context. DON'T STATE the disease, get info just from treatment. Also state a treatment if it's planed to be done, as a pre-treatment step.",
    "treatment_time": "time or phase relative to treatment. if you state a quantitative information state if it it post, pre or on treatment",
    "response": "type of reaction to the treatment, can be: no treatment, unknown, stable, progressive, success",
    "age": "sample donor age if human. if not possible to infer, state 'unknown'. Can be quantitative (range or exact age) or qualitative (eg: child, teenage, adult, senior, ETC). careful to find an age not just a random number",
    "sex": "sample donor sex if human. if not possible to infer, state 'unknown'",
    "ethnicity": "sample donor ethnicity if human (e.g. caucasian, black, asian). if not possible to infer, state 'unknown'",
    "is_cancer": "return 'True' if the disease is cancer related, 'False' otherwise"
}

STATIC_PROMPT = """Run accession: {run_accession}
Summary: {context}

Categories and definitions:
- {definition_cat}

For each category below:
- Extract information from the summary if possible
- If one value is impossible to extract, even by deducing it, return "unknown"

BE CAREFUL: Sometimes the information concerns several samples from the same study. It is important to distinguish between them and semantically extract what applies to the current run, so everything must be consistent.
FOR EACH CATEGORY SEVERAL ANSWERS CAN BE POSSIBLE, CITE THEM ALL WITH A ',' OR ';' SEPARATOR

Here is the output:
"""

def ensure_out_dir(path: str):
    os.makedirs(path, exist_ok=True)

def build_prompt(run_accession: str, context: str, category: str) -> str:
    definition_line = f"{category}: {DEFINITIONS[category]}"
    return STATIC_PROMPT.format(
        run_accession=str(run_accession) if run_accession is not None else "",
        context=str(context) if context is not None else "",
        definition_cat=definition_line
    )

def make_outputs_for_file(input_csv: str, split_name: str, out_dir: str):
    df = pd.read_csv(input_csv, dtype=str, keep_default_na=False)
    required_cols = {"run_accession", "phrase"} | set(CATEGORIES)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {input_csv}: {missing}")

    for cat in CATEGORIES:
        prompts = []
        outputs = []

        for _, row in df.iterrows():
            run_acc = row.get("run_accession", "")
            context = row.get("phrase", "")
            prompt_text = build_prompt(run_acc, context, cat)
            output_val = row.get(cat, "")
            prompts.append(prompt_text)
            outputs.append("" if output_val is None else str(output_val))

        out_df = pd.DataFrame({"prompt": prompts, "output": outputs})
        out_path = os.path.join(out_dir, f"{split_name}_{cat}.tsv")
        out_df.to_csv(out_path, sep="\t", index=False)

def main():
    ensure_out_dir(OUT_DIR)
    make_outputs_for_file(TRAIN_CSV, "train", OUT_DIR)
    make_outputs_for_file(VAL_CSV,   "val",   OUT_DIR)

if __name__ == "__main__":
    main()
