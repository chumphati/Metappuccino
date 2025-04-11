import csv
import re


def get_value(row, key):
    return row.get(key, '').strip() or 'nan'


def create_prompt(run_accession, metadata):
    # prompt = f"""Run accession: {run_accession}
    #         Metadata to analyze: {metadata}
    #
    #         For each row in the metadata line (the first line contains the column names), extract and format the following information concisely. For each missing category, provide a single answer without redundancy. Each category **MUST** have one distinct and explicit answer, even if inferred. **Do not leave any category empty.** Do not repeat information already provided in previous categories. Remove redundant text.
    #         "Organ – Provide me the organ(s) concerned by this study, in the UBERON GTEX terminology for the tissue type (e.g., Lung, Liver, etc...). If not specified, deduce from context, or search one related to the tissue."
    #
    #         If any information is missing in the metadata and can't be inferred, specify 'nan'. Don't double the answer. I want only one answer per category.
    #
    #         Strict output format (no additional text or special characters, no duplicated answers), ONLY print the answer. Do not elaborate.:
    #         OUTPUT IN THIS FORM= Organ: [single unique answer]
    #
    #         Respond with exactly one line. Do not elaborate. Only one word (or 3 max) is allowed after "Organ:".
    #
    #         Here is the output:"""

    prompt = f"""Run accession: {run_accession}
            Metadata to analyze: {metadata}

            For each row in the metadata line (the first line contains the column names), extract and format the following information concisely. For each missing category, provide a single answer without redundancy. Each category **MUST** have one distinct and explicit answer, even if inferred. **Do not leave any category empty.** Do not repeat information already provided in previous categories. Remove redundant text.
            Tissue type – The tissue type from which the sample originates (e.g., liver, lung, brain). If not specified, deduce from context in the two last columns.
            Cell line – Specify the cell line, or state 'Primary tissue' if the sample is from a primary tissue and not a cell line.
            Cell type – The type of cell in the sample (e.g., neuron, fibroblast, CD8 T cell, CD4 T cell, monocyte NK cell, mast cell, melanocyte, dendritic cell, etc...). If not provided, deduce based on the tissue type and the rest of the context and state the inference. Use thee Cell Ontology terms terminology.
            Organ – Provide me the organ(s) concerned by this study, in the UBERON GTEX terminology for the tissue type (e.g., Lung, Liver, etc...). If not specified, deduce from context, or search one related to the tissue.
            Disease Ontology Term – Return the Disease Ontology term corresponding to the disease associated with the sample in the format DOID:XXXXX + Disease Name. If the sample is explicitly described as 'normal' or 'healthy', or something similar do not infer any disease. In this case, do not search for disease-related information in the context. If the sample is not explicitly labeled as 'normal' or 'healthy' or 'no disease' etc, infer the disease from the context only if it is directly related to the sample (e.g., sample title, description, or metadata fields directly describing the sample). In case of cancer, something adjacent means that it's healthy. Non-disease conditions (e.g., pregnancy, aging, lifestyle factors) should be placed in the Donor information output column instead of the Disease Ontology Term field. DO NOT JUST STATE 'DISEASE' without inferring the type of disease. If nothing says there is a disease or any problem, state 'normal'
            Phenotype - Based on the given context, determine if the phenotype classification is 'parental' (Refers to the original, untreated cell line or population, which has not been exposed to selective pressure (such as drug treatment). Typically represents the baseline phenotype.) or 'persistant' (Refers to cells or populations that have survived treatment and exhibit drug persistence or resistance, often through adaptive mechanisms rather than genetic mutations.). Choose between those two possibilities, use your knowledge if the answer is not clear."
            Library selection fixed - Based on the given context, determine the library selection fixed category by searching for specific keywords or synonyms that match one of the five strict categories: 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or 'other'. Assign 'polyA' if the context contains any of the following terms OR SIMIILAR MEANING THAT CAN BE INFERRED: 'PolyA', 'poly.A', 'oligo.dT', 'oligodT', 'truseq.mrna', 'truseq.stranded.mrna', 'truseq.standard.mrna', 'smarter.mRNA', 'stran ded.mRNA'. Assign 'inverse rRNA' if the context mentions depletion of ribosomal RNA with any of these terms OR SIMIILAR MEANING THAT CAN BE INFERRED: 'ribominus', 'ribodep', 'ribozero', 'ribo.zero', 'riboerase', 'ribogone', 'ribocop', 'ribo-dep', 'ribo-mi', 'ribo minus', 'depleted ribosom', 'remove ribosom', 'TruSeq.Stranded.Total', 'TruSeq.Total', 'SMARTer.Stranded.Total', 'SMARTer.Total'. Assign 'hybrid selection' if the context refers to hybrid capture or exon selection using any of these terms OR SIMIILAR MEANING THAT CAN BE INFERRED: 'Hybrid.Selection', 'Exon.capture', 'Exome.capture', 'RNA.Exome', 'geoMX'. Assign 'small RNA' if the context refers to small RNA isolation with keywords such as 'TruSeq.Small', 'size.fraction' OR SIMIILAR MEANING THAT CAN BE INFERRED. Assign 'other' if none of the above terms are found. Return only the exact category name: 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or 'other', with no additional text.
            Library source - Based on the given context, determine the library source category by searching for specific keywords that match one of the two strict categories: 'single-cell' or 'bulk'. Assign 'single-cell' if the context contains any of the following terms: 'TRANSCRIPTOMIC SINGLE CELL', 'chromium', '10x', 'single.cell' OR SIMIILAR MEANING THAT CAN BE INFERRED. Assign 'bulk' if none of the above terms are found. Return only the exact category name: 'single-cell' or 'bulk', with no additional text.

            If any information is missing in the metadat can't be inferred for previous instruction, specify 'nan'. Don't double the answer. I want only one answer per category.
            Strict output format (no additional text or special characters, no duplicated answers), ONLY print the answer. Do not elaborate.:
            Output in this form: Cetegory: [single unique answer]

            Respond with exactly one line. Do not elaborate. Only one word (or 3 max) is allowed after the "Category:".
            """
    return prompt


input_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/data/raw/annotated_totalRNA.csv'
output_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/finetune_data.csv'

metadata_columns = [
    "run_accession",
    "sample_title",
    "sample_description",
    "description",
    "study_title",
    "sample metadata ncbi",
    "age"
]

with open(input_file, 'r', encoding='ISO-8859-1') as infile, \
        open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile, delimiter=';')
    writer = csv.writer(outfile)

    writer.writerow(['prompt', 'output'])

    for row in reader:
        metadata_values = [get_value(row, col) for col in metadata_columns]
        metadata = ", ".join(metadata_values)
        prompt = create_prompt(get_value(row, "run_accession"), metadata)

        raw_output = get_value(row, "Gtex")
        output = f"Organ: {raw_output}"

        writer.writerow([prompt, output])