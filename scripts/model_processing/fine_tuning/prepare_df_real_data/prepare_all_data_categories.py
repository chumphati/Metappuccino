import csv

metadata_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results_train/METADATA/cleaned_metadata_sra.txt'
input_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results_train/ena_results_formatted.csv'
output_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results_train/FINE_TUNING_COMPLETE/finetune_data.csv'

keywords = ['cell_type', 'host_body_site', 'tissue_type', 'cell_line', 'organ', 'disease', 'host_phenotype', 'library_selection', 'library_source', 'transcriptomic']

def create_prompt(run_accession, metadata_line):
    prompt = f"""Run accession: {run_accession}
            Metadata to analyze: {metadata_line}

            For each row in the metadata line (the first line contains the column names), extract and format the following information concisely. For the 8 following categories, provide a single answer without redundancy. Each category **MUST** have one distinct and explicit answer, even if inferred. **Do not leave any category empty.** Do not repeat information already provided in previous categories. Remove redundant text.
            cell_type – The type of cell in the sample (e.g., neuron, fibroblast, CD8 T cell, CD4 T cell, monocyte NK cell, mast cell, melanocyte, dendritic cell, etc...). If not provided, deduce based on the tissue type and the rest of the context and state the inference. Use thee Cell Ontology terms terminology.
            tissue_type – The tissue type from which the sample originates (e.g., liver, lung, brain). If not specified, deduce from context in the two last columns.
            cell_line – Specify the cell line, or state 'Primary tissue' if the sample is from a primary tissue and not a cell line.
            organ - Provide me the organ(s) concerned by this study, in the UBERON GTEX terminology for the tissue type (e.g., lungs, ... name of the organ). If not specified, deduce from context, or search one related to the tissue.
            disease – Return the Disease Ontology term corresponding to the disease associated with the sample in the format DOID:XXXXX + Disease Name. If the sample is explicitly described as 'normal' or 'healthy', or something similar do not infer any disease. In this case, do not search for disease-related information in the context. If the sample is not explicitly labeled as 'normal' or 'healthy' or 'no disease' etc, infer the disease from the context only if it is directly related to the sample (e.g., sample title, description, or metadata fields directly describing the sample). In case of cancer, something adjacent means that it's healthy. Non-disease conditions (e.g., pregnancy, aging, lifestyle factors) should be placed in the Donor information output column instead of the Disease Ontology Term field. DO NOT JUST STATE 'DISEASE' without inferring the type of disease. If nothing says there is a disease or any problem, state 'normal'.
            host_phenotype - Based on the given context, determine if the phenotype classification is 'parental' (Refers to the original, untreated cell line or population, which has not been exposed to selective pressure (such as drug treatment). Typically represents the baseline phenotype.) or 'persistant' (Refers to cells or populations that have survived treatment and exhibit drug persistence or resistance, often through adaptive mechanisms rather than genetic mutations.). Choose between those two possibilities, use your knowledge if the answer is not clear."
            library_selection - Based on the given context, determine the library selection fixed category by searching for specific keywords or synonyms that match one of the five strict categories: 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or 'other'. Assign 'polyA' if the context contains any of the following terms OR SIMIILAR MEANING THAT CAN BE INFERRED: 'PolyA', 'poly.A', 'oligo.dT', 'oligodT', 'truseq.mrna', 'truseq.stranded.mrna', 'truseq.standard.mrna', 'smarter.mRNA', 'stran ded.mRNA'. Assign 'inverse rRNA' if the context mentions depletion of ribosomal RNA with any of these terms OR SIMIILAR MEANING THAT CAN BE INFERRED: 'ribominus', 'ribodep', 'ribozero', 'ribo.zero', 'riboerase', 'ribogone', 'ribocop', 'ribo-dep', 'ribo-mi', 'ribo minus', 'depleted ribosom', 'remove ribosom', 'TruSeq.Stranded.Total', 'TruSeq.Total', 'SMARTer.Stranded.Total', 'SMARTer.Total'. Assign 'hybrid selection' if the context refers to hybrid capture or exon selection using any of these terms OR SIMIILAR MEANING THAT CAN BE INFERRED: 'Hybrid.Selection', 'Exon.capture', 'Exome.capture', 'RNA.Exome', 'geoMX'. Assign 'small RNA' if the context refers to small RNA isolation with keywords such as 'TruSeq.Small', 'size.fraction' OR SIMIILAR MEANING THAT CAN BE INFERRED. Assign 'other' if none of the above terms are found. Return only the exact category name: 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or 'other', with no additional text.
            library_source - Based on the given context, determine the library source category by searching for specific keywords that match one of the two strict categories: 'single-cell' or 'bulk'. Assign 'single-cell' if the context contains any of the following terms: 'TRANSCRIPTOMIC SINGLE CELL', 'chromium', '10x', 'single.cell' OR SIMIILAR MEANING THAT CAN BE INFERRED. Assign 'bulk' if none of the above terms are found. Return only the exact category name: 'single-cell' or 'bulk', with no additional text.

            If any information is missing in the metadat can't be inferred for previous instruction, specify 'nan'. Don't double the answer. I want only one answer per category.
            Strict output format (no additional text or special characters, no duplicated answers), ONLY print the answer. Do not elaborate.:
            Output in this form: Cetegory: [single unique answer]

            Respond with exactly one line. Do not elaborate. Only one word (or 3 max) is allowed after the "Category:".
            
            (one line per category, no repetition):
            Here is the output:
            """
    return prompt


metadata_dict = {}
with open(metadata_file, 'r', encoding='utf-8') as metafile:
    reader = csv.DictReader(metafile, delimiter='\t')
    selected_headers = [h for h in reader.fieldnames if any(kw in h.lower() for kw in keywords)]

    for row in reader:
        run = row.get("run_accession", "").strip()
        if run:
            filtered_values = [row[k].strip().replace('NA', 'nan') for k in selected_headers if row[k].strip()]
            metadata_line = ", ".join(filtered_values) if filtered_values else "nan"
            metadata_dict[run] = metadata_line

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
    writer.writerow(['prompt', 'output'])
    next(reader)

    for line in reader:
        if len(line) < 2:
            continue

        run_accession = line[0].strip()
        output_value = line[1].replace(';', '\n').strip()
        metadata_line = metadata_dict.get(run_accession, 'nan')
        prompt = create_prompt(run_accession, metadata_line)
        writer.writerow([prompt, output_value])