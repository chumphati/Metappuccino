import csv

metadata_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results_train/METADATA/cleaned_metadata_sra.txt'
input_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results_train/ena_results_formatted.csv'
output_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results_train/FINE_TUNING_COMPLETE/finetune_data.csv'

keywords = ['cell_type', 'tissue_type', 'cell_line']

def create_prompt(run_accession, metadata_line):
    prompt = f"""Run accession: {run_accession}
            Metadata to analyze: {metadata_line}

            For each row in the metadata line (the first line contains the column names), extract and format the following information concisely. For the 8 following categories, provide a single answer without redundancy. Each category **MUST** have one distinct and explicit answer, even if inferred. **Do not leave any category empty.** Do not repeat information already provided in previous categories. Remove redundant text.
            cell_type – The type of cell in the sample (e.g., neuron, fibroblast, CD8 T cell, CD4 T cell, monocyte NK cell, mast cell, melanocyte, dendritic cell, etc...). If not provided, deduce based on the tissue type and the rest of the context and state the inference. Use thee Cell Ontology terms terminology.
            tissue_type – The tissue type from which the sample originates (e.g., liver, lung, brain). If not specified, deduce from context in the two last columns.
            cell_line – Specify the cell line, or state 'Primary tissue' if the sample is from a primary tissue and not a cell line.
            
            If any information is missing in the metadat can't be inferred for previous instruction, specify 'nan'. Don't double the answer. I want only one answer per category.
            Strict output format (no additional text or special characters, no duplicated answers), ONLY print the answer. Do not elaborate.:
            Output in this form, with the category and prediction SEPARATED BY DOUBLE POINTS LIKE: Cetegory: [single unique answer]

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