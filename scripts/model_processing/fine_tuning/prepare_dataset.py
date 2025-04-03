import csv
import re


def get_value(row, key):
    return row.get(key, '').strip() or 'nan'


def create_prompt(run_accession, metadata):
    prompt = f"""Run accession: {run_accession}
            Metadata to analyze: {metadata}

            For each row in the metadata line (the first line contains the column names), extract and format the following information concisely. For each missing category, provide a single answer without redundancy. Each category **MUST** have one distinct and explicit answer, even if inferred. **Do not leave any category empty.** Do not repeat information already provided in previous categories. Remove redundant text.
            "Organ – Provide me the organ(s) concerned by this study, in the UBERON GTEX terminology for the tissue type (e.g., Lung, Liver, etc...). If not specified, deduce from context, or search one related to the tissue."

            If any information is missing in the metadata and can't be inferred, specify 'nan'. Don't double the answer. I want only one answer per category.

            Strict output format (no additional text or special characters, no duplicated answers), ONLY print the answer. Do not elaborate.:
            OUTPUT IN THIS FORM= Organ: [single unique answer]

            Respond with exactly one line. Do not elaborate. Only one word (or 3 max) is allowed after "Organ:".

            Here is the output:"""
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
