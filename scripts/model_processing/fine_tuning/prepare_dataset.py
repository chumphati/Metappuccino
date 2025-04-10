import csv
import re

input_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/data/raw/annotated_totalRNA.csv'
output_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/finetune_data.csv'

# output_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/finetune_train.csv'
# test_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/finetune_test.csv'

# runs_to_compare = [
#     'SRR1424691', 'SRR19095764', 'SRR16013060', 'SRR22847384', 'SRR25387888', 'SRR4785829',
#     'ERR3549178', 'SRR7094291', 'SRR20695355', 'SRR18363185', 'SRR6937757', 'SRR18363171',
#     'SRR22966269', 'SRR9007553', 'SRR22925398', 'SRR7094282', 'SRR1424662', 'SRR15013557',
#     'SRR22283244', 'SRR20695347', 'SRR25387901', 'SRR24709842', 'SRR22283281', 'SRR7430742',
#     'SRR6937753', 'SRR13572929', 'SRR19256827', 'SRR22283291', 'SRR22966271', 'SRR18363179',
#     'SRR915768', 'SRR22532373', 'SRR25387886', 'SRR8518134', 'SRR4785835', 'SRR22925192',
#     'SRR19666460', 'SRR6937800', 'SRR25387924', 'SRR7167724', 'SRR8518355', 'SRR1424692',
#     'SRR4785815', 'SRR4785838', 'SRR23920437', 'SRR23630185', 'SRR13485950', 'SRR15013650',
#     'SRR15013485', 'SRR22301776', 'SRR15013571', 'ERR5320490', 'ERR1514452', 'SRR22283234',
#     'SRR22925347', 'SRR15013491', 'SRR25098099', 'SRR11547383', 'SRR22532386', 'SRR13518181',
#     'SRR5259639', 'SRR15013474', 'SRR20140291', 'SRR12817269', 'SRR7430738', 'SRR15013470',
#     'SRR1603664', 'ERR3549198', 'SRR13518178', 'SRR8518278', 'SRR16013057', 'ERR5285553',
#     'SRR11547421', 'ERR1883116', 'SRR7767519', 'SRR11049435', 'SRR24709855', 'SRR22283241',
#     'SRR15013559', 'SRR15013496', 'SRR22283249', 'SRR18363169', 'SRR25098111', 'SRR15013462',
#     'SRR8932009', 'SRR13485967', 'ERR5285538', 'SRR13572928', 'SRR1721309', 'SRR5591607',
#     'SRR9007567', 'SRR16013068', 'SRR15013513', 'SRR7430749', 'SRR1424687', 'SRR15013627',
#     'SRR15013562', 'SRR8518148', 'SRR3393497', 'SRR15013486', 'SRR22283222', 'SRR7012369',
#     'SRR7094294', 'SRR26436583', 'SRR14027942', 'SRR15013514', 'SRR20695353', 'SRR5259643',
#     'SRR15013677', 'SRR7094278', 'SRR3393521', 'SRR15013524', 'SRR3703021', 'SRR25387875',
#     'SRR16013087', 'SRR22925215', 'SRR4240761', 'SRR6937766', 'SRR22301777', 'SRR1721301',
#     'SRR15013575', 'SRR22532397', 'SRR22301788', 'SRR15013615', 'SRR14362387', 'SRR19243478',
#     'SRR26436585', 'SRR1424670', 'DRR326900', 'SRR16212321', 'SRR7430734', 'SRR14027938',
#     'SRR1603663', 'SRR1424672', 'SRR22925281', 'SRR9007537', 'SRR15013598', 'SRR23630255',
#     'SRR14362390', 'SRR15013606', 'SRR23630234', 'SRR19779072', 'SRR25387899', 'SRR22532395',
#     'SRR13485933', 'ERR1993159'
# ]


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
        open(output_file, 'w', newline='', encoding='utf-8') as outfile, \
        open(test_file, 'w', newline='', encoding='utf-8') as testfile:
    reader = csv.DictReader(infile, delimiter=';')
    writer = csv.writer(outfile)
    writer.writerow(['prompt', 'output'])
    test_writer = csv.writer(testfile)
    test_writer.writerow(['prompt', 'output'])

    for row in reader:
        run_id = get_value(row, "run_accession")
        metadata_values = [get_value(row, col) for col in metadata_columns]
        metadata = ", ".join(metadata_values)
        prompt = create_prompt(run_id, metadata)
        raw_output = get_value(row, "Gtex")
        output = f"Organ: {raw_output}"
        writer.writerow([prompt, output])
        # if run_id in runs_to_compare:
        #     test_writer.writerow([prompt, output])
        # else:
        #     writer.writerow([prompt, output])
