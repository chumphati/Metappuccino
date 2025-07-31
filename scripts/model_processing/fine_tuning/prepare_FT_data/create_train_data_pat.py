##########################################################################################
#IMPORT
import pandas as pd
import random

##########################################################################################
#PATHS
train_input = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/train_metadata_replaced_table.csv'
val_input   = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/val_metadata_replaced_table.csv'
test_input   = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/test_metadata_replaced_table.csv'
train_output = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/finetune_data_train_corrected.csv'
val_output   = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/finetune_data_val_corrected.csv'
test_output = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/finetune_data_test_corrected.csv'

##########################################################################################
#MAIN
static_prompt = """Run accession: {run_accession}
Summary: {context}

Categories and definitions:
- library_selection: one of: 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or extract other rare value (exclude cDNA or similar that are previous steps before real library selection)
- sequencing_source: one of: 'spatial', 'bulk', 'single cell'. search for transcriptomics information in context
- biopsy_site: organ, body part or fluid WHERE TISSUE WAS SAMPLED
- biopsy_type: one of: 'primary', 'metastasis', 'blood'. If in situ, it's primary unless blood related information is mentioned.
- cell_line: exact cell line code
- cell_type: extract cell type: if known, specify it (e.g., 'T cell'); otherwise, write 'primary tissue'.
- organ: organ studied or affected (not where the sample is from, very different from biopsy_site)
- disease: report associated disease or 'healthy' status (be careful to specific vocabulary that could indicate that the sample is healthy, for eg. adjacent is something next to the disease, or normal, etc...)
- treatment: treatment applied (can be molecules, specific techniques, control, etc...)
- treatment_time: time or phase relative to treatment (qualitative or quantitative information)
- response: treatment response, state of the cell after treatment, without mention again the treatment any kind of event after treatment if applicable
- age: sample donor age. Can be quantitative (range or exact age) or qualitative (eg: child, teenage, adult, senior, ETC)
- sex: sample donor sex
- ethnicity: sample donor ethnicity (origins, genetics)
- localization: all geographical information available
- is_cancer: return 'True' if the disease is cancer related, 'False' otherwise

For each category below:
- Infer from the summary if possible
- The value can be not applicable ONLY FOR: treatment_time and response (if treatment = no treatment) AND cell_line (if cell_type = primary tissue), RETURN "not applicable" for those categories
- If one value is impossible to infer, return "unknown", applicable for all categories

Respond strictly in a few words with valid JSON (double quotes around keys and values), no extra keys.

Here is the output:
"""

def process_file(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    records = []
    for _, row in df.iterrows():
        run_acc = f"TRR{random.randint(100000,999999)}"
        prompt = static_prompt.format(run_accession=run_acc, context=row['phrase'])
        fields = [
            "library_selection", "sequencing_source", "biopsy_site", "biopsy_type",
            "cell_line", "cell_type", "organ", "disease", "treatment",
            "treatment_time", "response", "age", "sex", "ethnicity", "localization", "is_cancer"
        ]
        output = '\n'.join(f"{field}: {row[field]}" for field in fields)
        records.append({'prompt': prompt, 'output': output})
    pd.DataFrame(records).to_csv(output_csv, index=False)

if __name__ == "__main__":
    process_file(train_input, train_output)
    process_file(val_input, val_output)
    process_file(test_input, test_output)
