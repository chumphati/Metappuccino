##########################################################################################
#IMPORT
import pandas as pd
import random

##########################################################################################
#PATHS
train_input = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/store/Metappuccino_store/results_ft_patron_semi/train_metadata_replaced_table.csv'
val_input   = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/store/Metappuccino_store/results_ft_patron_semi/val_metadata_replaced_table.csv'
train_output = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/store/Metappuccino_store/results_ft_patron_semi/train_prompt_output.csv'
val_output   = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/store/Metappuccino_store/results_ft_patron_semi/val_prompt_output.csv'

##########################################################################################
#MAIN
static_prompt = """Run accession: {run_accession}
Metadata to analyze: {context}

For each row in the metadata line (the first line contains the column names), extract and format the following information concisely. For the 8 following categories, provide a single answer without redundancy. Each category **MUST** have one distinct and explicit answer, even if inferred. **Do not leave any category empty.** Do not repeat information already provided in previous categories. Remove redundant text.
cell_type - The type of cell in the sample. It can be among connective cells, fat cells, specialized integrated cells, migratory cells, kidney cells, muscle cells, bone cells, cartilage cells, stomach cells, sex cells, lung cells, pancreatic cells, liver cells (= hepatic cells), intestinal cells (= enterocytes), nerve cells, blood cells (= blood elements). First cite one of those huge categories, and if possible specify more specifically into each categorie (example: resident cells (fibroblasts, fibrocytes, tendinocytes, keratocytes), migratory cells (lymphocytes, histiocytes, melanocytes, natural killer cells), fat cells (mesenchymal cells, white adipocytes, brown adipocytes), specialized integrated cells (neuroepithelial cells, myoepithelial cells, goblet cells), kidney cells (podocytes, distal tubular cells, proximal tubular cells), muscle cells (smooth muscle cells, cardiac muscle cells, skeletal muscle cells), bone cells (osteoblasts, osteoclasts, osteocytes), cartilage cells (chondroblasts, hypertrophic chondrocytes), connective tissue cells (type A synoviocytes, type B synoviocytes), stomach cells (chief cells, parietal cells), sex cells (spermatozoa, spermatocytes, oocytes, sertoli cells, leydig cells), lung cells (type 1 pneumocytes, type 2 pneumocytes), pancreatic cells (alpha cells, beta cells, delta cells), liver cells (hepatocytes, kupffer cells), intestinal cells (enterocytes), nerve cells (neurons, neuroblasts, astrocytes, oligodendrocytes, schwann cells), blood cells (erythrocytes, thrombocytes, leukocytes, monocytes, eosinophils, basophils, neutrophils)). If not provided, deduce based on the tissue type and the rest of the context and state the inference. Use the Cell Ontology terms terminology. WARNING: IF CELL LINE CAN'T BE DETERMINE AND IF IT IS A PRIMARY TISSUE THERE, mark this category as 'Primary tissue' too.
tissue_type – The tissue type from which the sample originates (e.g., epithelial tissue, connective tissue, muscle tissue, nervous tissue). If not specified, deduce from context, from the organ or the type of cell it comes from.
cell_line – Specify the cell line which is a strict code, or state 'Primary tissue' if the sample is from a primary tissue and not a cell line.
organ - Provide me the organ(s) concerned by this study, in the UBERON GTEX terminology for the tissue type (e.g., lungs, liver, heart, etc.). If not specified, deduce from context, or search one related to the tissue.
disease – Return the Disease Ontology term corresponding to the disease associated with the sample in the format of the disease name. If the sample is explicitly described as 'normal' or 'healthy', do not infer any disease. If no disease, state 'normal'. In case of cancer, something adjacent means that it's healthy. Non-disease conditions (e.g., pregnancy, aging, lifestyle factors) should be placed in the Donor information output column instead of the Disease Ontology Term field.
host_phenotype - Determine if 'parental' or 'persistent'.
library_selection - Determine 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or 'other'.
library_source - Determine 'single-cell' or 'bulk'.
treatment – If the sample received a treatment, specify it; otherwise 'no treatment'.
treatment_time – State the treatment time or 'no treatment'/'nan'.
response – State the response or 'nan'.
donor_information – Specify donor info or 'nan'.

If missing, specify 'nan'. Only one answer per category.
Strict output format:
Category: [single unique answer]

(one line per category, no repetition):
Here is the output:
"""

def process_file(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    records = []
    for _, row in df.iterrows():
        run_acc = f"TRR{random.randint(100000,999999)}"
        prompt = static_prompt.format(run_accession=run_acc, context=row['phrase_text'])
        fields = [
            'cell_type','tissue_type','organ','cell_line','disease',
            'treatment','treatment_time','response','host_phenotype',
            'library_selection','library_source','donor_information','instrument_platform'
        ]
        output = '\n'.join(f"{field}: {row[field]}" for field in fields)
        records.append({'prompt': prompt, 'output': output})
    pd.DataFrame(records).to_csv(output_csv, index=False)

if __name__ == "__main__":
    process_file(train_input, train_output)
    process_file(val_input, val_output)
