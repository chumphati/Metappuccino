import csv
import re

tissue_type_dict = {
    "bone marrow": "connective",
    "cerebellum": "nervous",
    "pineal gland": "nervous",
    "thyroid": "epithelial",
    "neck": "connective",
    "placenta": "epithelial",
    "thymus": "lymphoid",
    "pulmonary artery": "connective",
    "mesothelium": "epithelial",
    "aorta": "connective",
    "plasma": "connective",
    "right parietal lobe": "nervous",
    "cell line": "epithelial",
    "pancreas": "epithelial",
    "sublingual gland": "epithelial",
    "t-lymphocytes": "lymphoid",
    "pituitary gland": "epithelial",
    "substantia nigra": "nervous",
    "pelvis": "connective",
    "heart": "muscular",
    "mammary gland/breast": "epithelial",
    "endometrium": "epithelial",
    "minor salivary gland": "epithelial",
    "hipsc-derive tissues in mouse kidney": "epithelial",
    "missing": "nan",
    "esophagus": "epithelial",
    "breast adenocarcinoma cell": "epithelial",
    "breast": "epithelial",
    "umbilical vein": "connective",
    "the non-target tumor": "nan",
    "the target tumor": "nan",
    "submandibular gland": "epithelial",
    "pdac": "epithelial",
    "lung": "epithelial",
    "testis": "epithelial",
    "adrenal gland": "epithelial",
    "cervix": "epithelial",
    "left ventricle tissue sample": "muscular",
    "cell culture": "nan",
    "mammary gland": "epithelial",
    "embryonic kideny": "epithelial",
    "esophageal squamous cell carcinomas": "epithelial",
    "lymphocytes": "lymphoid",
    "ureteric bud": "epithelial",
    "hela cell": "epithelial",
    "normal-density granulocytes": "lymphoid",
    "tongue": "epithelial",
    "superficial subcutaneous adipose tissues": "connective",
    "pancreas": "epithelial",
    "lesional skin": "epithelial",
    "biliary tract": "epithelial",
    "ascitic fluid": "connective",
    "esophageal": "epithelial",
    "breast cancer": "epithelial",
    "foreskin": "epithelial",
    "lung cancer": "epithelial",
    "kidney, cortex/proximal tubule": "epithelial",
    "liver cancer": "epithelial",
    "connective tissue": "connective",
    "hipsc": "epithelial",
    "pancreatic ductal adenocarcinoma": "epithelial",
    "low-density granulocytes": "lymphoid",
    "whole blood": "connective",
    "pbl": "lymphoid",
    "parotid gland": "epithelial",
    "stromal": "connective",
    "left temporal lobe": "nervous",
    "b cell from peripheral blood": "lymphoid",
    "pbmc": "lymphoid",
    "fibroblast": "connective",
    "omentum": "connective",
    "colon cancer": "epithelial",
    "brain (frontal cortex)": "nervous",
    "skin": "epithelial",
    "right temporal lobe": "nervous",
    "retina": "nervous",
    "liver": "epithelial",
    "glioblastoma": "nervous",
    "ascites": "connective",
    "cell from blood": "lymphoid",
    "bladder tumor assembloid": "epithelial",
    "pleural effusion": "connective",
    "dermis": "epithelial",
    "peripheral blood": "lymphoid",
    "brain: superior temporal gyrus": "nervous",
    "melanoma": "epithelial",
    "cultured cells": "epithelial",
    "myocardium of left ventricular": "muscular",
    "peripheral blood mononuclear cells": "lymphoid",
    "corneal endothelium": "epithelial",
    "the endothelium of blood vessels": "epithelial",
    "lymph node": "lymphoid",
    "b cells": "lymphoid",
    "cell": "epithelial",
    "liver": "epithelial",
    "missing: control sample": "nan",
    "primary tumor": "epithelial",
    "kidney": "epithelial",
    "lymphoid": "lymphoid",
    "human nasopharyngeal": "epithelial",
    "macrophage": "lymphoid",
    "gallbladder mucosa": "epithelial",
    "colon": "epithelial",
    "rna_ffpe": "nan",
    "stomach": "epithelial",
    "blood": "connective",
    "bladder": "epithelial",
    "nan": "nan",
    "muscle": "muscular",
    "connective": "connective",
    "nervous": "nervous",
    "epithelial": "epithelial"
}

input_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/finetune_clean_cl_data.csv'
output_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/finetune_clean_cl_tt_data.csv'


def normalize(text):
    return re.sub(r'\s+', ' ', text.strip().lower())


with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
        open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    fieldnames = reader.fieldnames
    writer = csv.DictWriter(outfile, fieldnames=fieldnames)
    writer.writeheader()

    for row in reader:
        output_text = row['output']
        match = re.search(r'tissue_type:\s*([^\n\r]+)', output_text, re.IGNORECASE)
        if match:
            tissue_value = normalize(match.group(1))
            standardized_type = tissue_type_dict.get(tissue_value)
            if standardized_type:
                output_text = re.sub(r'(tissue_type:\s*)[^\n\r]+', r'\1' + standardized_type, output_text,
                                     flags=re.IGNORECASE)
                row['output'] = output_text
        writer.writerow(row)
