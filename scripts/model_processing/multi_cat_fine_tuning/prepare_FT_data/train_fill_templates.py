import os
import json
import random
import pandas as pd
import re

DATA_PATH = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/data_templates_training/metadata_templates_train_withoutkeys.json'
EXPANDED_PATH = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/metadata_templates_train_expanded.json'
OUTPUT_CSV  = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/train_metadata_replaced_table.csv'

CATEGORIES = [
    "library_selection", "sequencing_source",
    "organ",
    "biopsy_site",
    "biopsy_type",
    "cell_line", "cell_type", "disease", "treatment",
    "treatment_time", "response", "age", "sex", "ethnicity", "localization", "is_cancer"
]


LIB_SEL = ["polyA","inverse rRNA","hybrid selection"]
SEQ_SRC = ["single-cell", "bulk", "spatial"]
BIOPSY_SITE = ["blood", "lung","heart","kidney","brain","stomach","intestine"]
BIOPSY_TYPE = ['primary', 'metastasis', 'blood']
CELL_LINE = ["primary tissue","HEK293","HeLa","HepG2","MCF7","A549","K562","U2OS","PC3","Jurkat","HCT116","SHSY5Y","C2C12","THP1","not applicable"]
CELL_TYPE = ["muscle cells","liver cells","blood cells","kidney cells","nerve cells","connective cells","fat cells","bone cells","cartilage cells","specialized integrated cells","fibroblasts","migratory cells","stomach cells","lung cells"]
ORGAN = ["liver","lung","heart","kidney","brain","muscle","spleen","pancreas","colon","stomach","intestine","skin","bone marrow","blood"]
DISEASE = ["lung cancer","hepatocellular carcinoma","breast cancer","leukemia","lymphoma","prostate cancer","pancreatic adenocarcinoma","glioblastoma","cirrhosis","chronic kidney disease","Crohn's disease","ulcerative colitis","myocardial infarction","diabetes mellitus"]
TREATMENT = ["no treatment","cisplatin","doxorubicin","paclitaxel","sorafenib","imatinib","erlotinib","tamoxifen","methotrexate","5-fluorouracil","amoxicillin","gentamicin","irradiation","dexamethasone"]
TREAT_TIME = ["no treatment","not applicable","pre treatment","on treatment","post treatment","relapse","unknown","24 hours","48 hours","1 week","3 months","1 year"]
RESPONSE = ["nan","resistance","not applicable","sensitivity","partial response","stable disease","progressive disease"]
AGE = ["34 years","62 years","47","born in 1982", "55 years", "19 years","44 years","67 years","41 years"]
SEX = ["male", "female"]
ETHNICITY = ["Asian", "Hispanic", "European", "African", "American"]
LOCALIZATION = ["North Africa", "Paris", "French", "English","Australia"]
IS_CANCER = ["true","false"]

SYNONYMS = {
    "muscle cells": ["myocytes", "contractile fibers", "myofibrils", "muscular cells", "myofibers"], "liver cells": ["hepatocytes", "hepatic parenchyma", "liver parenchyma cells", "liver tissue cells", "hepatic cells"], "blood cells": ["hematocytes", "corpuscles", "erythrocytes", "leukocytes", "circulating cells"], "kidney cells": ["renal cells", "nephron cells", "tubular cells", "glomerular cells", "kidney parenchymal cells"], "nerve cells": ["neurons", "neurocytes", "nerve fibers", "neural cells", "neuroblasts"], "connective cells": ["stromal cells", "mesenchymal cells", "support cells", "CT cells", "matrix cells"], "fat cells": ["adipocytes", "lipid cells", "adipose cells", "white adipose cells", "brown adipose cells"], "bone cells": ["osteoblasts", "osteoclasts", "osteocytes", "skeletal cells", "bone-forming cells"], "cartilage cells": ["chondrocytes", "chondroblasts", "hypertrophic chondrocytes", "cartilage-forming cells", "chondrogenic cells"], "specialized integrated cells": ["neuroepithelial cells", "myoepithelial cells", "goblet cells", "Merkel cells", "Paneth cells"], "fibroblasts": ["fibrocytes", "tendinocytes", "keratocytes", "stromal fibroblasts", "fibroblastic cells"], "migratory cells": ["lymphocytes", "histiocytes", "melanocytes", "natural killer cells", "leukocyte migrants"], "stomach cells": ["chief cells", "parietal cells", "mucous neck cells", "G cells", "enteroendocrine cells"], "lung cells": ["type 1 pneumocytes", "type 2 pneumocytes", "alveolar epithelial cells", "bronchial epithelial cells", "Clara cells"],
    "epithelial tissue": ["epithelium", "epithelial layer", "lining tissue", "surface tissue", "barrier tissue"], "connective tissue": ["stromal tissue", "supportive tissue", "interstitial tissue", "mesenchymal tissue", "CT"], "muscle tissue": ["muscular tissue", "myogenic tissue", "contractile tissue", "muscle fibers", "myotissue"], "nervous tissue": ["neural tissue", "neuronal tissue", "gray matter", "white matter", "neuroglial tissue"], "blood tissue": ["vascular tissue", "circulatory tissue", "hemal tissue", "blood elements", "blood corpuscles"], "lymphatic tissue": ["lymphoid tissue", "immune tissue", "lymphatic network", "lymphoid organ", "lymphatic organ tissue"], "adipose tissue": ["fatty tissue", "lipid tissue", "adipose layer", "white adipose tissue", "brown adipose tissue"], "cartilage tissue": ["cartilaginous tissue", "hyaline tissue", "elastic cartilage", "fibrocartilage tissue", "cartilage matrix"], "bone tissue": ["osseous tissue", "osseous matrix", "skeletal tissue", "bone matrix", "osseum"], "glandular tissue": ["secretory tissue", "gland epithelium", "endocrine tissue", "exocrine tissue", "glandular epithelium"], "fibrous tissue": ["fibrous connective tissue", "dense connective tissue", "fibrotic tissue", "fibroelastic tissue", "fibrotissue"], "hematopoietic tissue": ["blood-forming tissue", "hemopoietic tissue", "myeloid tissue", "lymphoid tissue", "bone marrow tissue"], "smooth muscle tissue": ["visceral muscle", "involuntary muscle", "non-striated muscle", "smooth fibers", "visceral tissue"], "cardiac muscle tissue": ["myocardial tissue", "heart muscle", "cardiac fibers", "myocardium", "cardiomyocytes"],
    "primary tissue": ["primary cells", "ex vivo tissue", "native tissue", "primary sample", "direct tissue"], "HEK293": ["293 cells", "HEK-293", "human embryonic kidney 293", "HEK 293T", "293T variant"], "HeLa": ["HeLa S3", "HeLa CCL-2", "Henrietta Lacks cells", "HeLa Kyoto", "HeLa derivative"], "HepG2": ["hepatocellular carcinoma cell line", "Hep G2", "HepG-2", "liver carcinoma cells", "HepG2/C3A"], "MCF7": ["Michigan Cancer Foundation-7", "MCF-7", "breast adenocarcinoma cells", "MCF7/LCC1", "MCF7-TAMR"], "A549": ["A-549", "adenocarcinomic human alveolar basal epithelial cells", "lung carcinoma line", "A549/DDP", "A549-shRNA"], "K562": ["K-562", "chronic myelogenous leukemia cells", "leukemia line", "K562-Luc", "K562/ADR"], "U2OS": ["U-2 OS", "osteosarcoma cells", "U2OSp53", "U2OS Flp-In", "U2OS-GFP"], "PC3": ["PC-3", "prostate adenocarcinoma cells", "PC3-MM2", "PC3-PSMA", "PC3/NZW"], "Jurkat": ["Jurkat E6-1", "T lymphocyte cells", "acute T cell leukemia line", "Jurkat clone E6", "Jurkat/Jurkat"], "HCT116": ["HCT-116", "colorectal carcinoma cells", "HCT116 p53+", "HCT116-Dp53", "HCT116 p21-/-"], "SHSY5Y": ["SH-SY5Y", "neuroblastoma cells", "SHSY5Y-BRN2", "SHSY5Y/RA", "SHSY5Y-derived"], "C2C12": ["C2 C12", "mouse myoblasts", "C2C12 myotubes", "C2C12 MB", "C2C12-GFP"], "THP1": ["THP-1", "monocytic leukemia cells", "THP1-derived macrophages", "THP1-shRNA", "THP1-DM"],
    "liver": ["hepatic organ", "hepatic tissue", "liver parenchyma", "hepatic lobes", "hepatobiliary organ"], "lung": ["pulmonary organ", "respiratory organ", "pulmonary lobes", "pulmo", "bronchial organ"], "heart": ["cardiac organ", "cardial organ", "myocardium", "cardiac muscle", "heart chamber"], "kidney": ["renal organ", "renal cortex", "nephron organ", "kidney parenchyma", "renal lobule"], "brain": ["encephalon", "cerebrum", "cerebral organ", "central nervous system", "brain tissue"], "muscle": ["skeletal muscle", "muscle mass", "myotissue", "muscular system", "contractile organ"], "spleen": ["splenic organ", "splenic tissue", "lien", "splenic parenchyma", "splenic pulp"], "pancreas": ["pancreatic organ", "pancreatic gland", "exocrine gland", "endocrine gland", "pancreatic tissue"], "colon": ["large intestine", "colonic organ", "colonic tissue", "colon mucosa", "intestinal segment"], "stomach": ["gastric organ", "gastric sac", "gastric tissue", "stomach mucosa", "gastrointestinal organ"], "intestine": ["intestinal tract", "gut", "bowel", "enteric organ", "intestinal tube"], "skin": ["cutaneous tissue", "dermis", "epidermis", "skin layer", "integument"], "bone marrow": ["medullary cavity", "myeloid tissue", "hematopoietic marrow", "bone medulla", "marrow tissue"], "blood": ["circulating fluid", "bloodstream", "hemal fluid", "vascular fluid", "blood plasma"],
    "lung cancer": ["pulmonary carcinoma", "bronchogenic carcinoma", "lung neoplasm", "pulmonary malignancy", "airway cancer"],
    "hepatocellular carcinoma": ["HCC", "hepatoma", "liver cancer", "malignant hepatoma", "hepatocellular neoplasm"],
    "breast cancer": ["mammary carcinoma", "ductal carcinoma", "breast neoplasm", "mammary gland cancer", "breast malignancy"],
    "leukemia": ["blood cancer", "hematologic malignancy", "leukemic disease", "leukaemia", "leukemia neoplasm"],
    "lymphoma": ["lymphatic cancer", "lymphosarcoma", "lymphoma neoplasm", "lymphatic malignancy", "malignant lymphoma"],
    "prostate cancer": ["prostatic carcinoma", "prostate adenocarcinoma", "prostatic neoplasm", "prostate malignancy", "adenocarcinoma of prostate"],
    "pancreatic adenocarcinoma": ["pancreatic cancer", "pancreatic ductal carcinoma", "PDAC", "exocrine pancreatic carcinoma", "pancreatic neoplasm"],
    "glioblastoma": ["glioblastoma multiforme", "GBM", "astrocytoma grade IV", "malignant glioma", "astrocytic glioma"],
    "cirrhosis": ["liver cirrhosis", "hepatic cirrhosis", "cirrhotic liver", "chronic liver disease", "cirrhotic disorder"],
    "chronic kidney disease": ["CKD", "chronic renal failure", "chronic kidney failure", "renal disease", "chronic nephropathy"],
    "Crohn's disease": ["regional enteritis", "granulomatous colitis", "Crohn’s", "ileitis", "Crohn disease"],
    "ulcerative colitis": ["UC", "colitis ulcerosa", "idiopathic ulcerative colitis", "inflammatory bowel disease", "colonic ulceration"],
    "myocardial infarction": ["heart attack", "MI", "acute myocardial infarction", "cardiac infarction", "coronary thrombosis"],
    "diabetes mellitus": ["diabetes", "DM", "metabolic disorder", "hyperglycemia", "endocrine disorder"],
    "parental": ["wild type", "WT", "baseline", "native", "untreated lineage"],
    "persistent": ["persister", "tolerant", "drug-persistent", "survivor cells", "resistant population"],
    "polyA": ["poly-A selected", "oligo-dT capture", "mRNA enrichment", "polyA+", "polyadenylation"],
    "inverse rRNA": ["ribozero", "rRNA depletion", "ribominus", "riboerase", "riboCOP"],
    "hybrid selection": ["hybrid capture", "exon capture", "RNA exome", "SeqCap", "capture probe"],
    "no treatment": ["untreated", "vehicle", "control", "mock", "NA"],
    "cisplatin": ["CDDP", "cis-Pt", "cisplatinum", "platin", "cis diamminedichloroplatinum"],
    "doxorubicin": ["adriamycin", "DOX", "anthracycline", "rubidomycin", "doxorubicine"],
    "paclitaxel": ["taxol", "PTX", "paclitaxel-lipid", "paclitax", "paclitaxel prodrug"],
    "sorafenib": ["BAY 43-9006", "Nexavar", "sorafenib tosylate", "sorafenib free base", "sorafen"],
    "imatinib": ["Gleevec", "STI571", "imatinib mesylate", "CGP57148B", "imatinib free base"],
    "erlotinib": ["Tarceva", "OSI-774", "erlotinib hydrochloride", "CP-358774", "erlotinib free base"],
    "tamoxifen": ["Nolvadex", "TAM", "tamoxifen citrate", "ICI 46,474", "tamoxifene"],
    "methotrexate": ["MTX", "amethopterin", "mexate", "methotrexatum", "methotrexate hydrate"],
    "5-fluorouracil": ["5-FU", "Adrucil", "Flurocil", "5-Fluoro-2'-deoxyuridine", "fluorouracil"],
    "amoxicillin": ["Amoxil", "Trimox", "amoxicillin trihydrate", "Moxatag", "amox"],
    "gentamicin": ["Garamycin", "gentamycin", "gentamicin sulfate", "GEN", "gentavi"],
    "irradiation": ["radiation", "ionizing radiation", "gamma rays", "X-ray", "rad-exposure"],
    "dexamethasone": ["Decadron", "DXM", "dexamethasone sodium phosphate", "dexone", "maxidex"],
    "pre treatment": ["pre-treatment", "baseline", "before treatment", "pretreatment", "pre therapy"],
    "on treatment": ["during treatment", "treatment-phase", "under treatment", "on therapy", "treatment ongoing"],
    "post treatment": ["post-treatment", "after treatment", "treatment ended", "post therapy", "after therapy"],
    "relapse": ["recurrence", "disease recurrence", "relapsing", "return", "re-emergence"],
    "unknown": ["undefined", "unspecified", "not defined", "NA", "missing"],
    "24 hours": ["24h", "one day", "day 1", "24 hrs", "t=24h"],
    "48 hours": ["48h", "two days", "day 2", "48 hrs", "t=48h"],
    "1 week": ["7 days", "one week", "week 1", "w01", "7d"],
    "3 months": ["3mo", "quarterly", "3-month", "90 days", "three months"],
    "1 year": ["12 months", "annual", "one year", "yearly", "365 days"],
    "nan": ["NA", "none", "not available", "unknown", "missing"],
    "resistance": ["resistant", "drug-resistant", "non-responsive", "refractory", "tolerant"], "sensitivity": ["sensitive", "drug-sensitive", "responsive", "susceptible", "high response"], "partial response": ["PR", "partial remission", "subpartial response", "decrease", "responding partially"], "stable disease": ["SD", "disease stabilization", "no progression", "stable condition", "unchanged"], "progressive disease": ["PD", "disease progression", "worsening", "advancing disease", "progression"],
    "male": ["M", "♂", "man", "adult male", "male donor"],
    "34 years": ["34 y.o.", "age 34", "34-year-old", "34yrs", "34yo"],
    "female": ["F", "♀", "woman", "adult female", "female donor"],
    "62 years": ["62 y.o.", "age 62", "62-year-old", "62yrs", "62yo"],
    "47": ["47 y.o.", "age 47", "47-year-old", "47yrs", "47yo"],
    "North Africa": ["Maghreb", "North-African region", "Northern Africa", "NA region", "North African"],
    "born in 1982": ["b. 1982", "birthyear 1982", "1982-born", "1982 cohort", "1982"],
    "Paris": ["Île-de-France", "city of Paris", "Parisian", "Paris, France", "Paris region"],
    "Asian": ["Asiatic", "East Asian", "South Asian", "Asian descent", "Asian ethnicity"],
    "Hispanic": ["Latino", "Latina", "Hispanic ethnicity", "Hispanic origin", "Latinx"],
    "European": ["Eurasian", "EU origin", "European descent", "Caucasian (Europe)", "Europe resident"],
    "vegan": ["plant-based diet", "vegan diet", "no animal products", "strictly vegan", "vegan lifestyle"],
    "African": ["Afro-descendant", "African origin", "Sub-Saharan African", "African ethnicity", "African continent"],
    "American": ["USA resident", "US citizen", "North American", "American origin", "American ethnicity"],
    "55 years": ["55 y.o.", "age 55", "55-year-old", "55yrs", "55yo"],
    "French, hypertensive": ["France origin, high blood pressure", "French donor with hypertension", "hypertensive French", "French & hypertensive", "hypertension, French"],
    "English": ["UK origin", "British", "England resident", "English ethnicity", "Anglo"],
    "44 years": ["44 y.o.", "age 44", "44-year-old", "44yrs", "44yo"],
    "67 years": ["67 y.o.", "age 67", "67-year-old", "67yrs", "67yo"],
    "Australia": ["Australian", "AUS", "Down Under", "Australia resident", "Oceania"],
    "41 years": ["41 y.o.", "age 41", "41-year-old", "41yrs", "41yo"],
    "Illumina NovaSeq 6000": ["NovaSeq6000", "Illumina NS6000", "NS6000", "NovaSeq", "Illumina NovaSeq"],
    "Illumina HiSeq 4000": ["HiSeq4000", "Illumina HS4000", "HS4000", "HiSeq", "Illumina HiSeq"],
    "NextSeq 500": ["NextSeq500", "NS500", "Illumina NextSeq", "NextSeq-500", "Next Seq 500"],
    "NextSeq 2000": ["NextSeq2000", "NS2000", "Illumina NextSeq 2000", "Next Seq 2000", "NextSeq-2000"],
    "MiSeq": ["MiSeq System", "Illumina MiSeq", "Mi-Seq", "MiSeq Sequencer", "MiSeq platform"],
    "primary": ["primary", "Primary", "PRIMARY", "native lesion"],
    "metastasis": ["metastasis", "Metastasis", "MET", "secondary tumor"],
    "blood": ["blood", "Blood sample", "peripheral blood", "venous blood"],
    "spatial": ["spatial", "spatial‑seq", "spatial RNA‑seq", "Visium"]
}

SEM = {
    "cell_type": {
        "organ": {
            "muscle cells": ["muscle", "heart"],
            "myocytes": ["muscle", "heart"],
            "liver cells": ["liver"],
            "hepatocytes": ["liver"],
            "blood cells": ["blood", "bone marrow"],
            "erythrocytes": ["blood", "bone marrow"],
            "bone cells": ["bone marrow", "bone"],
            "osteoblasts": ["bone"],
            "cartilage cells": ["cartilage"],
            "chondrocytes": ["cartilage"],
            "nerve cells": ["brain"],
            "neurons": ["brain"],
            "fat cells": ["skin"],
            "adipocytes": ["skin"],
            "fibroblasts": ["skin", "muscle", "connective tissue"],
            "connective cells": ["skin", "muscle", "connective tissue"],
            "lung cells": ["lung"],
            "alveolar epithelial cells": ["lung"],
            "stomach cells": ["stomach"],
            "chief cells": ["stomach"],
            "migratory cells": ["blood", "spleen"],
            "lymphocytes": ["spleen", "blood"],
            "specialized integrated cells": ["pancreas", "colon", "lung", "glandular tissue"]
        }
    },
    "disease": {
        "organ": {
            "lung cancer": ["lung"],
            "pulmonary carcinoma": ["lung"],
            "bronchogenic carcinoma": ["lung"],
            "hepatocellular carcinoma": ["liver"],
            "hepatoma": ["liver"],
            "liver cancer": ["liver"],
            "breast cancer": ["skin"],
            "mammary carcinoma": ["skin"],
            "glioblastoma": ["brain"],
            "glioblastoma multiforme": ["brain"],
            "astrocytoma grade IV": ["brain"],
            "cirrhosis": ["liver"],
            "liver cirrhosis": ["liver"],
            "chronic kidney disease": ["kidney"],
            "CKD": ["kidney"],
            "Crohn's disease": ["intestine", "colon"],
            "regional enteritis": ["intestine", "colon"],
            "ulcerative colitis": ["colon", "intestine"],
            "colitis ulcerosa": ["colon"],
            "myocardial infarction": ["heart"],
            "heart attack": ["heart"],
            "diabetes mellitus": ["pancreas"],
            "DM": ["pancreas"],
            "pancreatic adenocarcinoma": ["pancreas"],
            "PDAC": ["pancreas"],
            "prostate cancer": ["kidney"],
            "prostatic carcinoma": ["kidney"],
            "lymphoma": ["spleen"],
            "lymphosarcoma": ["spleen"],
            "leukemia": ["blood", "bone marrow"],
            "blood cancer": ["blood", "bone marrow"]
        }
    },
    "cell_line": {
        "organ": {
            "HepG2": ["liver"],
            "A549": ["lung"],
            "MCF7": ["skin"],
            "U2OS": ["bone marrow"],
            "Jurkat": ["blood", "bone marrow"],
            "K562": ["blood", "bone marrow"],
            "SHSY5Y": ["brain"],
            "C2C12": ["muscle"],
            "PC3": ["kidney"],
            "HCT116": ["colon"],
            "HeLa": ["skin"],
            "THP1": ["blood", "bone marrow"],
            "HEK293": ["kidney"],
            "primary tissue": ["liver", "lung", "heart", "kidney", "brain", "muscle", "spleen", "pancreas", "colon", "stomach", "intestine", "skin", "bone marrow", "blood"]
        }
    },
    "treatment": {
        "disease": {
            "cisplatin": [
                "lung cancer", "hepatocellular carcinoma", "breast cancer", "prostate cancer",
                "pancreatic adenocarcinoma", "glioblastoma", "leukemia", "lymphoma"
            ],
            "doxorubicin": [
                "breast cancer", "leukemia", "lymphoma", "glioblastoma", "sarcoma"
            ],
            "paclitaxel": [
                "lung cancer", "breast cancer", "ovarian cancer", "prostate cancer"
            ],
            "sorafenib": [
                "hepatocellular carcinoma", "renal cancer", "thyroid cancer"
            ],
            "imatinib": [
                "leukemia", "chronic kidney disease"
            ],
            "erlotinib": [
                "lung cancer", "pancreatic adenocarcinoma"
            ],
            "tamoxifen": [
                "breast cancer"
            ],
            "methotrexate": [
                "leukemia", "lymphoma", "breast cancer"
            ],
            "5-fluorouracil": [
                "colorectal cancer", "breast cancer", "pancreatic adenocarcinoma"
            ],
            "amoxicillin": [
                "none", "ulcerative colitis", "Crohn's disease"
            ],
            "gentamicin": [
                "none", "infection", "Crohn's disease"
            ],
            "irradiation": [
                "glioblastoma", "breast cancer", "lung cancer"
            ],
            "dexamethasone": [
                "glioblastoma", "leukemia", "ulcerative colitis", "Crohn's disease"
            ],
            "no treatment": [
                "normal", "unknown"
            ]
        }
    },
    "treatment_time": {
        "treatment": {
            "no treatment": ["not applicable"],
            "cisplatin": ["pre treatment", "on treatment", "post treatment", "relapse", "unknown", "24 hours", "48 hours", "1 week", "3 months", "1 year"],
            "doxorubicin": ["pre treatment", "on treatment", "post treatment", "relapse", "unknown", "24 hours", "48 hours", "1 week", "3 months", "1 year"],
            "irradiation": ["pre treatment", "on treatment", "post treatment", "relapse", "24 hours", "48 hours", "1 week"],
            "tamoxifen": ["pre treatment", "on treatment", "post treatment", "relapse"],
        }
    },
    "response": {
        "treatment": {
            "no treatment": ["not applicable"],
            "cisplatin": ["resistance", "sensitivity", "partial response", "stable disease", "progressive disease"],
            "doxorubicin": ["resistance", "sensitivity", "partial response", "stable disease", "progressive disease"],
            "paclitaxel": ["resistance", "sensitivity", "partial response", "stable disease"],
            "irradiation": ["resistance", "sensitivity", "partial response", "stable disease"],
        }
    },
    "biopsy_site": {
        "organ": {
            "blood": ["liver", "muscle", "spleen", "pancreas", "colon",
                      "skin", "bone marrow", "blood"],
            "lung": ["lung"],
            "heart": ["heart"],
            "kidney": ["kidney", "spleen"],
            "brain": ["brain"],
            "stomach": ["stomach"],
            "intestine": ["colon", "intestine", "stomach"]
        }
    },
    "library_source": {
        "cell_type": {
            "single-cell": [
                "blood cells", "nerve cells", "liver cells", "muscle cells", "fibroblasts", "bone cells",
                "cartilage cells", "connective cells", "lung cells", "kidney cells"
            ],
            "bulk": [
                "specialized integrated cells", "fat cells", "migratory cells", "stomach cells"
            ]
        }
    }
}


###################################################################################################
#MAIN

_all_vals = (
    LIB_SEL + SEQ_SRC + BIOPSY_SITE + BIOPSY_TYPE + CELL_LINE + CELL_TYPE + ORGAN + DISEASE +
    TREATMENT + TREAT_TIME + RESPONSE + AGE + SEX + ETHNICITY + LOCALIZATION + IS_CANCER
)
for _val in _all_vals:
    SYNONYMS.setdefault(_val, [_val])

CATEGORY_LISTS = {
    'library_selection': LIB_SEL,
    'sequencing_source': SEQ_SRC,
    'biopsy_site': BIOPSY_SITE,
    'biopsy_type': BIOPSY_TYPE,
    'cell_line': CELL_LINE,
    'cell_type': CELL_TYPE,
    'organ': ORGAN,
    'disease': DISEASE,
    'treatment': TREATMENT,
    'treatment_time': TREAT_TIME,
    'response': RESPONSE,
    'age': AGE,
    'sex': SEX,
    'ethnicity': ETHNICITY,
    'localization': LOCALIZATION,
    'is_cancer': IS_CANCER
}

CTX_WRAP = {
    'biopsy_site': [
        "targeted dissection of the {val} was performed to isolate transcript-rich zones",
        "the biopsy was scheduled based on abnormal imaging in the {val}",
        "samples were surgically obtained from the {val} following preoperative consent",
        "was extracted from tissue blocks originating in the {val}",
        "initial diagnosis prompted a biopsy of the {val}",
        "needle-guided retrieval was performed on the {val}",
        "sections from the {val} were fixed and preserved for sequencing",
        "collection involved multiple regions including the {val}",
        "sampling included perilesional and core areas of the {val}",
        "a sample from the {val} was included due to lesion proximity",
        "frozen sections from the {val} were used for nucleic acid extraction",
        "microdissection targeted specific compartments of the {val}"
    ],
    'organ': [
        "transcriptional profiles were enriched for genes expressed in the {val}",
        "the {val} appeared frequently altered in affected individuals",
        "molecular patterns in the {val} were compared across conditions",
        "analysis revealed cell-type heterogeneity in the {val}",
        "the functional role of the {val} was explored using expression datasets",
        "disease impact was localized predominantly to the {val}",
        "the {val} contributed significantly to overall expression variance",
        "inferred cellular dynamics suggested activation within the {val}",
        "inter-organ comparisons confirmed changes centered in the {val}",
        "sequencing highlighted regulatory elements active in the {val}",
        "the {val} exhibited distinct signatures relative to baseline tissue",
        "the experimental paradigm focused on perturbations within the {val}"
    ],
    'response': [
        'assessment concluded {val}',
        'biomarker trends were consistent with {val}',
        'status was updated to {val}',
        'board classification recorded {val}',
        'best overall response was adjudicated as {val}',
        'investigator assessment documented {val}',
        'progress notes annotated {val}',
        'case report form captured {val}',
        'report finalized with {val}',
        'treatment summary denoted {val}',
        'care plan was adjusted in light of {val}',
        'sponsor review logged {val} as outcome',
        'trial database entry reflected {val}',
        'protocol-defined endpoint corresponded to {val}',
        'safety review committee noted {val}',
        'assessment resulted in {val}',
        'follow-up consultation stated {val}',
        'clinical monitoring cycle reported {val}',
        'health record update listed {val}'
    ]
}

def inject_value(sentence: str, insertion: str) -> str:
    words = sentence.split()
    idx = random.randint(0, len(words))
    return ' '.join(words[:idx] + [insertion] + words[idx:])

with open(DATA_PATH) as f:
    templates = json.load(f)
expanded = random.choices(templates, k=1500)
with open(EXPANDED_PATH, 'w') as f:
    json.dump(expanded, f, indent=2)

CANCER_DISEASES = {
    "lung cancer", "hepatocellular carcinoma", "breast cancer", "leukemia", "lymphoma",
    "prostate cancer", "pancreatic adenocarcinoma", "glioblastoma"
}

rows = []
for template, _ in expanded:
    phrase = template
    record, chosen = {}, {}
    for cat in CATEGORIES:
        if cat == 'organ':
            ct = chosen.get('cell_type')
            cl = chosen.get('cell_line')
            opts_ct = SEM['cell_type']['organ'].get(ct, [])
            opts_cl = SEM['cell_line']['organ'].get(cl, [])
            candidates = list(set(opts_ct) & set(opts_cl)) or opts_ct or opts_cl or ORGAN
            raw = random.choice(candidates)
        elif cat == 'cell_type':
            organ = chosen.get('organ')
            valid_ct = [ct for ct, organs in SEM['cell_type']['organ'].items() if organ in organs]
            raw = random.choice(valid_ct) if valid_ct else random.choice(CELL_TYPE)
        elif cat == 'disease':
            org = chosen.get('organ')
            valid = [d for d in DISEASE if org in SEM['disease']['organ'].get(d, [])]
            raw = random.choice(valid) if valid else random.choice(DISEASE)
        elif cat == 'treatment_time':
            tr = chosen.get('treatment')
            raw = random.choice(SEM['treatment_time']['treatment'].get(tr, TREAT_TIME))
        elif cat == 'response':
            tr = chosen.get('treatment')
            raw = random.choice(SEM['response']['treatment'].get(tr, RESPONSE))
        elif cat == 'biopsy_site':
            org = chosen.get('organ')
            valid_sites = [site for site, organs in SEM.get("biopsy_site", {}).get("organ", {}).items() if
                           org in organs]
            raw = random.choice(valid_sites) if valid_sites else random.choice(BIOPSY_SITE)
        elif cat == 'is_cancer':
            raw = 'true' if chosen.get('disease') in CANCER_DISEASES else 'false'
        elif cat == 'treatment':
            disease = chosen.get('disease')
            valid_treatments = [t for t, diseases in SEM['treatment']['disease'].items() if disease in diseases]
            raw = random.choice(valid_treatments) if valid_treatments else "no treatment"

        else:
            raw = random.choice(CATEGORY_LISTS[cat])

        chosen[cat] = raw

        if cat != 'is_cancer':
            alt_options = SYNONYMS.get(raw, [raw])
            alt = random.choice(alt_options) if random.random() < 0.8 else raw
            if cat in CTX_WRAP:
                wrapper = random.choice(CTX_WRAP[cat])
                alt_phrase = wrapper.format(val=alt)
            else:
                alt_phrase = alt

            phrase = inject_value(phrase, alt_phrase)

        record[cat] = raw

    record['phrase'] = phrase
    rows.append(record)

pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)