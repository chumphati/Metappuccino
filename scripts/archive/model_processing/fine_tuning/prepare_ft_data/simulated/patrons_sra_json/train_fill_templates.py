import os
import json
import random
import pandas as pd
import re

# Paths
DATA_PATH = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/patrons_training/metadata_templates_train.json'
CORRECTED_PATH = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_ft_patron_semi/metadata_templates_train_corrected.json'
EXPANDED_PATH = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_ft_patron_semi/metadata_templates_train_expanded.json'
OUTPUT_CSV  = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_ft_patron_semi/metadata_replaced_table.csv'

# Categories and placeholders
CATEGORIES = [
    'cell_type', 'tissue_type', 'cell_line', 'organ', 'disease',
    'host_phenotype', 'library_selection', 'library_source',
    'treatment', 'treatment_time', 'response', 'donor_information', 'instrument_platform'
]
PH = {cat: f"__{cat.upper()}__" for cat in CATEGORIES}

# Pools
CELL_TYPE     = ["muscle cells","liver cells","blood cells","kidney cells","nerve cells","connective cells","fat cells","bone cells","specialized integrated cells","fibroblasts","migratory cells","stomach cells","lung cells"]
TISSUE_TYPE   = ["epithelial tissue","connective tissue","muscle tissue","nervous tissue","blood tissue","lymphatic tissue","adipose tissue","bone tissue","glandular tissue","fibrous tissue","hematopoietic tissue","smooth muscle tissue","cardiac muscle tissue"]
CELL_LINE     = ["Primary tissue","HEK293","HeLa","HepG2","MCF7","A549","K562","U2OS","PC3","Jurkat","HCT116","SHSY5Y","C2C12","THP1"]
ORGAN         = ["liver","lung","heart","kidney","brain","muscle","spleen","pancreas","colon","stomach","intestine","skin","bone marrow","blood"]
DISEASE       = ["lung cancer","hepatocellular carcinoma","breast cancer","leukemia","lymphoma","prostate cancer","pancreatic adenocarcinoma","glioblastoma","cirrhosis","chronic kidney disease","Crohn's disease","ulcerative colitis","myocardial infarction","diabetes mellitus"]
HOST_PHENO    = ["parental","persistent"]
LIB_SEL       = ["polyA","inverse rRNA","hybrid selection"]
LIB_SRC       = ["single-cell","bulk"]
TREATMENT     = ["no treatment","cisplatin","doxorubicin","paclitaxel","sorafenib","imatinib","erlotinib","tamoxifen","methotrexate","5-fluorouracil","amoxicillin","gentamicin","irradiation","dexamethasone"]
TREAT_TIME    = ["no treatment","pre treatment","on treatment","post treatment","relapse","undefined","24 hours","48 hours","1 week","3 months","1 year"]
RESPONSE      = ["nan","resistance","sensitivity","partial response","stable disease","progressive disease"]
DONOR_INFO    = ["male","34 years","female","62 years","47","North Africa","born in 1982","Paris","Asian","non-smoker","Hispanic","BMI 27","European","vegan","African","American","55 years","no information","19 years, runner","French, hypertensive","English","44 years","67 years","Australia","smoker","41 years"]
PLATFORMS     = ["Illumina NovaSeq 6000","Illumina HiSeq 4000","NextSeq 500","NextSeq 2000","MiSeq"]

POOLS = {
    'cell_type': CELL_TYPE,
    'tissue_type': TISSUE_TYPE,
    'cell_line': CELL_LINE,
    'organ': ORGAN,
    'disease': DISEASE,
    'host_phenotype': HOST_PHENO,
    'library_selection': LIB_SEL,
    'library_source': LIB_SRC,
    'treatment': TREATMENT,
    'treatment_time': TREAT_TIME,
    'response': RESPONSE,
    'donor_information': DONOR_INFO,
    'instrument_platform': PLATFORMS
}

# Synonyms mapping
SYNONYMS = {
    "muscle cells": ["myocytes", "contractile fibers", "myofibrils", "muscular cells", "myofibers"], "liver cells": ["hepatocytes", "hepatic parenchyma", "liver parenchyma cells", "liver tissue cells", "hepatic cells"], "blood cells": ["hematocytes", "corpuscles", "erythrocytes", "leukocytes", "circulating cells"], "kidney cells": ["renal cells", "nephron cells", "tubular cells", "glomerular cells", "kidney parenchymal cells"], "nerve cells": ["neurons", "neurocytes", "nerve fibers", "neural cells", "neuroblasts"], "connective cells": ["stromal cells", "mesenchymal cells", "support cells", "CT cells", "matrix cells"], "fat cells": ["adipocytes", "lipid cells", "adipose cells", "white adipose cells", "brown adipose cells"], "bone cells": ["osteoblasts", "osteoclasts", "osteocytes", "skeletal cells", "bone-forming cells"], "specialized integrated cells": ["neuroepithelial cells", "myoepithelial cells", "goblet cells", "Merkel cells", "Paneth cells"], "fibroblasts": ["fibrocytes", "tendinocytes", "keratocytes", "stromal fibroblasts", "fibroblastic cells"], "migratory cells": ["lymphocytes", "histiocytes", "melanocytes", "natural killer cells", "leukocyte migrants"], "stomach cells": ["chief cells", "parietal cells", "mucous neck cells", "G cells", "enteroendocrine cells"], "lung cells": ["type 1 pneumocytes", "type 2 pneumocytes", "alveolar epithelial cells", "bronchial epithelial cells", "Clara cells"],
    "epithelial tissue": ["epithelium", "epithelial layer", "lining tissue", "surface tissue", "barrier tissue"], "connective tissue": ["stromal tissue", "supportive tissue", "interstitial tissue", "mesenchymal tissue", "CT"], "muscle tissue": ["muscular tissue", "myogenic tissue", "contractile tissue", "muscle fibers", "myotissue"], "nervous tissue": ["neural tissue", "neuronal tissue", "gray matter", "white matter", "neuroglial tissue"], "blood tissue": ["vascular tissue", "circulatory tissue", "hemal tissue", "blood elements", "blood corpuscles"], "lymphatic tissue": ["lymphoid tissue", "immune tissue", "lymphatic network", "lymphoid organ", "lymphatic organ tissue"], "adipose tissue": ["fatty tissue", "lipid tissue", "adipose layer", "white adipose tissue", "brown adipose tissue"], "bone tissue": ["osseous tissue", "osseous matrix", "skeletal tissue", "bone matrix", "osseum"], "glandular tissue": ["secretory tissue", "gland epithelium", "endocrine tissue", "exocrine tissue", "glandular epithelium"], "fibrous tissue": ["fibrous connective tissue", "dense connective tissue", "fibrotic tissue", "fibroelastic tissue", "fibrotissue"], "hematopoietic tissue": ["blood-forming tissue", "hemopoietic tissue", "myeloid tissue", "lymphoid tissue", "bone marrow tissue"], "smooth muscle tissue": ["visceral muscle", "involuntary muscle", "non-striated muscle", "smooth fibers", "visceral tissue"], "cardiac muscle tissue": ["myocardial tissue", "heart muscle", "cardiac fibers", "myocardium", "cardiomyocytes"],
    "Primary tissue": ["primary cells", "ex vivo tissue", "native tissue", "primary sample", "direct tissue"], "HEK293": ["293 cells", "HEK-293", "human embryonic kidney 293", "HEK 293T", "293T variant"], "HeLa": ["HeLa S3", "HeLa CCL-2", "Henrietta Lacks cells", "HeLa Kyoto", "HeLa derivative"], "HepG2": ["hepatocellular carcinoma cell line", "Hep G2", "HepG-2", "liver carcinoma cells", "HepG2/C3A"], "MCF7": ["Michigan Cancer Foundation-7", "MCF-7", "breast adenocarcinoma cells", "MCF7/LCC1", "MCF7-TAMR"], "A549": ["A-549", "adenocarcinomic human alveolar basal epithelial cells", "lung carcinoma line", "A549/DDP", "A549-shRNA"], "K562": ["K-562", "chronic myelogenous leukemia cells", "leukemia line", "K562-Luc", "K562/ADR"], "U2OS": ["U-2 OS", "osteosarcoma cells", "U2OSp53", "U2OS Flp-In", "U2OS-GFP"], "PC3": ["PC-3", "prostate adenocarcinoma cells", "PC3-MM2", "PC3-PSMA", "PC3/NZW"], "Jurkat": ["Jurkat E6-1", "T lymphocyte cells", "acute T cell leukemia line", "Jurkat clone E6", "Jurkat/Jurkat"], "HCT116": ["HCT-116", "colorectal carcinoma cells", "HCT116 p53+", "HCT116-Dp53", "HCT116 p21-/-"], "SHSY5Y": ["SH-SY5Y", "neuroblastoma cells", "SHSY5Y-BRN2", "SHSY5Y/RA", "SHSY5Y-derived"], "C2C12": ["C2 C12", "mouse myoblasts", "C2C12 myotubes", "C2C12 MB", "C2C12-GFP"], "THP1": ["THP-1", "monocytic leukemia cells", "THP1-derived macrophages", "THP1-shRNA", "THP1-DM"],
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
    "undefined": ["unknown", "unspecified", "not defined", "NA", "missing"],
    "24 hours": ["24h", "one day", "day 1", "24 hrs", "t=24h"],
    "48 hours": ["48h", "two days", "day 2", "48 hrs", "t=48h"],
    "1 week": ["7 days", "one week", "week 1", "w01", "7d"],
    "3 months": ["3mo", "quarterly", "3-month", "90 days", "three months"],
    "1 year": ["12 months", "annual", "one year", "yearly", "365 days"],
    "nan": ["NA", "none", "not available", "undefined", "missing"], "resistance": ["resistant", "drug-resistant", "non-responsive", "refractory", "tolerant"], "sensitivity": ["sensitive", "drug-sensitive", "responsive", "susceptible", "high response"], "partial response": ["PR", "partial remission", "subpartial response", "decrease", "responding partially"], "stable disease": ["SD", "disease stabilization", "no progression", "stable condition", "unchanged"], "progressive disease": ["PD", "disease progression", "worsening", "advancing disease", "progression"],
    "male": ["M", "♂", "man", "adult male", "male donor"],
    "34 years": ["34 y.o.", "age 34", "34-year-old", "34yrs", "34yo"],
    "female": ["F", "♀", "woman", "adult female", "female donor"],
    "62 years": ["62 y.o.", "age 62", "62-year-old", "62yrs", "62yo"],
    "47": ["47 y.o.", "age 47", "47-year-old", "47yrs", "47yo"],
    "North Africa": ["Maghreb", "North-African region", "Northern Africa", "NA region", "North African"],
    "born in 1982": ["b. 1982", "birthyear 1982", "1982-born", "1982 cohort", "1982"],
    "Paris": ["Île-de-France", "city of Paris", "Parisian", "Paris, France", "Paris region"],
    "Asian": ["Asiatic", "East Asian", "South Asian", "Asian descent", "Asian ethnicity"],
    "non-smoker": ["never smoked", "smoke-free", "no smoking history", "non smoker", "never-smoker"],
    "Hispanic": ["Latino", "Latina", "Hispanic ethnicity", "Hispanic origin", "Latinx"],
    "BMI 27": ["BMI=27", "body mass index 27", "27 BMI", "overweight BMI", "27.0 BMI"],
    "European": ["Eurasian", "EU origin", "European descent", "Caucasian (Europe)", "Europe resident"],
    "vegan": ["plant-based diet", "vegan diet", "no animal products", "strictly vegan", "vegan lifestyle"],
    "African": ["Afro-descendant", "African origin", "Sub-Saharan African", "African ethnicity", "African continent"],
    "American": ["USA resident", "US citizen", "North American", "American origin", "American ethnicity"],
    "55 years": ["55 y.o.", "age 55", "55-year-old", "55yrs", "55yo"],
    "no information": ["unknown", "NA", "not available", "missing data", "unspecified"],
    "19 years, runner": ["19 y.o. runner", "19 years old athlete", "runner age 19", "19yo runner", "19-year-old runner"],
    "French, hypertensive": ["France origin, high blood pressure", "French donor with hypertension", "hypertensive French", "French & hypertensive", "hypertension, French"],
    "English": ["UK origin", "British", "England resident", "English ethnicity", "Anglo"],
    "44 years": ["44 y.o.", "age 44", "44-year-old", "44yrs", "44yo"],
    "67 years": ["67 y.o.", "age 67", "67-year-old", "67yrs", "67yo"],
    "Australia": ["Australian", "AUS", "Down Under", "Australia resident", "Oceania"],
    "smoker": ["current smoker", "tobacco user", "smoking history", "smoker status", "active smoker"],
    "41 years": ["41 y.o.", "age 41", "41-year-old", "41yrs", "41yo"],
    "Illumina NovaSeq 6000": ["NovaSeq6000", "Illumina NS6000", "NS6000", "NovaSeq", "Illumina NovaSeq"],
    "Illumina HiSeq 4000": ["HiSeq4000", "Illumina HS4000", "HS4000", "HiSeq", "Illumina HiSeq"],
    "NextSeq 500": ["NextSeq500", "NS500", "Illumina NextSeq", "NextSeq-500", "Next Seq 500"],
    "NextSeq 2000": ["NextSeq2000", "NS2000", "Illumina NextSeq 2000", "Next Seq 2000", "NextSeq-2000"],
    "MiSeq": ["MiSeq System", "Illumina MiSeq", "Mi-Seq", "MiSeq Sequencer", "MiSeq platform"],
    "single-cell": ["single cell", "unicellular", "droplet-based", "FACS-isolated", "scRNA-seq", "single-nucleus sequencing"],
    "bulk": ["population-level", "whole-tissue", "ensemble", "mass", "aggregated", "population RNA-seq"]
}
# ensure every main maps to itself too
for main in list(SYNONYMS.keys()):
    SYNONYMS.setdefault(main, [])
    SYNONYMS[main].append(main)

def choose_disp(val):
    syns = SYNONYMS.get(val, [])
    if syns and random.random()<0.6:
        return random.choice(syns)
    return val

# Rich semantic flows
SEM = {
    # cell_type ⇒ tissue_type & organ
    "cell_type": {
        "tissue_type": {
            "muscle cells":           ["muscle tissue", "smooth muscle tissue", "cardiac muscle tissue"],
            "myocytes":               ["muscle tissue", "smooth muscle tissue", "cardiac muscle tissue"],
            "liver cells":            ["epithelial tissue", "glandular tissue"],
            "hepatocytes":            ["epithelial tissue", "glandular tissue"],
            "blood cells":            ["blood tissue", "hematopoietic tissue"],
            "erythrocytes":           ["blood tissue", "hematopoietic tissue"],
            "kidney cells":           ["epithelial tissue"],
            "renal cells":            ["epithelial tissue"],
            "nerve cells":            ["nervous tissue"],
            "neurons":                ["nervous tissue"],
            "connective cells":       ["connective tissue", "fibrous tissue"],
            "stromal cells":          ["connective tissue", "fibrous tissue"],
            "fat cells":              ["adipose tissue"],
            "adipocytes":             ["adipose tissue"],
            "bone cells":             ["bone tissue"],
            "osteoblasts":            ["bone tissue"],
            "fibroblasts":            ["connective tissue", "fibrous tissue"],
            "fibrocytes":             ["connective tissue", "fibrous tissue"],
            "migratory cells":        ["blood tissue", "lymphatic tissue"],
            "lymphocytes":            ["blood tissue", "lymphatic tissue"],
            "stomach cells":          ["epithelial tissue", "glandular tissue"],
            "chief cells":            ["epithelial tissue", "glandular tissue"],
            "lung cells":             ["epithelial tissue", "glandular tissue"],
            "alveolar epithelial cells": ["epithelial tissue"],
            "specialized integrated cells": ["epithelial tissue", "glandular tissue"]
        },
        "organ": {
            "muscle cells":           ["muscle", "heart"],
            "myocytes":               ["muscle", "heart"],
            "liver cells":            ["liver"],
            "hepatocytes":            ["liver"],
            "blood cells":            ["blood", "bone marrow"],
            "erythrocytes":           ["blood", "bone marrow"],
            "kidney cells":           ["kidney"],
            "renal cells":            ["kidney"],
            "nerve cells":            ["brain"],
            "neurons":                ["brain"],
            "connective cells":       ["skin", "muscle"],
            "stromal cells":          ["skin", "muscle"],
            "fat cells":              ["skin"],
            "adipocytes":             ["skin"],
            "bone cells":             ["bone", "bone marrow"],
            "osteoblasts":            ["bone"],
            "chondrocytes":           ["bone"],
            "fibroblasts":            ["skin", "muscle"],
            "fibrocytes":             ["skin", "muscle"],
            "migratory cells":        ["blood", "spleen"],
            "lymphocytes":            ["blood", "spleen"],
            "stomach cells":          ["stomach"],
            "chief cells":            ["stomach"],
            "lung cells":             ["lung"],
            "alveolar epithelial cells": ["lung"],
            "specialized integrated cells": ["pancreas", "colon", "lung"]
        }
    },

    # cell_line ⇒ organ
    "cell_line": {
        "organ": {
            "Primary tissue": ["liver","lung","heart","kidney","brain","muscle","spleen","pancreas","colon","stomach","intestine","skin","bone marrow","blood"],
            "HEK293":         ["kidney"],
            "HeLa":           ["skin"],
            "HepG2":          ["liver"],
            "MCF7":           ["skin"],
            "A549":           ["lung"],
            "K562":           ["blood","bone marrow"],
            "U2OS":           ["bone","bone marrow"],
            "PC3":            ["kidney"],
            "Jurkat":         ["blood","bone marrow"],
            "HCT116":         ["colon"],
            "SHSY5Y":         ["brain"],
            "C2C12":          ["muscle"],
            "THP1":           ["blood","bone marrow"]
        },
        "disease": {
            "Primary tissue":          DISEASE,
            "HEK293":                  ["chronic kidney disease"],
            "HeLa":                    ["breast cancer"],
            "HepG2":                   ["hepatocellular carcinoma"],
            "MCF7":                    ["breast cancer"],
            "A549":                    ["lung cancer"],
            "K562":                    ["leukemia"],
            "Jurkat":                  ["leukemia"],
            "U2OS":                    ["leukemia","lymphoma"],
            "PC3":                     ["prostate cancer"],
            "HCT116":                  ["Crohn's disease","ulcerative colitis"],
            "SHSY5Y":                  ["glioblastoma"],
            "C2C12":                   ["myocardial infarction"],
            "THP1":                    ["leukemia"]
        }
    },

    # disease ⇒ organ
    "disease": {
        "organ": {
            "lung cancer":               ["lung"],
            "pulmonary carcinoma":       ["lung"],
            "hepatocellular carcinoma":  ["liver"],
            "liver cancer":              ["liver"],
            "breast cancer":             ["skin"],
            "mammary carcinoma":         ["skin"],
            "leukemia":                  ["blood","bone marrow"],
            "lymphoma":                  ["spleen"],
            "prostate cancer":           ["kidney"],
            "pancreatic adenocarcinoma": ["pancreas"],
            "glioblastoma":              ["brain"],
            "cirrhosis":                 ["liver"],
            "chronic kidney disease":    ["kidney"],
            "Crohn's disease":           ["colon","intestine"],
            "ulcerative colitis":        ["colon","intestine"],
            "myocardial infarction":     ["heart"],
            "diabetes mellitus":         ["pancreas"]
        }
    },

    # treatment ⇒ disease
    "treatment": {
        "disease": {
            "cisplatin":      ["lung cancer","hepatocellular carcinoma","breast cancer","leukemia","lymphoma"],
            "doxorubicin":    ["breast cancer","leukemia","lymphoma","glioblastoma"],
            "paclitaxel":     ["lung cancer","breast cancer","prostate cancer"],
            "sorafenib":      ["hepatocellular carcinoma","chronic kidney disease"],
            "imatinib":       ["leukemia","chronic kidney disease"],
            "erlotinib":      ["lung cancer","pancreatic adenocarcinoma"],
            "tamoxifen":      ["breast cancer"],
            "methotrexate":   ["leukemia","lymphoma","breast cancer"],
            "5-fluorouracil": ["colorectal cancer","breast cancer","pancreatic adenocarcinoma"],
            "amoxicillin":    ["infection","Crohn's disease","ulcerative colitis"],
            "gentamicin":     ["infection","Crohn's disease"],
            "irradiation":    ["glioblastoma","breast cancer","lung cancer","prostate cancer"],
            "dexamethasone":  ["glioblastoma","leukemia","Crohn's disease","ulcerative colitis"],
            "no treatment":   DISEASE
        }
    },

    # treatment_time ⇒ treatment
    "treatment_time": {
        "treatment": {
            "no treatment":   ["no treatment"],
            "cisplatin":      TREAT_TIME,
            "doxorubicin":    TREAT_TIME,
            "paclitaxel":     TREAT_TIME,
            "erlotinib":      TREAT_TIME,
            "irradiation":    ["pre treatment","on treatment","post treatment","relapse","24 hours","48 hours","1 week"],
            "tamoxifen":      ["pre treatment","on treatment","post treatment","relapse"]
        }
    },

    # response ⇒ treatment
    "response": {
        "treatment": {
            "no treatment":  ["nan"],
            "cisplatin":     RESPONSE,
            "doxorubicin":   RESPONSE,
            "paclitaxel":    RESPONSE,
            "irradiation":   RESPONSE,
            "tamoxifen":     RESPONSE
        }
    },

    # host_phenotype ⇒ treatment
    "host_phenotype": {
        "treatment": {
            "no treatment":  ["parental"],
            "cisplatin":     HOST_PHENO,
            "doxorubicin":   HOST_PHENO,
            "paclitaxel":    HOST_PHENO,
            "irradiation":   HOST_PHENO
        }
    },

    # library_selection ⇒ tissue_type
    "library_selection": {
        "tissue_type": {
            "polyA": [
                "epithelial tissue",
                "muscle tissue",
                "nervous tissue",
                "glandular tissue"
            ],
            "inverse rRNA": [
                "blood tissue",
                "lymphatic tissue",
                "hematopoietic tissue",
                "fibrous tissue"
            ],
            "hybrid selection": [
                "muscle tissue",
                "nervous tissue",
                "glandular tissue"
            ]
        }
    },

    # library_source ⇒ cell_type
    "library_source": {
        "cell_type": {
            "single-cell": CELL_TYPE,
            "bulk":        ["specialized integrated cells","fat cells","migratory cells","stomach cells"]
        }
    }
}


# --- Charger & corriger les templates ---
with open(DATA_PATH) as f:
    templates = json.load(f)

corrected_tpl = []
for tpl, cnt in templates:
    for ph in PH.values():
        if ph not in tpl:
            tpl += " " + ph
    corrected_tpl.append((tpl, cnt))

with open(CORRECTED_PATH, 'w') as f:
    json.dump(corrected_tpl, f, indent=2)

# --- Génération d'une cinquantaine de combos de base cohérents ---
base_combos = []
while len(base_combos) < 50:
    combo = {}
    # 1) cell_type
    ct = random.choice(CELL_TYPE)
    combo['cell_type'] = ct

    # 2) tissue_type
    combo['tissue_type'] = random.choice(
        SEM['cell_type']['tissue_type'].get(ct, TISSUE_TYPE)
    )

    # 3) organ
    org = random.choice(
        SEM['cell_type']['organ'].get(ct, ORGAN)
    )
    combo['organ'] = org

    # 4) cell_line  ← ici on inverse correctement la map
    cl_opts = [
        cl for cl, orgs in SEM['cell_line']['organ'].items()
        if org in orgs
    ]
    if cl_opts:
        combo['cell_line'] = random.choice(cl_opts)
    else:
        combo['cell_line'] = random.choice(CELL_LINE)

    # 5) disease
    ds_opts = [
        d for d, orgs in SEM['disease']['organ'].items()
        if org in orgs
    ]
    combo['disease'] = random.choice(ds_opts or DISEASE)

    # 6) treatment
    combo['treatment'] = random.choice(
        SEM['treatment']['disease'].get(combo['disease'], TREATMENT)
    )

    # 7) treatment_time
    combo['treatment_time'] = random.choice(
        SEM['treatment_time']['treatment'].get(combo['treatment'], TREAT_TIME)
    )

    # 8) response
    combo['response'] = random.choice(
        SEM['response']['treatment'].get(combo['treatment'], RESPONSE)
    )

    # 9) host_phenotype
    combo['host_phenotype'] = random.choice(
        SEM['host_phenotype']['treatment'].get(combo['treatment'], HOST_PHENO)
    )

    # 10) library_selection
    combo['library_selection'] = random.choice(
        SEM['library_selection']['tissue_type']
           .get(combo['tissue_type'], LIB_SEL)
    )

    # 11) library_source
    combo['library_source'] = random.choice(
        SEM['library_source']['cell_type']
           .get(combo['cell_type'], LIB_SRC)
    )

    # 12) donor_information & 13) instrument_platform
    combo['donor_information']   = random.choice(DONOR_INFO)
    combo['instrument_platform'] = random.choice(PLATFORMS)

    base_combos.append(combo)

# --- Expansion à 1500 phrases ---
records = []
for _ in range(1500):
    combo  = random.choice(base_combos)
    tpl     = random.choice(corrected_tpl)[0]
    phrase  = tpl
    for cat, val in combo.items():
        disp = choose_disp(val)
        phrase = re.sub(re.escape(PH[cat]), disp, phrase)
    rec = {'phrase_text': phrase}
    rec.update(combo)
    records.append(rec)

# --- Sauvegarde ---
with open(EXPANDED_PATH, 'w') as f:
    json.dump([r['phrase_text'] for r in records], f, indent=2)

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Généré {len(records)} enregistrements cohérents.")