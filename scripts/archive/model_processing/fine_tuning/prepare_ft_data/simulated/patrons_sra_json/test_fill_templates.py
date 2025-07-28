import os
import json
import random
import pandas as pd
import re

DATA_PATH = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/patrons_training/metadata_templates_test.json'
CORRECTED_PATH = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_ft_patron_semi/metadata_templates_test_corrected.json'
EXPANDED_PATH = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_ft_patron_semi/metadata_templates_test_expanded.json'
OUTPUT_CSV  = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_ft_patron_semi/test_metadata_replaced_table.csv'

CATEGORIES = [
    'cell_type', 'tissue_type', 'cell_line', 'organ', 'disease',
    'host_phenotype', 'library_selection', 'library_source',
    'treatment', 'treatment_time', 'response', 'donor_information', 'instrument_platform'
]
PH = {cat: f"__{cat.upper()}__" for cat in CATEGORIES}

CELL_TYPE = ["melanocytes", "keratinocytes", "osteoclasts", "osteoblasts", "Kupffer cells", "Langerhans cells", "pancreatic beta cells", "alveolar macrophages", "choroid plexus epithelial cells", "germinal center B cells"]
TISSUE_TYPE = ["epidermal tissue", "dermal connective tissue", "bone tissue", "hepatic sinusoidal tissue", "islet of Langerhans tissue", "alveolar tissue", "lymphoid follicle tissue", "choroid plexus tissue", "germinal center tissue"]
CELL_LINE = ["HaCaT", "MIN6", "Capan-1", "LNCaP", "DU145", "HL-60", "MCF10A", "MDCK", "H9", "Raji"]
ORGAN = ["bladder", "thyroid", "adrenal gland", "lymph node", "prostate", "placenta", "pituitary gland", "tonsil", "salivary gland", "pancreatic duct"]
DISEASE = ["melanoma", "psoriasis", "osteoporosis", "Hashimoto's thyroiditis", "multiple sclerosis", "HIV infection", "tuberculosis", "hepatitis C", "rheumatoid arthritis", "cystic fibrosis"]
TREATMENT = ["surgical excision", "topical corticosteroids", "bone marrow transplantation", "antiretroviral therapy", "enzyme replacement therapy", "plasma exchange", "monoclonal antibody therapy", "phototherapy", "anti-TNF therapy", "hormone suppression therapy"]
LIB_SEL = ["ribodepletion", "targeted capture", "total RNA", "small RNA-free",]
LIB_SRC = ["nuclei sequencing", "exosomal RNA", "cytosolic RNA", "mitochondrial RNA", "total cell extract"]
TREAT_TIME = ["12 hours", "36 hours", "15 days", "none", "10 months", "18 months", "unknown"]
RESPONSE = ["complete response", "no evidence of disease", "disease stabilization", "clinical benefit", "treatment-related mortality", "mixed response"]
DONOR_INFO = ["28-year-old", "smoker", "60-year-old", "vegetarian", "pregnant woman", "male donor", "type O", "female", "donor type AB", "diabetic", "BMI 22", "child", "senior citizen"]
HOST_PHENO = ["wild-type", "knockout", "drug-sensitive", "drug-resistant", "mutant", "hypomorphic"]
PLATFORMS = ["PacBio Sequel II", "Oxford Nanopore PromethION", "BGISEQ-500", "Ion Torrent Proton", "MGI DNBSEQ-T7"]

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

SYNONYMS = {
    "melanocytes":                        ["melanin-producing cells", "skin pigment cells", "melanocytic cells", "melanocyte cells", "melanocytes"],
    "keratinocytes":                      ["epidermal keratinocytes", "skin keratin cells", "epidermal cells", "keratin-forming cells", "keratinocytes"],
    "osteoclasts":                        ["bone resorbing cells", "multinucleated bone cells", "osteoclast cells", "bone osteoclasts", "osteoclasts"],
    "osteoblasts":                        ["bone forming cells", "osteoblast cells", "osteogenic cells", "bone osteoblasts", "osteoblasts"],
    "Kupffer cells":                      ["liver macrophages", "hepatic Kupffer cells", "Kupffer macrophages", "hepatic macrophages", "Kupffer cells"],
    "Langerhans cells":                   ["skin dendritic cells", "epidermal APCs", "LCs", "cutaneous Langerhans cells", "Langerhans cells"],
    "pancreatic beta cells":              ["islet beta cells", "beta islet cells", "insulin-secreting cells", "pancreatic β cells", "pancreatic beta cells"],
    "alveolar macrophages":               ["lung macrophages", "alveolar immune cells", "pulmonary macrophages", "airway macrophages", "alveolar macrophages"],
    "choroid plexus epithelial cells":    ["CPE cells", "choroid plexus epithelium", "ventricular epithelial cells", "CPEC cells", "choroid plexus epithelial cells"],
    "germinal center B cells":            ["GC B cells", "B lymphocytes", "follicular B cells", "germinal center B lymphocytes", "germinal center B cells"],
    "epidermal tissue":                   ["epidermis", "epidermal layer", "skin epidermis", "outer skin layer", "epidermal tissue"],
    "dermal connective tissue":           ["dermis", "dermal layer", "skin connective tissue", "connective dermal tissue", "dermal connective tissue"],
    "bone tissue":                        ["osseous tissue", "skeletal tissue", "bony tissue", "compact/spongy bone", "bone tissue"],
    "hepatic sinusoidal tissue":          ["liver sinusoidal tissue", "hepatic sinusoids", "sinusoid tissue", "sinusoidal hepatic tissue", "hepatic sinusoidal tissue"],
    "islet of Langerhans tissue":         ["pancreatic islets", "islets of Langerhans", "endocrine pancreas tissue", "Langerhans islet tissue", "islet of Langerhans tissue"],
    "alveolar tissue":                    ["lung alveoli tissue", "pulmonary alveolar tissue", "air sac tissue", "alveolar epithelium", "alveolar tissue"],
    "lymphoid follicle tissue":           ["lymph node follicles", "follicular lymphoid tissue", "germinal center tissue", "lymphoid follicle structure", "lymphoid follicle tissue"],
    "choroid plexus tissue":              ["CPE tissue", "choroid plexus", "ventricular epithelium tissue", "CPEC tissue", "choroid plexus tissue"],
    "germinal center tissue":             ["GC tissue", "follicular germinal center", "lymphoid germinal centers", "germinal center region", "germinal center tissue"],
    "HaCaT":                              ["HaCaT keratinocyte line", "immortalized keratinocytes", "HaCaT cells", "keratinocyte HaCaT", "HaCaT"],
    "MIN6":                               ["MIN6 beta cell line", "mouse insulinoma MIN6", "MIN6 cells", "beta MIN6 line", "MIN6"],
    "Capan-1":                            ["Capan1 cells", "pancreatic adenocarcinoma Capan-1", "Capan-1 line", "Capan1 line", "Capan-1"],
    "LNCaP":                              ["LNCaP prostate cancer cells", "LNCaP line", "LNCaP cells", "prostate LNCaP", "LNCaP"],
    "DU145":                              ["DU-145 prostate cancer cells", "DU145 line", "DU-145 line", "prostate DU145", "DU145"],
    "HL-60":                              ["HL60 cells", "promyelocytic leukemia HL-60", "HL 60 line", "HL60 line", "HL-60"],
    "MCF10A":                             ["MCF-10A cells", "non-tumorigenic breast epithelial", "MCF10A line", "MCF-10A line", "MCF10A"],
    "MDCK":                               ["MDCK epithelial cells", "canine kidney MDCK", "MDCK line", "MDCK cells", "MDCK"],
    "H9":                                 ["H9 T-cell line", "H9 cells", "T-cell H9 line", "H9 leukemia line", "H9"],
    "Raji":                               ["Raji B-cell line", "Burkitt lymphoma Raji", "Raji cells", "Raji lymphoma cells", "Raji"],
    "bladder":                            ["urinary bladder", "vesica urinaria", "bladder organ", "bladder tissue", "bladder"],
    "thyroid":                            ["thyroid gland", "endocrine thyroid gland", "glandula thyreoidea", "thyroid tissue", "thyroid"],
    "adrenal gland":                      ["suprarenal gland", "adrenal cortex", "adrenal organ", "suprarenal tissue", "adrenal gland"],
    "lymph node":                         ["lymphatic node", "lymph gland", "lymph node tissue", "lymphatic organ", "lymph node"],
    "prostate":                           ["prostatic gland", "male prostate", "prostate organ", "prostate tissue", "prostate"],
    "placenta":                           ["placental tissue", "chorionic villi", "placental organ", "placenta", "placental"],
    "pituitary gland":                    ["hypophysis", "pituitary", "pituitary organ", "pituitary tissue", "pituitary gland"],
    "tonsil":                             ["pharyngeal tonsil", "palatine tonsil", "tonsillar tissue", "tonsil", "tonsil organ"],
    "salivary gland":                     ["salivary gland tissue", "salivary organ", "glandula salivaria", "salivary tissue", "salivary gland"],
    "pancreatic duct":                    ["duct of pancreas", "pancreatic ductal tissue", "ductus pancreaticus", "pancreatic duct", "pancreatic duct"],
    "melanoma":                           ["skin melanoma", "malignant melanoma", "cutaneous melanoma", "melanoma", "melanoma skin cancer"],
    "psoriasis":                          ["psoriatic disease", "chronic plaque psoriasis", "psoriasis vulgaris", "psoriasis", "psoriatic dermatitis"],
    "osteoporosis":                       ["bone density loss", "porotic bone", "osteoporotic disease", "osteoporosis", "low bone mass disease"],
    "Hashimoto's thyroiditis":            ["autoimmune thyroiditis", "chronic lymphocytic thyroiditis", "Hashimoto thyroiditis", "Hashimoto's thyroiditis", "Hashimoto disease"],
    "multiple sclerosis":                 ["MS", "disseminated sclerosis", "demyelinating disease", "multiple sclerosis", "MS disease"],
    "HIV infection":                      ["human immunodeficiency virus", "HIV positive", "HIV/AIDS", "HIV infection", "HIV disease"],
    "tuberculosis":                       ["TB", "Mycobacterium tuberculosis infection", "pulmonary TB", "tuberculous disease", "tuberculosis"],
    "hepatitis C":                        ["HCV infection", "chronic hepatitis C", "hepatitis C virus", "hepatitis C", "HCV disease"],
    "rheumatoid arthritis":               ["RA", "autoimmune arthritis", "chronic rheumatoid disease", "rheumatoid arthritis", "rheumatoid disease"],
    "cystic fibrosis":                    ["CF", "mucoviscidosis", "cystic fibrosis", "CF disease", "cystic fibrosis"],
    "surgical excision":                  ["excisional surgery", "surgical removal", "tissue excision", "surgical excision", "excision procedure"],
    "topical corticosteroids":            ["steroid cream", "topical steroids", "anti-inflammatory cream", "topical corticosteroids", "steroid ointment"],
    "bone marrow transplantation":        ["BMT", "hematopoietic stem cell transplant", "bone marrow transplant", "stem cell transplantation", "bone marrow transplantation"],
    "antiretroviral therapy":             ["ART", "HIV therapy", "antiretroviral treatment", "HAART", "antiretroviral therapy"],
    "enzyme replacement therapy":         ["ERT", "enzyme therapy", "recombinant enzyme therapy", "enzyme replacement therapy", "replacement therapy"],
    "plasma exchange":                    ["plasmapheresis", "plasma filtration", "therapeutic plasma exchange", "PEX", "plasma exchange"],
    "monoclonal antibody therapy":        ["mAb therapy", "biologic therapy", "monoclonal therapy", "antibody therapy", "monoclonal antibody therapy"],
    "phototherapy":                       ["UV therapy", "light therapy", "PUVA", "narrowband UVB", "phototherapy"],
    "anti-TNF therapy":                   ["TNF blocker", "tumor necrosis factor inhibitor", "TNF-alpha inhibitor", "anti-TNF", "anti-TNF therapy"],
    "hormone suppression therapy":        ["endocrine suppression", "hormone blockade", "androgen deprivation therapy", "estrogen suppression", "hormone suppression therapy"],
    "ribodepletion":                      ["rRNA depletion", "ribosomal RNA removal", "ribodepletion", "rRNA removal", "depletion protocol"],
    "targeted capture":                   ["hybrid capture", "targeted sequencing", "capture-based library", "sequence capture", "targeted capture"],
    "total RNA":                          ["whole transcriptome", "total transcriptome", "total RNA library", "RNA-seq total", "total RNA"],
    "small RNA-free":                     ["no small RNA", "smallRNA depleted", "small RNA depletion", "smallRNA free", "small RNA-free"],
    "nuclei sequencing":                  ["nuclear RNA-seq", "nucleus sequencing", "single nucleus RNA", "nuc-seq", "nuclei sequencing"],
    "exosomal RNA":                       ["extracellular vesicle RNA", "exosome RNA", "EV-RNA", "microvesicle RNA", "exosomal RNA"],
    "cytosolic RNA":                      ["cell cytoplasm RNA", "cytosolic transcriptome", "cytosol RNA", "cytosolic RNA", "cytoplasmic RNA"],
    "mitochondrial RNA":                  ["mtRNA", "mitochondrial transcriptome", "mitoRNA", "mitochondrial RNA", "mt transcript"],
    "total cell extract":                 ["whole cell extract", "total cell lysate", "cell extract", "whole extract", "total cell extract"],    "12 hours":                           ["12h", "twelve hours", "half day", "12-hour", "12 hours"],
    "36 hours":                           ["36h", "thirty-six hours", "1.5 days", "36-hour", "36 hours"],
    "15 days":                            ["15d", "fifteen days", "over two weeks", "15-day", "15 days"],
    "none":                               ["no treatment", "untreated", "none", "no therapy", "untreated sample"],
    "10 months":                          ["10mo", "ten months", "10-month", "10 months", "10mo"],
    "18 months":                          ["18mo", "eighteen months", "18-month", "18 months", "18mo"],
    "unknown":                            ["NA", "not available", "unknown", "missing", "unspecified"],
    "complete response":                  ["CR", "complete remission", "full response", "complete response", "complete remission"],
    "no evidence of disease":             ["NED", "disease-free", "no disease evidence", "no evidence of disease", "NED status"],
    "disease stabilization":              ["stable disease", "disease stable", "stabilization", "disease stabilization", "stable"],
    "clinical benefit":                   ["CB", "clinical response", "benefit", "clinical benefit", "benefit observed"],
    "treatment-related mortality":        ["TRM", "treatment mortality", "therapy-related death", "treatment-related mortality", "treatment death"],
    "mixed response":                     ["mixed outcome", "partial response", "mixed response", "heterogeneous response", "mixed results"],
    "28-year-old":                        ["28 y.o.", "age 28", "28 year old", "28yrs", "28-year-old"],
    "smoker":                             ["current smoker", "tobacco user", "smoking donor", "smoking", "smoker"],
    "60-year-old":                        ["60 y.o.", "age 60", "60 year old", "60yrs", "60-year-old"],
    "vegetarian":                         ["plant-based diet", "veg diet", "vegetarian", "veg donor", "vegetarian"],
    "pregnant woman":                     ["pregnant female", "expectant mother", "gravida", "preg", "pregnant woman"],
    "male donor":                         ["male", "man donor", "adult male donor", "M donor", "male donor"],
    "type O":                             ["O blood type", "blood type O", "O Rh", "O donor", "type O"],
    "female":                             ["F", "woman", "female donor", "adult female", "female"],
    "donor type AB":                      ["AB donor", "blood type AB", "AB Rh", "AB blood group", "donor type AB"],
    "diabetic":                           ["type 2 diabetic", "diabetes mellitus", "diabetic donor", "diabetes", "diabetic"],
    "BMI 22":                             ["BMI22", "body mass index 22", "BMI = 22", "BMI22 donor", "BMI 22"],
    "child":                              ["pediatric donor", "child donor", "juvenile donor", "underage donor", "child"],
    "senior citizen":                     ["elderly donor", "aged donor", "senior donor", "older adult", "senior citizen"],
    "wild-type":                          ["WT", "wild type", "native genotype", "unmodified", "wild-type"],
    "knockout":                           ["KO", "gene knockout", "null mutant", "knockout", "knock-out"],
    "drug-sensitive":                     ["sensitive", "drug sensitive", "therapy sensitive", "sensitive phenotype", "drug-sensitive"],
    "drug-resistant":                     ["resistant", "drug resistance", "therapy resistant", "resistant phenotype", "drug-resistant"],
    "mutant":                             ["mt", "mutant genotype", "gene mutant", "mutant", "mut"],
    "hypomorphic":                        ["hypomorph", "partial loss-of-function", "reduced function", "hypomorphic", "hypo"],
    "PacBio Sequel II":                   ["Sequel II", "PacBio II", "Pacific Biosciences Sequel II", "PacBio sequel2", "PacBio Sequel II"],
    "Oxford Nanopore PromethION":         ["PromethION", "Oxford PromethION", "Nanopore PromethION", "ONT PromethION", "Oxford Nanopore PromethION"],
    "BGISEQ-500":                         ["BGISEQ-500", "BGI Seq 500", "BGISEQ500", "BGI-SEQ 500", "BGISEQ-500"],
    "Ion Torrent Proton":                 ["Ion Proton", "Proton sequencer", "Ion Torrent P1", "Proton platform", "Ion Torrent Proton"],
    "MGI DNBSEQ-T7":                      ["DNBSEQ-T7", "MGI T7", "MGI DNBSEQ T7", "DNBSEQ T-7", "MGI DNBSEQ-T7"]
}

for main in list(SYNONYMS.keys()):
    SYNONYMS.setdefault(main, [])
    SYNONYMS[main].append(main)

def choose_disp(val):
    syns = SYNONYMS.get(val, [])
    if syns and random.random()<0.85:
        return random.choice(syns)
    return val

SEM = {
    "cell_type": {
        "tissue_type": {
            "melanocytes":                     ["epidermal tissue"],
            "keratinocytes":                   ["epidermal tissue"],
            "osteoclasts":                     ["bone tissue"],
            "osteoblasts":                     ["bone tissue"],
            "Kupffer cells":                   ["hepatic sinusoidal tissue"],
            "Langerhans cells":                ["epidermal tissue"],
            "pancreatic beta cells":           ["islet of Langerhans tissue"],
            "alveolar macrophages":            ["alveolar tissue"],
            "choroid plexus epithelial cells": ["choroid plexus tissue"],
            "germinal center B cells":         ["lymphoid follicle tissue"]
        },
        "organ": {
            "melanocytes":                     ["salivary gland"],
            "keratinocytes":                   ["salivary gland"],
            "osteoclasts":                     ["placenta"],
            "osteoblasts":                     ["placenta"],
            "Kupffer cells":                   ["adrenal gland"],
            "Langerhans cells":                ["tonsil"],
            "pancreatic beta cells":           ["pancreatic duct"],
            "alveolar macrophages":            ["bladder"],
            "choroid plexus epithelial cells": ["pituitary gland"],
            "germinal center B cells":         ["lymph node"]
        }
    },

    "cell_line": {
        "organ": {
            "HaCaT":   ["salivary gland"],
            "MIN6":    ["pancreatic duct"],
            "Capan-1": ["pancreatic duct"],
            "LNCaP":   ["prostate"],
            "DU145":   ["prostate"],
            "HL-60":   ["lymph node"],
            "MCF10A":  ["salivary gland"],
            "MDCK":    ["bladder"],
            "H9":      ["lymph node"],
            "Raji":    ["lymph node"]
        },
        "disease": {
            "HaCaT":   ["psoriasis"],
            "MIN6":    ["diabetes mellitus"],
            "Capan-1": ["hepatitis C"],
            "LNCaP":   ["prostate cancer"],
            "DU145":   ["prostate cancer"],
            "HL-60":   ["leukemia"],
            "MCF10A":  ["breast cancer"],
            "MDCK":    ["tuberculosis"],
            "H9":      ["HIV infection"],
            "Raji":    ["rheumatoid arthritis"]
        }
    },

    "disease": {
        "organ": {
            "melanoma":                  ["salivary gland"],
            "psoriasis":                 ["salivary gland"],
            "osteoporosis":              ["adrenal gland"],
            "Hashimoto's thyroiditis":   ["thyroid"],
            "multiple sclerosis":        ["pituitary gland"],
            "HIV infection":             ["tonsil"],
            "tuberculosis":              ["tonsil"],
            "hepatitis C":               ["pancreatic duct"],
            "rheumatoid arthritis":      ["adrenal gland"],
            "cystic fibrosis":           ["pancreatic duct"]
        }
    },

    "treatment": {
        "disease": {
            "melanoma":                  ["surgical excision"],
            "psoriasis":                 ["topical corticosteroids", "phototherapy"],
            "osteoporosis":              ["enzyme replacement therapy"],
            "Hashimoto's thyroiditis":   ["plasma exchange"],
            "multiple sclerosis":        ["hormone suppression therapy"],
            "HIV infection":             ["antiretroviral therapy"],
            "tuberculosis":              ["enzyme replacement therapy"],
            "hepatitis C":               ["antiviral therapy"],
            "rheumatoid arthritis":      ["monoclonal antibody therapy", "anti-TNF therapy"],
            "cystic fibrosis":           ["enzyme replacement therapy"]
        }
    },

    "treatment_time": {
        "treatment": {
            "surgical excision":            ["12 hours", "36 hours", "4 days"],
            "topical corticosteroids":      ["8 days", "15 days", "4 weeks"],
            "bone marrow transplantation":  ["8 weeks", "10 months", "18 months"],
            "antiretroviral therapy":       ["4 weeks", "8 weeks", "10 months"],
            "enzyme replacement therapy":   ["4 weeks", "8 weeks", "10 months"],
            "plasma exchange":              ["4 days", "8 days", "15 days"],
            "monoclonal antibody therapy":  ["15 days", "4 weeks", "8 weeks"],
            "phototherapy":                 ["12 hours", "36 hours", "4 days"],
            "anti-TNF therapy":             ["4 weeks", "8 weeks", "15 days"],
            "hormone suppression therapy":  ["4 weeks", "8 weeks", "15 days"]
        }
    },

    "response": {
        "treatment": {
            "surgical excision":            ["complete response", "no evidence of disease"],
            "topical corticosteroids":      ["disease stabilization", "mixed response"],
            "bone marrow transplantation":  ["clinical benefit", "mixed response"],
            "antiretroviral therapy":       ["clinical benefit", "mixed response"],
            "enzyme replacement therapy":   ["clinical benefit", "mixed response"],
            "plasma exchange":              ["mixed response", "disease stabilization"],
            "monoclonal antibody therapy":  ["clinical benefit", "mixed response"],
            "phototherapy":                 ["disease stabilization", "mixed response"],
            "anti-TNF therapy":             ["clinical benefit", "mixed response"],
            "hormone suppression therapy":  ["clinical benefit", "mixed response"]
        }
    },

    "host_phenotype": {
        "treatment": {
            "surgical excision":            ["wild-type", "knockout"],
            "topical corticosteroids":      ["drug-sensitive", "drug-resistant"],
            "bone marrow transplantation":  ["wild-type", "hypomorphic"],
            "antiretroviral therapy":       ["drug-sensitive", "drug-resistant"],
            "enzyme replacement therapy":   ["wild-type", "hypomorphic"],
            "plasma exchange":              ["drug-sensitive", "drug-resistant"],
            "monoclonal antibody therapy":  ["drug-resistant", "mutant"],
            "phototherapy":                 ["drug-sensitive", "drug-resistant"],
            "anti-TNF therapy":             ["drug-resistant", "mutant"],
            "hormone suppression therapy":  ["drug-sensitive", "drug-resistant"]
        }
    },

    "library_selection": {
        "tissue_type": {
            "epidermal tissue":              ["ribodepletion", "targeted capture"],
            "dermal connective tissue":      ["total RNA", "ribodepletion"],
            "bone tissue":                   ["total RNA", "targeted capture"],
            "hepatic sinusoidal tissue":     ["ribodepletion", "small RNA-free"],
            "islet of Langerhans tissue":    ["targeted capture", "total RNA"],
            "alveolar tissue":               ["small RNA-free", "ribodepletion"],
            "lymphoid follicle tissue":      ["targeted capture", "total RNA"],
            "choroid plexus tissue":         ["ribodepletion", "small RNA-free"],
            "germinal center tissue":        ["total RNA", "targeted capture"]
        }
    },

    "library_source": {
        "cell_type": {
            "melanocytes":                     ["exosomal RNA", "cytosolic RNA"],
            "keratinocytes":                   ["cytosolic RNA", "nuclei sequencing"],
            "osteoclasts":                     ["total cell extract", "mitochondrial RNA"],
            "osteoblasts":                     ["total cell extract", "mitochondrial RNA"],
            "Kupffer cells":                   ["exosomal RNA", "cytosolic RNA"],
            "Langerhans cells":                ["nuclei sequencing", "cytosolic RNA"],
            "pancreatic beta cells":           ["mitochondrial RNA", "total cell extract"],
            "alveolar macrophages":            ["exosomal RNA", "cytosolic RNA"],
            "choroid plexus epithelial cells": ["nuclei sequencing", "cytosolic RNA"],
            "germinal center B cells":         ["exosomal RNA", "nuclei sequencing"]
        }
    },

    "donor_information": {
        "organ": {
            "bladder":        ["male donor", "smoker", "28-year-old"],
            "thyroid":        ["60-year-old", "female", "pregnant woman"],
            "adrenal gland":  ["male donor", "vegetarian", "28-year-old"],
            "lymph node":     ["child", "senior citizen", "male donor"],
            "prostate":       ["male donor", "60-year-old", "diabetic"],
            "placenta":       ["pregnant woman", "female", "child"],
            "pituitary gland":["female", "unknown", "senior citizen"],
            "tonsil":         ["male donor", "child", "child under 5"],
            "salivary gland": ["28-year-old", "athlete", "vegetarian"],
            "pancreatic duct":["BMI 22", "diabetic", "senior citizen"]
        }
    }
}

###################################################################################################
# MAIN

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

base_combos = []
while len(base_combos) < 50:
    combo = {}
    ct = random.choice(CELL_TYPE)
    combo['cell_type'] = ct
    combo['tissue_type'] = random.choice(
        SEM['cell_type']['tissue_type'].get(ct, TISSUE_TYPE)
    )
    org = random.choice(
        SEM['cell_type']['organ'].get(ct, ORGAN)
    )
    combo['organ'] = org
    cl_opts = [
        cl for cl, orgs in SEM['cell_line']['organ'].items()
        if org in orgs
    ]
    if cl_opts:
        combo['cell_line'] = random.choice(cl_opts)
    else:
        combo['cell_line'] = random.choice(CELL_LINE)
    ds_opts = [
        d for d, orgs in SEM['disease']['organ'].items()
        if org in orgs
    ]
    combo['disease'] = random.choice(ds_opts or DISEASE)
    combo['treatment'] = random.choice(
        SEM['treatment']['disease'].get(combo['disease'], TREATMENT)
    )
    combo['treatment_time'] = random.choice(
        SEM['treatment_time']['treatment'].get(combo['treatment'], TREAT_TIME)
    )
    combo['response'] = random.choice(
        SEM['response']['treatment'].get(combo['treatment'], RESPONSE)
    )
    combo['host_phenotype'] = random.choice(
        SEM['host_phenotype']['treatment'].get(combo['treatment'], HOST_PHENO)
    )
    combo['library_selection'] = random.choice(SEM['library_selection']['tissue_type'].get(combo['tissue_type'], LIB_SEL))
    combo['library_source'] = random.choice(SEM['library_source']['cell_type'].get(combo['cell_type'], LIB_SRC))
    combo['donor_information']   = random.choice(DONOR_INFO)
    combo['instrument_platform'] = random.choice(PLATFORMS)
    donor_opts = [d for d in DONOR_INFO
                  if org in SEM.get("donor_information", {}).get("organ", {}).get(d, ORGAN)]
    combo["donor_information"] = random.choice(donor_opts or DONOR_INFO)

    base_combos.append(combo)

records = []
for _ in range(5000):
    combo  = random.choice(base_combos)
    tpl     = random.choice(corrected_tpl)[0]
    phrase  = tpl
    for cat, val in combo.items():
        disp = choose_disp(val)
        phrase = re.sub(re.escape(PH[cat]), disp, phrase)
    rec = {'phrase_text': phrase}
    rec.update(combo)
    records.append(rec)

with open(EXPANDED_PATH, 'w') as f:
    json.dump([r['phrase_text'] for r in records], f, indent=2)
df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
