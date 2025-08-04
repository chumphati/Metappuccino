import os
import json
import random
import pandas as pd
import re

DATA_PATH = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/data_templates_training/metadata_templates_test_withoutkeys.json'
EXPANDED_PATH = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/metadata_templates_test_expanded.json'
OUTPUT_CSV  = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/test_metadata_replaced_table.csv'

CATEGORIES = [
    "library_selection", "sequencing_source",
    "organ", "biopsy_site", "biopsy_type",
    "cell_line", "cell_type", "disease", "treatment",
    "treatment_time", "response", "age", "sex", "ethnicity", "localization", "is_cancer"
]

LIB_SEL = ["small RNA", "other"]
SEQ_SRC = ["single-cell", "bulk", "spatial"]
BIOPSY_SITE = ["blood", "eye", "ear", "ovaries", "cartilage"]
BIOPSY_TYPE = ['primary', 'metastasis', 'blood']
CELL_LINE = ["KYSE-30", "TE-1", "ARPE-19", "HEI-OC1", "KGN", "NT2/D1", "Ishikawa", "ECC-1", "SW1353", "ATDC5"]
CELL_TYPE = [
    "cartilage cells", "chondrocytes", "esophageal epithelial cells", "retinal ganglion cells",
    "photoreceptor cells", "retinal pigment epithelial cells", "cochlear hair cells", "granulosa cells",
    "theca cells", "oocytes", "Leydig cells", "Sertoli cells", "spermatogonia", "spermatocytes",
    "endometrial epithelial cells", "decidual cells"
]
ORGAN = ["esophagus", "eye", "ear", "ovaries", "testes", "uterus", "cartilage"]
DISEASE = [
    "Barrett's esophagus", "esophageal squamous cell carcinoma", "age-related macular degeneration",
    "glaucoma", "retinitis pigmentosa", "sensorineural hearing loss", "polycystic ovary syndrome",
    "testicular germ cell tumor", "endometriosis", "osteoarthritis"
]
TREATMENT = [
    "esophagectomy", "photodynamic therapy", "anti-VEGF intravitreal injection", "cochlear implant",
    "laparoscopic ovarian cystectomy", "in vitro fertilization", "testosterone replacement therapy",
    "hysterectomy", "arthroscopic debridement", "mesenchymal stem cell cartilage regeneration"
]
TREAT_TIME = ["72 hours", "2 weeks", "6 months"]
RESPONSE = ["relapse", "adverse event"]
AGE = ["39 years", "36 years", "unknown"]
SEX = ["male", "female"]
ETHNICITY = ["American Indian", "Pacific Islander", "unknown"]
LOCALIZATION = ["Mediterranean", "unknown"]
IS_CANCER = ["true","false"]

SYNONYMS = {
    "cartilage cells": [
        "cartilage cell population",
        "chondrocyte population",
        "articular cartilage cells",
        "cartilaginous cells",
        "cartilage cells",
    ],
    "chondrocytes": [
        "cartilage chondrocytes",
        "cartilage-forming cells",
        "chondrocyte cells",
        "cartilage-producing cells",
        "chondrocytes",
    ],
    "esophageal epithelial cells": [
        "esophageal epithelium cells",
        "esophageal lining cells",
        "esophagus epithelial cells",
        "esophageal mucosal cells",
        "esophageal epithelial cells",
    ],
    "retinal ganglion cells": [
        "RGCs",
        "ganglion cells of retina",
        "retina ganglion neurons",
        "optic-nerve ganglion cells",
        "retinal ganglion cells",
    ],
    "photoreceptor cells": [
        "retinal photoreceptors",
        "rod and cone cells",
        "light-sensing cells",
        "visual photoreceptor cells",
        "photoreceptor cells",
    ],
    "retinal pigment epithelial cells": [
        "RPE cells",
        "pigmented retinal epithelial cells",
        "retina RPE cells",
        "retinal pigmented epithelium cells",
        "retinal pigment epithelial cells",
    ],
    "cochlear hair cells": [
        "inner ear hair cells",
        "auditory hair cells",
        "organ of Corti hair cells",
        "sensory hair cells",
        "cochlear hair cells",
    ],
    "granulosa cells": [
        "ovarian granulosa cells",
        "follicular granulosa cells",
        "granulosa cell layer",
        "granulosa cell population",
        "granulosa cells",
    ],
    "theca cells": [
        "ovarian theca cells",
        "follicular theca cells",
        "theca interna cells",
        "steroidogenic theca cells",
        "theca cells",
    ],
    "oocytes": [
        "egg cells",
        "female germ cells",
        "ovum",
        "oocyte cells",
        "oocytes",
    ],
    "Leydig cells": [
        "interstitial Leydig cells",
        "testicular Leydig cells",
        "Leydig cell population",
        "testis Leydig cells",
        "Leydig cells",
    ],
    "Sertoli cells": [
        "nurse cells",
        "testicular Sertoli cells",
        "seminiferous Sertoli cells",
        "Sertoli cell population",
        "Sertoli cells",
    ],
    "spermatogonia": [
        "male germ stem cells",
        "spermatogonial cells",
        "testicular spermatogonia",
        "undifferentiated germ cells",
        "spermatogonia",
    ],
    "spermatocytes": [
        "primary spermatocytes",
        "secondary spermatocytes",
        "meiotic germ cells",
        "spermatocyte cells",
        "spermatocytes",
    ],
    "endometrial epithelial cells": [
        "uterine epithelial cells",
        "endometrial lining cells",
        "endometrium epithelial cells",
        "uterine endometrial cells",
        "endometrial epithelial cells",
    ],
    "decidual cells": [
        "maternal decidual cells",
        "decidua cells",
        "decidual stromal cells",
        "pregnancy decidual cells",
        "decidual cells",
    ],
    "cartilage tissue": [
        "cartilaginous tissue",
        "cartilage matrix",
        "hyaline cartilage",
        "articular cartilage tissue",
        "cartilage tissue",
    ],
    "articular cartilage": [
        "joint cartilage",
        "articular hyaline cartilage",
        "synovial cartilage",
        "joint surface cartilage",
        "articular cartilage",
    ],
    "esophageal mucosa": [
        "esophagus mucosa",
        "mucosal layer of esophagus",
        "esophageal lining",
        "esophagus mucosal tissue",
        "esophageal mucosa",
    ],
    "esophageal submucosa": [
        "esophagus submucosa",
        "submucosal layer of esophagus",
        "esophageal connective layer",
        "esophageal submucosal tissue",
        "esophageal submucosa",
    ],
    "retinal tissue": [
        "retina tissue",
        "neural retina",
        "ocular retinal tissue",
        "retinal layer",
        "retinal tissue",
    ],
    "cochlear sensory epithelium": [
        "organ of Corti",
        "cochlear epithelium",
        "auditory sensory epithelium",
        "inner ear sensory epithelium",
        "cochlear sensory epithelium",
    ],
    "ovarian follicle tissue": [
        "ovarian follicular tissue",
        "follicle wall",
        "ovarian follicle",
        "follicular tissue",
        "ovarian follicle tissue",
    ],
    "testicular seminiferous tubules": [
        "seminiferous tubule tissue",
        "testis seminiferous tubules",
        "germinal tubules",
        "sperm-producing tubules",
        "testicular seminiferous tubules",
    ],
    "endometrial tissue": [
        "uterine endometrium",
        "endometrium tissue",
        "uterine lining tissue",
        "endometrial layer",
        "endometrial tissue",
    ],
    "myometrium": [
        "uterine muscle layer",
        "uterine myometrium",
        "myometrial tissue",
        "smooth muscle of uterus",
        "myometrium",
    ],
    "KYSE-30": [
        "KYSE30 cells",
        "KYSE-30 esophageal carcinoma cells",
        "KYSE 30 line",
        "esophageal squamous line KYSE-30",
        "KYSE-30",
    ],
    "TE-1": [
        "TE1 cells",
        "TE-1 esophageal carcinoma cells",
        "TE 1 line",
        "esophageal cancer line TE-1",
        "TE-1",
    ],
    "ARPE-19": [
        "ARPE19 cells",
        "ARPE-19 retinal pigment epithelial cells",
        "ARPE 19 line",
        "human RPE line ARPE-19",
        "ARPE-19",
    ],
    "HEI-OC1": [
        "HEI OC1 cells",
        "HEI-OC1 inner ear cells",
        "HEI OC 1 line",
        "cochlear line HEI-OC1",
        "HEI-OC1",
    ],
    "KGN": [
        "KGN cells",
        "KGN granulosa cell line",
        "KGN ovarian cells",
        "human granulosa line KGN",
        "KGN",
    ],
    "NT2/D1": [
        "NT2D1 cells",
        "NT2/D1 testicular carcinoma cells",
        "NT2 D1 line",
        "pluripotent line NT2/D1",
        "NT2/D1",
    ],
    "Ishikawa": [
        "Ishikawa cells",
        "Ishikawa endometrial cells",
        "endometrial carcinoma line Ishikawa",
        "uterine cancer line Ishikawa",
        "Ishikawa",
    ],
    "ECC-1": [
        "ECC1 cells",
        "ECC-1 endometrial carcinoma cells",
        "ECC 1 line",
        "endometrial cancer line ECC-1",
        "ECC-1",
    ],
    "SW1353": [
        "SW-1353 cells",
        "SW1353 chondrosarcoma cells",
        "SW 1353 line",
        "cartilage tumor line SW1353",
        "SW1353",
    ],
    "ATDC5": [
        "ATDC-5 cells",
        "ATDC5 chondrogenic cells",
        "ATDC 5 line",
        "mouse chondroprogenitor line ATDC5",
        "ATDC5",
    ],
    "esophagus": [
        "oesophagus",
        "esophageal organ",
        "esophagus",
    ],
    "eye": [
        "ocular organ",
        "eyeball",
        "visual organ",
        "globe",
        "eye",
    ],
    "ear": [
        "auditory organ",
        "auricle",
        "hearing organ",
        "otologic organ",
        "ear",
    ],
    "ovaries": [
        "ovary organs",
        "female gonads",
        "ovarian glands",
        "ovary pair",
        "ovaries",
    ],
    "testes": [
        "male gonads",
        "testicles",
        "testis organs",
        "male testicular organs",
        "testes",
    ],
    "uterus": [
        "womb",
        "uterine organ",
        "uterine cavity",
        "female uterus",
        "uterus",
    ],
    "cartilage": [
        "cartilaginous tissue",
        "joint cartilage",
        "cartilage matrix",
        "cartilage structure",
        "cartilage",
    ],
    "Barrett's esophagus": [
        "Barrett oesophagus",
        "intestinal metaplasia of esophagus",
        "Barrett syndrome",
        "Barrett disease",
        "Barrett's esophagus",
    ],
    "esophageal squamous cell carcinoma": [
        "ESCC",
        "esophageal SCC",
        "squamous carcinoma of esophagus",
        "oesophageal squamous carcinoma",
        "esophageal squamous cell carcinoma",
    ],
    "age-related macular degeneration": [
        "AMD",
        "ARMD",
        "macular degeneration",
        "senile macular degeneration",
        "age-related macular degeneration",
    ],
    "glaucoma": [
        "ocular hypertension",
        "glaucomatous optic neuropathy",
        "glaucoma disease",
        "increased intraocular pressure",
        "glaucoma",
    ],
    "retinitis pigmentosa": [
        "RP",
        "pigmentary retinopathy",
        "retinal dystrophy",
        "retinitis pigmentosa disease",
        "retinitis pigmentosa",
    ],
    "sensorineural hearing loss": [
        "SNHL",
        "nerve deafness",
        "sensorineural deafness",
        "inner ear hearing loss",
        "sensorineural hearing loss",
    ],
    "polycystic ovary syndrome": [
        "PCOS",
        "Stein–Leventhal syndrome",
        "polycystic ovarian disease",
        "hyperandrogenic syndrome",
        "polycystic ovary syndrome",
    ],
    "testicular germ cell tumor": [
        "TGCT",
        "testicular GCT",
        "germ cell cancer of testis",
        "testicular germ cell cancer",
        "testicular germ cell tumor",
    ],
    "endometriosis": [
        "endometrial implants",
        "ectopic endometrium",
        "endometriotic lesions",
        "pelvic endometriosis",
        "endometriosis",
    ],
    "osteoarthritis": [
        "OA",
        "degenerative joint disease",
        "osteoarthrosis",
        "arthrosis deformans",
        "osteoarthritis",
    ],
    "esophagectomy": [
        "esophageal resection",
        "surgical removal of esophagus",
        "esophagogastrectomy",
        "oesophagectomy",
        "esophagectomy",
    ],
    "photodynamic therapy": [
        "PDT",
        "light-activated therapy",
        "photosensitizer therapy",
        "photochemotherapy",
        "photodynamic therapy",
    ],
    "anti-VEGF intravitreal injection": [
        "VEGF inhibitor injection",
        "intravitreal anti-VEGF",
        "anti-VEGF shot",
        "anti-vascular endothelial growth factor therapy",
        "anti-VEGF intravitreal injection",
    ],
    "cochlear implant": [
        "CI",
        "cochlear prosthesis",
        "inner ear implant",
        "electronic ear implant",
        "cochlear implant",
    ],
    "laparoscopic ovarian cystectomy": [
        "lap ovarian cyst removal",
        "laparoscopic cyst removal",
        "ovarian cystectomy",
        "lap cyst excision",
        "laparoscopic ovarian cystectomy",
    ],
    "in vitro fertilization": [
        "IVF",
        "test-tube fertilization",
        "assisted fertilization",
        "in vitro conception",
        "in vitro fertilization",
    ],
    "testosterone replacement therapy": [
        "TRT",
        "androgen replacement therapy",
        "testosterone therapy",
        "hormone replacement testosterone",
        "testosterone replacement therapy",
    ],
    "hysterectomy": [
        "uterus removal surgery",
        "uterine resection",
        "surgical hysterectomy",
        "removal of womb",
        "hysterectomy",
    ],
    "arthroscopic debridement": [
        "arthroscopy debridement",
        "joint debridement",
        "arthroscopic cleaning",
        "arthroscopic lavage",
        "arthroscopic debridement",
    ],
    "mesenchymal stem cell cartilage regeneration": [
        "MSC cartilage repair",
        "mesenchymal cell cartilage therapy",
        "stem cell cartilage regeneration",
        "MSC-based chondrogenesis",
        "mesenchymal stem cell cartilage regeneration",
    ],
    "small RNA":              ["sRNA", "smallRNA", "microRNA-seq", "miRNA profiling", "small RNA"],
    "other":                  ["miscellaneous", "alternative", "non-standard", "varied", "other"],
    "single-cell":            ["scRNA-seq", "single-cell sequencing", "single cell assay", "sc-seq", "single-cell"],
    "bulk":                   ["bulk RNA-seq", "population sequencing", "ensemble sequencing", "bulk assay", "bulk"],
    "72 hours":               ["72h", "3 days", "three-day", "seventy-two hours", "72 hours"],
    "2 weeks":                ["14 days", "fortnight", "biweekly", "two-week", "2 weeks"],
    "6 months":               ["half year", "26 weeks", "semiannual", "six-month", "6 months"],
    "relapse":                ["recurrence", "disease recurrence", "recurrent disease", "relapsing", "relapse"],
    "adverse event":          ["AE", "side effect", "unwanted reaction", "treatment-related event", "adverse event"],
    "female":                 ["F", "woman", "female donor", "female"],
    "Mediterranean":          ["Med descent", "Southern European", "Med Sea region", "Mediterranean origin", "Mediterranean"],
    "normal":                 ["healthy", "control", "baseline", "reference", "normal"],
    "39 years":               ["39 y.o.", "age 39", "39-year-old", "39yrs", "39 years"],
    "American Indian":        ["Native American", "Indigenous American", "American Indigenous", "Native Amer", "American Indian"],
    "Pacific Islander":       ["Polynesian", "Micronesian", "Melanesian", "Pacific descent", "Pacific Islander"],
    "36 years":               ["36 y.o.", "age 36", "36-year-old", "36yrs", "36 years"],
    "male":                   ["M", "man", "male donor", "male"],
    "unknown":                ["NA", "not available", "unspecified", "missing data", "unknown"],
    "metastasis": ["metastasis", "Metastasis", "MET", "secondary tumor"],
    "blood": ["blood", "Blood sample", "peripheral blood", "venous blood"],
    "spatial": ["spatial", "spatial‑seq", "spatial RNA‑seq", "Visium"]
}

SEM = {
    "cell_type": {
        "organ": {
            "cartilage cells": ["cartilage"],
            "chondrocytes": ["cartilage"],
            "esophageal epithelial cells": ["esophagus"],
            "retinal ganglion cells": ["eye"],
            "photoreceptor cells": ["eye"],
            "retinal pigment epithelial cells": ["eye"],
            "cochlear hair cells": ["ear"],
            "granulosa cells": ["ovaries"],
            "theca cells": ["ovaries"],
            "oocytes": ["ovaries"],
            "Leydig cells": ["testes"],
            "Sertoli cells": ["testes"],
            "spermatogonia": ["testes"],
            "spermatocytes": ["testes"],
            "endometrial epithelial cells": ["uterus"],
            "decidual cells": ["uterus"]
        }
    },
    "disease": {
        "organ": {
            "Barrett's esophagus": ["esophagus"],
            "esophageal squamous cell carcinoma": ["esophagus"],
            "age-related macular degeneration": ["eye"],
            "glaucoma": ["eye"],
            "retinitis pigmentosa": ["eye"],
            "sensorineural hearing loss": ["ear"],
            "polycystic ovary syndrome": ["ovaries"],
            "testicular germ cell tumor": ["testes"],
            "endometriosis": ["uterus"],
            "osteoarthritis": ["cartilage"]
        }
    },
    "cell_line": {
        "organ": {
            "KYSE-30": ["esophagus"],
            "TE-1": ["esophagus"],
            "ARPE-19": ["eye"],
            "HEI-OC1": ["ear"],
            "KGN": ["ovaries"],
            "NT2/D1": ["testes"],
            "Ishikawa": ["uterus"],
            "ECC-1": ["uterus"],
            "SW1353": ["cartilage"],
            "ATDC5": ["cartilage"]
        }
    },
    "treatment": {
        "disease": {
            "esophagectomy": ["esophageal squamous cell carcinoma", "Barrett's esophagus"],
            "photodynamic therapy": ["Barrett's esophagus"],
            "anti-VEGF intravitreal injection": ["age-related macular degeneration", "glaucoma"],
            "cochlear implant": ["sensorineural hearing loss"],
            "laparoscopic ovarian cystectomy": ["polycystic ovary syndrome"],
            "in vitro fertilization": ["polycystic ovary syndrome", "endometriosis"],
            "testosterone replacement therapy": ["testicular germ cell tumor"],
            "hysterectomy": ["endometriosis"],
            "arthroscopic debridement": ["osteoarthritis"],
            "mesenchymal stem cell cartilage regeneration": ["osteoarthritis"],
            "no treatment": ["none", "unknown"]
        }
    },
    "treatment_time": {
        "treatment": {
            "esophagectomy": ["72 hours", "2 weeks", "6 months"],
            "photodynamic therapy": ["72 hours", "2 weeks", "6 months"],
            "anti-VEGF intravitreal injection": ["72 hours", "2 weeks", "6 months"],
            "cochlear implant": ["72 hours", "2 weeks", "6 months"],
            "laparoscopic ovarian cystectomy": ["72 hours", "2 weeks", "6 months"],
            "in vitro fertilization": ["72 hours", "2 weeks", "6 months"],
            "testosterone replacement therapy": ["72 hours", "2 weeks", "6 months"],
            "hysterectomy": ["72 hours", "2 weeks", "6 months"],
            "arthroscopic debridement": ["72 hours", "2 weeks", "6 months"],
            "mesenchymal stem cell cartilage regeneration": ["72 hours", "2 weeks", "6 months"]
        }
    },
    "response": {
        "treatment": {
            "esophagectomy": ["relapse", "adverse event"],
            "photodynamic therapy": ["relapse", "adverse event"],
            "anti-VEGF intravitreal injection": ["relapse", "adverse event"],
            "cochlear implant": ["relapse", "adverse event"],
            "laparoscopic ovarian cystectomy": ["relapse", "adverse event"],
            "in vitro fertilization": ["relapse", "adverse event"],
            "testosterone replacement therapy": ["relapse", "adverse event"],
            "hysterectomy": ["relapse", "adverse event"],
            "arthroscopic debridement": ["relapse", "adverse event"],
            "mesenchymal stem cell cartilage regeneration": ["relapse", "adverse event"]
        }
    },
    "biopsy_site": {
        "organ": {
            "blood": ["blood", "ovaries", "testes", "uterus", "cartilage"],
            "eye": ["eye"],
            "ear": ["ear"],
            "ovaries": ["ovaries"],
            "cartilage": ["cartilage"],
            "esophagus": ["esophagus"],
        }
    },
    "library_source": {
        "cell_type": {
            "single-cell": [
                "granulosa cells", "photoreceptor cells", "cochlear hair cells",
                "endometrial epithelial cells", "spermatogonia"
            ],
            "bulk": [
                "cartilage cells", "decidual cells", "retinal pigment epithelial cells"
            ],
            "spatial": [
                "Leydig cells", "Sertoli cells", "retinal ganglion cells"
            ]
        }
    },
    "sex": {
        "disease": {
            "polycystic ovary syndrome": ["female"],
            "endometriosis": ["female"],
            "testicular germ cell tumor": ["male"]
        },
        "organ": {
            "ovaries": ["female"],
            "uterus": ["female"],
            "testes": ["male"]
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
        'histological samples were retrieved from the {val} for downstream analysis',
        'prior to RNA extraction, tissue fragments were isolated from the {val}',
        'the biopsy procedure targeted the {val} using minimally invasive techniques',
        'material harvested during surgery included sections of the {val}',
        'molecular profiling was initiated after collecting {val}-specific biopsies',
        'local anesthesia was applied before obtaining samples from the {val}',
        'fine-needle aspiration from the {val} yielded sufficient RNA',
        'the patient’s {val} was selected as the anatomical site for sample retrieval',
        'transcriptomic libraries were derived from biopsies of the {val}',
        'sectioning of the {val} was performed to isolate cellular components',
        'RNA was stabilized immediately following {val}-directed sampling',
        'genetic material from the {val} was considered representative of the pathology'
    ],
    'organ': [
        'gene dysregulation was prominently noted in pathways related to the {val}',
        'pathophysiological changes were concentrated in the {val}',
        'the {val} presented abnormal transcriptomic signatures compared to controls',
        'expression profiles implicated the {val} in disease progression',
        'differential analysis revealed organ-specific shifts in the {val}',
        'samples were enriched for genes characteristic of {val}-associated function',
        'inflammatory markers in the {val} were found to be upregulated',
        'focus was placed on the {val} due to its known involvement in similar conditions',
        'mutations affecting {val} integrity were cataloged',
        'transcriptional disruptions were mapped onto the {val} regulatory network',
        'several {val}-related genes were identified as differentially expressed',
        'analysis emphasized cellular pathways inherent to the {val}'
    ],
    'response': [
        'post-treatment evaluation indicated {val}',
        'follow-up imaging demonstrated {val}',
        'the clinical course was characterized by {val}',
        'physician assessment at follow-up documented {val}',
        'outcome monitoring recorded {val}',
        'safety surveillance noted {val}',
        'the patient trajectory showed evidence of {val}',
        'on-treatment review highlighted {val}',
        'end-of-cycle assessment reported {val}',
        'study visit documentation referenced {val}',
        'chart review confirmed {val} following intervention',
        'adjudication of outcomes concluded {val}',
        'longitudinal tracking revealed {val} after therapy',
        'the recorded endpoint was consistent with {val}',
        'patient-reported outcome logs captured {val}',
        'the adverse event log included {val}',
        'discharge summary listed {val} as the observed response',
        'clinical narrative described {val} during follow-up',
        'the response profile was summarized as {val}',
        'progress notes explicitly mentioned {val}'
    ]
}

def inject_value(sentence: str, insertion: str) -> str:
    words = sentence.split()
    idx = random.randint(0, len(words))
    return ' '.join(words[:idx] + [insertion] + words[idx:])

with open(DATA_PATH) as f:
    templates = json.load(f)
expanded = random.choices(templates, k=5000)
with open(EXPANDED_PATH, 'w') as f:
    json.dump(expanded, f, indent=2)

CANCER_DISEASES = {
    "esophageal squamous cell carcinoma",
    "testicular germ cell tumor"
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
        elif cat == 'sex':
            org = chosen.get('organ')
            disease = chosen.get('disease')
            opts_from_org = SEM["sex"]["organ"].get(org, [])
            opts_from_disease = SEM["sex"]["disease"].get(disease, [])
            candidates = list(set(opts_from_org) & set(opts_from_disease)) or opts_from_org or opts_from_disease or SEX
            raw = random.choice(candidates)

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
