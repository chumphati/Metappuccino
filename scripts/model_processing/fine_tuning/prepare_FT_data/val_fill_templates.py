import os
import json
import random
import pandas as pd
import re

DATA_PATH = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/data_templates_training/metadata_templates_val_withoutkeys.json'
EXPANDED_PATH = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/metadata_templates_val_expanded.json'
OUTPUT_CSV  = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/val_metadata_replaced_table.csv'

CATEGORIES = [
    "library_selection", "sequencing_source",
    "organ", "biopsy_site", "biopsy_type",
    "cell_line", "cell_type", "disease", "treatment",
    "treatment_time", "response", "age", "sex", "ethnicity", "localization", "is_cancer"
]

LIB_SEL = ["small RNA", "other"]
SEQ_SRC = ["single-cell", "bulk", "spatial"]
BIOPSY_SITE = ["blood", "eye", "ear", "ovaries", "cartilage", "esophagus"]
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
        "chondrocyte population",
        "articular cartilage cells",
        "cartilaginous cells",
    ],
    "chondrocytes": [
        "cartilage chondrocytes",
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
        "RGC",
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
        "female germ cells",
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
        "spermatogonia",
    ],
    "spermatocytes": [
        "primary spermatocytes",
        "secondary spermatocytes",
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
        "cartilage tissue",
    ],
    "articular cartilage": [
        "articular hyaline cartilage",
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
        "endometriosis",
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
    "female":                 ["F", "woman", "female", "female donor"],
    "Mediterranean":          ["Med descent", "Southern European", "Med Sea region", "Mediterranean origin", "Mediterranean"],
    "normal":                 ["healthy", "control", "baseline", "reference", "normal"],
    "39 years":               ["39 y.o.", "age 39", "39-year-old", "39yrs", "39 years"],
    "American Indian":        ["Native American", "Indigenous American", "American Indigenous", "Native Amer", "American Indian"],
    "Pacific Islander":       ["Polynesian", "Melanesian", "Pacific descent", "Pacific Island"],
    "36 years":               ["36 y.o.", "age 36", "36-year-old", "36yrs", "36 years"],
    "male":                   ["M", "man", "male", "male donor"],
    "unknown":                ["NA", "not available", "unspecified", "missing data", "unknown"],
    "metastasis": ["metastasis", "Metastasis", "MET", "secondary tumor"],
    "blood": ["blood", "Blood sample", "peripheral blood", "venous blood"],
    "spatial": ["spatial", "spatial-seq", "spatial RNA-seq", "Visium"]
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
            "blood": ["blood", "ovaries", "testes", "uterus", "cartilage", "esophagus", "eye", "ear"],
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
    },

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
target_n = 400
expanded = random.choices(templates, k=max(1200, target_n*3))
with open(EXPANDED_PATH, 'w') as f:
    json.dump(expanded, f, indent=2)

CANCER_DISEASES = {
    "esophageal squamous cell carcinoma",
    "testicular germ cell tumor"
}

def eq_quota(values, total):
    k=len(values); base=total//k; r=total-base*k
    lst=list(values); random.shuffle(lst)
    q={v:base for v in lst}
    for v in lst[:r]: q[v]+=1
    return q

NOISE = [
    "sequencer lane rebalanced for duplex bias","insert size ~320bp","sample barcoded with UMIs","QC passed at RIN 8.1",
    "library molarity 7.5 nM","coverage uniformity within 1.3x","FASTQ trimmed (Phred≥30)","adapter clipping enabled",
    "lane bleedthrough negligible","5' bias not detected","batch randomization across plates","ERCC spike-ins present",
    "index-hopping mitigated","PCR duplicates removed","alignment on GRCh38","multi-mapping reads <2%","FRiP 0.21",
    "median insert 286","sequencing kit v3","freeze-thaw cycles=1","lane clustering density nominal","negative control clean",
    "RNase-free handling confirmed","unique fragments retained","flowcell ID anonymized"
]

def fuzzy_token_patterns(values):
    pats = []
    for v in values:
        base = re.sub(r'\s+', r'\\s*', re.escape(v))
        base = base.replace(r'\-', r'[-_ ]*').replace(r'_', r'[_ ]*')
        pats.append(r'(?i)(?<![A-Za-z0-9])' + base + r'(?![A-Za-z0-9])')
        if re.search(r'\d', v):
            letters = re.sub(r'[^A-Za-z]+', '', v)
            digits  = ''.join(re.findall(r'\d+', v))
            if letters and digits:
                pats.append(r'(?i)(?<![A-Za-z0-9])' + re.escape(letters) + r'\s*[-_ ]*' + re.escape(digits) + r'(?![A-Za-z0-9])')
    return pats

def allowed_terms(value):
    return set([value] + SYNONYMS.get(value, []))

def sanitize_phrase(phrase, rec):
    keep = {
        "organ": allowed_terms(rec["organ"]),
        "disease": allowed_terms(rec["disease"]),
        "cell_line": allowed_terms(rec["cell_line"]),
        "cell_type": allowed_terms(rec["cell_type"]),
        "treatment": allowed_terms(rec["treatment"]),
        "biopsy_site": allowed_terms(rec["biopsy_site"]),
        "sequencing_source": allowed_terms(rec["sequencing_source"]),
        "library_selection": allowed_terms(rec["library_selection"]),
    }
    forbidden = []
    all_sets = {
        "organ": set(ORGAN),
        "disease": set(DISEASE),
        "cell_line": set(CELL_LINE),
        "cell_type": set(CELL_TYPE),
        "treatment": set(TREATMENT),
        "biopsy_site": set(BIOPSY_SITE),
        "sequencing_source": set(SEQ_SRC),
        "library_selection": set(LIB_SEL),
    }
    for cat, universe in all_sets.items():
        for val in universe:
            if val in keep.get(cat,set()):
                continue
            forbidden.extend([val] + SYNONYMS.get(val, []))
    patterns = fuzzy_token_patterns(forbidden)
    for pat in patterns:
        phrase = re.sub(pat, '', phrase)
    phrase = re.sub(r'\s{2,}', ' ', phrase).strip().strip(',').strip()
    return phrase

def normalize_is_cancer(d):
    return "true" if d in CANCER_DISEASES else "false"

def pick_template(expanded_list):
    item = random.choice(expanded_list)
    if isinstance(item, (list, tuple)):
        return item[0]
    if isinstance(item, dict):
        return item.get("template", next(iter(item.values())))
    return item

def allowed_sources_for_org(org):
    if org in {"blood","bone marrow"}:
        return [x for x in SEQ_SRC if x!="spatial"]
    return list(SEQ_SRC)

def allowed_libraries_for_source(src):
    if src in {"single-cell","spatial"}:
        return ["other"]
    # bulk
    return ["small RNA","other"]

def allowed_celltypes_for_org(org):
    cands = [ct for ct, orgs in SEM["cell_type"]["organ"].items() if org in orgs]
    return cands if cands else list(CELL_TYPE)

def allowed_celllines_for_org(org):
    cands = [cl for cl, orgs in SEM["cell_line"]["organ"].items() if org in orgs]
    return cands if cands else list(CELL_LINE)

def primary_sites_for_org(org):
    sites = [site for site, orgs in SEM["biopsy_site"]["organ"].items() if org in orgs]
    return sites if sites else list(BIOPSY_SITE)

def metastasis_sites_for_org(org):
    return [site for site in BIOPSY_SITE if site != org]

def allowed_treatments_for_disease(d):
    feas = [t for t, ds in SEM["treatment"]["disease"].items() if d in ds and t in TREATMENT]
    return feas if feas else list(TREATMENT)

def allowed_tt_for_treatment(tr):
    return SEM["treatment_time"]["treatment"].get(tr, TREAT_TIME)

def allowed_rr_for_treatment(tr):
    return SEM["response"]["treatment"].get(tr, RESPONSE)

def allowed_sex(d, org):
    by_d = SEM["sex"]["disease"].get(d, [])
    by_o = SEM["sex"]["organ"].get(org, [])
    if by_d and by_o:
        inter = sorted(set(by_d) & set(by_o))
        if inter: return inter
    union = sorted(set(by_d) | set(by_o))
    return union if union else list(SEX)

# ----- quotas -----
quota = {cat: eq_quota(vals, target_n) for cat, vals in CATEGORY_LISTS.items()}
remaining = {k: {v:c for v,c in q.items()} for k,q in quota.items()}

def pop_any(cat):
    pool=[v for v,c in remaining[cat].items() if c>0]
    if not pool: return None
    v=random.choice(pool); remaining[cat][v]-=1; return v

def pop_from(cat, valid):
    pool=[v for v in valid if remaining[cat].get(v,0)>0]
    if pool:
        v=random.choice(pool); remaining[cat][v]-=1; return v
    return pop_any(cat)

def phrase_with_context(phrase, rec):
    # comme le train: injecter quelques catégories ciblées + NOISE, puis sanitize
    cats = ['biopsy_site','organ','response','treatment','disease','cell_type','cell_line']
    for cat in cats:
        val = rec[cat]
        alt = random.choice(SYNONYMS.get(val,[val])) if random.random()<0.8 else val
        if cat in CTX_WRAP:
            alt = random.choice(CTX_WRAP[cat]).format(val=alt)
        phrase = inject_value(phrase, alt)
    for _ in range(random.randint(3,5)):
        phrase = inject_value(phrase, random.choice(NOISE))
    phrase = sanitize_phrase(phrase, rec)
    return phrase

rows = []

for i in range(target_n):
    d = pop_any('disease')
    if d is None: break

    isc = normalize_is_cancer(d)
    # rééquilibrage 50/50 si quota saturé
    if remaining['is_cancer'][isc]==0:
        alt = 'false' if isc=='true' else 'true'
        if remaining['is_cancer'][alt]>0:
            isc = alt
    remaining['is_cancer'][isc]-=1

    organs = SEM["disease"]["organ"].get(d, ORGAN) or ORGAN
    org = pop_from('organ', organs) or random.choice(ORGAN)

    ct = pop_from('cell_type', allowed_celltypes_for_org(org)) or random.choice(CELL_TYPE)
    cl = pop_from('cell_line', allowed_celllines_for_org(org)) or random.choice(CELL_LINE)

    src = pop_from('sequencing_source', allowed_sources_for_org(org)) or random.choice(SEQ_SRC)
    lib = pop_from('library_selection', allowed_libraries_for_source(src)) or random.choice(LIB_SEL)

    # Biopsy type/site avec logique partagée
    bt = pop_from('biopsy_type', BIOPSY_TYPE) or random.choice(BIOPSY_TYPE)
    if bt == 'blood':
        bs = pop_from('biopsy_site', ['blood']) or 'blood'
    elif bt == 'primary':
        bs = pop_from('biopsy_site', primary_sites_for_org(org)) or pop_any('biopsy_site') or random.choice(BIOPSY_SITE)
    else:  # metastasis
        if isc == 'true':
            mets = metastasis_sites_for_org(org)
            bs = pop_from('biopsy_site', mets) or pop_from('biopsy_site', primary_sites_for_org(org)) or pop_any('biopsy_site') or random.choice(BIOPSY_SITE)
            if bs in primary_sites_for_org(org):
                bt = 'primary'
        else:
            bt = 'primary'
            bs = pop_from('biopsy_site', primary_sites_for_org(org)) or pop_any('biopsy_site') or random.choice(BIOPSY_SITE)

    # spatial doit être site==org
    if src == 'spatial' and bs != org:
        bt = 'primary'
        bs = pop_from('biopsy_site', [org]) or bs  # si possible, forcer site==org

    tr = pop_from('treatment', allowed_treatments_for_disease(d)) or random.choice(TREATMENT)
    tt = pop_from('treatment_time', allowed_tt_for_treatment(tr)) or random.choice(TREAT_TIME)
    rr = pop_from('response',      allowed_rr_for_treatment(tr))  or random.choice(RESPONSE)

    sx_allowed = allowed_sex(d, org)
    sex = pop_from('sex', sx_allowed) or random.choice(SEX)

    age = pop_any('age') or random.choice(AGE)
    eth = pop_any('ethnicity') or random.choice(ETHNICITY)
    loc = pop_any('localization') or random.choice(LOCALIZATION)

    record = {
        "library_selection":lib,
        "sequencing_source":src,
        "organ":org,
        "biopsy_site":bs,
        "biopsy_type":bt,
        "cell_line":cl,
        "cell_type":ct,
        "disease":d,
        "treatment":tr,
        "treatment_time":tt,
        "response":rr,
        "age":age,
        "sex":sex,
        "ethnicity":eth,
        "localization":loc,
        "is_cancer":isc
    }

    template = pick_template(expanded)
    phrase = template if isinstance(template, str) else str(template)
    record["phrase"] = phrase_with_context(phrase, record)
    rows.append(record)

pd.DataFrame(rows, columns=CATEGORIES+["phrase"]).to_csv(OUTPUT_CSV, index=False)
