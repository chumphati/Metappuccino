import os
import json
import random
import pandas as pd
import re

DATA_PATH = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/patrons_training/metadata_templates_val.json'
CORRECTED_PATH = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_ft_patron_semi/metadata_templates_val_corrected.json'
EXPANDED_PATH = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_ft_patron_semi/metadata_templates_val_expanded.json'
OUTPUT_CSV  = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_ft_patron_semi/val_metadata_replaced_table.csv'

CATEGORIES = [
    'cell_type', 'tissue_type', 'cell_line', 'organ', 'disease',
    'host_phenotype', 'library_selection', 'library_source',
    'treatment', 'treatment_time', 'response', 'donor_information', 'instrument_platform'
]
PH = {cat: f"__{cat.upper()}__" for cat in CATEGORIES}

CELL_TYPE = ["cartilage cells", "chondrocytes", "esophageal epithelial cells", "retinal ganglion cells", "photoreceptor cells", "retinal pigment epithelial cells", "cochlear hair cells", "granulosa cells", "theca cells", "oocytes", "Leydig cells", "Sertoli cells", "spermatogonia", "spermatocytes", "endometrial epithelial cells", "decidual cells"]
TISSUE_TYPE = ["cartilage tissue", "articular cartilage", "esophageal mucosa", "esophageal submucosa", "retinal tissue", "cochlear sensory epithelium", "ovarian follicle tissue", "testicular seminiferous tubules", "endometrial tissue", "myometrium"]
CELL_LINE = [ "KYSE-30", "TE-1", "ARPE-19", "HEI-OC1", "KGN", "NT2/D1", "Ishikawa", "ECC-1", "SW1353", "ATDC5"]
ORGAN = ["esophagus", "eye", "ear", "ovaries", "testes", "uterus", "cartilage"]
DISEASE = ["Barrett's esophagus", "esophageal squamous cell carcinoma", "age-related macular degeneration", "glaucoma", "retinitis pigmentosa", "sensorineural hearing loss", "polycystic ovary syndrome", "testicular germ cell tumor", "endometriosis", "osteoarthritis"]
TREATMENT = ["esophagectomy", "photodynamic therapy", "anti-VEGF intravitreal injection", "cochlear implant", "laparoscopic ovarian cystectomy", "in vitro fertilization", "testosterone replacement therapy", "hysterectomy", "arthroscopic debridement", "mesenchymal stem cell cartilage regeneration"]
LIB_SEL = ["small RNA", "other"]
TREAT_TIME = ["72 hours", "2 weeks", "6 months"]
RESPONSE = ["relapse", "adverse event"]
DONOR_INFO = ["female", "pregnant", "Mediterranean", "normal", "39 years", "immunosuppressed", "American Indian", "athlete", "Pacific Islander", "36 years", "male", "ex-smoker", "unknown"]
HOST_PHENO = ["parental","persistent"]
LIB_SELECTION_VAL = ["small RNA", "other"]
LIB_SRC = ["single-cell","bulk"]
PLATFORMS = ["Illumina NovaSeq 9000","Illumina 500","NextSeq 500","MiSeq 200"]

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
    # ------------------------------------------------------------------
    # CELL_TYPE
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # TISSUE_TYPE
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # CELL_LINE
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # ORGAN
    # ------------------------------------------------------------------
    "esophagus": [
        "gullet",
        "food pipe",
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

    # ------------------------------------------------------------------
    # DISEASE
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # TREATMENT
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # LIBRARY SELECTION & SOURCE
    # ------------------------------------------------------------------
    "small RNA":              ["sRNA", "smallRNA", "microRNA-seq", "miRNA profiling", "small RNA"],
    "other":                  ["miscellaneous", "alternative", "non-standard", "varied", "other"],
    "single-cell":            ["scRNA-seq", "single-cell sequencing", "single cell assay", "sc-seq", "single-cell"],
    "bulk":                   ["bulk RNA-seq", "population sequencing", "ensemble sequencing", "bulk assay", "bulk"],

    # ------------------------------------------------------------------
    # TREATMENT TIME
    # ------------------------------------------------------------------
    "72 hours":               ["72h", "3 days", "three-day", "seventy-two hours", "72 hours"],
    "2 weeks":                ["14 days", "fortnight", "biweekly", "two-week", "2 weeks"],
    "6 months":               ["half year", "26 weeks", "semiannual", "six-month", "6 months"],

    # ------------------------------------------------------------------
    # RESPONSE
    # ------------------------------------------------------------------
    "relapse":                ["recurrence", "disease recurrence", "recurrent disease", "relapsing", "relapse"],
    "adverse event":          ["AE", "side effect", "unwanted reaction", "treatment-related event", "adverse event"],

    # ------------------------------------------------------------------
    # DONOR INFO
    # ------------------------------------------------------------------
    "female":                 ["F", "woman", "adult female", "female donor", "female"],
    "pregnant":               ["gestating", "expectant", "gravida", "preg", "pregnant"],
    "Mediterranean":          ["Med descent", "Southern European", "Med Sea region", "Mediterranean origin", "Mediterranean"],
    "normal":                 ["healthy", "control", "baseline", "reference", "normal"],
    "39 years":               ["39 y.o.", "age 39", "39-year-old", "39yrs", "39 years"],
    "immunosuppressed":       ["immunocompromised", "immune suppressed", "IS", "immune deficient", "immunosuppressed"],
    "American Indian":        ["Native American", "Indigenous American", "American Indigenous", "Native Amer", "American Indian"],
    "athlete":                ["sportsperson", "sportsman", "sportswoman", "competitive athlete", "athlete"],
    "Pacific Islander":       ["Polynesian", "Micronesian", "Melanesian", "Pacific descent", "Pacific Islander"],
    "36 years":               ["36 y.o.", "age 36", "36-year-old", "36yrs", "36 years"],
    "male":                   ["M", "man", "adult male", "male donor", "male"],
    "ex-smoker":              ["former smoker", "past smoker", "quit smoking", "ex tobacco user", "ex-smoker"],
    "unknown":                ["NA", "not available", "unspecified", "missing data", "unknown"],

    # ------------------------------------------------------------------
    # HOST PHENOTYPE
    # ------------------------------------------------------------------
    "parental":               ["wild type", "WT", "baseline", "native", "parental"],
    "persistent":             ["persister", "tolerant", "survivor cells", "drug-persistent", "persistent"],

    # ------------------------------------------------------------------
    # SEQUENCING PLATFORMS
    # ------------------------------------------------------------------
    "Illumina NovaSeq 9000":  ["NovaSeq 9000", "NS9000", "Illumina NS9000", "Illumina NovaSeq", "Illumina NovaSeq 9000"],
    "Illumina 500":           ["HiSeq 500", "HS500", "Illumina HiSeq 500", "Illumina HS500", "Illumina 500"],
    "NextSeq 500":            ["NS500", "Next Seq 500", "NextSeq-500", "Illumina NextSeq", "NextSeq 500"],
    "MiSeq 200":              ["MiSeq200", "MiSeq 200", "Illumina MiSeq", "Mi-Seq 200", "MiSeq 200"],
}

for main in list(SYNONYMS.keys()):
    SYNONYMS.setdefault(main, [])
    SYNONYMS[main].append(main)

def choose_disp(val):
    syns = SYNONYMS.get(val, [])
    if syns and random.random()<0.7:
        return random.choice(syns)
    return val

SEM = {
    "cell_type": {
        "tissue_type": {
            "cartilage cells":                ["cartilage tissue", "articular cartilage"],
            "chondrocytes":                   ["cartilage tissue", "articular cartilage"],
            "esophageal epithelial cells":    ["esophageal mucosa", "esophageal submucosa"],
            "retinal ganglion cells":         ["retinal tissue",   "retinal tissue"],
            "photoreceptor cells":            ["retinal tissue",   "retinal tissue"],
            "retinal pigment epithelial cells":["retinal tissue",  "retinal tissue"],
            "cochlear hair cells":            ["cochlear sensory epithelium", "cochlear sensory epithelium"],
            "granulosa cells":                ["ovarian follicle tissue", "ovarian follicle tissue"],
            "theca cells":                    ["ovarian follicle tissue", "ovarian follicle tissue"],
            "oocytes":                        ["ovarian follicle tissue", "ovarian follicle tissue"],
            "Leydig cells":                   ["testicular seminiferous tubules", "testicular seminiferous tubules"],
            "Sertoli cells":                  ["testicular seminiferous tubules", "testicular seminiferous tubules"],
            "spermatogonia":                  ["testicular seminiferous tubules", "testicular seminiferous tubules"],
            "spermatocytes":                  ["testicular seminiferous tubules", "testicular seminiferous tubules"],
            "endometrial epithelial cells":   ["endometrial tissue", "myometrium"],
            "decidual cells":                 ["endometrial tissue", "myometrium"]
        },
        "organ": {
            "cartilage cells":                ["cartilage"],
            "chondrocytes":                   ["cartilage"],
            "esophageal epithelial cells":    ["esophagus"],
            "retinal ganglion cells":         ["eye"],
            "photoreceptor cells":            ["eye"],
            "retinal pigment epithelial cells":["eye"],
            "cochlear hair cells":            ["ear"],
            "granulosa cells":                ["ovaries"],
            "theca cells":                    ["ovaries"],
            "oocytes":                        ["ovaries"],
            "Leydig cells":                   ["testes"],
            "Sertoli cells":                  ["testes"],
            "spermatogonia":                  ["testes"],
            "spermatocytes":                  ["testes"],
            "endometrial epithelial cells":   ["uterus"],
            "decidual cells":                 ["uterus"]
        }
    },

    "cell_line": {
        "organ": {
            "KYSE-30":   ["esophagus"],
            "TE-1":      ["esophagus"],
            "ARPE-19":   ["eye"],
            "HEI-OC1":   ["ear"],
            "KGN":       ["ovaries"],
            "NT2/D1":    ["testes"],
            "Ishikawa":  ["uterus"],
            "ECC-1":     ["uterus"],
            "SW1353":    ["cartilage"],
            "ATDC5":     ["cartilage"]
        },
        "disease": {
            "KYSE-30":   ["esophageal squamous cell carcinoma"],
            "TE-1":      ["esophageal squamous cell carcinoma"],
            "ARPE-19":   ["age-related macular degeneration"],
            "HEI-OC1":   ["sensorineural hearing loss"],
            "KGN":       ["polycystic ovary syndrome"],
            "NT2/D1":    ["testicular germ cell tumor"],
            "Ishikawa":  ["endometriosis"],
            "ECC-1":     ["endometriosis"],
            "SW1353":    ["osteoarthritis"],
            "ATDC5":     ["osteoarthritis"]
        }
    },

    "disease": {
        "organ": {
            "Barrett's esophagus":                 ["esophagus"],
            "esophageal squamous cell carcinoma":  ["esophagus"],
            "age-related macular degeneration":    ["eye"],
            "glaucoma":                            ["eye"],
            "retinitis pigmentosa":                ["eye"],
            "sensorineural hearing loss":          ["ear"],
            "polycystic ovary syndrome":           ["ovaries"],
            "testicular germ cell tumor":          ["testes"],
            "endometriosis":                       ["uterus"],
            "osteoarthritis":                      ["cartilage"]
        }
    },

    "treatment": {
        "disease": {
            "Barrett's esophagus":                 ["esophagectomy", "photodynamic therapy"],
            "esophageal squamous cell carcinoma":  ["esophagectomy", "photodynamic therapy"],
            "age-related macular degeneration":    ["anti-VEGF intravitreal injection"],
            "glaucoma":                            ["anti-VEGF intravitreal injection"],
            "sensorineural hearing loss":          ["cochlear implant"],
            "polycystic ovary syndrome":           ["laparoscopic ovarian cystectomy", "in vitro fertilization"],
            "testicular germ cell tumor":          ["testosterone replacement therapy"],
            "endometriosis":                       ["hysterectomy"],
            "osteoarthritis":                      ["arthroscopic debridement", "mesenchymal stem cell cartilage regeneration"]
          }
    },

    "treatment_time": {
        "treatment": {
            "esophagectomy":                             ["72 hours", "2 weeks", "6 months"],
            "photodynamic therapy":                      ["72 hours", "2 weeks", "6 months"],
            "anti-VEGF intravitreal injection":          ["72 hours", "2 weeks", "6 months"],
            "cochlear implant":                          ["72 hours", "2 weeks", "6 months"],
            "laparoscopic ovarian cystectomy":           ["72 hours", "2 weeks", "6 months"],
            "in vitro fertilization":                    ["72 hours", "2 weeks", "6 months"],
            "testosterone replacement therapy":          ["72 hours", "2 weeks", "6 months"],
            "hysterectomy":                              ["72 hours", "2 weeks", "6 months"],
            "arthroscopic debridement":                  ["72 hours", "2 weeks", "6 months"],
            "mesenchymal stem cell cartilage regeneration": ["72 hours", "2 weeks", "6 months"]
        }
    },

    "response": {
        "treatment": {
            "esophagectomy":                             ["relapse", "adverse event"],
            "photodynamic therapy":                      ["relapse", "adverse event"],
            "anti-VEGF intravitreal injection":          ["relapse", "adverse event"],
            "cochlear implant":                          ["relapse", "adverse event"],
            "laparoscopic ovarian cystectomy":           ["relapse", "adverse event"],
            "in vitro fertilization":                    ["relapse", "adverse event"],
            "testosterone replacement therapy":          ["relapse", "adverse event"],
            "hysterectomy":                              ["relapse", "adverse event"],
            "arthroscopic debridement":                  ["relapse", "adverse event"],
            "mesenchymal stem cell cartilage regeneration": ["relapse", "adverse event"]
        }
    },

    "host_phenotype": {
        "treatment": {
            "esophagectomy":                             ["parental", "persistent"],
            "photodynamic therapy":                      ["parental", "persistent"],
            "anti-VEGF intravitreal injection":          ["parental", "persistent"],
            "cochlear implant":                          ["parental", "persistent"],
            "laparoscopic ovarian cystectomy":           ["parental", "persistent"],
            "in vitro fertilization":                    ["parental", "persistent"],
            "testosterone replacement therapy":          ["parental", "persistent"],
            "hysterectomy":                              ["parental", "persistent"],
            "arthroscopic debridement":                  ["parental", "persistent"],
            "mesenchymal stem cell cartilage regeneration": ["parental", "persistent"]
        }
    },

    "library_selection": {
        "tissue_type": {
            "small RNA": [
                "cartilage tissue", "articular cartilage", "esophageal mucosa",
                "esophageal submucosa", "retinal tissue", "cochlear sensory epithelium",
                "ovarian follicle tissue", "testicular seminiferous tubules",
                "endometrial tissue", "myometrium"
            ],
            "other": [
                "cartilage tissue", "articular cartilage", "esophageal mucosa",
                "esophageal submucosa", "retinal tissue", "cochlear sensory epithelium",
                "ovarian follicle tissue", "testicular seminiferous tubules",
                "endometrial tissue", "myometrium"
            ]
        }
    },

    "library_source": {
        "cell_type": {
            "single-cell": [
                "cartilage cells", "chondrocytes", "esophageal epithelial cells",
                "retinal ganglion cells", "photoreceptor cells", "retinal pigment epithelial cells",
                "cochlear hair cells", "granulosa cells", "theca cells", "oocytes",
                "Leydig cells", "Sertoli cells", "spermatogonia", "spermatocytes",
                "endometrial epithelial cells", "decidual cells"
            ],
            "bulk": [
                "cartilage cells", "chondrocytes", "esophageal epithelial cells",
                "retinal ganglion cells", "photoreceptor cells", "retinal pigment epithelial cells",
                "cochlear hair cells", "granulosa cells", "theca cells", "oocytes",
                "Leydig cells", "Sertoli cells", "spermatogonia", "spermatocytes",
                "endometrial epithelial cells", "decidual cells"
            ]
        }
    },

    "donor_information": {
        "organ": {
            "male":   ["testes", "esophagus", "eye", "ear", "cartilage"],
            "female": ["ovaries", "uterus", "esophagus", "eye", "ear", "cartilage"],
            "pregnant": ["uterus", "ovaries"]
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
for _ in range(400):
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
