##########################################################################################
# IMPORT
import os
import json
import random
import pandas as pd
import re

##########################################################################################
#PATHS
DATA_PATH       = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/patrons_training/metadata_templates_train.json'
CORRECTED_PATH  = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_ft_patron_semi/metadata_templates_train_corrected.json'
EXPANDED_PATH   = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_ft_patron_semi/metadata_templates_train_expanded.json'
OUTPUT_CSV      = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_ft_patron_semi/metadata_replaced_table.csv'

CATEGORIES = [
    'cell_type','tissue_type','cell_line','organ','disease',
    'treatment','treatment_time','response','host_phenotype',
    'library_selection','library_source','donor_information','instrument_platform'
]
PH = {cat: f"__{cat.upper()}__" for cat in CATEGORIES}

CELL_TYPE   = ["endothelial cells","microglia","schwann cells","oligodendrocytes","pericytes"]
TISSUE_TYPE = ["endocrine tissue","reticular connective tissue","mesothelial tissue","pigmented epithelium","basal lamina"]
CELL_LINE   = ["PC-12","SK-BR-3","COS-7","HUVEC","G361"]
ORGAN       = ["testis","ovary","thymus","bladder","thyroid"]
DISEASE     = ["thyroiditis","orchitis","interstitial cystitis","myasthenia gravis","polycystic ovary syndrome"]
TREATMENT   = ["methimazole","azathioprine","penicillin","levothyroxine","rapamycin"]
TREAT_TIME  = ["72 hours","2 weeks","6 months"]
RESPONSE    = ["relapse","adverse event"]
HOST_PHENO  = ["parental","persistent"]
LIB_SEL     = ["small RNA","other"]
LIB_SRC     = ["single-cell","bulk"]
DONOR_INFO  = ["female","pregnant","Mediterranean","normal","39 years",
               "immunosuppressed","American Indian","athlete","Pacific Islander",
               "36 years","male","ex-smoker","unknown"]
PLATFORMS   = ["Illumina NovaSeq 9000","Illumina 500","NextSeq 500","MiSeq 200"]

POOLS = {
    'cell_type':           CELL_TYPE,
    'tissue_type':         TISSUE_TYPE,
    'cell_line':           CELL_LINE,
    'organ':               ORGAN,
    'disease':             DISEASE,
    'treatment':           TREATMENT,
    'treatment_time':      TREAT_TIME,
    'response':            RESPONSE,
    'host_phenotype':      HOST_PHENO,
    'library_selection':   LIB_SEL,
    'library_source':      LIB_SRC,
    'donor_information':   DONOR_INFO,
    'instrument_platform': PLATFORMS
}

# Synonyms mapping (new values)
SYNONYMS = {
    "endothelial cells":    ["vascular endothelial cells","blood vessel lining","capillary cells","arterial endothelial cells","mesenteric endothelial cells"],
    "microglia":            ["brain macrophages","CNS immune cells","resident microglia","neuroimmune cells","resting microglia"],
    "schwann cells":        ["peripheral glia","myelinating glia","Schwann glia","neurolemmocytes","PNS support cells"],
    "oligodendrocytes":     ["CNS myelinating cells","oligodendroglia","myelin-forming cells","white matter glia","OL cells"],
    "pericytes":            ["vascular mural cells","capillary support cells","blood–brain barrier cells","vascular pericytes","perivascular cells"],

    "endocrine tissue":          ["hormone-secreting tissue","glandular endocrine tissue","endocrine gland","endocrine layer","hormonal epithelium"],
    "reticular connective tissue":["reticular tissue","lymphoid connective framework","supportive reticular network","reticular fibers","stroma reticulum"],
    "mesothelial tissue":        ["serosal lining","peritoneal epithelium","mesothelium","pleural lining","pericardial epithelium"],
    "pigmented epithelium":      ["melanized layer","pigment cell layer","retinal pigment epithelium","uveal epithelium","pigment epithelium"],
    "basal lamina":              ["basement membrane","basal membrane","lamina densa","extracellular basal sheet","basal matrix"],

    "PC-12":     ["pheochromocytoma line","PC12 cells","rat adrenal line","PC-12 adrenal cells","PC12 pheochromocytoma"],
    "SK-BR-3":   ["breast carcinoma line","SKBR3 cells","HER2+ breast cancer","SKBR-3 carcinoma","human SK-BR-3"],
    "COS-7":     ["simian kidney line","SV40-transformed line","COS7 cells","monkey fibroblast line","COS-7 fibroblasts"],
    "HUVEC":     ["human umbilical vein cells","umbilical endothelial cells","HUVEC line","vascular HUVEC","primary HUVEC"],
    "G361":      ["human melanoma line","G361 melanoma","skin cancer line","G-361 cells","G361 melanoma cells"],

    "testis":    ["testicular tissue","male gonad","testes","seminiferous tissue","gonadal organ"],
    "ovary":     ["ovarian tissue","female gonad","ovaries","follicular tissue","gonadal organ"],
    "thymus":    ["thymic tissue","immune gland","thymus gland","lymphoid organ","thymic stroma"],
    "bladder":   ["urinary bladder","vesica urinaria","bladder organ","urothelium","bladder wall"],
    "thyroid":   ["thyroid gland","endocrine gland","thyroid organ","glandula thyreoidea","thyroid tissue"],

    "thyroiditis":               ["inflammation of thyroid","autoimmune thyroiditis","Hashimoto's thyroiditis","subacute thyroiditis","de Quervain thyroiditis"],
    "orchitis":                  ["testicular inflammation","epididymo-orchitis","acute orchitis","viral orchitis","bacterial orchitis"],
    "interstitial cystitis":     ["painful bladder syndrome","bladder pain syndrome","IC/BPS","Hunner’s ulcer disease","chronic cystitis"],
    "myasthenia gravis":         ["MG","autoimmune neuromuscular disease","ocular MG","seronegative MG","generalized myasthenia"],
    "polycystic ovary syndrome": ["PCOS","Stein–Leventhal syndrome","hyperandrogenic syndrome","polycystic ovaries","ovarian cyst disorder"],

    "methimazole":   ["MMI","thiouracil","anti-thyroid agent","methimazole therapy","methimazole drug"],
    "azathioprine":  ["AZA","Imuran","purine antagonist","immunosuppressive AZA","azathioprine therapy"],
    "penicillin":    ["beta-lactam antibiotic","penicillin G","penicillin V","penicillin therapy","benzylpenicillin"],
    "levothyroxine": ["T4 hormone","L-thyroxine","thyroxine replacement","synthetic thyroxine","levothyrox"],
    "rapamycin":     ["sirolimus","mTOR inhibitor","rapa","rapamycin therapy","rapamune"],

    "72 hours":      ["72h","three days","3 days","seventy-two hours","3-day"],
    "2 weeks":       ["14 days","fortnight","biweekly","two weeks","2-week"],
    "6 months":      ["half a year","26 weeks","six months","semiannual","6-month"],

    "relapse":       ["recurrence","disease return","relapsing","reappearance","recidivism"],
    "adverse event": ["AE","side effect","treatment-related effect","undesirable event","complication"],

    "female":           ["F","♀","adult female","woman","female donor"],
    "pregnant":         ["gestating","expectant","gravida","in utero donor","currently pregnant"],
    "Mediterranean":    ["Southern European","Med origin","Mediterranean descent","Med Sea region","Mediterranean ethnicity"],
    "normal":           ["healthy","baseline","control","reference","standard"],
    "39 years":         ["39 y.o.","age 39","39-year-old","39yrs","thirty-nine"],
    "immunosuppressed": ["immunocompromised","immune suppressed","IS","immunodeficient","weakened immunity"],
    "American Indian":  ["Native American","Indigenous American","American Indigenous","Native Amer.","AI donor"],
    "athlete":          ["sportsperson","athletic individual","sportsman","sportswoman","competitive athlete"],
    "Pacific Islander": ["Polynesian","Micronesian","Melanesian","PI descent","Pacific background"],
    "36 years":         ["36 y.o.","age 36","36-year-old","36yrs","thirty-six"],
    "male":             ["M","♂","adult male","man","male donor"],
    "ex-smoker":        ["former smoker","past smoker","quit smoking","ex tobacco user","ceased smoker"],
    "unknown":          ["NA","not available","unspecified","missing","unknown donor"],

    "Illumina NovaSeq 9000":["NovaSeq 9000","NS9000","Illumina NS9000","NovaSeq","Illumina NovaSeq"],
    "Illumina 500":        ["HiSeq 500","HS500","Illumina HiSeq 500","Illumina HS500","HiSeq"],
    "NextSeq 500":         ["NS500","Next Seq 500","NextSeq-500","Illumina NextSeq","NextSeq"],
    "MiSeq 200":           ["MiSeq200","MiSeq 200","Illumina MiSeq","Mi-Seq 200","MiSeq System"]
}

def choose_disp(val):
    syns = SYNONYMS.get(val, [])
    return random.choice(syns) if syns and random.random()<0.6 else val

SEM = {
    "cell_type": {
        "tissue_type": {
            "endothelial cells":  ["reticular connective tissue","basal lamina"],
            "microglia":          ["nervous tissue"],
            "schwann cells":      ["nervous tissue"],
            "oligodendrocytes":   ["nervous tissue"],
            "pericytes":          ["reticular connective tissue"]
        },
        "organ": {
            "endothelial cells":  ["thyroid","bladder","testis"],
            "microglia":          ["brain"],
            "schwann cells":      ["brain"],
            "oligodendrocytes":   ["brain"],
            "pericytes":          ["thyroid","bladder"]
        }
    },
    "cell_line": {
        "organ": {
            "PC-12":   ["thymus"],
            "SK-BR-3": ["ovary"],
            "COS-7":   ["bladder"],
            "HUVEC":   ["testis"],
            "G361":    ["thyroid"]
        },
        "disease": {
            "PC-12":   ["myasthenia gravis"],
            "SK-BR-3": ["polycystic ovary syndrome"],
            "COS-7":   ["interstitial cystitis"],
            "HUVEC":   ["orchitis"],
            "G361":    ["thyroiditis"]
        }
    },
    "disease": {
        "organ": {
            "orchitis":                  ["testis"],
            "thyroiditis":               ["thyroid"],
            "polycystic ovary syndrome": ["ovary"],
            "myasthenia gravis":         ["thymus"],
            "interstitial cystitis":     ["bladder"]
        }
    },
    "treatment": {
        "disease": {
            "methimazole":   ["thyroiditis"],
            "azathioprine":  ["orchitis","myasthenia gravis"],
            "penicillin":    ["orchitis","interstitial cystitis"],
            "levothyroxine": ["thyroiditis"],
            "rapamycin":     ["polycystic ovary syndrome"]
        }
    },
    "treatment_time": {
        "treatment": { t: TREAT_TIME for t in TREATMENT }
    },
    "response": {
        "treatment": { t: RESPONSE for t in TREATMENT }
    },
    "host_phenotype": {
        "treatment": { t: HOST_PHENO for t in TREATMENT }
    },
    "library_selection": {
        "tissue_type": {
            "small RNA": TISSUE_TYPE,
            "other":     TISSUE_TYPE
        }
    },
    "library_source": {
        "cell_type": {
            "single-cell": CELL_TYPE,
            "bulk":        CELL_TYPE
        }
    }
}

##########################################################################################
#MAIN
with open(DATA_PATH) as f:
    templates = json.load(f)

corrected = []
for phrase, cnt in templates:
    for ph in PH.values():
        if ph not in phrase:
            phrase += ' ' + ph
    corrected.append((phrase, cnt))

with open(CORRECTED_PATH, 'w') as f:
    json.dump(corrected, f, indent=2)

base_combos = []
while len(base_combos) < 50:
    combo = {}
    ct = random.choice(CELL_TYPE)
    combo['cell_type'] = ct
    combo['tissue_type'] = random.choice(
        SEM['cell_type']['tissue_type'][ct]
    )
    combo['organ'] = random.choice(
        SEM['cell_type']['organ'][ct]
    )
    opts = SEM['cell_line']['organ'].get(combo['cell_line'] if False else None, [])
    cl_opts = [cl for cl,orgs in SEM['cell_line']['organ'].items() if combo['organ'] in orgs]
    combo['cell_line'] = random.choice(cl_opts or CELL_LINE)
    ds_opts = SEM['cell_line']['disease'].get(combo['cell_line'], [])
    combo['disease'] = random.choice(ds_opts or DISEASE)
    combo['treatment'] = random.choice(
        SEM['treatment']['disease'][combo['disease']]
    )
    combo['treatment_time'] = random.choice(
        SEM['treatment_time']['treatment'][combo['treatment']]
    )
    combo['response'] = random.choice(
        SEM['response']['treatment'][combo['treatment']]
    )
    combo['host_phenotype'] = random.choice(
        SEM['host_phenotype']['treatment'][combo['treatment']]
    )
    combo['library_selection'] = random.choice(
        SEM['library_selection']['tissue_type'][combo['tissue_type']]
    )
    combo['library_source'] = random.choice(
        SEM['library_source']['cell_type'][combo['cell_type']]
    )
    combo['donor_information'] = random.choice(DONOR_INFO)
    combo['instrument_platform'] = random.choice(PLATFORMS)

    base_combos.append(combo)

records = []
for _ in range(1500):
    combo = random.choice(base_combos)
    tpl, _ = random.choice(corrected)
    phrase = tpl
    for cat, val in combo.items():
        disp   = choose_disp(val)
        phrase = re.sub(re.escape(PH[cat]), disp, phrase)
    rec = {'phrase_text': phrase, **combo}
    records.append(rec)

with open(EXPANDED_PATH, 'w') as f:
    json.dump([r['phrase_text'] for r in records], f, indent=2)

df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
