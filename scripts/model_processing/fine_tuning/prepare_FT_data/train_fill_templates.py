import os
import json
import random
import re
import pandas as pd

DATA_PATH = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/data_templates_training/metadata_templates_train_withoutkeys.json'
EXPANDED_PATH = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/metadata_templates_train_expanded.json'
OUTPUT_CSV  = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/train_metadata_replaced_table.csv'

CATEGORIES = [
    "library_selection","sequencing_source","organ","biopsy_site","biopsy_type",
    "cell_line","cell_type","disease","treatment","treatment_time","response",
    "age","sex","ethnicity","localization","is_cancer"
]

LIB_SEL = ["polyA","inverse rRNA","hybrid selection"]
SEQ_SRC = ["single-cell","bulk","spatial"]
BIOPSY_SITE = ["blood","lung","heart","kidney","brain","stomach","intestine","colon","skin","liver","spleen","bone marrow","pancreas","prostate"]
BIOPSY_TYPE = ["primary","metastasis","blood"]
CELL_LINE = ["primary tissue","HEK293","HeLa","HepG2","MCF7","A549","K562","U2OS","PC3","Jurkat","HCT116","SHSY5Y","C2C12","THP1","not applicable"]
CELL_TYPE = ["muscle cells","liver cells","blood cells","kidney cells","nerve cells","connective cells","fat cells","bone cells","cartilage cells","specialized integrated cells","fibroblasts","migratory cells","stomach cells","lung cells","primary tissue"]
ORGAN = ["liver","lung","heart","kidney","brain","muscle","spleen","pancreas","colon","stomach","intestine","skin","bone marrow","blood","prostate"]
DISEASE = ["lung cancer","hepatocellular carcinoma","breast cancer","leukemia","lymphoma","prostate cancer","pancreatic adenocarcinoma","glioblastoma","cirrhosis","chronic kidney disease","Crohn's disease","ulcerative colitis","myocardial infarction","diabetes mellitus"]
TREATMENT = ["no treatment","cisplatin","doxorubicin","paclitaxel","sorafenib","imatinib","erlotinib","tamoxifen","methotrexate","5-fluorouracil","amoxicillin","gentamicin","irradiation","dexamethasone"]
TREAT_TIME = ["pre treatment","on treatment","post treatment","relapse","unknown","24 hours","48 hours","1 week","3 months","1 year"]
RESPONSE = ["resistance","sensitivity","partial response","stable disease","progressive disease"]
AGE = ["34 years","62 years","47","born in 1982","55 years","19 years","44 years","67 years","41 years"]
SEX = ["male","female"]
ETHNICITY = ["Asian","Hispanic","European","African","American"]
LOCALIZATION = ["North Africa","Paris","French","English","Australia"]
IS_CANCER = ["true","false"]

SYNONYMS = {
    "muscle cells":["myocytes","contractile fibers","myofibers"],
    "liver cells":["hepatocytes","hepatic parenchyma"],
    "blood cells":["erythrocytes","leukocytes"],
    "kidney cells":["renal cells","tubular cells"],
    "nerve cells":["neurons","neural cells"],
    "connective cells":["stromal cells","mesenchymal cells"],
    "fat cells":["adipocytes"],
    "bone cells":["osteoblasts","osteocytes"],
    "cartilage cells":["chondrocytes"],
    "specialized integrated cells":["goblet cells","Paneth cells","myoepithelial cells"],
    "fibroblasts":["fibrocytes","stromal fibroblasts"],
    "migratory cells":["lymphocytes","natural killer cells"],
    "stomach cells":["parietal cells","chief cells"],
    "lung cells":["alveolar epithelial cells","bronchial epithelial cells"],
    "primary tissue":["primary sample","native tissue"],

    "HEK293":["HEK-293","293 cells","HEK 293","HEK293T","293T"],
    "HeLa":["HeLa S3","HeLa Kyoto","HeLa CCL-2"],
    "HepG2":["Hep G2","HepG-2","HepG2/C3A"],
    "MCF7":["MCF-7","MCF7/LCC1","MCF7-TAMR"],
    "A549":["A-549","A549/DDP"],
    "K562":["K-562","K562/ADR","K562-Luc"],
    "U2OS":["U-2 OS"],
    "PC3":["PC-3","PC3-MM2"],
    "Jurkat":["Jurkat E6-1"],
    "HCT116":["HCT-116","HCT116 p53+","HCT116 p21-/-"],
    "SHSY5Y":["SH-SY5Y","SHSY5Y/RA"],
    "C2C12":["C2 C12","C2C12 MB"],
    "THP1":["THP-1","THP1-DM"],

    "liver":["hepatic organ","hepatobiliary organ","liver parenchyma","hepatic lobes"],
    "lung":["pulmonary organ","respiratory organ","pulmonary lobes"],
    "heart":["myocardium","cardiac organ","heart muscle"],
    "kidney":["renal organ","renal cortex","nephron organ"],
    "brain":["encephalon","cerebrum","cerebral organ"],
    "muscle":["muscle mass","contractile organ","skeletal muscle"],
    "spleen":["splenic organ","splenic tissue"],
    "pancreas":["pancreatic organ","pancreatic gland"],
    "colon":["large intestine","colonic organ","colon mucosa"],
    "stomach":["gastric organ","stomach mucosa"],
    "intestine":["intestinal tract","gut","bowel"],
    "skin":["cutaneous tissue","epidermis","skin layer"],
    "bone marrow":["hematopoietic marrow","myeloid tissue","marrow tissue"],
    "blood":["bloodstream","vascular fluid","circulating fluid"],
    "prostate":["prostatic organ","prostate gland"],

    "lung cancer":["pulmonary carcinoma","lung neoplasm"],
    "hepatocellular carcinoma":["HCC","hepatoma","liver cancer"],
    "breast cancer":["mammary carcinoma","breast neoplasm"],
    "leukemia":["blood cancer","hematologic malignancy"],
    "lymphoma":["lymphatic malignancy"],
    "prostate cancer":["prostatic carcinoma","prostate adenocarcinoma"],
    "pancreatic adenocarcinoma":["PDAC","pancreatic cancer"],
    "glioblastoma":["GBM","malignant glioma","astrocytoma grade IV"],
    "cirrhosis":["hepatic cirrhosis"],
    "chronic kidney disease":["CKD","renal disease"],
    "Crohn's disease":["regional enteritis","Crohn disease"],
    "ulcerative colitis":["UC","colitis ulcerosa"],
    "myocardial infarction":["heart attack","AMI"],
    "diabetes mellitus":["diabetes","DM"],

    "polyA":["polyA+","mRNA enrichment","oligo-dT capture"],
    "inverse rRNA":["rRNA depletion","RiboZero","ribominus"],
    "hybrid selection":["hybrid capture","RNA exome","exon capture"],

    "cisplatin":["CDDP"],
    "doxorubicin":["adriamycin","DOX"],
    "paclitaxel":["taxol","PTX"],
    "sorafenib":["Nexavar"],
    "imatinib":["Gleevec"],
    "erlotinib":["Tarceva"],
    "tamoxifen":["TAM"],
    "methotrexate":["MTX"],
    "5-fluorouracil":["5-FU"],
    "amoxicillin":["Amoxil"],
    "gentamicin":["Garamycin"],
    "irradiation":["radiation","gamma rays","X-ray"],
    "dexamethasone":["Decadron","DXM"],

    "pre treatment":["pretreatment","before treatment"],
    "on treatment":["during treatment","treatment-phase"],
    "post treatment":["after treatment","post-therapy"],
    "relapse":["recurrence","re-emergence"],

    "resistance":["refractory","non-responsive"],
    "sensitivity":["responsive","susceptible"],
    "partial response":["PR","partial remission"],
    "stable disease":["SD","disease stabilization"],
    "progressive disease":["PD","disease progression"],

    "male":["man","male donor"],
    "female":["woman","female donor"],

    "North Africa":["Maghreb","North-African region"],
    "Paris":["Île-de-France","Paris, France","Paris region"],
    "French":["France","French origin"],
    "English":["UK origin","British"],
    "Australia":["AUS","Australia resident"],

    "34 years":["34-year-old","34yo","34 yrs"],
    "62 years":["62-year-old","62yo","62 yrs"],
    "47":["47-year-old","47yo","47 yrs"],
    "55 years":["55-year-old","55yo","55 yrs"],
    "19 years":["19-year-old","19yo","19 yrs"],
    "44 years":["44-year-old","44yo","44 yrs"],
    "67 years":["67-year-old","67yo","67 yrs"],
    "41 years":["41-year-old","41yo","41 yrs"],
}

SEM = {
    "cell_type":{"organ":{
        "muscle cells":["muscle","heart"],
        "liver cells":["liver"],
        "blood cells":["blood","bone marrow"],
        "bone cells":["bone marrow"],
        "cartilage cells":["skin"],
        "nerve cells":["brain"],
        "fat cells":["skin"],
        "fibroblasts":["skin","muscle"],
        "connective cells":["skin","muscle"],
        "lung cells":["lung"],
        "stomach cells":["stomach"],
        "migratory cells":["blood","spleen"],
        "specialized integrated cells":["pancreas","colon","lung","skin","prostate"],
        "kidney cells":["kidney"],
        "primary tissue":["liver","lung","heart","kidney","brain","muscle","spleen","pancreas","colon","stomach","intestine","skin","bone marrow","blood","prostate"]
    }},
    "disease":{"organ":{
        "lung cancer":["lung"],
        "hepatocellular carcinoma":["liver"],
        "breast cancer":["skin"],
        "leukemia":["blood","bone marrow"],
        "lymphoma":["spleen"],
        "prostate cancer":["prostate"],
        "pancreatic adenocarcinoma":["pancreas"],
        "glioblastoma":["brain"],
        "cirrhosis":["liver"],
        "chronic kidney disease":["kidney"],
        "Crohn's disease":["intestine","colon"],
        "ulcerative colitis":["colon","intestine"],
        "myocardial infarction":["heart"],
        "diabetes mellitus":["pancreas"]
    }},
    "cell_line":{"organ":{
        "HepG2":["liver"],
        "A549":["lung"],
        "MCF7":["skin"],
        "U2OS":["bone marrow"],
        "Jurkat":["blood","bone marrow"],
        "K562":["blood","bone marrow"],
        "SHSY5Y":["brain"],
        "C2C12":["muscle"],
        "PC3":["prostate"],
        "HCT116":["colon"],
        "HeLa":["skin"],
        "THP1":["blood","bone marrow"],
        "HEK293":["kidney"],
        "primary tissue":["liver","lung","heart","kidney","brain","muscle","spleen","pancreas","colon","stomach","intestine","skin","bone marrow","blood","prostate"]
    }},
    "treatment":{"disease":{
        "cisplatin":["lung cancer","hepatocellular carcinoma","breast cancer","prostate cancer","pancreatic adenocarcinoma","glioblastoma","leukemia","lymphoma"],
        "doxorubicin":["breast cancer","leukemia","lymphoma","glioblastoma"],
        "paclitaxel":["lung cancer","breast cancer","prostate cancer"],
        "sorafenib":["hepatocellular carcinoma"],
        "imatinib":["leukemia","chronic kidney disease"],
        "erlotinib":["lung cancer","pancreatic adenocarcinoma"],
        "tamoxifen":["breast cancer"],
        "methotrexate":["leukemia","lymphoma","breast cancer"],
        "5-fluorouracil":["pancreatic adenocarcinoma","breast cancer","ulcerative colitis"],
        "amoxicillin":["ulcerative colitis","Crohn's disease"],
        "gentamicin":["Crohn's disease"],
        "irradiation":["glioblastoma","breast cancer","lung cancer"],
        "dexamethasone":["glioblastoma","leukemia","ulcerative colitis","Crohn's disease"],
        "no treatment":["cirrhosis","myocardial infarction","diabetes mellitus"]
    }},
    "treatment_time":{"treatment":{
        "cisplatin":["pre treatment","on treatment","post treatment","relapse","unknown","24 hours","48 hours","1 week","3 months","1 year"],
        "doxorubicin":["pre treatment","on treatment","post treatment","relapse","unknown","24 hours","48 hours","1 week","3 months","1 year"],
        "paclitaxel":["pre treatment","on treatment","post treatment","relapse","unknown","24 hours","48 hours","1 week","3 months","1 year"],
        "sorafenib":["pre treatment","on treatment","post treatment","unknown","1 week","3 months","1 year"],
        "imatinib":["pre treatment","on treatment","post treatment","unknown","1 week","3 months"],
        "erlotinib":["pre treatment","on treatment","post treatment","unknown","1 week"],
        "tamoxifen":["pre treatment","on treatment","post treatment","unknown"],
        "methotrexate":["pre treatment","on treatment","post treatment","relapse","unknown","24 hours","48 hours","1 week"],
        "5-fluorouracil":["pre treatment","on treatment","post treatment","unknown","24 hours","48 hours"],
        "amoxicillin":["pre treatment","on treatment","post treatment","unknown","24 hours","1 week"],
        "gentamicin":["pre treatment","on treatment","post treatment","unknown","24 hours","48 hours"],
        "irradiation":["pre treatment","on treatment","post treatment","unknown","24 hours","1 week"],
        "dexamethasone":["pre treatment","on treatment","post treatment","unknown","24 hours"],
        "no treatment":["not applicable"]
    }},
    "response":{"treatment":{
        "cisplatin":["resistance","sensitivity","partial response","stable disease","progressive disease"],
        "doxorubicin":["resistance","sensitivity","partial response","stable disease","progressive disease"],
        "paclitaxel":["resistance","sensitivity","partial response","stable disease","progressive disease"],
        "sorafenib":["resistance","sensitivity","partial response","stable disease","progressive disease"],
        "imatinib":["resistance","sensitivity","partial response","stable disease","progressive disease"],
        "erlotinib":["resistance","sensitivity","partial response","stable disease","progressive disease"],
        "tamoxifen":["resistance","sensitivity","partial response","stable disease","progressive disease"],
        "methotrexate":["resistance","sensitivity","partial response","stable disease","progressive disease"],
        "5-fluorouracil":["resistance","sensitivity","partial response","stable disease","progressive disease"],
        "amoxicillin":["partial response","stable disease","progressive disease"],
        "gentamicin":["partial response","stable disease","progressive disease"],
        "irradiation":["partial response","stable disease","progressive disease"],
        "dexamethasone":["partial response","stable disease","progressive disease"],
        "no treatment":["not applicable"]
    }},
    "biopsy_site":{"organ":{
        "blood":["blood","bone marrow","spleen","liver","muscle","pancreas","colon","skin"],
        "lung":["lung"],
        "heart":["heart"],
        "kidney":["kidney"],
        "brain":["brain"],
        "stomach":["stomach"],
        "intestine":["intestine","colon","stomach"],
        "colon":["colon","intestine"],
        "skin":["skin"],
        "liver":["liver"],
        "spleen":["spleen"],
        "bone marrow":["bone marrow","blood"],
        "pancreas":["pancreas"],
        "prostate":["prostate"]
    }},
    "library_source":{"cell_type":{
        "single-cell":["blood cells","nerve cells","liver cells","muscle cells","fibroblasts","bone cells","cartilage cells","connective cells","lung cells","kidney cells","migratory cells"],
        "bulk":["blood cells","migratory cells","specialized integrated cells","fat cells","stomach cells","primary tissue","fibroblasts","liver cells","lung cells","kidney cells","muscle cells","bone cells"],
        "spatial":["primary tissue","liver cells","lung cells","kidney cells","muscle cells","stomach cells","connective cells","fibroblasts","specialized integrated cells"]
    }}
}

DISEASE_SEX = {"prostate cancer":"male","breast cancer":"female"}
CANCER_DISEASES = {"lung cancer","hepatocellular carcinoma","breast cancer","leukemia","lymphoma","prostate cancer","pancreatic adenocarcinoma","glioblastoma"}
HEMATO_DISEASES = {"leukemia","lymphoma"}
DISEASE_MET_SITES = {
    "pancreatic adenocarcinoma": {"liver", "lung", "bone marrow"},
    "lung cancer": {"brain", "liver", "bone marrow"},
    "breast cancer": {"bone marrow", "lung", "liver", "brain"},
    "prostate cancer": {"bone marrow", "liver", "lung"},
    "hepatocellular carcinoma": {"lung", "bone marrow"},
    "glioblastoma": set(),
    "lymphoma": set(),
    "leukemia": set(),
}

PREFERRED_CT_BY_CL = {
    "HepG2":["liver cells","specialized integrated cells","primary tissue"],
    "A549":["lung cells","specialized integrated cells","primary tissue"],
    "MCF7":["specialized integrated cells","fibroblasts","primary tissue"],
    "U2OS":["bone cells","fibroblasts","primary tissue"],
    "Jurkat":["blood cells","migratory cells","primary tissue"],
    "K562":["blood cells","migratory cells","primary tissue"],
    "SHSY5Y":["nerve cells","primary tissue"],
    "C2C12":["muscle cells","primary tissue"],
    "PC3":["specialized integrated cells","primary tissue"],
    "HCT116":["specialized integrated cells","primary tissue"],
    "HeLa":["specialized integrated cells","primary tissue"],
    "THP1":["blood cells","migratory cells","primary tissue"],
    "HEK293":["kidney cells","primary tissue"]
}

CTX_WRAP = {
    "biopsy_site":[
        "sections from the {val} were fixed and preserved for sequencing",
        "initial diagnosis prompted a biopsy of the {val}",
        "a sample from the {val} was included due to lesion proximity",
        "needle-guided retrieval was performed on the {val}"
    ],
    "organ":[
        "the {val} appeared frequently altered in affected individuals",
        "transcriptional profiles were enriched for genes expressed in the {val}",
        "inter-organ comparisons confirmed changes centered in the {val}"
    ],
    "response":[
        "investigator assessment documented {val}",
        "trial database entry reflected {val}",
        "assessment concluded {val}"
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

def pick_organ_from_disease(d):
    return random.choice(SEM["disease"]["organ"].get(d, ORGAN))

def pick_organ_from_cell_line(cl):
    return random.choice(SEM["cell_line"]["organ"].get(cl, ORGAN))

def pick_cell_type_from_organ(org):
    if random.random() < 0.25:
        return "primary tissue"
    cands = [ct for ct, orgs in SEM["cell_type"]["organ"].items() if org in orgs]
    return random.choice(cands) if cands else "primary tissue"

def pick_cell_line_from_organ(org):
    cands = [cl for cl, orgs in SEM["cell_line"]["organ"].items() if org in orgs and cl in CELL_LINE and cl not in ["primary tissue","not applicable"]]
    return random.choice(cands) if cands and random.random()<0.6 else "not applicable"

def pick_seq_source_from_cell_type(ct, org):
    if org in {"blood","bone marrow"}:
        if ct in SEM["library_source"]["cell_type"]["single-cell"]:
            return "single-cell" if random.random()<0.7 else "bulk"
        return "bulk"
    if ct in SEM["library_source"]["cell_type"]["single-cell"] and ct not in SEM["library_source"]["cell_type"]["bulk"]:
        return "single-cell" if random.random()<0.8 else "bulk"
    if ct in SEM["library_source"]["cell_type"]["spatial"] and random.random()<0.25:
        return "spatial"
    if ct in SEM["library_source"]["cell_type"]["bulk"]:
        return "bulk"
    return random.choice(["single-cell","bulk"])

def pick_library_selection(src):
    if src=="single-cell":
        return "polyA"
    if src=="spatial":
        return random.choice(["polyA","hybrid selection"])
    return random.choice(["polyA","inverse rRNA"])

def pick_disease_from_organ(org):
    cands = [d for d in DISEASE if org in SEM["disease"]["organ"].get(d,[])]
    return random.choice(cands) if cands else random.choice(DISEASE)

def pick_treatment_from_disease(d):
    cands = [t for t,ds in SEM["treatment"]["disease"].items() if d in ds]
    if not cands or random.random()<0.2:
        if d in SEM["treatment"]["disease"].get("no treatment",[]):
            return "no treatment"
    return random.choice(cands) if cands else "no treatment"

def pick_timing_and_response(tr):
    if tr=="no treatment":
        return "not applicable","not applicable"
    tt = random.choice(SEM["treatment_time"]["treatment"].get(tr,TREAT_TIME))
    rr = random.choice(SEM["response"]["treatment"].get(tr,RESPONSE))
    return tt, rr

def pick_biopsy_site(org):
    if random.random() < 0.7:
        return org
    if random.random() < 0.5:
        return "blood"
    cands = [site for site, organs in SEM["biopsy_site"]["organ"].items() if org in organs and site != org]
    return random.choice(cands) if cands else org

def decide_biopsy_type(site, organ, disease, is_cancer):
    if site == "blood":
        return "blood"
    if disease in HEMATO_DISEASES or disease == "glioblastoma":
        return "primary"
    if not is_cancer:
        return "primary"
    if site != organ and site not in {"blood", "bone marrow"}:
        plausible = site in DISEASE_MET_SITES.get(disease, set())
        return "metastasis" if plausible else "primary"
    return "primary"

def normalize_is_cancer(d):
    return "true" if d in CANCER_DISEASES else "false"

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
        "cell_line": allowed_terms(rec["cell_line"]) if rec["cell_line"]!="not applicable" else set(),
        "cell_type": allowed_terms(rec["cell_type"]),
        "treatment": allowed_terms(rec["treatment"]),
        "biopsy_site": allowed_terms(rec["biopsy_site"])
    }
    forbidden = []

    all_sets = {
        "organ": set(ORGAN),
        "disease": set(DISEASE),
        "cell_line": set([c for c in CELL_LINE if c not in ["primary tissue","not applicable"]]),
        "cell_type": set([c for c in CELL_TYPE if c!="primary tissue"]),
        "treatment": set([t for t in TREATMENT if t!="no treatment"]),
        "biopsy_site": set(BIOPSY_SITE)
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

def adjust_celltype_for_cellline(ct, cl, org):
    if cl in PREFERRED_CT_BY_CL:
        prefs = PREFERRED_CT_BY_CL[cl]
        valid = [x for x in prefs if org in SEM["cell_type"]["organ"].get(x,[])]
        if valid:
            return random.choice(valid)
        return prefs[0]
    return ct

def validate(rec):
    d = rec["disease"]; org = rec["organ"]; cl = rec["cell_line"]; ct = rec["cell_type"]
    site = rec["biopsy_site"]; btype = rec["biopsy_type"]; src = rec["sequencing_source"]
    tr = rec["treatment"]; tt = rec["treatment_time"]; rr = rec["response"]
    sex = rec["sex"]; isc = rec["is_cancer"]

    if org not in SEM["disease"]["organ"].get(d,[]):
        return False
    if ct!="primary tissue" and org not in SEM["cell_type"]["organ"].get(ct,[]):
        return False
    if cl!="not applicable" and org not in SEM["cell_line"]["organ"].get(cl,[]):
        return False
    if site=="blood" and btype!="blood":
        return False
    if site!="blood" and btype=="blood":
        return False
    if isc!=normalize_is_cancer(d):
        return False
    if d in DISEASE_SEX and sex!=DISEASE_SEX[d]:
        return False
    if ct == "primary tissue" and cl != "not applicable":
        return False
    if tr == "no treatment":
        if tt != "not applicable" or rr != "not applicable":
            return False
    else:
        if tt == "not applicable" or rr == "not applicable":
            return False
        if rr not in RESPONSE:
            return False

    if isc == "true" and d not in HEMATO_DISEASES:
        if site == org and btype == "metastasis":
            return False
        plausible = site in DISEASE_MET_SITES.get(d, set())
        if site not in {"blood", "bone marrow"} and site != org:
            if plausible and btype != "metastasis":
                return False
            if not plausible and btype == "metastasis":
                return False

    if org in {"blood","bone marrow"} and src=="spatial":
        return False
    sc_ok = ct in SEM["library_source"]["cell_type"]["single-cell"]
    bulk_ok = ct in SEM["library_source"]["cell_type"]["bulk"]
    spatial_ok = ct in SEM["library_source"]["cell_type"]["spatial"]
    if src=="single-cell" and not sc_ok:
        return False
    if src=="bulk" and not bulk_ok:
        return False
    if src=="spatial" and not spatial_ok:
        return False
    if d in HEMATO_DISEASES and btype=="metastasis":
        return False
    if d=="glioblastoma" and btype=="metastasis":
        return False
    if site == "bone marrow" and d not in HEMATO_DISEASES and org != "bone marrow":
        return False

    if src == "spatial" and site != org:
        return False

    if site not in {"blood", "bone marrow"} and site != org:
        if isc == "true" and d not in HEMATO_DISEASES:
            plausible = site in DISEASE_MET_SITES.get(d, set())
            if not plausible or btype != "metastasis":
                return False
        else:
            return False

    if site == org and btype == "metastasis":
        return False

    return True

def phrase_with_context(phrase, rec):
    for cat in CATEGORIES:
        if cat == "is_cancer":
            continue
        val = rec[cat]
        if val in {"not applicable", "unknown"} or (cat == "treatment" and val == "no treatment"):
            continue
        alt = random.choice(SYNONYMS.get(val, [val])) if random.random() < 0.8 else val
        if cat in CTX_WRAP:
            alt = random.choice(CTX_WRAP[cat]).format(val=alt)
        phrase = inject_value(phrase, alt)
    phrase = sanitize_phrase(phrase, rec)
    return phrase

rows = []
target_n = 1500
attempts_cap = 300000
attempts = 0

for template, _ in expanded:
    if len(rows)>=target_n: break

while len(rows)<target_n and attempts<attempts_cap:
    attempts += 1
    template, _ = random.choice(expanded)
    phrase = template
    record = {}

    anchor = random.random()
    if anchor < 0.4:
        d = random.choice(DISEASE)
        org = pick_organ_from_disease(d)
    elif anchor < 0.8:
        org = random.choice(ORGAN)
        d = pick_disease_from_organ(org)
    else:
        cl0 = random.choice([x for x in CELL_LINE if x not in ["primary tissue","not applicable"]])
        org = pick_organ_from_cell_line(cl0)
        d = pick_disease_from_organ(org)

    sex = DISEASE_SEX.get(d, random.choice(SEX))
    ct = pick_cell_type_from_organ(org)
    cl = pick_cell_line_from_organ(org)
    if ct=="primary tissue":
        cl="not applicable"
    if cl!="not applicable":
        ct = adjust_celltype_for_cellline(ct, cl, org)

    src = pick_seq_source_from_cell_type(ct, org)
    lib = pick_library_selection(src)
    tr = pick_treatment_from_disease(d)
    tt, rr = pick_timing_and_response(tr)
    site = pick_biopsy_site(org)
    isc = normalize_is_cancer(d)
    age = random.choice(AGE);
    eth = random.choice(ETHNICITY);
    loc = random.choice(LOCALIZATION)

    btype = decide_biopsy_type(site, org, d, isc == "true")

    record.update({
        "library_selection":lib,
        "sequencing_source":src,
        "organ":org,
        "biopsy_site":site,
        "biopsy_type":btype,
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
    })

    if not validate(record):
        continue

    record["phrase"] = phrase_with_context(phrase, record)
    rows.append(record)

df = pd.DataFrame(rows, columns=CATEGORIES+["phrase"])
df.to_csv(OUTPUT_CSV, index=False)
