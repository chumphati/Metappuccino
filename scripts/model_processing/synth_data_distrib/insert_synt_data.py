import os
import re
import random
import pandas as pd
from copy import deepcopy

TRAIN_TSV_IN = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/results/model_replace_final/out/train_masked.cleaned.tsv"
VAL_TSV_IN   = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/results/model_replace_final/out/val_masked.cleaned.tsv"

TRAIN_CSV_OUT = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/results/model_replace_final/out/train_metadata_replaced_table.csv"
VAL_CSV_OUT   = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/results/model_replace_final/out/val_metadata_replaced_table.csv"

CATS = [
    "library_selection","sequencing_source","biopsy_site","biopsy_type",
    "cell_line","cell_type","organ","disease",
    "treatment","treatment_time","response","age","sex","ethnicity"
]
OPEN_CATS = {
    "biopsy_site","cell_line","cell_type","organ","disease",
    "treatment","treatment_time","age","ethnicity"
}
CAT_TO_KEY = {
    "library_selection": "LIB_SEL",
    "sequencing_source": "SEQ_SRC",
    "biopsy_site": "BIOPSY_SITE",
    "biopsy_type": "BIOPSY_TYPE",
    "cell_line": "CELL_LINE",
    "cell_type": "CELL_TYPE",
    "organ": "ORGAN",
    "disease": "DISEASE",
    "treatment": "TREATMENT",
    "treatment_time": "TREATMENT_TIME",
    "response": "RESPONSE",
    "age": "AGE",
    "sex": "SEX",
    "ethnicity": "ETHNICITY",
}

def build_category_matchers():
    uppers = {}
    for cat in CATS:
        u = cat.upper()
        uppers[cat] = {"upper": u, "upper_no_us": u.replace("_", "")}
    return uppers

UPPERS = build_category_matchers()
UPPER_TOKEN_RE = re.compile(r"(?<![A-Z0-9_])[A-Z][A-Z0-9_]*(?![A-Z0-9_])")
UPPER_TRASH_RE = re.compile(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9_-]{3,}(?![A-Za-z0-9])")

def detect_token_category(token: str):
    tok_norm = token.replace("_", "")
    if tok_norm.endswith("TIME") or "TREATMENTTIME" in tok_norm:
        return "treatment_time"
    matches = []
    for cat, forms in UPPERS.items():
        pat = forms["upper_no_us"]
        if pat and pat in tok_norm:
            matches.append((cat, len(pat)))
    if not matches:
        return None
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[0][0]

def inject_value(sentence: str, insertion: str) -> str:
    if not insertion:
        return sentence
    parts = sentence.split()
    if not parts:
        return insertion
    idx = random.randint(0, len(parts))
    return " ".join(parts[:idx] + [insertion] + parts[idx:])

def replace_time_fragments(text: str, surface_time: str):
    if not surface_time:
        return text
    text = re.sub(
        r'(?<![A-Za-z0-9])TREATMENT\s*[_-]?\s*TIME(?![A-Za-z0-9])',
        surface_time,
        text
    )
    text = re.sub(
        r'(?<![A-Za-z0-9])_?TIME(?![A-Za-z0-9])',
        surface_time,
        text
    )
    return text

def replace_markers_with_values(text: str, cat_to_surface: dict):
    replaced_cats = set()
    def repl(m):
        tok = m.group(0)
        cat = detect_token_category(tok)
        if cat and cat in cat_to_surface and cat_to_surface[cat]:
            replaced_cats.add(cat)
            return cat_to_surface[cat]
        return tok
    new_text = UPPER_TOKEN_RE.sub(repl, text)
    return new_text, replaced_cats

def cleanup_uppercase_garbage(text: str):
    text = UPPER_TRASH_RE.sub("", text)
    text = re.sub(r"\s{2,}", " ", text).strip().strip(",").strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text

def regex_for_surface(surface: str):
    if not surface:
        return None
    pat = re.escape(surface)
    pat = re.sub(r"\\\s+", r"\\s+", pat)
    pat = pat.replace(r"\ ", r"\s+")
    return re.compile(r"(?i)\b" + pat + r"\b")

def remove_numbers_around_surface_v2(text: str, surface: str, window=2):
    if not surface:
        return text
    pat = regex_for_surface(surface)
    if not pat:
        return text
    for _ in range(2):
        text = re.sub(r'((?:\b\S*\d\S*\b\s+){1,'+str(window)+'})(' + pat.pattern[4:-2] + r')', r'\2', text)
        text = re.sub(r'(' + pat.pattern[4:-2] + r')((?:\s+\b\S*\d\S*\b){1,'+str(window)+'})', r'\1', text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def remove_age_inline_numbers(text: str, inserted_numbers: set, window_tokens=4):
    toks = text.split()
    to_delete = set()
    for i, tok in enumerate(toks):
        if tok.lower() == "age":
            after_idx = []
            before_idx = []
            j = i + 1
            steps = 0
            while j < len(toks) and steps < window_tokens and len(after_idx) < 2:
                if re.search(r"\d", toks[j]):
                    after_idx.append(j)
                j += 1
                steps += 1
            j = i - 1
            steps = 0
            while j >= 0 and steps < window_tokens and len(before_idx) < 2:
                if re.search(r"\d", toks[j]):
                    before_idx.append(j)
                j -= 1
                steps += 1
            for idx_group in [after_idx, before_idx]:
                if not idx_group:
                    continue
                has_inserted = any(any(num in toks[k] for num in inserted_numbers) for k in idx_group)
                if has_inserted:
                    for k in idx_group:
                        if not any(num in toks[k] for num in inserted_numbers):
                            to_delete.add(k)
                else:
                    for k in idx_group:
                        to_delete.add(k)
    if not to_delete:
        return text
    kept = [t for idx, t in enumerate(toks) if idx not in to_delete]
    out = " ".join(kept)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return re.sub(r"\s{2,}", " ", out).strip()

def scrub_age_following_numbers(text: str, age_inserted_numbers: set, max_numbers=2):
    toks = text.split()
    to_delete = set()
    i = 0
    while i < len(toks):
        if toks[i].lower() == "age":
            picked = []
            j = i + 1
            while j < len(toks) and len(picked) < max_numbers:
                if re.search(r"\d", toks[j]):
                    num_tokens = re.findall(r"\d+", toks[j])
                    if not any(n in age_inserted_numbers for n in num_tokens):
                        picked.append(j)
                j += 1
            if picked:
                to_delete.add(i)
                for k in picked:
                    to_delete.add(k)
        i += 1
    if not to_delete:
        return text
    kept = [t for idx, t in enumerate(toks) if idx not in to_delete]
    out = " ".join(kept)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return re.sub(r"\s{2,}", " ", out).strip()

def remove_years_months_with_numbers(text: str):
    text = re.sub(r'\b\d+\s*(?:year|years|month|months)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def remove_repeated_uppercase_lots(text: str):
    repeated = set()
    for m in re.finditer(r'\b([A-Z]{2,})\b(?:\s+\1\b)+', text):
        repeated.add(m.group(1))
    if not repeated:
        return text
    for w in repeated:
        text = re.sub(rf'\b{re.escape(w)}\b', '', text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()

def add_syn(d, k, vals):
    d["SYNONYMS"].setdefault(k, [k])
    d["SYNONYMS"][k] = list(dict.fromkeys(d["SYNONYMS"][k] + vals))

ORIG_TRAIN = {
    "LIB_SEL": ['polyA','inverse rRNA','hybrid selection','small RNA','other','unknown'],
    "SEQ_SRC": ['single-cell','bulk','spatial','unknown'],
    "BIOPSY_SITE": ['blood','lung','heart','kidney','brain','stomach','intestine','colon','skin','liver','spleen','bone marrow','pancreas','prostate','breast','cervix','bone','cartilage','unknown'],
    "BIOPSY_TYPE": ['primary','metastasis','blood'],
    "CELL_LINE": ['HEK293','HeLa','HepG2','MCF7','A549','K562','U2OS','PC3','Jurkat','HCT116','SHSY5Y','C2C12','THP1','not applicable','unknown','Raji','Daudi','Ramos','SU-DHL-4','U937'],
    "CELL_TYPE": ['muscle cells','liver cells','blood cells','kidney cells','nerve cells','connective cells','fat cells','bone cells','cartilage cells','specialized integrated cells','fibroblasts','migratory cells','stomach cells','lung cells','primary tissue','unknown','lymphocytes'],
    "ORGAN": ['liver','lung','heart','kidney','brain','muscle','spleen','pancreas','colon','stomach','intestine','skin','bone marrow','blood','prostate','breast','cervix','bone','cartilage','unknown'],
    "DISEASE": ['lung cancer','hepatocellular carcinoma','breast cancer','leukemia','lymphoma','prostate cancer','pancreatic adenocarcinoma','glioblastoma','cirrhosis','chronic kidney disease',"Crohn's disease",'ulcerative colitis','myocardial infarction','diabetes mellitus','healthy','unknown'],
    "TREATMENT": ['no treatment','cisplatin','doxorubicin','paclitaxel','sorafenib','imatinib','erlotinib','tamoxifen','methotrexate','5-fluorouracil','amoxicillin','gentamicin','irradiation','dexamethasone','unknown'],
    "TREATMENT_TIME": ['pre treatment','on treatment','post treatment','relapse','unknown','24 hours','48 hours','1 week','3 months','1 year','not applicable'],
    "RESPONSE": ['no treatment','unknown','stable','progressive','success'],
    "AGE": ['34 years','62 years','47','born in 1982','55 years','19 years','44 years','67 years','41 years','adult','senior','unknown'],
    "SEX": ['male','female','unknown'],
    "ETHNICITY": ['Asian','Hispanic','European','African','American','unknown'],
    "SYNONYMS": {}
}
for lst in ["LIB_SEL","SEQ_SRC","BIOPSY_SITE","BIOPSY_TYPE","CELL_LINE","CELL_TYPE","ORGAN","DISEASE","TREATMENT","TREATMENT_TIME","RESPONSE","AGE","SEX","ETHNICITY"]:
    for v in ORIG_TRAIN[lst]:
        ORIG_TRAIN["SYNONYMS"].setdefault(v,[v])
add_syn(ORIG_TRAIN,"polyA",['polyA+','mRNA enrichment','oligo-dT capture','mrna polyadenylated','truseq mrna','truseq stranded mrna'])
add_syn(ORIG_TRAIN,"inverse rRNA",['rRNA depletion','RiboZero','ribominus','ribo-zero','total RNA','ribodep','riboerase','ribogone','ribocop'])
add_syn(ORIG_TRAIN,"hybrid selection",['hybrid capture','RNA exome','exon capture','exome capture','geomx'])
add_syn(ORIG_TRAIN,"small RNA",['small-RNA','miRNA-focused','short RNA','size fraction','truseq small'])
add_syn(ORIG_TRAIN,"other",['unspecified','miscellaneous','unknown method'])
add_syn(ORIG_TRAIN,"male",['man','male donor','M donor'])
add_syn(ORIG_TRAIN,"female",['woman','F donor'])
add_syn(ORIG_TRAIN,"primary tissue",['primary sample','native tissue'])
add_syn(ORIG_TRAIN,"lung cancer",['pulmonary carcinoma','lung neoplasm'])
add_syn(ORIG_TRAIN,"hepatocellular carcinoma",['HCC','hepatoma','liver cancer'])
add_syn(ORIG_TRAIN,"breast cancer",['mammary carcinoma','breast neoplasm'])
add_syn(ORIG_TRAIN,"glioblastoma",['GBM','malignant glioma','astrocytoma grade IV'])
add_syn(ORIG_TRAIN,"ulcerative colitis",['UC','colitis ulcerosa'])
add_syn(ORIG_TRAIN,"Crohn's disease",['regional enteritis','Crohn disease'])
add_syn(ORIG_TRAIN,"partial response",['PR','partial remission'])
add_syn(ORIG_TRAIN,"stable disease",['SD','disease stabilization'])
add_syn(ORIG_TRAIN,"progressive disease",['PD','disease progression'])
add_syn(ORIG_TRAIN,"success",['complete response','response achieved'])
add_syn(ORIG_TRAIN,"liver",['hepatic','liver tissue'])
add_syn(ORIG_TRAIN,"lung",['pulmonary','lung tissue'])
add_syn(ORIG_TRAIN,"kidney",['renal','kidney tissue'])
add_syn(ORIG_TRAIN,"heart",['cardiac','myocardial'])
add_syn(ORIG_TRAIN,"brain",['cerebral','brain tissue'])
add_syn(ORIG_TRAIN,"skin",['cutaneous','dermal'])
add_syn(ORIG_TRAIN,"intestine",['intestinal','gut'])
add_syn(ORIG_TRAIN,"colon",['colonic'])
add_syn(ORIG_TRAIN,"stomach",['gastric'])
add_syn(ORIG_TRAIN,"spleen",['splenic'])
add_syn(ORIG_TRAIN,"pancreas",['pancreatic'])
add_syn(ORIG_TRAIN,"bone marrow",['marrow','BM','medulla ossea'])
add_syn(ORIG_TRAIN,"blood",['whole blood','venous blood'])
add_syn(ORIG_TRAIN,"HepG2",['HEPG2'])
add_syn(ORIG_TRAIN,"A549",['A-549'])
add_syn(ORIG_TRAIN,"K562",['K-562'])
add_syn(ORIG_TRAIN,"5-fluorouracil",['5FU','5-FU'])
add_syn(ORIG_TRAIN,"Raji",['RAJI'])
add_syn(ORIG_TRAIN,"Daudi",['DAUDI'])
add_syn(ORIG_TRAIN,"Ramos",['RAMOS'])
add_syn(ORIG_TRAIN,"SU-DHL-4",['SUDHL4','SU DDL 4','SU D-HL-4'])
add_syn(ORIG_TRAIN,"U937",['U-937'])
add_syn(ORIG_TRAIN,"lymphocytes",['B cells','T cells','lymphoid cells','lymphocyte'])
for a in list(ORIG_TRAIN["AGE"]):
    if re.search(r'\d', a):
        n = re.findall(r'\d+', a)
        if n:
            n = n[0]
            add_syn(ORIG_TRAIN, a, [f"{n}-year-old", f"aged {n}", f"{n} yo", f"{n} yrs"])

BASE_EVAL = {
    "LIB_SEL": ['polyA','inverse rRNA','hybrid selection','small RNA','other','unknown'],
    "SEQ_SRC": ['single-cell','bulk','spatial','unknown'],
    "BIOPSY_SITE": ["blood","eye","ear","ovaries","cartilage","esophagus"],
    "BIOPSY_TYPE": ['primary','metastasis','blood'],
    "CELL_LINE": ["KYSE-30","TE-1","ARPE-19","HEI-OC1","KGN","NT2/D1","Ishikawa","ECC-1","SW1353","ATDC5"],
    "CELL_TYPE": ["cartilage cells","chondrocytes","esophageal epithelial cells","retinal ganglion cells","photoreceptor cells","retinal pigment epithelial cells","cochlear hair cells","granulosa cells","theca cells","oocytes","Leydig cells","Sertoli cells","spermatogonia","spermatocytes","endometrial epithelial cells","decidual cells"],
    "ORGAN": ["esophagus","eye","ear","ovaries","testes","uterus","cartilage"],
    "DISEASE": ["Barrett's esophagus","esophageal squamous cell carcinoma","age-related macular degeneration","glaucoma","retinitis pigmentosa","sensorineural hearing loss","polycystic ovary syndrome","testicular germ cell tumor","endometriosis","osteoarthritis","healthy"],
    "TREATMENT": ["esophagectomy","photodynamic therapy","anti-VEGF intravitreal injection","cochlear implant","laparoscopic ovarian cystectomy","in vitro fertilization","testosterone replacement therapy","hysterectomy","arthroscopic debridement","mesenchymal stem cell cartilage regeneration"],
    "TREATMENT_TIME": ["72 hours","2 weeks","6 months","unknown"],
    "RESPONSE": ['no treatment','unknown','stable','progressive','success'],
    "AGE": ["39 years","36 years","adult","unknown"],
    "SEX": ["male","female","unknown"],
    "ETHNICITY": ["American Indian","Pacific Islander","unknown"],
    "SYNONYMS": {}
}
for lst in ["LIB_SEL","SEQ_SRC","BIOPSY_SITE","BIOPSY_TYPE","CELL_LINE","CELL_TYPE","ORGAN","DISEASE","TREATMENT","TREATMENT_TIME","RESPONSE","AGE","SEX","ETHNICITY"]:
    for v in BASE_EVAL[lst]:
        BASE_EVAL["SYNONYMS"].setdefault(v,[v])
def add_syn_b(v, vals):
    add_syn(BASE_EVAL, v, vals)
add_syn_b("small RNA",['small-RNA','miRNA-focused','short RNA','size fraction','truseq small'])
add_syn_b("other",['unspecified','miscellaneous'])
add_syn_b("single-cell",['scRNA-seq','single cell'])
add_syn_b("bulk",['bulk RNA-seq'])
add_syn_b("spatial",['spatial transcriptomics'])
add_syn_b("blood",['peripheral blood','whole blood'])
add_syn_b("esophagus",['oesophagus','esophageal'])
add_syn_b("eye",['ocular'])
add_syn_b("ear",['auricular'])
add_syn_b("ovaries",['ovarian'])
add_syn_b("cartilage",['articular cartilage'])
add_syn_b("KYSE-30",['KYSE30'])
add_syn_b("TE-1",['TE1'])
add_syn_b("ARPE-19",['ARPE19','RPE line'])
add_syn_b("SW1353",['SW-1353'])
for a in list(BASE_EVAL["TREATMENT_TIME"]):
    if re.search(r'\d', a):
        a_num = re.findall(r'\d+', a)[0]
        if "hour" in a:
            add_syn_b(a,[f"{a_num}h",f"{a_num} hours"])
        if "week" in a:
            add_syn_b(a,[f"{a_num} wk",f"{a_num}w",f"{a_num}-week"])
        if "month" in a:
            add_syn_b(a,[f"{a_num} mo",f"{a_num}mos",f"{a_num}-month"])
for a in list(BASE_EVAL["AGE"]):
    if re.search(r'\d', a):
        n = re.findall(r'\d+', a)[0]
        add_syn_b(a,[f"{n}-year-old", f"aged {n}", f"{n} yo", f"{n} yrs"])

TEST2 = {
    "LIB_SEL": ['polyA','inverse rRNA','hybrid selection','small RNA','other','unknown'],
    "SEQ_SRC": ['single-cell','bulk','spatial','unknown'],
    "BIOPSY_SITE": ["thyroid","adrenal gland","placenta","pituitary","parathyroid","thymus"],
    "BIOPSY_TYPE": ['primary','metastasis','blood'],
    "CELL_LINE": ["Nthy-ori 3-1","TPC-1","NCI-H295R","SW-13","JEG-3","BeWo","HTR-8/SVneo","GH3","HP75","sHPT-1","TT"],
    "CELL_TYPE": ["thyrocytes","adrenocortical cells","trophoblasts","syncytiotrophoblasts","pituitary endocrine cells","somatotrophs","lactotrophs","parathyroid chief cells","thymic epithelial cells"],
    "ORGAN": ["thyroid","adrenal gland","placenta","pituitary","parathyroid","thymus"],
    "DISEASE": ["papillary thyroid carcinoma","Hashimoto's thyroiditis","Cushing's syndrome","Addison's disease","preeclampsia","pituitary adenoma","hyperparathyroidism","thymoma","adrenocortical carcinoma","healthy"],
    "TREATMENT": ["radioiodine ablation","levothyroxine","methimazole","transsphenoidal surgery","hydrocortisone therapy","ketoconazole therapy","thymectomy","parathyroidectomy","magnesium sulfate","antihypertensive therapy","mitotane"],
    "TREATMENT_TIME": ["96 hours","5 days","10 days","4 weeks","8 weeks","9 months","18 months","unknown"],
    "RESPONSE": ['no treatment','unknown','stable','progressive','success'],
    "AGE": ["28 years","33 years","45 years","52 years","63 years","71 years","adult","unknown"],
    "SEX": ["male","female","unknown"],
    "ETHNICITY": ["Middle Eastern","Nordic","Eastern European","Sub-Saharan African","South Asian","Caribbean","unknown"],
    "SYNONYMS": {}
}
for lst in ["LIB_SEL","SEQ_SRC","BIOPSY_SITE","BIOPSY_TYPE","CELL_LINE","CELL_TYPE","ORGAN","DISEASE","TREATMENT","TREATMENT_TIME","RESPONSE","AGE","SEX","ETHNICITY"]:
    for v in TEST2[lst]:
        TEST2["SYNONYMS"].setdefault(v,[v])

def expand_synonyms_variants(syn):
    out = {}
    for k, vs in syn.items():
        acc = set(vs)
        for v in list(vs):
            acc.add(v.lower())
            acc.add(v.title())
            acc.add(v.replace(" ", "_"))
            acc.add(v.replace(" ", "-"))
            acc.add(v.replace("-", " "))
            acc.add(v.replace("_", " "))
        out[k] = list(dict.fromkeys([k] + list(acc)))
    return out

add_syn(TEST2,"thyroid",['thyroid gland','thyroidal'])
add_syn(TEST2,"adrenal gland",['adrenal','suprarenal','suprarenal gland'])
add_syn(TEST2,"placenta",['placental'])
add_syn(TEST2,"pituitary",['pituitary gland','hypophysis','hypophyseal'])
add_syn(TEST2,"parathyroid",['parathyroid gland','parathyroidal'])
add_syn(TEST2,"thymus",['thymic'])
for a in list(TEST2["TREATMENT_TIME"]):
    if re.search(r'\d', a):
        a_num = re.findall(r'\d+', a)[0]
        if "hour" in a:
            add_syn(TEST2,a,[f"{a_num}h",f"{a_num} hours"])
        if "day" in a:
            add_syn(TEST2,a,[f"{a_num} d",f"{a_num}days"])
        if "week" in a:
            add_syn(TEST2,a,[f"{a_num} wk",f"{a_num}w",f"{a_num}-week"])
        if "month" in a:
            add_syn(TEST2,a,[f"{a_num} mo",f"{a_num}mos",f"{a_num}-month"])
for a in list(TEST2["AGE"]):
    if re.search(r'\d', a):
        n = re.findall(r'\d+', a)[0]
        add_syn(TEST2, a, [f"{n}-year-old", f"aged {n}", f"{n} yo", f"{n} yrs"])

ORIG_TRAIN["SYNONYMS"] = expand_synonyms_variants(ORIG_TRAIN["SYNONYMS"])
BASE_EVAL["SYNONYMS"]  = expand_synonyms_variants(BASE_EVAL["SYNONYMS"])
TEST2["SYNONYMS"]      = expand_synonyms_variants(TEST2["SYNONYMS"])

SEM_ORIG = {
    "cell_type": {"organ":{
        'muscle cells':['muscle','heart'],
        'liver cells':['liver'],
        'blood cells':['blood','bone marrow','spleen'],
        'bone cells':['bone'],
        'cartilage cells':['cartilage'],
        'nerve cells':['brain'],
        'fat cells':['skin'],
        'fibroblasts':['skin','muscle','breast'],
        'connective cells':['skin','muscle','breast'],
        'lung cells':['lung'],
        'stomach cells':['stomach'],
        'migratory cells':['blood','spleen'],
        'specialized integrated cells':['pancreas','colon','lung','skin','prostate','breast','cervix'],
        'kidney cells':['kidney'],
        'primary tissue':['liver','lung','heart','kidney','brain','muscle','spleen','pancreas','colon','stomach','intestine','skin','bone marrow','blood','prostate','breast','cervix','bone','cartilage'],
        'unknown':['unknown'],
        'lymphocytes':['spleen','blood','bone marrow']
    }},
    "disease":{"organ":{
        'lung cancer':['lung'],
        'hepatocellular carcinoma':['liver'],
        'breast cancer':['breast'],
        'leukemia':['blood','bone marrow'],
        'lymphoma':['spleen','blood','bone marrow'],
        'prostate cancer':['prostate'],
        'pancreatic adenocarcinoma':['pancreas'],
        'glioblastoma':['brain'],
        'cirrhosis':['liver'],
        'chronic kidney disease':['kidney'],
        "Crohn's disease":["intestine","colon"],
        'ulcerative colitis':['colon','intestine'],
        'myocardial infarction':['heart'],
        'diabetes mellitus':['pancreas'],
        'healthy':['liver','lung','heart','kidney','brain','muscle','spleen','pancreas','colon','stomach','intestine','skin','bone marrow','blood','prostate','breast','cervix','bone','cartilage']
    }},
    "cell_line":{"organ":{
        'HepG2':['liver'],'A549':['lung'],'MCF7':['breast'],'U2OS':['bone'],'Jurkat':['blood','bone marrow'],'K562':['blood','bone marrow'],
        'SHSY5Y':['brain'],'C2C12':['muscle'],'PC3':['prostate'],'HCT116':['colon'],'HeLa':['cervix'],'THP1':['blood','bone marrow'],
        'HEK293':['kidney'],'not applicable':['unknown'],'unknown':['unknown'],
        'Raji':['spleen','blood','bone marrow'],'Daudi':['spleen','blood','bone marrow'],'Ramos':['spleen','blood','bone marrow'],
        'SU-DHL-4':['spleen','blood','bone marrow'],'U937':['blood','bone marrow']
    }},
    "treatment":{"disease":{
        'cisplatin':['lung cancer','hepatocellular carcinoma','breast cancer','prostate cancer','pancreatic adenocarcinoma','glioblastoma','leukemia','lymphoma'],
        'doxorubicin':['breast cancer','leukemia','lymphoma','glioblastoma'],
        'paclitaxel':['lung cancer','breast cancer','prostate cancer'],
        'sorafenib':['hepatocellular carcinoma'],
        'imatinib':['leukemia','chronic kidney disease'],
        'erlotinib':['lung cancer','pancreatic adenocarcinoma'],
        'tamoxifen':['breast cancer'],
        'methotrexate':['leukemia','lymphoma','breast cancer'],
        '5-fluorouracil':['pancreatic adenocarcinoma','breast cancer','ulcerative colitis'],
        'amoxicillin':['ulcerative colitis',"Crohn's disease"],
        'gentamicin':["Crohn's disease"],
        'irradiation':['glioblastoma','breast cancer','lung cancer'],
        'dexamethasone':['glioblastoma','leukemia','ulcerative colitis',"Crohn's disease"],
        'no treatment':['cirrhosis','myocardial infarction','diabetes mellitus','healthy','unknown'],
        'unknown':['unknown','healthy']
    }},
    "treatment_time":{"treatment":{
        'no treatment':['unknown'],'tamoxifen':['unknown'],'unknown':['unknown']
    }},
    "response":{"treatment":{
        'no treatment':['unknown'],'tamoxifen':['unknown'],'unknown':['unknown']
    }},
    "biopsy_site":{"organ":{
        'blood':['blood','bone marrow'],
        'bone marrow':['bone marrow','blood'],
        'lung':['lung'],'heart':['heart'],'kidney':['kidney'],'brain':['brain'],'stomach':['stomach'],
        'intestine':['intestine','colon','stomach'],'colon':['colon','intestine'],'skin':['skin'],'liver':['liver'],
        'spleen':['spleen'],'pancreas':['pancreas'],'prostate':['prostate'],
        'breast':['breast'],'cervix':['cervix'],'bone':['bone','bone marrow'],'cartilage':['cartilage'],'unknown':['unknown']
    }},
    "library_source":{"cell_type":{
        'single-cell':['blood cells','nerve cells','liver cells','muscle cells','fibroblasts','bone cells','cartilage cells','connective cells','lung cells','kidney cells','migratory cells','primary tissue','stomach cells','specialized integrated cells','lymphocytes'],
        'bulk':['cartilage cells','retinal pigment epithelial cells','primary tissue','fibroblasts','liver cells','lung cells','kidney cells','muscle cells','bone cells','connective cells','nerve cells','blood cells','stomach cells','lymphocytes'],
        'spatial':['primary tissue','liver cells','lung cells','kidney cells','muscle cells','stomach cells','connective cells','fibroblasts','specialized integrated cells','cartilage cells']
    }},
    "sex":{
        "organ":{'prostate':['male'],'testes':['male'],'cervix':['female'],'ovaries':['female'],'uterus':['female'],'placenta':['female']},
        "disease":{'prostate cancer':['male'],'preeclampsia':['female'],'testicular germ cell tumor':['male']}
    }
}

SEM_VAL = {
    "cell_type":{"organ":{
        "cartilage cells":["cartilage"],"chondrocytes":["cartilage"],"esophageal epithelial cells":["esophagus"],
        "retinal ganglion cells":["eye"],"photoreceptor cells":["eye"],"retinal pigment epithelial cells":["eye"],
        "cochlear hair cells":["ear"],"granulosa cells":["ovaries"],"theca cells":["ovaries"],"oocytes":["ovaries"],
        "Leydig cells":["testes"],"Sertoli cells":["testes"],"spermatogonia":["testes"],"spermatocytes":["testes"],
        "endometrial epithelial cells":["uterus"],"decidual cells":["uterus"]
    }},
    "disease":{"organ":{
        "Barrett's esophagus":["esophagus"],"esophageal squamous cell carcinoma":["esophagus"],
        "age-related macular degeneration":["eye"],"glaucoma":["eye"],"retinitis pigmentosa":["eye"],
        "sensorineural hearing loss":["ear"],"polycystic ovary syndrome":["ovaries"],
        "testicular germ cell tumor":["testes"],"endometriosis":["uterus"],"osteoarthritis":["cartilage"],"healthy":["esophagus","eye","ear","ovaries","testes","uterus","cartilage"]
    }},
    "cell_line":{"organ":{
        "KYSE-30":["esophagus"],"TE-1":["esophagus"],"ARPE-19":["eye"],"HEI-OC1":["ear"],"KGN":["ovaries"],"NT2/D1":["testes"],
        "Ishikawa":["uterus"],"ECC-1":["uterus"],"SW1353":["cartilage"],"ATDC5":["cartilage"]
    }},
    "treatment":{"disease":{
        "esophagectomy":["esophageal squamous cell carcinoma","Barrett's esophagus"],
        "photodynamic therapy":["Barrett's esophagus"],
        "anti-VEGF intravitreal injection":["age-related macular degeneration","glaucoma"],
        "cochlear implant":["sensorineural hearing loss"],
        "laparoscopic ovarian cystectomy":["polycystic ovary syndrome"],
        "in vitro fertilization":["polycystic ovary syndrome","endometriosis"],
        "testosterone replacement therapy":["testicular germ cell tumor"],
        "hysterectomy":["endometriosis"],
        "arthroscopic debridement":["osteoarthritis"],
        "mesenchymal stem cell cartilage regeneration":["osteoarthritis"],
        "unknown":["unknown","healthy"]
    }},
    "treatment_time":{"treatment":{"unknown":["unknown"]}},
    "response":{"treatment":{"unknown":["unknown"]}},
    "biopsy_site":{"organ":{
        "blood":["blood","ovaries","testes","uterus","cartilage","esophagus","eye","ear"],
        "eye":["eye"],"ear":["ear"],"ovaries":["ovaries"],"cartilage":["cartilage"],"esophagus":["esophagus"]
    }},
    "library_source":{"cell_type":{
        "single-cell":["granulosa cells","photoreceptor cells","cochlear hair cells","endometrial epithelial cells","spermatogonia","retinal ganglion cells"],
        "bulk":["cartilage cells","decidual cells","retinal pigment epithelial cells"],
        "spatial":["Leydig cells","Sertoli cells","retinal ganglion cells","cartilage cells","esophageal epithelial cells"]
    }},
    "sex":{
        "organ":{"ovaries":["female"],"uterus":["female"],"testes":["male"]},
        "disease":{"polycystic ovary syndrome":["female"],"endometriosis":["female"],"testicular germ cell tumor":["male"]}
    }
}

SEM_TEST2 = {
    "cell_type":{"organ":{
        "thyrocytes":["thyroid"],"adrenocortical cells":["adrenal gland"],
        "trophoblasts":["placenta"],"syncytiotrophoblasts":["placenta"],
        "pituitary endocrine cells":["pituitary"],"somatotrophs":["pituitary"],"lactotrophs":["pituitary"],
        "parathyroid chief cells":["parathyroid"],"thymic epithelial cells":["thymus"]
    }},
    "disease":{"organ":{
        "papillary thyroid carcinoma":["thyroid"],"Hashimoto's thyroiditis":["thyroid"],
        "Cushing's syndrome":["adrenal gland"],"Addison's disease":["adrenal gland"],
        "preeclampsia":["placenta"],"pituitary adenoma":["pituitary"],
        "hyperparathyroidism":["parathyroid"],"thymoma":["thymus"],
        "adrenocortical carcinoma":["adrenal gland"],"healthy":["thyroid","adrenal gland","placenta","pituitary","parathyroid","thymus"]
    }},
    "cell_line":{"organ":{
        "Nthy-ori 3-1":["thyroid"],"TPC-1":["thyroid"],"NCI-H295R":["adrenal gland"],"SW-13":["adrenal gland"],
        "JEG-3":["placenta"],"BeWo":["placenta"],"HTR-8/SVneo":["placenta"],
        "GH3":["pituitary"],"HP75":["pituitary"],"sHPT-1":["parathyroid"],"TT":["thyroid"]
    }},
    "treatment":{"disease":{
        "radioiodine ablation":["papillary thyroid carcinoma"],
        "levothyroxine":["Hashimoto's thyroiditis"],
        "methimazole":["Hashimoto's thyroiditis"],
        "transsphenoidal surgery":["pituitary adenoma"],
        "hydrocortisone therapy":["Addison's disease"],
        "ketoconazole therapy":["Cushing's syndrome"],
        "thymectomy":["thymoma"],
        "parathyroidectomy":["hyperparathyroidism"],
        "magnesium sulfate":["preeclampsia"],
        "antihypertensive therapy":["preeclampsia"],
        "mitotane":["adrenocortical carcinoma"],
        "unknown":["unknown","healthy"]
    }},
    "treatment_time":{"treatment":{"unknown":["unknown"]}},
    "response":{"treatment":{"unknown":["unknown"]}},
    "biopsy_site":{"organ":{
        "thyroid":["thyroid"],"adrenal gland":["adrenal gland"],"placenta":["placenta"],
        "pituitary":["pituitary"],"parathyroid":["parathyroid"],"thymus":["thymus"]
    }},
    "library_source":{"cell_type":{
        "single-cell":["thyrocytes","adrenocortical cells","trophoblasts","syncytiotrophoblasts","pituitary endocrine cells","somatotrophs","lactotrophs","parathyroid chief cells","thymic epithelial cells"],
        "bulk":["adrenocortical cells","trophoblasts","thyrocytes"],
        "spatial":["trophoblasts","syncytiotrophoblasts","thyrocytes","thymic epithelial cells"]
    }},
    "sex":{
        "organ":{"placenta":["female"]},
        "disease":{"preeclampsia":["female"]}
    }
}

CANCER_DISEASES_TRAIN = {
    'lung cancer','hepatocellular carcinoma','breast cancer','leukemia','lymphoma',
    'prostate cancer','pancreatic adenocarcinoma','glioblastoma',
    'papillary thyroid carcinoma','adrenocortical carcinoma','thymoma'
}
CANCER_DISEASES_VAL = {'esophageal squamous cell carcinoma','testicular germ cell tumor'}

CL_TO_CT = {
    'HepG2':'liver cells','A549':'lung cells','MCF7':'specialized integrated cells','HeLa':'specialized integrated cells',
    'K562':'blood cells','Jurkat':'blood cells','THP1':'blood cells','HCT116':'specialized integrated cells',
    'SHSY5Y':'nerve cells','C2C12':'muscle cells','U2OS':'bone cells','PC3':'specialized integrated cells','HEK293':'kidney cells',
    'KYSE-30':'esophageal epithelial cells','TE-1':'esophageal epithelial cells','ARPE-19':'retinal pigment epithelial cells',
    'HEI-OC1':'cochlear hair cells','KGN':'granulosa cells','NT2/D1':'spermatogonia','Ishikawa':'endometrial epithelial cells',
    'ECC-1':'endometrial epithelial cells','SW1353':'chondrocytes','ATDC5':'chondrocytes',
    'Nthy-ori 3-1':'thyrocytes','TPC-1':'thyrocytes','NCI-H295R':'adrenocortical cells','SW-13':'adrenocortical cells',
    'JEG-3':'trophoblasts','BeWo':'trophoblasts','HTR-8/SVneo':'trophoblasts',
    'GH3':'pituitary endocrine cells','HP75':'pituitary endocrine cells','sHPT-1':'parathyroid chief cells','TT':'thyrocytes',
    'Raji':'lymphocytes','Daudi':'lymphocytes','Ramos':'lymphocytes','SU-DHL-4':'lymphocytes','U937':'blood cells'
}

FUZZY_TEMPLATES = {
    "metastasis": [
        "metastatic focus documented",
        "compatible with metastatic sampling",
        "suggestive of dissemination",
        "secondary lesion context",
        "metastasis noted",
        "biopsy from distant site",
        "evidence of spread",
        "distant involvement recorded",
        "lesion away from the primary",
        "metastatic sampling"
    ],
    "blood": [
        "circulating specimen",
        "liquid biopsy",
        "peripheral blood sampling",
        "blood-derived material",
        "plasma-based sample",
        "serum collection",
        "cell-free fraction",
        "venous draw",
        "hematologic specimen",
        "blood sample"
    ]
}

def stable_union(a, b):
    seen, out = set(), []
    for v in a + b:
        if v not in seen:
            out.append(v); seen.add(v)
    return out

def merge_synonyms(sa, sb):
    keys = set(sa.keys()) | set(sb.keys())
    merged = {}
    for k in keys:
        la, lb = sa.get(k, []), sb.get(k, [])
        merged[k] = stable_union(la, lb) if (la or lb) else [k]
    return merged

def merge_SEM(sem_list):
    out = {}
    for sem in sem_list:
        for k, v in sem.items():
            if k not in out: out[k] = {}
            for k2, v2 in v.items():
                if k2 not in out[k]: out[k][k2] = {}
                for key_leaf, mapping in v2.items():
                    lst = out[k][k2].setdefault(key_leaf, [])
                    for itm in mapping:
                        if itm not in lst: lst.append(itm)
    return out

def build_train_cfg():
    cfg = {}
    for cat in CATS:
        key = CAT_TO_KEY[cat]
        a = ORIG_TRAIN.get(key, [])
        b = TEST2.get(key, [])
        cfg[key] = stable_union(a, b)
    cfg["SYNONYMS"] = expand_synonyms_variants(merge_synonyms(ORIG_TRAIN["SYNONYMS"], TEST2["SYNONYMS"]))
    cfg["SEM"] = merge_SEM([SEM_ORIG, SEM_TEST2])
    cfg["CANCER"] = set(CANCER_DISEASES_TRAIN)
    return cfg

def build_val_cfg(train_cfg):
    cfg = {}
    for cat in CATS:
        key = CAT_TO_KEY[cat]
        base_vals = BASE_EVAL.get(key, [])
        if cat in OPEN_CATS:
            tr_vals = set(train_cfg.get(key, []))
            vals = [v for v in base_vals if v not in tr_vals]
            cfg[key] = vals if vals else base_vals[:]
        else:
            cfg[key] = base_vals[:]
    syn = {}
    for v, sv in BASE_EVAL["SYNONYMS"].items():
        owner_keys = sum([v in cfg[CAT_TO_KEY[c]] for c in CATS if CAT_TO_KEY.get(c)])
        if owner_keys: syn[v] = sv[:]
    cfg["SYNONYMS"] = expand_synonyms_variants(syn)
    cfg["SEM"] = deepcopy(SEM_VAL)
    cfg["CANCER"] = set(CANCER_DISEASES_VAL)
    return cfg

class Balancer:
    def __init__(self):
        self.counts = {}
        self.cyclers = {}
    def inc(self, cat, val):
        self.counts.setdefault(cat, {})
        self.counts[cat][val] = self.counts[cat].get(val, 0) + 1
    def pick_min(self, cat, allowed):
        if not allowed:
            self.inc(cat, "unknown")
            return "unknown"
        self.counts.setdefault(cat, {})
        minc = min(self.counts[cat].get(v, 0) for v in allowed)
        cands = [v for v in allowed if self.counts[cat].get(v, 0) == minc]
        choice = random.choice(cands)
        self.inc(cat, choice)
        return choice
    def pick_cycle(self, cat, allowed):
        if not allowed:
            self.inc(cat, "unknown")
            return "unknown"
        self.cyclers.setdefault(cat, 0)
        idx = self.cyclers[cat] % len(allowed)
        self.cyclers[cat] += 1
        choice = allowed[idx]
        self.inc(cat, choice)
        return choice

class FuzzyManager:
    def __init__(self, total_limit=50):
        self.total_limit = total_limit
        self.used = 0
    def allow(self):
        if self.used >= self.total_limit:
            return False
        self.used += 1
        return True

def adjust_time_response(rec, cfg):
    t = rec.get("treatment")
    tt = rec.get("treatment_time")
    rr = rec.get("response")
    if t in {"no treatment","unknown",""}:
        rec["treatment_time"] = "unknown"
        rec["response"] = "unknown"
        return
    if tt in {"pre treatment"}:
        rec["response"] = "unknown"
    relapse_like = {"relapse","recurrence"}
    success_like = {"success","complete response"}
    if tt in relapse_like and rr in success_like:
        alt_times = [x for x in cfg["TREATMENT_TIME"] if x not in {"relapse","not applicable","unknown"}]
        rec["treatment_time"] = random.choice(alt_times) if alt_times else "post treatment"
    if tt in relapse_like and rr in {"stable disease","sensitivity","partial response"}:
        rec["response"] = "progressive disease" if "progressive disease" in cfg["RESPONSE"] else "unknown"
    if rr == "not applicable":
        rec["response"] = "unknown"
    if tt == "not applicable":
        rec["treatment_time"] = "unknown"

def adjust_library_for_source(rec, cfg):
    src = rec.get("sequencing_source","")
    lib = rec.get("library_selection","")
    libs = [v for v in cfg["LIB_SEL"] if v in {"polyA","inverse rRNA","hybrid selection","small RNA","other","unknown"}]
    if src in {"single-cell","spatial"}:
        bad = {"inverse rRNA","hybrid selection"}
        if lib in bad:
            cand = [x for x in libs if x not in bad]
            rec["library_selection"] = random.choice(cand) if cand else "other"

def choose_coherent_record(cfg, text_upper_tokens, bal: Balancer):
    SEM = cfg["SEM"]
    diseases = [v for v in cfg["DISEASE"]]
    d = bal.pick_cycle("DISEASE", diseases)
    is_cancer = "true" if d in cfg["CANCER"] else "false"
    d_orgs = SEM.get("disease",{}).get("organ",{}).get(d, cfg["ORGAN"]) or cfg["ORGAN"]
    org = bal.pick_min("ORGAN", d_orgs)
    cl_candidates = []
    for cl, orgs in SEM.get("cell_line",{}).get("organ",{}).items():
        if cl in cfg["CELL_LINE"] and org in orgs:
            cl_candidates.append(cl)
    if cl_candidates:
        cl = bal.pick_min("CELL_LINE", cl_candidates)
    else:
        cl = "not applicable" if "not applicable" in cfg["CELL_LINE"] else bal.pick_min("CELL_LINE", [])
    cl_orgs = SEM.get("cell_line",{}).get("organ",{}).get(cl, [org])
    ct_map = CL_TO_CT.get(cl, "")
    ct_allowed_orgs = SEM.get("cell_type",{}).get("organ",{})
    if ct_map and org not in ct_allowed_orgs.get(ct_map, [org]):
        org_opts = [o for o in d_orgs if o in cl_orgs and o in ct_allowed_orgs.get(ct_map, [o])]
        if org_opts:
            org = bal.pick_min("ORGAN", org_opts)
    if ct_map and ct_map in ct_allowed_orgs.get(org, cfg["CELL_TYPE"]):
        ct = ct_map
    else:
        ct = bal.pick_min("CELL_TYPE", ct_allowed_orgs.get(org, cfg["CELL_TYPE"]) or cfg["CELL_TYPE"])
    if ct == "primary tissue" and "not applicable" in cfg["CELL_LINE"]:
        cl = "not applicable"
    inv = SEM.get("library_source",{}).get("cell_type",{})
    src_pref = []
    for s, cts in inv.items():
        if ct in cts:
            src_pref.append(s)
    src = bal.pick_min("SEQ_SRC", src_pref if src_pref else cfg["SEQ_SRC"])
    lib_all = [v for v in cfg["LIB_SEL"] if v in {"polyA","inverse rRNA","hybrid selection","small RNA","other","unknown"}]
    if src in {"single-cell","spatial"}:
        lib_candidates = [v for v in lib_all if v not in {"inverse rRNA","hybrid selection"}]
    else:
        lib_candidates = lib_all[:]
    library_selection = bal.pick_min("LIB_SEL", lib_candidates or ["other"])
    if is_cancer == "false":
        has_blood = ("BLOOD" in text_upper_tokens)
        if has_blood and "blood" in cfg["BIOPSY_SITE"]:
            bs = "blood"; bt = "blood"
        else:
            bs = org if org in cfg["BIOPSY_SITE"] else bal.pick_min("BIOPSY_SITE", cfg["BIOPSY_SITE"])
            bt = "primary"
    else:
        has_met = ("METASTASIS" in text_upper_tokens)
        has_blood = ("BLOOD" in text_upper_tokens)
        if has_blood and "blood" in cfg["BIOPSY_SITE"]:
            bs = "blood"; bt = "blood"
        elif has_met:
            alt_sites = [x for x in cfg["BIOPSY_SITE"] if x != org] or cfg["BIOPSY_SITE"]
            bs = bal.pick_min("BIOPSY_SITE", alt_sites)
            bt = "metastasis"
        else:
            bt_choice = bal.pick_min("BIOPSY_TYPE_CANCER", ["primary","metastasis","blood"])
            if bt_choice == "blood" and "blood" in cfg["BIOPSY_SITE"]:
                bs = "blood"; bt = "blood"
            elif bt_choice == "metastasis":
                alt_sites = [x for x in cfg["BIOPSY_SITE"] if x != org] or cfg["BIOPSY_SITE"]
                bs = bal.pick_min("BIOPSY_SITE", alt_sites)
                bt = "metastasis"
            else:
                bs = org if org in cfg["BIOPSY_SITE"] else bal.pick_min("BIOPSY_SITE", cfg["BIOPSY_SITE"])
                bt = "primary"
    tr_allowed = []
    tmap = SEM.get("treatment",{}).get("disease",{})
    for t, ds in tmap.items():
        if d in ds and t in cfg["TREATMENT"]:
            tr_allowed.append(t)
    treatment = bal.pick_min("TREATMENT", tr_allowed if tr_allowed else cfg["TREATMENT"])
    if treatment in {"no treatment","unknown",""}:
        tt = "unknown"
        rr = "unknown"
    else:
        tt_opts = SEM.get("treatment_time",{}).get("treatment",{}).get(treatment, cfg["TREATMENT_TIME"]) or cfg["TREATMENT_TIME"]
        rr_opts = SEM.get("response",{}).get("treatment",{}).get(treatment, cfg["RESPONSE"]) or cfg["RESPONSE"]
        tt_opts = [x for x in tt_opts if x != "not applicable"]
        rr_opts = [x for x in rr_opts if x != "not applicable"]
        tt = bal.pick_min("TREATMENT_TIME", tt_opts or ["unknown"])
        rr = bal.pick_min("RESPONSE", rr_opts or ["unknown"])
    age = bal.pick_min("AGE", cfg["AGE"])
    sex_allowed = cfg["SEX"]
    if org in SEM.get("sex",{}).get("organ",{}):
        sex_allowed = SEM["sex"]["organ"][org]
    if d in SEM.get("sex",{}).get("disease",{}):
        sex_allowed = SEM["sex"]["disease"][d]
    sex = bal.pick_min("SEX", sex_allowed)
    eth = bal.pick_min("ETHNICITY", cfg["ETHNICITY"])
    record = {
        "library_selection": library_selection if library_selection else "unknown",
        "sequencing_source": src if src else "unknown",
        "biopsy_site": bs if bs else "unknown",
        "biopsy_type": bt,
        "cell_line": cl if cl else "unknown",
        "cell_type": ct if ct else "unknown",
        "organ": org if org else "unknown",
        "disease": d if d else "unknown",
        "treatment": treatment if treatment else "unknown",
        "treatment_time": tt if tt else "unknown",
        "response": rr if rr else "unknown",
        "age": age if age else "unknown",
        "sex": sex if sex else "unknown",
        "ethnicity": eth if eth else "unknown",
        "is_cancer": is_cancer,
    }
    adjust_time_response(record, cfg)
    adjust_library_for_source(record, cfg)
    return record

def choose_surface(cfg, cat, value, rec):
    if not value:
        return ""
    if cat in {"treatment","treatment_time","response"} and rec.get("treatment") in {"no treatment","unknown",""}:
        return ""
    if value in {"unknown","no treatment"}:
        return ""
    syns = cfg["SYNONYMS"].get(value, [value])
    return random.choice(syns) if random.random() < 0.7 and syns else value

def insert_biopsy_tag_adjacent(text: str, biopsy_type: str, tag: str) -> str:
    if biopsy_type == "metastasis":
        kw = "metastasis"
    elif biopsy_type == "blood":
        kw = "blood"
    else:
        return text
    pat = re.compile(rf"(?i)\b{re.escape(kw)}\b")
    if pat.search(text):
        before = random.random() < 0.5
        def repl(m):
            return (f"{tag} {m.group(0)}") if before else (f"{m.group(0)} {tag}")
        return cleanup_uppercase_garbage(pat.sub(repl, text, count=1))
    if random.random() < 0.5:
        return cleanup_uppercase_garbage(f"{kw} {tag} {text}")
    else:
        return cleanup_uppercase_garbage(f"{text} {kw} {tag}")

def process_tsv(in_path, out_path, cfg, fuzzyman):
    df = pd.read_csv(in_path, sep="\t", dtype=str).fillna("")
    bal = Balancer()
    rows = []
    for _, row in df.iterrows():
        run_acc = row.get("run_accession","")
        raw_text = row.get("summary_new","")
        text = remove_years_months_with_numbers(raw_text)
        upper_tokens = set([t for t in re.findall(r"[A-Z][A-Z0-9_]*", text)])
        rec = choose_coherent_record(cfg, upper_tokens, bal)
        surfaces = {cat: choose_surface(cfg, cat, rec[cat], rec) for cat in CATS}
        if surfaces.get("treatment_time"):
            text = replace_time_fragments(text, surfaces["treatment_time"])
        replaced_text, replaced_cats = replace_markers_with_values(text, surfaces)
        for cat in CATS:
            if cat not in replaced_cats and surfaces.get(cat):
                replaced_text = inject_value(replaced_text, surfaces[cat])
        if rec.get("treatment_time") and rec.get("treatment_time") != "unknown":
            replaced_text = re.sub(
                r'(?<![A-Za-z0-9])TREATMENT\s*[_-]?\s*TIME(?![A-Za-z0-9])',
                "",
                replaced_text
            )
            replaced_text = re.sub(
                r'(?<![A-Za-z0-9])TREATMENT(?![A-Za-z0-9])\s*',
                "",
                replaced_text
            )
        if rec.get("age") == "unknown":
            replaced_text = re.sub(r"(?<![A-Za-z0-9])AGE(?![A-Za-z0-9])", "", replaced_text)
        if rec.get("sex") == "unknown":
            replaced_text = re.sub(r"(?<![A-Za-z0-9])SEX(?![A-Za-z0-9])", "", replaced_text)
        if surfaces.get("age"):
            replaced_text = remove_numbers_around_surface_v2(replaced_text, surfaces["age"], window=2)
        age_inserted_numbers = set(re.findall(r"\d+", surfaces.get("age","")))
        replaced_text = scrub_age_following_numbers(replaced_text, age_inserted_numbers, max_numbers=2)
        inserted_numbers_all = set(re.findall(r"\d+", " ".join([v for v in surfaces.values() if v])))
        replaced_text = remove_age_inline_numbers(replaced_text, inserted_numbers_all, window_tokens=4)
        final_phrase = cleanup_uppercase_garbage(replaced_text)
        if rec["is_cancer"] == "true" and rec["biopsy_type"] in {"metastasis","blood"} and fuzzyman.allow():
            tag = random.choice(FUZZY_TEMPLATES[rec["biopsy_type"]])
            final_phrase = insert_biopsy_tag_adjacent(final_phrase, rec["biopsy_type"], tag)
        if not final_phrase.strip():
            pool = [surfaces.get(k,"") for k in ["organ","disease","cell_type","cell_line","biopsy_type","biopsy_site"] if surfaces.get(k)]
            final_phrase = " ".join(pool) if pool else rec["biopsy_type"]
        final_phrase = remove_repeated_uppercase_lots(final_phrase)
        out_row = {"run_accession": run_acc, **{c: rec[c] for c in CATS}, "is_cancer": rec["is_cancer"], "phrase": final_phrase}
        rows.append(out_row)
    out_cols = ["run_accession"] + CATS + ["is_cancer","phrase"]
    out_df = pd.DataFrame(rows, columns=out_cols).fillna("unknown")
    for c in CATS + ["is_cancer","phrase"]:
        if c in out_df:
            out_df[c] = out_df[c].replace({"": "unknown"}) if c != "phrase" else out_df[c].replace({"": " "}).str.strip()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_df.to_csv(out_path, index=False)

if __name__ == "__main__":
    random.seed(42)
    TRAIN_CFG = build_train_cfg()
    VAL_CFG   = build_val_cfg(TRAIN_CFG)
    fuzzyman = FuzzyManager(total_limit=50)
    process_tsv(TRAIN_TSV_IN, TRAIN_CSV_OUT, TRAIN_CFG, fuzzyman)
    process_tsv(VAL_TSV_IN,   VAL_CSV_OUT,   VAL_CFG,   fuzzyman)
