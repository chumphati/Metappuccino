#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, csv, random, sys, time
from pathlib import Path
from typing import List, Tuple, Dict, Iterable, Set
from concurrent.futures import ProcessPoolExecutor, TimeoutError

INFILES = [
    # "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/results/model_replace_final/train_masked.tsv",
    "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/results/model_replace_final/val_masked.tsv",
]
OUTDIR = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/results/model_replace_final/out_final_seq2"

PARALLEL = True
N_PROCS = 50
PER_ROW_TIMEOUT = 60
RESUME = True
PROGRESS_EVERY = 50

STRICT_MODE = True
SOFT_COLLAPSE_TOKENS = 2 if STRICT_MODE else 1
HARD_COLLAPSE_TOKENS = 4 if STRICT_MODE else 6
SAFE_RADIUS_WORDS = 4 if STRICT_MODE else 3
ROLLBACK_LEN_FRAC = 0.35 if STRICT_MODE else 0.45
ROLLBACK_CHRF_MIN = 0.65 if STRICT_MODE else 0.55

CATS = [
    "library_selection","sequencing_source","biopsy_site","biopsy_type",
    "cell_line","cell_type","organ","disease",
    "treatment","treatment_time","response","age","sex","ethnicity"
]
CATS_UP = [c.upper() for c in CATS]
CATS_ALL_UP = CATS_UP + ["IS_CANCER"]

MAX_PER_CAT = {c: 2 for c in CATS_UP}
for c in ["AGE","SEX","ETHNICITY"]:
    MAX_PER_CAT[c] = 1

RNG = random.Random(1337)

def _normalize_list_blob(blob: str) -> Set[str]:
    if not blob:
        return set()
    terms = set()
    cleaned = blob.replace("“","\"").replace("”","\"").replace("’","'")
    parts = re.split(r"[\n\r]+|(?<!\w)\"|\"(?!\w)", cleaned)
    for p in parts:
        if not p: continue
        subparts = re.split(r"\s*,\s*", p)
        for s in subparts:
            s = s.strip()
            if not s or s.lower() == "null":
                continue
            s = re.sub(r"([a-z])([A-Z])", r"\1 \2", s)
            s = re.sub(r"\s+", " ", s).strip(" .;|,")
            if len(s) < 2:
                continue
            terms.add(s)
            atoms = re.split(r"\b(?:of|from|the|a|an|and|,|;)\b", s, flags=re.I)
            for a in atoms:
                a = re.sub(r"\s+", " ", a).strip(" .;|,")
                if a and a.lower() != "null" and len(a) >= 2:
                    terms.add(a)
    out = set()
    for t in terms:
        t2 = t.strip(" .;|,")
        if t2 and t2.lower() != "null":
            out.add(t2)
    return out

_BLOB_ORGAN_TISSUE = r'''
Skin, dermis Facelift skin blood platelet adrenal gland pancreas Foreskin, skin liver skin peripheral blood Salivary gland, submandibular gland Peripheral blood bone marrow bone marrow adrenal gland pleural effusion liver Hand, skin Face, skin, sebaceous gland blood Eye, retina, retinal pigment epithelium blood Pamlar dermis mobilized peripheral blood bone marrow ES-derived cardiac progenitor cells blood Pericardial effusion Eye, retina, retinal pigment epithelium Astrocytes blood Bone, tibia Peripheral blood liver abdomen colon cervical lymph node placenta adrenal gland left supraclavicular lymph node adrenal gland bone marrow adrenal gland pleural effusion liver Skin pleural effusion brain Peripheral blood Placental Villous Tissue Oral cavity, gingiva, epithelium lung blastocyst Normal Prostate bone marrow fat brain renal cortex colon Uterus, cervix colon liver Skin, dermis lung Fetal kidney tonsil bone marrow hypodermis CRC liver metastase Foreskin, skin lung blood Leg, skin Oral cavity, buccal mucosa pleural effusion Eye, retina, retinal pigment epithelium Peripheral blood Prostate Placenta, arterial endothelium Uterus, cervix liver lung Primary immune cells subcutaneous xenograft Peripheral blood left supraclavicular lymph node pleural effusion thyroid gland colon pleural effusion Nodule from palmer fascia Fetal kidney Healthy Mucosa colon Fetal lung colon Disease-modeled retinal pigment epithelial cells Peripheral blood pancreas Colorectal lung not collected FFPE placenta Peripheral blood ocular colon tumor stomach bone marrow Kidney, proximal tubule cell line normal colorectal mucosa colon liver Uterus, cervix tumor biopsy Skin Uterus, cervix sputum lung buccal mucosa epithelium peripheral blood pleural effusion Total clear cell renal cell carcinoma Homogenate gastrocnemius blood tumor blood organoid Foreskin, skin Sessile serrated adenoma/polyp abdomen kidney blood vessel smooth muscle blastocyst pleural effusion Skin liver kidney Fetal kidney adult stem cell blood cervical lymph node pleural effusion MDA-MB-231 Eye, uvea Muscle cell Trophoblast tissue_2 blood thymus blood kidney Kidney organoid Eye, globe umbilical vein breast cancer Uterus, cervix blood peripheral blood Endometrium, epithelium trabecular bone tissue pleural effusion umbilical vein bone marrow Back, skin, epidermis liver pleural effusion Skin lung blood cartilage lung pleural effusion nasopharynx liver supraclavicular adipose tissue Peripheral blood PBMCs Peripheral blood mononuclear cells primary tumor FFPE Peripheral blood lung Skin pleural effusion glioblatoma brain breast liver primary cell type skin Cumulus cells Fetal kidney pleural effusion bone marrow adrenal gland Prostate, epithelium colon Face, skin, sebaceous gland blood Placenta, arterial endothelium mucosa of stomach pleural effusion bone marrow human bronchial epithelia kidney placental artery bone marrow primary fibroblasts lung pleural effusion Pancreas, islets of Langerhans pleural effusion liver Uterus, cervix sperm Placenta, arterial endothelium colon Normal Testicular Microenvironment heart lymph node lung pleural effusion breast lymph node liver esophagus Breast cancer adjacent tissue Placenta, arterial endothelium liver Muscle ovary colon blood Arm, skin blood Ascites adrenal gland hu-mice placenta immune cells (TCR) Peripheral blood SJSA-1 T cells Fetal kidney Eye, uvea liver primary tumor tissue from high grade serous ovarian tumors CTC Transmural LV Apex pleural effusion PBMC pancreas Tumor tissue
'''
_BLOB_CELL_LINES_TYPES = r'''
201B7 TIG3 human dermal fibroblast PC12 HTT-SW2-Q73-TagRFP clone 17 HFF-1 Hep 3B2.1-7 HCC1954 BL A-253 Peripheral blood mononuclear OCI-AML-5 ARP-1 MCF-7 SNU-398 adipose stromal HSC-1 PBMC SZ95 hTERT-RPE1 CD34+ Organoid NCI-H441 Astrocytes U2OS THP-1 IMR-32 Caco-2 HSC-4 cholangiocyte organoids LNCaP SK-N-BE(2)-C HMEC HDMB03 SK-MEL-28 MDA-MB-231 T98G H9 hTERT TIGKs A-549 HS401 KIT-dependent GIST GIST-T1 Airway basal MDS-L primary U373 MG HCT 116 HeLa Acute myeloid leukemia Huh-7 201B7 Human primary adipose stem NCI-H1975 HEK293T iPSC-derived neurons Germinal center B CD19+ CD38+ HUVEC MOLM-1 human embryonic stem TCC-MESO-2 BJ-TERT NCI-H1573 DN3 S18 A-375 NB-1 Human Leukocytes NALM-6 PF179T CAF 293T 3T3 Dual THP-1 PAE-iPS-4 HeLa hiPSC-ECs A-549 PBMCs Primary T LNCaP NCI-H929 FTC-133 H1 IMR-90 Metastatic Human Prostate Cancer Bone marrow mononuclear LL/2 Plasmablast human iPSCs-derived motor neurons BxPC-3 iPSC-derived astrocytes iPS cardiomyocyte acute monocytic leukemia N91 MOLM-13 CA1 M263 Ovarian adenocarcinoma AGS SK-N-AS Keratinocytes RPTEC/TERT1 Neutrophils hepatic CD161+CD8+ non-MAIT HeLa glioblastoma culture G01 501-mel SiHa vitiligo melanocytes alveolar type 2 hMECs neutrophils K-562 SKCO1 jurkat embryonic stem H9 PAAF CCD-1077Sk Primary melanoma RH4 MCF-7 HEK293 T-REx Flp-In HCA2-hTert QGP-1 T MDA-MB-468 Hep-G2 Caki-1 hPSC cardiomyocytes platelet BJ chondrocyte K-562 C666-1 UFRJi008-A SK-MEL-28 immune hPSC1 Huh-7 BT-474 Suction Blisters Granulosa MDA-MB-468 K562 CCL-243 patient-derived xenograft cl_3188 RWPE-1 primary human trabacular meshwork SZ95 Antigen Presenting SCLC CD19-specific CD4+ CAR-T MCF-7 CD138+ Airway basal 786-O endothelial SH-SY5Y A-549 Primary myotube K-562 QGP-1 T MDA-MB-468 Hep-G2 HeLa S3 UM-SCC-1 PAE-iPS-4 derived from iPSCs Leydig Sertoli PMC iPSC-EC monocyte-macrophage lineage MNCs A-427 A-549 MCF-7 hiPSC-L1 H9 SUM159PT OCI-Ly8 Hep 3B2.1-7 TYUC-1 Breast tissue PAE-iPS-4 Hep-G2 A-673 PBMC A2780 HCT 116 NCI-H1963 KYS510 MDCK-SIAT1 A549 CV-hIPS-B CS-THL1 Akata THP-1 macrophages lymphoblastoid line MCF10A THP-1 Primary T HEK293 92-1 OMM2.3 EpCAM- and PSMA- U-937 WI-38 Epithelium from hiPSCs-derived kidney organoid RUES2 CCNB1-eGFP reporter cardiomyocytes
'''
_BLOB_DISEASES = r'''
head and neck squamous cell carcinoma HNSCC pheochromocytoma childhood hepatocellular carcinoma Submandibular gland squamous cell carcinoma idiopathic pulmonary disease acute myeloid leukemia Plasma cell myeloma Invasive breast carcinoma of no special type adult hepatocellular carcinoma skin squamous cell carcinoma Healthy Dupuytrens disease aged lung papillary adenocarcinoma Early Parkinson's disease PD glioblastoma Plasmodium vivax malaria osteosarcoma Childhood acute monocytic leukemia intrahepatic cholangiocarcinoma neuroblastoma colon adenocarcinoma tongue squamous cell carcinoma prostate carcinoma skin melanoma breast adenocarcinoma Sezary's disease lung adenocarcinoma myelodysplastic syndrome colon carcinoma Human papillomavirus-related endocervical adenocarcinoma normal Blast phase chronic myelogenous leukemia BCR-ABL1 positive Pleural epithelioid mesothelioma IgG4-related disease amelanotic melanoma Buccal mucosa squamous cell carcinoma B-cell adult acute lymphocytic leukemia TKI-resistant Malignant tumors of the mouse pulmonary system Rheumatoid Arthritis RA pancreatic ductal adenocarcinoma Li-Fraumeni syndrome LFS Idiopathic pulmonary arterial hypertension IPAH osteoarthritis gastric adenocarcinoma healthy Endometriosis infertility multiple myeloma IPF renal cell carcinoma lean pancreatic somatostatinoma uveal melanoma Ewing sarcoma retinoblastoma dilated cardiomyopathy breast pleomorphic carcinoma diffuse large B-cell lymphoma germinal center B-cell type Esophageal small cell carcinoma High grade B-cell lymphoma with MYC BCL2 and BCL6 rearrangements EBV-related Burkitt lymphoma hepatoblastoma endometriosis nasopharynx carcinoma neuroendocrine esophageal carcinoma allergy biliary tract cancer Australian Chronic Allograft Dysfunction neuroblastoma
'''

def _build_lexicons():
    org_terms = _normalize_list_blob(_BLOB_ORGAN_TISSUE)
    cell_terms = _normalize_list_blob(_BLOB_CELL_LINES_TYPES)
    disease_terms = _normalize_list_blob(_BLOB_DISEASES)

    def _single_words(terms: Set[str]) -> Set[str]:
        out = set()
        for t in terms:
            for w in re.findall(r"[A-Za-z][A-Za-z\-']{1,}", t):
                if len(w) >= 3:
                    out.add(w.lower())
        return out

    organ_single = _single_words(org_terms)
    cell_single = _single_words(cell_terms)

    base_def = {"tissue","biopsy","biopsies","sample","specimen","organ","cell","cells","pbmc","pbmcs","fibroblast","skin","blood","serum","plasma"}
    base_def |= {w for w in organ_single if w not in {"the","and","null"}}
    base_def |= {w for w in cell_single if w not in {"the","and","null"}}

    dis_phr = sorted([d for d in disease_terms if len(d) <= 60], key=len, reverse=True)
    dis_alt = "|".join(re.escape(d) for d in dis_phr) if dis_phr else ""

    term2cat = {}
    for t in org_terms:
        term2cat[t.lower()] = "ORGAN"
    for t in cell_terms:
        key = t.lower()
        if re.search(r"\b(fibroblast|cell|cells|astrocyte|keratinocyte|neutrophil|t[- ]?cell|b[- ]?cell|melanocyte|endothelial|epithelial)\b", key):
            term2cat[key] = "CELL_TYPE"
        else:
            term2cat[key] = "CELL_LINE"
    for t in disease_terms:
        term2cat[t.lower()] = "DISEASE"

    # Noms communs candidats à supprimer (mots simples uniquement)
    COMMON_NOUNS = set()
    for ww in (organ_single | cell_single):
        if ww.isalpha() and len(ww) >= 3 and ww not in {"the","and","null"}:
            COMMON_NOUNS.add(ww)
    # Quelques génériques très fréquents déjà inclus : fibroblast, skin, blood, plasma, serum, placenta, kidney, colon, lung, etc.

    return base_def, dis_alt, term2cat, COMMON_NOUNS

_DEF_TERMS_SET, _DISEASE_ALT, _TERM2CAT, _COMMON_NOUNS = _build_lexicons()
_CATS_LOWER = {c.lower() for c in CATS_UP}

def read_tsv(path: str) -> Iterable[Dict[str,str]]:
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            yield row

def prepare_outputs(base_name: str):
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    out_clean = os.path.join(OUTDIR, f"{base_name}.cleaned.tsv")
    out_metrics = os.path.join(OUTDIR, f"{base_name}.cleaned.metrics.tsv")
    cleaned_exists = os.path.exists(out_clean)
    metrics_exists = os.path.exists(out_metrics)
    f_clean = open(out_clean, "a", encoding="utf-8", newline="")
    f_metrics = open(out_metrics, "a", encoding="utf-8", newline="")
    w_clean = csv.writer(f_clean, delimiter="\t")
    w_metrics = csv.writer(f_metrics, delimiter="\t")
    if not cleaned_exists:
        w_clean.writerow(["run_accession","summary_original","summary_new"])
    if not metrics_exists:
        w_metrics.writerow(["run_accession","auc_struct","chrf","jaccard_token"])
    return f_clean, w_clean, f_metrics, w_metrics, out_clean, out_metrics

def count_done_rows(out_path: str) -> int:
    if not RESUME or not os.path.exists(out_path):
        return 0
    with open(out_path, "r", encoding="utf-8") as f:
        n = sum(1 for _ in f)
    return max(0, n - 1)

def remove_edge_chunks_and_brackets(text: str) -> str:
    text = re.sub(r"^(?:\s*(?:CHUNK|\[[^\[\]]*\])\s*)+", "", text)
    text = re.sub(r"(?:\s*(?:CHUNK|\[[^\[\]]*\])\s*)+$", "", text)
    return text

def strip_all_bracket_numbers(text: str) -> str:
    return re.sub(r"\[[^\[\]]*\]", "", text)

def normalize_spaces_light(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" ?([,;|.]) ?", r"\1 ", text)
    return text.strip()

def restore_missing_spaces(text: str) -> str:
    text = re.sub(r"([,.;|])(?=\S)", r"\1 ", text)
    text = re.sub(r"(?<=\d)(?=[A-Za-z])", " ", text)
    text = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", text)
    for cat in CATS_ALL_UP:
        text = re.sub(rf"(?<!\s)(?={re.escape(cat)}\b)", " ", text)
        text = re.sub(rf"(?<=\b{re.escape(cat)})(?!\s|[,.;|])", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def collapse_chunk_storm(text: str, max_chunks: int = 40) -> str:
    text = re.sub(r"\bCHUNK\b\s*\[\d+\]", "CHUNK", text)
    text = re.sub(r"(?:\bCHUNK\b[\s,;.|]*){2,}", " CHUNK ", text)
    chunks = list(re.finditer(r"\bCHUNK\b", text))
    if len(chunks) > max_chunks:
        keep = set(m.span() for m in chunks[:max_chunks])
        out = []
        i = 0
        for m in chunks:
            s, e = m.span()
            out.append(text[i:s])
            if (s, e) in keep:
                out.append("CHUNK")
            i = e
        out.append(text[i:])
        text = "".join(out)
    return normalize_spaces_light(text)

def cat_regex_union(boundary_safe: bool = True) -> re.Pattern:
    cats = "|".join(map(re.escape, CATS_ALL_UP))
    if boundary_safe:
        return re.compile(rf"(?<![A-Z0-9_])({cats})(?![A-Z0-9_])")
    else:
        return re.compile(rf"({cats})")

CAT_BOUND = cat_regex_union(True)

def find_all_cat_occurrences(text: str) -> List[Tuple[str,int,int]]:
    return [(m.group(1), m.start(), m.end()) for m in CAT_BOUND.finditer(text.upper())]

BROKEN_PATTERNS = [
    (re.compile(r"(?<![A-Z0-9_])LIBRARY\s*[_\-]\s*SELECTION(?![A-Z0-9_])", re.I), "LIBRARY_SELECTION"),
    (re.compile(r"(?<![A-Z0-9_])SEQUENCING\s*[_\-]\s*SOURCE(?![A-Z0-9_])", re.I), "SEQUENCING_SOURCE"),
    (re.compile(r"(?<![A-Z0-9_])BIOPSY\s*[_\-]\s*SITE(?![A-Z0-9_])", re.I), "BIOPSY_SITE"),
    (re.compile(r"(?<![A-Z0-9_])CELL\s*[_\-]\s*TYPE(?![A-Z0-9_])", re.I), "CELL_TYPE"),
    (re.compile(r"(?<![A-Z0-9_])T\s*[_\-]\s*TIME(?![A-Z0-9_])", re.I), "TREATMENT_TIME"),
    (re.compile(r"(?<![A-Z0-9_])TREAT\s*MENT?\s*[_\-]\s*TIME(?![A-Z0-9_])", re.I), "TREATMENT_TIME"),
    (re.compile(r"(?<![A-Z0-9_])EORGAN(?![A-Z0-9_])", re.I), "ORGAN"),
    (re.compile(r"(?<![A-Z0-9_])CELL\s*[_\-]?\s*TYP(?![A-Z0-9_])", re.I), "CELL_TYPE"),
]

def repair_cut_categories(text: str) -> str:
    for pat, rep in BROKEN_PATTERNS:
        text = pat.sub(rep, text)
    return text

def normalize_cat_tokens(text: str) -> str:
    text = re.sub(r"\bT\s*[_\-]\s*TIME\b", "TREATMENT_TIME", text)
    text = re.sub(r"\bTREAT\s*MENT?\s*[_\-]\s*TIME\b", "TREATMENT_TIME", text, flags=re.I)
    for cat in CATS_UP + ["IS_CANCER"]:
        if "_" in cat:
            a, b = cat.split("_", 1)
            text = re.sub(rf"(?<![A-Z0-9_]){a}\s*[_]\s*{b}(?![A-Z0-9_])", cat, text, flags=re.I)
    return text

_SPLIT_CAT_CAND = re.compile(r"\b([A-Za-z]{3,})\s+(on|in)\s+([A-Za-z]{2,})(?:\s*(’s|'s))?\b")

def _looks_like_category_value_joined(c: str) -> bool:
    c = c.lower()
    if c.endswith(("’s","'s")): c = c[:-2]
    if re.search(r"(parkinson|alzheimer|huntington)s?$", c): return True
    if re.search(r"(itis|oma|carcinoma|sarcoma|glioma|leukemia|lymphoma)$", c): return True
    if re.search(r"(virus|viral|influenza|coronavirus)$", c): return True
    if re.search(r"(insulin|interferon|dexamethasone|tamoxifen|cisplatin)$", c): return True
    return False

def drop_split_category_values(text: str) -> str:
    def _repl(m: re.Match) -> str:
        joined = f"{m.group(1)}{m.group(2)}{m.group(3)}{m.group(4) or ''}"
        return "" if _looks_like_category_value_joined(joined) else m.group(0)
    return _SPLIT_CAT_CAND.sub(_repl, text)

DEF_TERMS = r"(?:%s)" % "|".join(sorted({re.escape(t) for t in _DEF_TERMS_SET}))
def redact_around_definition_terms(text: str) -> str:
    text = re.sub(rf"\b[A-Za-z]{{2,}}\s+({DEF_TERMS})\b", r"\1", text, flags=re.I)
    text = re.sub(rf"\b({DEF_TERMS})\s+[A-Za-z]{{2,}}\b", r"\1", text, flags=re.I)
    text = re.sub(rf"\b({DEF_TERMS})\s+(of|in|on|the|a|an|and)\s+\w+\b", r"\1", text, flags=re.I)
    return text

SEP_AFTER = re.compile(r"\s*[:=\-–—]\s*")
STOP_STRONG = re.compile(r"(?:\bCHUNK\b|[;,.|])")
WORD = re.compile(r"\S+")

def _is_cat_like(tok: str) -> bool:
    return tok.upper() in CATS_ALL_UP

def collapse_values_after_categories(text: str) -> Tuple[str, List[int]]:
    freed = []; i = 0; out = []
    while True:
        m = CAT_BOUND.search(text, i)
        if not m:
            out.append(text[i:]); break
        out.append(text[i:m.start()])
        cat = m.group(1).upper()
        if cat == "IS_CANCER":
            i = m.end()
            ms = SEP_AFTER.match(text, i)
            if ms:
                j = ms.end(); tokens = 0
                while tokens < HARD_COLLAPSE_TOKENS:
                    ms2 = STOP_STRONG.search(text, j)
                    m2 = CAT_BOUND.search(text, j)
                    nxt = min(ms2.start() if ms2 else len(text), m2.start() if m2 else len(text))
                    mtok = WORD.search(text, j, nxt)
                    if not mtok: break
                    j = mtok.end(); tokens += 1
                i = j
            else:
                j = m.end()
                mt = WORD.match(text, j)
                if mt and not _is_cat_like(mt.group(0)) and not STOP_STRONG.match(mt.group(0)):
                    i = mt.end()
            continue
        out.append(cat)
        ms = SEP_AFTER.match(text, m.end())
        if ms:
            j = ms.end(); start_val = j; tokens = 0
            while tokens < HARD_COLLAPSE_TOKENS:
                ms2 = STOP_STRONG.search(text, j)
                m2 = CAT_BOUND.search(text, j)
                nxt = min(ms2.start() if ms2 else len(text), m2.start() if m2 else len(text))
                mtok = WORD.search(text, j, nxt)
                if not mtok: break
                j = mtok.end(); tokens += 1
            if j > start_val: freed.append(start_val)
            i = j; continue
        j = m.end(); start_val = j; tokens = 0
        while tokens < SOFT_COLLAPSE_TOKENS:
            mt = WORD.match(text, j)
            if not mt: break
            tok = mt.group(0)
            if _is_cat_like(tok) or STOP_STRONG.match(tok): break
            if any(c.islower() for c in tok):
                j = mt.end(); tokens += 1
            else:
                break
        if j > start_val:
            freed.append(start_val); i = j
        else:
            i = m.end()
    return "".join(out), freed

def collapse_values_final_pass(text: str) -> str:
    i = 0; out = []
    while True:
        m = CAT_BOUND.search(text, i)
        if not m:
            out.append(text[i:]); break
        out.append(text[i:m.start()])
        cat = m.group(1).upper()
        if cat == "IS_CANCER":
            i = m.end(); continue
        out.append(cat)
        ms = SEP_AFTER.match(text, m.end())
        if ms:
            j = ms.end(); tokens = 0
            while tokens < HARD_COLLAPSE_TOKENS:
                ms2 = STOP_STRONG.search(text, j)
                m2 = CAT_BOUND.search(text, j)
                nxt = min(ms2.start() if ms2 else len(text), m2.start() if m2 else len(text))
                mtok = WORD.search(text, j, nxt)
                if not mtok: break
                j = mtok.end(); tokens += 1
            i = j; continue
        j = m.end(); tokens = 0
        while tokens < SOFT_COLLAPSE_TOKENS:
            mt = WORD.match(text, j)
            if not mt: break
            tok = mt.group(0)
            if _is_cat_like(tok) or STOP_STRONG.match(tok): break
            if any(c.islower() for c in tok):
                j = mt.end(); tokens += 1
            else:
                break
        i = j
    return "".join(out)

def tokenize_words(text: str) -> List[Tuple[int,int]]:
    return [m.span() for m in re.finditer(r"\S+", text)]

def chunk_positions(text: str) -> List[Tuple[int,int]]:
    return [m.span() for m in re.finditer(r"\bCHUNK\b", text)]

def cat_positions(text: str) -> List[Tuple[int,int,str]]:
    return [(m.start(), m.end(), m.group(1).upper()) for m in CAT_BOUND.finditer(text)]

def _snap_to_boundary(text: str, pos: int) -> int:
    if 0 < pos < len(text) and text[pos-1].isalnum() and text[pos].isalnum():
        l = pos
        while l > 0 and text[l-1].isalnum(): l -= 1
        r = pos
        while r < len(text) and text[r].isalnum(): r += 1
        return r if (pos - l) > (r - pos) else l
    if 0 < pos < len(text) and (text[pos-1] == "_" or text[pos] == "_"):
        return pos+1 if text[pos] == "_" else pos-1
    return pos

def choose_safe_chunk_for_insertion(text: str) -> Tuple[int,int]:
    ch = [(_snap_to_boundary(text, s), _snap_to_boundary(text, e)) for (s,e) in chunk_positions(text)]
    if not ch: return (-1,-1)
    cats = cat_positions(text)
    words = tokenize_words(text)
    def widx(pos):
        for i,(s,e) in enumerate(words):
            if s <= pos < e: return i
        return -1
    for cs,ce in ch:
        wi = widx(cs); close = False
        for as_,ae,_ in cats:
            wj = widx(as_)
            if wj != -1 and wi != -1 and abs(wi - wj) <= SAFE_RADIUS_WORDS:
                close = True; break
        if not close:
            return (cs,ce)
    return ch[0]

def dedupe_categories(text: str) -> Tuple[str, Dict[str,List[Tuple[int,int]]], List[int]]:
    n = len(text)
    occ = find_all_cat_occurrences(text)
    by = {c: [] for c in CATS_UP}
    for cat,s,e in occ:
        if cat == "IS_CANCER": continue
        by[cat].append((s,e, 1.0 - min(abs((s+e)/2 - n/2)/(n/2 if n else 1), 1.0)))
    to_remove, kept = [], {c: [] for c in CATS_UP}
    for cat, spans in by.items():
        if not spans: continue
        limit = MAX_PER_CAT[cat]
        spans_sorted = sorted(spans, key=lambda x:(x[2], -x[0]), reverse=True)
        kept[cat] = [(s,e) for (s,e,_) in spans_sorted[:limit]]
        to_remove += [(s,e) for (s,e,_) in spans_sorted[limit:]]
    vacancies = []
    for s,e in sorted(to_remove, key=lambda x:x[0], reverse=True):
        vacancies.append(s)
        text = text[:s] + text[e:]
    for m in list(CAT_BOUND.finditer(text)):
        if m.group(1).upper() == "IS_CANCER":
            text = text[:m.start()] + text[m.end():]
    return text, kept, vacancies

def _allowed_boundaries(text: str) -> List[int]:
    seps = set(".;|")
    bounds = [0] + [m.end() for m in re.finditer(r"\S+(?:\s+|$)", text)]
    out = []
    for pos in bounds:
        if pos <= 0 or pos >= len(text):
            continue
        i = pos - 1
        while i >= 0 and text[i].isspace():
            i -= 1
        j = pos
        while j < len(text) and text[j].isspace():
            j += 1
        left = text[i] if 0 <= i < len(text) else ""
        right = text[j] if 0 <= j < len(text) else ""
        if left in seps or right in seps:
            continue
        out.append(pos)
    return out

def _scatter_insert_missing(text: str, missing: List[str]) -> str:
    if not missing:
        return text
    allowed = _allowed_boundaries(text)
    if not allowed:
        return text
    RNG.shuffle(allowed)
    picks, used = [], set()
    for pos in allowed:
        if any(abs(pos - p) < 2 for p in used):
            continue
        picks.append(pos)
        used.add(pos)
        if len(picks) >= len(missing):
            break
    while len(picks) < len(missing):
        picks.append(RNG.choice(allowed))
    pairs = list(zip(picks, missing))
    pairs.sort(key=lambda x: x[0])
    out = []
    last = 0
    for pos, cat in pairs:
        pos = _snap_to_boundary(text, pos)
        if pos <= 0: pos = 1
        if pos >= len(text): pos = len(text) - 1
        out.append(text[last:pos])
        out.append(" " + cat + " ")
        last = pos
    out.append(text[last:])
    text = "".join(out)
    text = normalize_spaces_light(text)
    text = re.sub(r"^\s+", "", text)
    text = re.sub(r"\s+$", "", text)
    return text

def ensure_all_categories(text: str, kept: Dict[str,List[Tuple[int,int]]], vacancies: List[int]) -> str:
    missing = [c for c in CATS_UP if not re.search(rf"(?<![A-Z0-9_]){re.escape(c)}(?![A-Z0-9_])", text)]
    if not missing:
        return text
    tl = list(text); offset = 0
    for cat in list(missing):
        if not vacancies:
            break
        pos = _snap_to_boundary(text, vacancies.pop(0))
        ins = " " + cat + " "
        tl.insert(pos + offset, ins)
        offset += len(ins)
        missing.remove(cat)
    text = "".join(tl)
    if not missing:
        return normalize_spaces_light(text)
    for cat in list(missing):
        cs, ce = choose_safe_chunk_for_insertion(text)
        if cs != -1:
            text = text[:cs] + " " + cat + " " + text[ce:]
            missing.remove(cat)
    if not missing:
        return normalize_spaces_light(text)
    text = _scatter_insert_missing(text, missing)
    return normalize_spaces_light(text)

def purge_final(text: str) -> str:
    text = re.sub(r"\bCHUNK\b", " ", text)
    text = strip_all_bracket_numbers(text)
    return normalize_spaces_light(text)

_STOP2 = {"a","an","of","in","on","at","by","to","or","as","we","us","et","al"}
def clean_orphan_fragments(text: str) -> str:
    for cat in CATS_UP:
        text = re.sub(rf"(?<![A-Z0-9_]){re.escape(cat)}\s+[a-zA-Z]([^\w]|$)", rf"{cat}\1", text)
    def _kill_small(m):
        tok = m.group(1)
        return "" if tok.lower() not in _STOP2 else tok
    text = re.sub(r"\s([a-z]{1,2})(?=\s|[;,.|])", _kill_small, text)
    text = re.sub(r"\bORGAN\s+ism\b", "organism", text, flags=re.I)
    return text

# ————— suppression souple des noms communs (singulier/pluriel, ponctuation autorisée)
def _build_common_nouns_pattern() -> re.Pattern:
    if not _COMMON_NOUNS:
        return re.compile(r"$a")
    alts = []
    for w in sorted(_COMMON_NOUNS, key=len, reverse=True):
        esc = re.escape(w)
        # pluriels simples -s / -es + tolérance apostrophes
        alts.append(rf"{esc}(?:e?s)?")
    pat = r"(?i)\b(?:%s)\b" % "|".join(alts)
    return re.compile(pat)

_COMMON_NOUNS_PAT = _build_common_nouns_pattern()

def _remove_common_nouns(text: str) -> str:
    def repl(m: re.Match) -> str:
        g = m.group(0)
        # ne jamais supprimer une CAT majuscule
        if g.upper() in CATS_ALL_UP:
            return g
        return " "
    text = _COMMON_NOUNS_PAT.sub(repl, text)
    return normalize_spaces_light(restore_missing_spaces(text))

# ————— renforcement DISEASE via listes
_DISEASE_LIKE = re.compile(
    r"""
    (?:\b[A-Z][a-z]+(?:’s|'s)?\s+disease\b)|
    (?:\b[a-z][a-z\-]*(?:itis|oma|carcinoma|sarcoma|glioma|leukemia|lymphoma)\b)|
    (?:\b(parkinson|alzheimer|huntington)(?:’s|'s)?\b)
    """ + (f"|(?:\\b(?:{_DISEASE_ALT})\\b)" if _DISEASE_ALT else ""),
    re.I | re.X
)

def redact_plain_disease_values(text: str, original: str) -> str:
    cand = _DISEASE_LIKE.sub("", text)
    def _chrf(a,b):
        def ngrams(s, n): s = s.replace(" ",""); return [s[i:i+n] for i in range(len(s)-n+1)]
        from collections import Counter
        vals=[]
        for n in range(2,7):
            R=Counter(ngrams(a,n)); H=Counter(ngrams(b,n))
            overlap=sum((R&H).values()); p=overlap/max(1,sum(H.values())); r=overlap/max(1,sum(R.values()))
            vals.append(0 if p+r==0 else (1+4)*p*r/(4*p+r))
        return sum(vals)/len(vals)
    chrf_before = _chrf(original, text)
    chrf_after  = _chrf(original, cand)
    dlen = 0.0 if len(original)==0 else abs(len(cand)-len(original))/len(original)
    if (chrf_after + 1e-6) >= chrf_before*0.97 and dlen <= 0.35:
        return cand
    return text

STUDY_PREFIXES = [
    "study shows","study describes","study details","study outlines","study investigates",
    "study evaluates","study examines","study highlights","study reports","study presents",
    "study summarizes","study documents","study characterizes","study assesses","study explores",
    "study focuses on","study provides","study reviews","study discusses","study depicts",
    "study narrates","study reveals","study indicates","study notes","study records",
    "study catalogs","study enumerates","study compiles","study drafts","study sketches",
    "study frames","study profiles","study portrays","study establishes","study relates",
    "study clarifies","study contextualizes","study annotates","study references","study indicates overview of",
    "study briefs","study synopsizes","study itemizes","study condenses","study logs",
    "study chronicles","study delineates"
]

def add_study_prefixes_to_sentences(text: str) -> str:
    parts = re.split(r"([.;|])", text)
    out, last_prefixed, idx = [], False, 0
    def has_cat(s): return any(re.search(rf"(?<![A-Z0-9_]){re.escape(c)}(?![A-Z0-9_])", s.upper()) for c in CATS_UP)
    def token_count(s): return len(re.findall(r"\S+", s))
    for i in range(0, len(parts), 2):
        sent = parts[i]; sep = parts[i+1] if i+1 < len(parts) else ""
        if sent.strip() == "":
            out.append(sent + sep); last_prefixed = False; continue
        if (not has_cat(sent)) and token_count(sent) >= 5 and not sent.strip().lower().startswith("study "):
            if not last_prefixed:
                pref = STUDY_PREFIXES[idx % len(STUDY_PREFIXES)]; idx += 1
                sent = pref + " " + sent
                last_prefixed = True
        else:
            last_prefixed = False
        out.append(sent + sep)
    return "".join(out).strip()

def lcs_length(a: str, b: str) -> int:
    n, m = len(a), len(b)
    dp = [0]*(m+1)
    for i in range(1, n+1):
        prev = 0; ai = a[i-1]
        for j in range(1, m+1):
            tmp = dp[j]
            dp[j] = prev + 1 if ai == b[j-1] else (dp[j] if dp[j] >= dp[j-1] else dp[j-1])
            prev = tmp
    return dp[m]

def auc_structural(original: str, new: str, steps: int = 20) -> float:
    if not original or not new: return 0.0
    T = min(len(original), len(new))
    if T == 0: return 0.0
    ys = []
    for k in range(1, steps+1):
        t = max(1, (k*T)//steps)
        s = lcs_length(original[:t], new[:t]) / t
        ys.append(s)
    auc = 0.0
    for i in range(1, len(ys)):
        auc += 0.5*(ys[i] + ys[i-1])*(1/steps)
    return round(auc, 6)

def chrF(original: str, new: str, max_n: int = 6, beta: float = 2.0) -> float:
    def ngrams(s, n):
        s = s.replace(" ", "")
        return [s[i:i+n] for i in range(len(s)-n+1)]
    def score_n(n):
        ref = ngrams(original, n); hyp = ngrams(new, n)
        if not ref or not hyp: return 0.0
        from collections import Counter
        R = Counter(ref); H = Counter(hyp)
        overlap = sum((R & H).values())
        p = overlap / max(1, sum(H.values()))
        r = overlap / max(1, sum(R.values()))
        if p+r == 0: return 0.0
        return (1+beta*beta)*p*r/(beta*beta*p + r)
    vals = [score_n(n) for n in range(2, max_n+1)]
    return round(sum(vals)/len(vals), 6)

def jaccard_token(original: str, new: str) -> float:
    tok = re.findall(r"\w+|[^\w\s]", original, flags=re.UNICODE)
    tok2 = re.findall(r"\w+|[^\w\s]", new, flags=re.UNICODE)
    S1, S2 = set(tok), set(tok2)
    inter = len(S1 & S2); union = len(S1 | S2) or 1
    return round(inter/union, 6)

def limit_mentions_per_category(text: str) -> str:
    for cat in CATS_UP + ["IS_CANCER"]:
        matches = list(re.finditer(rf"(?<![A-Z0-9_]){re.escape(cat)}(?![A-Z0-9_])", text))
        if cat == "IS_CANCER":
            for m in reversed(matches):
                text = text[:m.start()] + text[m.end():]
            continue
        limit = MAX_PER_CAT[cat]
        if len(matches) > limit:
            n = len(text)
            scored = [(m.start(), m.end(), 1.0 - min(abs((m.start()+m.end())/2 - n/2)/(n/2 if n else 1),1.0)) for m in matches]
            keep = set(sorted(scored, key=lambda x:(x[2], -x[0]), reverse=True)[:limit])
            for s,e,_ in sorted(scored, key=lambda x:x[0], reverse=True):
                if (s,e,_) not in keep:
                    text = text[:s] + text[e:]
    return text

def validate_and_fix(text: str) -> str:
    text = limit_mentions_per_category(text)
    text = normalize_cat_tokens(text)
    text = clean_orphan_fragments(text)
    U = text.upper()
    for cat in CATS_UP + ["IS_CANCER"]:
        for m in re.finditer(re.escape(cat), U):
            s, e = m.start(), m.end()
            if (s > 0 and re.match(r"[A-Z0-9_]", U[s-1] or "")) or (e < len(U) and re.match(r"[A-Z0-9_]", U[e:e+1] or "")):
                text = text[:s] + " " + cat + " " + text[e:]
                U = text.upper()
    text = restore_missing_spaces(text)
    return normalize_spaces_light(text)

def _build_phrase_pattern(terms: List[str]) -> re.Pattern:
    safe = []
    for t in terms:
        t = t.strip()
        if not t: continue
        esc = re.escape(t)
        esc = esc.replace(r"\ ", r"\s+")
        esc = re.sub(r"\\'", "['’]", esc)
        if re.search(r"[A-Za-z]$", t):
            esc = esc + r"(?:e?s)?"
        safe.append(esc)
    if not safe:
        return re.compile(r"$a")
    pat = r"(?i)\b(?:%s)\b" % "|".join(safe)
    return re.compile(pat)

def _replace_terms_with_categories(text: str) -> str:
    present_counts = {c: len(re.findall(rf"(?<![A-Z0-9_]){re.escape(c)}(?![A-Z0-9_])", text)) for c in CATS_UP}
    items: List[Tuple[str,str]] = []
    for term, cat in _TERM2CAT.items():
        if term in _CATS_LOWER:
            continue
        items.append((term, cat))
    items.sort(key=lambda x: len(x[0]), reverse=True)
    by_cat: Dict[str, List[str]] = {}
    for term, cat in items:
        by_cat.setdefault(cat, []).append(term)
    for cat, terms in by_cat.items():
        limit = MAX_PER_CAT[cat]
        if present_counts.get(cat, 0) >= limit:
            continue
        pat = _build_phrase_pattern(terms)
        def _do(m):
            nonlocal present_counts
            if present_counts.get(cat, 0) >= limit:
                return m.group(0)
            g = m.group(0)
            if g.lower() in _CATS_LOWER:
                return g
            present_counts[cat] = present_counts.get(cat, 0) + 1
            return " " + cat + " "
        text = pat.sub(_do, text)
    text = restore_missing_spaces(text)
    return normalize_spaces_light(text)

def _enforce_category_run_limit(text: str, max_run: int = 3) -> str:
    tokens = list(re.finditer(r"(?<![A-Z0-9_])(?:" + "|".join(map(re.escape, CATS_ALL_UP)) + r")(?![A-Z0-9_])", text))
    if not tokens:
        return text
    idxs = [m.span() for m in tokens]
    to_redistribute = []
    run = []
    prev_end = -2
    for s, e in idxs:
        if run and s == prev_end + 1 or (run and s <= prev_end + 1):
            run.append((s,e))
        else:
            if len(run) > max_run:
                to_redistribute += run[max_run:]
            run = [(s,e)]
        prev_end = e
    if len(run) > max_run:
        to_redistribute += run[max_run:]
    if not to_redistribute:
        return text
    kept = set()
    current = []
    prev_end = -2
    for s, e in idxs:
        if current and s <= prev_end + 1:
            current.append((s,e))
        else:
            if len(current) > max_run:
                for t in current[:max_run]:
                    kept.add(t)
            else:
                for t in current:
                    kept.add(t)
            current = [(s,e)]
        prev_end = e
    if current:
        if len(current) > max_run:
            for t in current[:max_run]:
                kept.add(t)
        else:
            for t in current:
                kept.add(t)
    slots = []
    last = 0
    for s,e in sorted(list(kept)):
        slots.append(text[last:s])
        slots.append(text[s:e])
        last = e
    slots.append(text[last:])
    rebuilt = "".join(slots)
    cats_redist = []
    for _ in to_redistribute:
        cats_redist.append(RNG.choice(CATS_UP))
    rebuilt = _scatter_insert_missing(rebuilt, cats_redist)
    rebuilt = restore_missing_spaces(rebuilt)
    return normalize_spaces_light(rebuilt)

# ————— insertion par PHRASE de toutes les CAT manquantes
_SENT_SPLIT = re.compile(r"([.;|])")

def _allowed_boundaries_in_span(span: str) -> List[int]:
    # positions internes “sûres” (pas début/fin, pas juste à côté d’un séparateur)
    if not span.strip():
        return []
    out = []
    for m in re.finditer(r"\S+(?:\s+|$)", span):
        pos = m.end()
        if pos <= 1 or pos >= len(span)-1:  # éviter tout début/fin
            continue
        left = span[pos-1] if pos-1 >= 0 else ""
        right = span[pos] if pos < len(span) else ""
        if left in ".;|" or right in ".;|":
            continue
        out.append(pos)
    return out

def _insert_cat_in_sentence(sent: str, cat: str) -> str:
    # priorité : remplacer un CHUNK local
    m = re.search(r"\bCHUNK\b", sent)
    if m:
        return restore_missing_spaces(sent[:m.start()] + " " + cat + " " + sent[m.end():])
    # sinon, choisir une frontière interne non adjacente à une autre CAT
    allowed = _allowed_boundaries_in_span(sent)
    RNG.shuffle(allowed)
    for pos in allowed[:10] + allowed:  # quelques essais
        left = sent[:pos].rstrip()
        right = sent[pos:].lstrip()
        if left.endswith(tuple(CATS_ALL_UP)) or any(re.search(rf"(?<![A-Z0-9_]){c}(?![A-Z0-9_])$", left) for c in CATS_ALL_UP):
            continue
        if right.startswith(tuple(CATS_ALL_UP)) or any(re.match(rf"^(?<![A-Z0-9_]){c}(?![A-Z0-9_])", right) for c in CATS_ALL_UP):
            continue
        return restore_missing_spaces(left + " " + cat + " " + right)
    # fallback : milieu
    mid = max(1, min(len(sent)-1, len(sent)//2))
    return restore_missing_spaces(sent[:mid] + " " + cat + " " + sent[mid:])

def ensure_all_categories_per_sentence(text: str) -> str:
    parts = _SENT_SPLIT.split(text)
    out = []
    for i in range(0, len(parts), 2):
        sent = parts[i]
        sep = parts[i+1] if i+1 < len(parts) else ""
        present = {c for c in CATS_UP if re.search(rf"(?<![A-Z0-9_]){re.escape(c)}(?![A-Z0-9_])", sent)}
        missing = [c for c in CATS_UP if c not in present]
        # insérer chaque CAT manquante (respect des limites globales déjà gérées ailleurs)
        for cat in missing:
            sent = _insert_cat_in_sentence(sent, cat)
        out.append(sent + sep)
    text2 = "".join(out)
    text2 = restore_missing_spaces(text2)
    text2 = normalize_spaces_light(text2)
    text2 = _enforce_category_run_limit(text2, 3)
    return text2

def process_summary_once(summary: str) -> str:
    if not summary: return ""
    text = summary
    text = repair_cut_categories(text)
    text = remove_edge_chunks_and_brackets(text)
    text = collapse_chunk_storm(text, max_chunks=40)
    if len(text) > 20000 or text.count("CHUNK") > 200:
        text = purge_final(text)
        text = restore_missing_spaces(text)
        text = collapse_values_final_pass(text)
        text = validate_and_fix(text)
        text = ensure_all_categories_per_sentence(text)
        return text
    text = redact_around_definition_terms(text)
    text = drop_split_category_values(text)

    # SUPPRESSION des noms communs avant remplacement (pour éviter résidus : ex. fibroblast)
    text = _remove_common_nouns(text)

    # Remplacement via lexiques → CAT (respect des quotas)
    text = _replace_terms_with_categories(text)

    text, freed = collapse_values_after_categories(text)
    text = normalize_cat_tokens(text)
    text, kept, freed2 = dedupe_categories(text)

    # Insertion globale des CAT manquantes
    text = ensure_all_categories(text, kept, freed + freed2)

    text = purge_final(text)
    text = validate_and_fix(text)

    # Renforcement maladies (suppression valeurs explicites) si ça n'altère pas trop
    text = redact_plain_disease_values(text, summary)

    text = add_study_prefixes_to_sentences(text)
    text = validate_and_fix(text)
    text = restore_missing_spaces(text)
    text = collapse_values_final_pass(text)
    text = validate_and_fix(text)

    # Pas d'acronymes majuscule non-CAT
    cats_union = "|".join(map(re.escape, CATS_ALL_UP))
    text = re.sub(rf"\b(?!(?:{cats_union})\b)[A-Z]{{4,}}\b", "", text)

    text = normalize_spaces_light(text)
    text = validate_and_fix(text)

    # **Nouvelle contrainte** : toutes les CAT au moins une fois **par phrase**
    text = ensure_all_categories_per_sentence(text)

    return text

def process_summary(summary: str) -> str:
    s_new = process_summary_once(summary)
    if not STRICT_MODE:
        return s_new
    auc = auc_structural(summary, s_new, steps=20)
    chrf = chrF(summary, s_new, max_n=6, beta=2.0)
    dlen = 0.0 if len(summary)==0 else abs(len(s_new) - len(summary))/len(summary)
    if chrf < ROLLBACK_CHRF_MIN or dlen > ROLLBACK_LEN_FRAC:
        global SOFT_COLLAPSE_TOKENS, HARD_COLLAPSE_TOKENS
        old_soft, old_hard = SOFT_COLLAPSE_TOKENS, HARD_COLLAPSE_TOKENS
        SOFT_COLLAPSE_TOKENS, HARD_COLLAPSE_TOKENS = 1, 2
        try:
            s_alt = process_summary_once(summary)
        finally:
            SOFT_COLLAPSE_TOKENS, HARD_COLLAPSE_TOKENS = old_soft, old_hard
        chrf_alt = chrF(summary, s_alt)
        return s_alt if chrf_alt >= chrf else s_new
    return s_new

def _fallback_from_row(row: Dict[str,str]) -> Tuple[Tuple[str,str,str], Tuple[str,float,float,float]]:
    run = (row.get("run_accession") or "").strip()
    s = row.get("summary") or ""
    t = remove_edge_chunks_and_brackets(s)
    t = collapse_chunk_storm(t, max_chunks=40)
    s_new = normalize_spaces_light(restore_missing_spaces(purge_final(t)))
    cleaned = (run, s, s_new)
    metrics = (run, 0.0, 0.0, 0.0)
    return cleaned, metrics

def _process_row_core(row: Dict[str,str]) -> Tuple[Tuple[str,str,str], Tuple[str,float,float,float]]:
    seed = (row.get("seed_summary") or "").strip()
    if seed:
        first_word = seed.split(maxsplit=1)[0]
        row["run_accession"] = first_word
    run = (row.get("run_accession") or "").strip()
    s = row.get("summary") or ""
    if len(s) > 5000 or s.count("CHUNK") > 200:
        t = remove_edge_chunks_and_brackets(s)
        t = collapse_chunk_storm(t, max_chunks=40)
        s_new = validate_and_fix(collapse_values_final_pass(restore_missing_spaces(purge_final(t))))
        s_new = ensure_all_categories_per_sentence(s_new)
    else:
        s_new = process_summary(s)
    auc = auc_structural(s, s_new, steps=20)
    chrf = chrF(s, s_new, max_n=6, beta=2.0)
    jac = jaccard_token(s, s_new)
    cleaned = (run, s, s_new)
    metrics = (run, auc, chrf, jac)
    return cleaned, metrics

def _work_row(row: Dict[str,str]) -> Tuple[str, Tuple[str,str,str], Tuple[str,float,float,float], str]:
    try:
        cleaned, metrics = _process_row_core(row)
        return ("ok", cleaned, metrics, (row.get("run_accession") or "").strip())
    except Exception:
        cleaned, metrics = _fallback_from_row(row)
        return ("err", cleaned, metrics, (row.get("run_accession") or "").strip())

def main():
    for infile in INFILES:
        base = os.path.basename(infile); name, _ = os.path.splitext(base)
        f_clean, w_clean, f_metrics, w_metrics, out_clean, out_metrics = prepare_outputs(name)
        already = count_done_rows(out_clean)
        total = sum(1 for _ in read_tsv(infile))
        print(f"[{name}] Total rows: {total}. Resuming from {already}.", flush=True)

        idx = -1
        processed = 0
        timed_out = 0

        if PARALLEL and os.name == "posix":
            with ProcessPoolExecutor(max_workers=N_PROCS) as ex:
                futures = []
                for row in read_tsv(infile):
                    idx += 1
                    if idx < already:
                        continue
                    futures.append((idx, ex.submit(_work_row, row)))
                for i_row, fut in futures:
                    try:
                        status, cleaned, metrics, run = fut.result(timeout=PER_ROW_TIMEOUT)
                        if status != "ok":
                            timed_out += 1
                    except TimeoutError:
                        timed_out += 1
                        cleaned = (f"row_{i_row}", "", "")
                        metrics = (f"row_{i_row}", 0.0, 0.0, 0.0)
                    except Exception:
                        timed_out += 1
                        cleaned = (f"row_{i_row}", "", "")
                        metrics = (f"row_{i_row}", 0.0, 0.0, 0.0)

                    w_clean.writerow(cleaned)
                    w_metrics.writerow(metrics)
                    f_clean.flush(); f_metrics.flush()
                    os.fsync(f_clean.fileno()); os.fsync(f_metrics.fileno())

                    processed += 1
                    done_now = already + processed
                    if processed % PROGRESS_EVERY == 0 or processed == (total - already):
                        pct = (done_now) / total * 100.0 if total else 100.0
                        print(f"[{name}] {done_now}/{total} ({pct:5.1f}%)  timeouts:{timed_out}", end="\r", flush=True)
        else:
            for row in read_tsv(infile):
                idx += 1
                if idx < already:
                    continue
                try:
                    cleaned, metrics = _work_row(row)[1:3]
                except Exception:
                    cleaned, metrics = _fallback_from_row(row)
                w_clean.writerow(cleaned)
                w_metrics.writerow(metrics)
                f_clean.flush(); f_metrics.flush()
                os.fsync(f_clean.fileno()); os.fync(f_metrics.fileno())
                processed += 1
                done_now = already + processed
                if processed % PROGRESS_EVERY == 0 or processed == (total - already):
                    pct = (done_now) / total * 100.0 if total else 100.0
                    print(f"[{name}] {done_now}/{total} ({pct:5.1f}%)  timeouts:{timed_out}", end="\r", flush=True)

        f_clean.close(); f_metrics.close()
        print(f"\n[{name}] Done. Wrote: {processed} rows. Timeouts/errs: {timed_out}", flush=True)

if __name__ == "__main__":
    main()
