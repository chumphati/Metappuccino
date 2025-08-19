import os
import json
import random
import re
import pandas as pd
from collections import Counter

TRAIN_DATA_PATH = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/data_templates_training/metadata_templates_train_withoutkeys.json'
TRAIN_EXPANDED  = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/metadata_templates_train_expanded.json'
TRAIN_CSV       = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/train_metadata_replaced_table.csv'

EVAL_DATA_PATH  = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/data_templates_training/metadata_templates_val_withoutkeys.json'
EVAL_EXPANDED   = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/metadata_templates_val_expanded.json'
EVAL_CSV        = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/val_metadata_replaced_table.csv'

TEST1_DATA_PATH = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/data_templates_training/metadata_templates_test_withoutkeys.json'
TEST1_EXPANDED  = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/metadata_templates_test_expanded.json'
TEST1_CSV       = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/test_metadata_replaced_table.csv'

TEST2_DATA_PATH = TEST1_DATA_PATH
TEST2_EXPANDED  = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/metadata_templates_test2_expanded.json'
TEST2_CSV       = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/test2_metadata_replaced_table.csv'

def eq_quota(values, total):
    k=len(values); base=total//k; r=total-base*k
    lst=list(values); random.shuffle(lst)
    q={v:base for v in lst}
    for v in lst[:r]: q[v]+=1
    return q

def pick_template(expanded_list):
    item = random.choice(expanded_list)
    if isinstance(item, (list, tuple)): return item[0]
    if isinstance(item, dict): return item.get('template', next(iter(item.values())))
    return item

def inject_value(sentence, insertion):
    words=sentence.split()
    idx=random.randint(0, len(words))
    return ' '.join(words[:idx]+[insertion]+words[idx:])

def fuzzy_token_patterns(values):
    pats=[]
    for v in values:
        base=re.sub(r'\s+', r'\\s*', re.escape(v))
        base=base.replace(r'\-', r'[-_ ]*').replace(r'_', r'[_ ]*')
        pats.append(r'(?i)(?<![A-Za-z0-9])'+base+r'(?![A-Za-z0-9])')
        if re.search(r'\d', v):
            letters=re.sub(r'[^A-Za-z]+','',v)
            digits=''.join(re.findall(r'\d+',v))
            if letters and digits:
                pats.append(r'(?i)(?<![A-Za-z0-9])'+re.escape(letters)+r'\s*[-_ ]*'+re.escape(digits)+r'(?![A-Za-z0-9])')
    return pats

def sanitize_phrase(phrase, rec, syno, sets_keep, all_sets):
    keep={}
    for cat in sets_keep.keys():
        v = rec.get(cat, None)
        if not v: continue
        keep.setdefault(cat, set()).update({v, *syno.get(v, [v])})
    forbidden=[]
    for cat, universe in all_sets.items():
        for val in universe:
            if val in keep.get(cat,set()): continue
            forbidden.extend([val] + syno.get(val, []))
    for pat in fuzzy_token_patterns(forbidden):
        phrase=re.sub(pat,'',phrase)
    phrase=re.sub(r'\s{2,}',' ',phrase).strip().strip(',').strip()
    return phrase

def phrase_with_context(phrase, rec, syno, noise, ctx_wrap):
    cats=['biopsy_site','organ','response','treatment','disease','cell_type','cell_line']
    for cat in cats:
        val=rec.get(cat,'')
        if not val or val in {'not applicable','unknown'} or (cat=='treatment' and val=='no treatment'):
            continue
        alt=random.choice(syno.get(val,[val])) if random.random()<0.8 else val
        if ctx_wrap and cat in ctx_wrap:
            alt = random.choice(ctx_wrap[cat]).format(val=alt)
        phrase=inject_value(phrase, alt)
    for _ in range(random.randint(3,5)):
        phrase=inject_value(phrase, random.choice(noise))
    return phrase

TRAIN = {}
TRAIN['CATEGORIES'] = ['library_selection','sequencing_source','organ','biopsy_site','biopsy_type','cell_line','cell_type','disease','treatment','treatment_time','response','age','sex','ethnicity','localization','is_cancer','phrase']
TRAIN['LIB_SEL'] = ['polyA','inverse rRNA','hybrid selection','unknown']
TRAIN['SEQ_SRC'] = ['single-cell','bulk','spatial','unknown']
TRAIN['BIOPSY_SITE'] = ['blood','lung','heart','kidney','brain','stomach','intestine','colon','skin','liver','spleen','bone marrow','pancreas','prostate','breast','cervix','bone','cartilage','unknown']
TRAIN['BIOPSY_TYPE'] = ['primary','metastasis','blood','unknown']
TRAIN['CELL_LINE'] = ['HEK293','HeLa','HepG2','MCF7','A549','K562','U2OS','PC3','Jurkat','HCT116','SHSY5Y','C2C12','THP1','not applicable','unknown']
TRAIN['CELL_TYPE'] = ['muscle cells','liver cells','blood cells','kidney cells','nerve cells','connective cells','fat cells','bone cells','cartilage cells','specialized integrated cells','fibroblasts','migratory cells','stomach cells','lung cells','primary tissue','unknown']
TRAIN['ORGAN'] = ['liver','lung','heart','kidney','brain','muscle','spleen','pancreas','colon','stomach','intestine','skin','bone marrow','blood','prostate','breast','cervix','bone','cartilage','unknown']
TRAIN['DISEASE'] = ['lung cancer','hepatocellular carcinoma','breast cancer','leukemia','lymphoma','prostate cancer','pancreatic adenocarcinoma','glioblastoma','cirrhosis','chronic kidney disease',"Crohn's disease",'ulcerative colitis','myocardial infarction','diabetes mellitus','unknown']
TRAIN['TREATMENT'] = ['no treatment','cisplatin','doxorubicin','paclitaxel','sorafenib','imatinib','erlotinib','tamoxifen','methotrexate','5-fluorouracil','amoxicillin','gentamicin','irradiation','dexamethasone','unknown']
TRAIN['TREAT_TIME'] = ['pre treatment','on treatment','post treatment','relapse','unknown','24 hours','48 hours','1 week','3 months','1 year','not applicable']
TRAIN['RESPONSE'] = ['resistance','sensitivity','partial response','stable disease','progressive disease','unknown','not applicable']
TRAIN['AGE'] = ['34 years','62 years','47','born in 1982','55 years','19 years','44 years','67 years','41 years','unknown']
TRAIN['SEX'] = ['male','female','unknown']
TRAIN['ETHNICITY'] = ['Asian','Hispanic','European','African','American','unknown']
TRAIN['LOCALIZATION'] = ['North Africa','Paris','French','English','Australia','unknown']
TRAIN['SYNONYMS'] = {}
for lst in ['LIB_SEL','SEQ_SRC','BIOPSY_SITE','BIOPSY_TYPE','CELL_LINE','CELL_TYPE','ORGAN','DISEASE','TREATMENT','TREAT_TIME','RESPONSE','AGE','SEX','ETHNICITY','LOCALIZATION']:
    for v in TRAIN[lst]:
        TRAIN['SYNONYMS'].setdefault(v,[v])
TRAIN['SYNONYMS']['polyA']+=['polyA+','mRNA enrichment','oligo-dT capture']
TRAIN['SYNONYMS']['inverse rRNA']+=['rRNA depletion','RiboZero','ribominus']
TRAIN['SYNONYMS']['hybrid selection']+=['hybrid capture','RNA exome','exon capture']
TRAIN['SYNONYMS']['male']+=['man','male donor','M donor']
TRAIN['SYNONYMS']['North Africa']+=['Maghreb','North-African']
TRAIN['SYNONYMS']['Paris']+=['Paris region']
TRAIN['SYNONYMS']['French']+=['France','French or']
TRAIN['SYNONYMS']['English']+=['UK origin','British']
TRAIN['SYNONYMS']['muscle cells']+=['myocytes','contractile fibers','myofibers']
TRAIN['SYNONYMS']['liver cells']+=['hepatocytes','hepatic parenchyma']
TRAIN['SYNONYMS']['blood cells']+=['erythrocytes','leukocytes']
TRAIN['SYNONYMS']['kidney cells']+=['renal cells','tubular cells']
TRAIN['SYNONYMS']['nerve cells']+=['neurons','neural cells']
TRAIN['SYNONYMS']['connective cells']+=['stromal cells','mesenchymal cells']
TRAIN['SYNONYMS']['fat cells']+=['adipocytes']
TRAIN['SYNONYMS']['bone cells']+=['osteoblasts','osteocytes']
TRAIN['SYNONYMS']['cartilage cells']+=['chondrocytes']
TRAIN['SYNONYMS']['primary tissue']+=['primary sample','native tissue']
TRAIN['SYNONYMS']['lung cancer']+=['pulmonary carcinoma','lung neoplasm']
TRAIN['SYNONYMS']['hepatocellular carcinoma']+=['HCC','hepatoma','liver cancer']
TRAIN['SYNONYMS']['breast cancer']+=['mammary carcinoma','breast neoplasm']
TRAIN['SYNONYMS']['leukemia']+=['blood cancer','hematologic malignancy']
TRAIN['SYNONYMS']['lymphoma']+=['lymphatic malignancy']
TRAIN['SYNONYMS']['prostate cancer']+=['prostatic carcinoma','prostate adenocarcinoma']
TRAIN['SYNONYMS']['pancreatic adenocarcinoma']+=['PDAC','pancreatic cancer']
TRAIN['SYNONYMS']['glioblastoma']+=['GBM','malignant glioma','astrocytoma grade IV']
TRAIN['SYNONYMS']['ulcerative colitis']+=['UC','colitis ulcerosa']
TRAIN['SYNONYMS']["Crohn's disease"]+=['regional enteritis','Crohn disease']
TRAIN['SYNONYMS']['myocardial infarction']+=['heart attack','AMI']
TRAIN['SYNONYMS']['diabetes mellitus']+=['diabetes','DM']
TRAIN['SYNONYMS']['cisplatin']+=['CDDP']
TRAIN['SYNONYMS']['doxorubicin']+=['adriamycin','DOX']
TRAIN['SYNONYMS']['paclitaxel']+=['taxol','PTX']
TRAIN['SYNONYMS']['erlotinib']+=['Tarceva']
TRAIN['SYNONYMS']['irradiation']+=['radiation','gamma rays','X-ray']
TRAIN['SYNONYMS']['dexamethasone']+=['Decadron','DXM']
TRAIN['SYNONYMS']['partial response']+=['PR','partial remission']
TRAIN['SYNONYMS']['stable disease']+=['SD','disease stabilization']
TRAIN['SYNONYMS']['progressive disease']+=['PD','disease progression']
TRAIN['SEM'] = {
    'cell_type':{'organ':{
        'muscle cells':['muscle','heart'],
        'liver cells':['liver'],
        'blood cells':['blood','bone marrow'],
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
        'unknown':TRAIN['ORGAN']
    }},
    'disease':{'organ':{
        'lung cancer':['lung'],
        'hepatocellular carcinoma':['liver'],
        'breast cancer':['breast'],
        'leukemia':['blood','bone marrow'],
        'lymphoma':['spleen'],
        'prostate cancer':['prostate'],
        'pancreatic adenocarcinoma':['pancreas'],
        'glioblastoma':['brain'],
        'cirrhosis':['liver'],
        'chronic kidney disease':['kidney'],
        "Crohn's disease":["intestine","colon"],
        'ulcerative colitis':['colon','intestine'],
        'myocardial infarction':['heart'],
        'diabetes mellitus':['pancreas'],
        'unknown':TRAIN['ORGAN']
    }},
    'cell_line':{'organ':{
        'HepG2':['liver'],
        'A549':['lung'],
        'MCF7':['breast'],
        'U2OS':['bone'],
        'Jurkat':['blood','bone marrow'],
        'K562':['blood','bone marrow'],
        'SHSY5Y':['brain'],
        'C2C12':['muscle'],
        'PC3':['prostate'],
        'HCT116':['colon'],
        'HeLa':['cervix'],
        'THP1':['blood','bone marrow'],
        'HEK293':['kidney'],
        'unknown':TRAIN['ORGAN']
    }},
    'treatment':{'disease':{
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
        'no treatment':['cirrhosis','myocardial infarction','diabetes mellitus','unknown'],
        'unknown':TRAIN['DISEASE']
    }},
    'treatment_time':{'treatment':{
        'cisplatin':TRAIN['TREAT_TIME'],'doxorubicin':TRAIN['TREAT_TIME'],'paclitaxel':TRAIN['TREAT_TIME'],'sorafenib':TRAIN['TREAT_TIME'],
        'imatinib':TRAIN['TREAT_TIME'],'erlotinib':TRAIN['TREAT_TIME'],'tamoxifen':TRAIN['TREAT_TIME'],'methotrexate':TRAIN['TREAT_TIME'],
        '5-fluorouracil':TRAIN['TREAT_TIME'],'amoxicillin':TRAIN['TREAT_TIME'],'gentamicin':TRAIN['TREAT_TIME'],'irradiation':TRAIN['TREAT_TIME'],
        'dexamethasone':TRAIN['TREAT_TIME'],'no treatment':['not applicable'],'unknown':TRAIN['TREAT_TIME']
    }},
    'response':{'treatment':{
        'cisplatin':TRAIN['RESPONSE'],'doxorubicin':TRAIN['RESPONSE'],'paclitaxel':TRAIN['RESPONSE'],'sorafenib':TRAIN['RESPONSE'],
        'imatinib':TRAIN['RESPONSE'],'erlotinib':TRAIN['RESPONSE'],'tamoxifen':TRAIN['RESPONSE'],'methotrexate':TRAIN['RESPONSE'],
        '5-fluorouracil':TRAIN['RESPONSE'],'amoxicillin':TRAIN['RESPONSE'],'gentamicin':TRAIN['RESPONSE'],'irradiation':TRAIN['RESPONSE'],
        'dexamethasone':TRAIN['RESPONSE'],'no treatment':['not applicable'],
        'unknown':['unknown','partial response','stable disease','progressive disease']
    }},
    'biopsy_site':{'organ':{
        'blood':['blood','bone marrow'],
        'lung':['lung'],'heart':['heart'],'kidney':['kidney'],'brain':['brain'],'stomach':['stomach'],
        'intestine':['intestine','colon','stomach'],'colon':['colon','intestine'],'skin':['skin'],'liver':['liver'],
        'spleen':['spleen'],'bone marrow':['bone marrow','blood'],'pancreas':['pancreas'],'prostate':['prostate'],
        'breast':['breast'],'cervix':['cervix'],'bone':['bone','bone marrow'],'cartilage':['cartilage'],'unknown':TRAIN['BIOPSY_SITE']
    }},
    'library_source':{'cell_type':{
        'single-cell':['blood cells','nerve cells','liver cells','muscle cells','fibroblasts','bone cells','cartilage cells','connective cells','lung cells','kidney cells','migratory cells'],
        'bulk':['blood cells','migratory cells','specialized integrated cells','fat cells','stomach cells','primary tissue','fibroblasts','liver cells','lung cells','kidney cells','muscle cells','bone cells'],
        'spatial':['primary tissue','liver cells','lung cells','kidney cells','muscle cells','stomach cells','connective cells','fibroblasts','specialized integrated cells']
    }}
}
TRAIN['CANCER_DISEASES'] = {'lung cancer','hepatocellular carcinoma','breast cancer','leukemia','lymphoma','prostate cancer','pancreatic adenocarcinoma','glioblastoma'}
TRAIN['HEMATO_DISEASES'] = {'leukemia','lymphoma'}
TRAIN['SOLID_DISEASES']  = {'lung cancer','hepatocellular carcinoma','breast cancer','prostate cancer','pancreatic adenocarcinoma','glioblastoma'}
TRAIN['NON_ONCO_NON_INF']={'cirrhosis','chronic kidney disease','myocardial infarction','diabetes mellitus'}
TRAIN['DISEASE_MET_SITES']={
    'pancreatic adenocarcinoma': {'liver','lung','bone marrow'},
    'lung cancer': {'brain','liver','bone marrow'},
    'breast cancer': {'bone marrow','lung','liver','brain'},
    'prostate cancer': {'bone marrow','liver','lung'},
    'hepatocellular carcinoma': {'lung','bone marrow'},
    'glioblastoma': set(),
    'lymphoma': set(),
    'leukemia': set(),
}
TRAIN['NOISE'] = [
    "sequencer lane rebalanced for duplex bias","insert size ~320bp","sample barcoded with UMIs","QC passed at RIN 8.1",
    "library molarity 7.5 nM","coverage uniformity within 1.3x","FASTQ trimmed (Phred≥30)","adapter clipping enabled",
    "lane bleedthrough negligible","5' bias not detected","batch randomization across plates","ERCC spike-ins present",
    "index-hopping mitigated","PCR duplicates removed","alignment on GRCh38","multi-mapping reads <2%","FRiP 0.21",
    "median insert 286","sequencing kit v3","freeze-thaw cycles=1","lane clustering density nominal","negative control clean",
    "RNase-free handling confirmed","unique fragments retained","flowcell ID anonymized"
]
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
BASE_EVAL = {}
BASE_EVAL['CATEGORIES'] = ['library_selection','sequencing_source','organ','biopsy_site','biopsy_type','cell_line','cell_type','disease','treatment','treatment_time','response','age','sex','ethnicity','localization','is_cancer','phrase']
BASE_EVAL['LIB_SEL'] = ["small RNA","other"]
BASE_EVAL['SEQ_SRC'] = ["single-cell","bulk","spatial"]
BASE_EVAL['BIOPSY_SITE'] = ["blood","eye","ear","ovaries","cartilage","esophagus"]
BASE_EVAL['BIOPSY_TYPE'] = ['primary','metastasis','blood']
BASE_EVAL['CELL_LINE'] = ["KYSE-30","TE-1","ARPE-19","HEI-OC1","KGN","NT2/D1","Ishikawa","ECC-1","SW1353","ATDC5"]
BASE_EVAL['CELL_TYPE'] = ["cartilage cells","chondrocytes","esophageal epithelial cells","retinal ganglion cells","photoreceptor cells","retinal pigment epithelial cells","cochlear hair cells","granulosa cells","theca cells","oocytes","Leydig cells","Sertoli cells","spermatogonia","spermatocytes","endometrial epithelial cells","decidual cells"]
BASE_EVAL['ORGAN'] = ["esophagus","eye","ear","ovaries","testes","uterus","cartilage"]
BASE_EVAL['DISEASE'] = ["Barrett's esophagus","esophageal squamous cell carcinoma","age-related macular degeneration","glaucoma","retinitis pigmentosa","sensorineural hearing loss","polycystic ovary syndrome","testicular germ cell tumor","endometriosis","osteoarthritis"]
BASE_EVAL['TREATMENT'] = ["esophagectomy","photodynamic therapy","anti-VEGF intravitreal injection","cochlear implant","laparoscopic ovarian cystectomy","in vitro fertilization","testosterone replacement therapy","hysterectomy","arthroscopic debridement","mesenchymal stem cell cartilage regeneration"]
BASE_EVAL['TREAT_TIME'] = ["72 hours","2 weeks","6 months"]
BASE_EVAL['RESPONSE'] = ["relapse","adverse event"]
BASE_EVAL['AGE'] = ["39 years","36 years","unknown"]
BASE_EVAL['SEX'] = ["male","female"]
BASE_EVAL['ETHNICITY'] = ["American Indian","Pacific Islander","unknown"]
BASE_EVAL['LOCALIZATION'] = ["Mediterranean","unknown"]
BASE_EVAL['SYNONYMS'] = {}
for lst in ['LIB_SEL','SEQ_SRC','BIOPSY_SITE','BIOPSY_TYPE','CELL_LINE','CELL_TYPE','ORGAN','DISEASE','TREATMENT','TREAT_TIME','RESPONSE','AGE','SEX','ETHNICITY','LOCALIZATION']:
    for v in BASE_EVAL[lst]:
        BASE_EVAL['SYNONYMS'].setdefault(v,[v])
BASE_EVAL['SEM'] = {
    "cell_type": {"organ": {
        "cartilage cells":["cartilage"],
        "chondrocytes":["cartilage"],
        "esophageal epithelial cells":["esophagus"],
        "retinal ganglion cells":["eye"],
        "photoreceptor cells":["eye"],
        "retinal pigment epithelial cells":["eye"],
        "cochlear hair cells":["ear"],
        "granulosa cells":["ovaries"],
        "theca cells":["ovaries"],
        "oocytes":["ovaries"],
        "Leydig cells":["testes"],
        "Sertoli cells":["testes"],
        "spermatogonia":["testes"],
        "spermatocytes":["testes"],
        "endometrial epithelial cells":["uterus"],
        "decidual cells":["uterus"]
    }},
    "disease": {"organ": {
        "Barrett's esophagus":["esophagus"],
        "esophageal squamous cell carcinoma":["esophagus"],
        "age-related macular degeneration":["eye"],
        "glaucoma":["eye"],
        "retinitis pigmentosa":["eye"],
        "sensorineural hearing loss":["ear"],
        "polycystic ovary syndrome":["ovaries"],
        "testicular germ cell tumor":["testes"],
        "endometriosis":["uterus"],
        "osteoarthritis":["cartilage"]
    }},
    "cell_line": {"organ": {
        "KYSE-30":["esophagus"],
        "TE-1":["esophagus"],
        "ARPE-19":["eye"],
        "HEI-OC1":["ear"],
        "KGN":["ovaries"],
        "NT2/D1":["testes"],
        "Ishikawa":["uterus"],
        "ECC-1":["uterus"],
        "SW1353":["cartilage"],
        "ATDC5":["cartilage"]
    }},
    "treatment": {"disease": {
        "esophagectomy":["esophageal squamous cell carcinoma","Barrett's esophagus"],
        "photodynamic therapy":["Barrett's esophagus"],
        "anti-VEGF intravitreal injection":["age-related macular degeneration","glaucoma"],
        "cochlear implant":["sensorineural hearing loss"],
        "laparoscopic ovarian cystectomy":["polycystic ovary syndrome"],
        "in vitro fertilization":["polycystic ovary syndrome","endometriosis"],
        "testosterone replacement therapy":["testicular germ cell tumor"],
        "hysterectomy":["endometriosis"],
        "arthroscopic debridement":["osteoarthritis"],
        "mesenchymal stem cell cartilage regeneration":["osteoarthritis"]
    }},
    "treatment_time": {"treatment": {
        "esophagectomy":["72 hours","2 weeks","6 months"],
        "photodynamic therapy":["72 hours","2 weeks","6 months"],
        "anti-VEGF intravitreal injection":["72 hours","2 weeks","6 months"],
        "cochlear implant":["72 hours","2 weeks","6 months"],
        "laparoscopic ovarian cystectomy":["72 hours","2 weeks","6 months"],
        "in vitro fertilization":["72 hours","2 weeks","6 months"],
        "testosterone replacement therapy":["72 hours","2 weeks","6 months"],
        "hysterectomy":["72 hours","2 weeks","6 months"],
        "arthroscopic debridement":["72 hours","2 weeks","6 months"],
        "mesenchymal stem cell cartilage regeneration":["72 hours","2 weeks","6 months"]
    }},
    "response": {"treatment": {
        "esophagectomy":["relapse","adverse event"],
        "photodynamic therapy":["relapse","adverse event"],
        "anti-VEGF intravitreal injection":["relapse","adverse event"],
        "cochlear implant":["relapse","adverse event"],
        "laparoscopic ovarian cystectomy":["relapse","adverse event"],
        "in vitro fertilization":["relapse","adverse event"],
        "testosterone replacement therapy":["relapse","adverse event"],
        "hysterectomy":["relapse","adverse event"],
        "arthroscopic debridement":["relapse","adverse event"],
        "mesenchymal stem cell cartilage regeneration":["relapse","adverse event"]
    }},
    "biopsy_site": {"organ": {
        "blood":["blood","ovaries","testes","uterus","cartilage","esophagus","eye","ear"],
        "eye":["eye"], "ear":["ear"], "ovaries":["ovaries"],
        "cartilage":["cartilage"], "esophagus":["esophagus"]
    }},
    "library_source": {"cell_type": {
        "single-cell": ["granulosa cells","photoreceptor cells","cochlear hair cells","endometrial epithelial cells","spermatogonia"],
        "bulk": ["cartilage cells","decidual cells","retinal pigment epithelial cells"],
        "spatial": ["Leydig cells","Sertoli cells","retinal ganglion cells"]
    }},
    "sex": {
        "disease": {"polycystic ovary syndrome":["female"], "endometriosis":["female"], "testicular germ cell tumor":["male"]},
        "organ": {"ovaries":["female"], "uterus":["female"], "testes":["male"]}
    }
}
BASE_EVAL['CANCER_DISEASES'] = {"esophageal squamous cell carcinoma","testicular germ cell tumor"}
BASE_EVAL['NOISE'] = TRAIN['NOISE']

TEST2 = {}
TEST2['CATEGORIES'] = BASE_EVAL['CATEGORIES'][:]
TEST2['LIB_SEL'] = BASE_EVAL['LIB_SEL'][:]
TEST2['SEQ_SRC'] = BASE_EVAL['SEQ_SRC'][:]
TEST2['BIOPSY_SITE'] = ["thyroid","adrenal gland","placenta","pituitary","parathyroid","thymus"]
TEST2['BIOPSY_TYPE'] = ['primary','metastasis','blood']
TEST2['CELL_LINE'] = ["Nthy-ori 3-1","TPC-1","NCI-H295R","SW-13","JEG-3","BeWo","HTR-8/SVneo","GH3","HP75","TT"]
TEST2['CELL_TYPE'] = ["thyrocytes","adrenocortical cells","trophoblasts","syncytiotrophoblasts","pituitary endocrine cells","somatotrophs","lactotrophs","parathyroid chief cells","thymic epithelial cells"]
TEST2['ORGAN'] = ["thyroid","adrenal gland","placenta","pituitary","parathyroid","thymus"]
TEST2['DISEASE'] = ["papillary thyroid carcinoma","Hashimoto's thyroiditis","Cushing's syndrome","Addison's disease","preeclampsia","pituitary adenoma","hyperparathyroidism","thymoma","adrenocortical carcinoma"]
TEST2['TREATMENT'] = ["radioiodine ablation","levothyroxine","methimazole","transsphenoidal surgery","hydrocortisone therapy","ketoconazole therapy","thymectomy","parathyroidectomy","magnesium sulfate","antihypertensive therapy","mitotane"]
TEST2['TREAT_TIME'] = ["96 hours","5 days","10 days","4 weeks","8 weeks","9 months","18 months"]
TEST2['RESPONSE'] = ["complete response","no response","symptom resolution","dose-limiting toxicity"]
TEST2['AGE'] = ["28 years","33 years","45 years","52 years","63 years","71 years"]
TEST2['SEX'] = ["male","female"]
TEST2['ETHNICITY'] = ["Middle Eastern","Nordic","Eastern European","Sub-Saharan African","South Asian","Caribbean"]
TEST2['LOCALIZATION'] = ["Scandinavia","Iberian Peninsula","Balkan region","Baltic states","Andes","Southeast Asia"]
TEST2['SYNONYMS'] = {}
for lst in ['LIB_SEL','SEQ_SRC','BIOPSY_SITE','BIOPSY_TYPE','CELL_LINE','CELL_TYPE','ORGAN','DISEASE','TREATMENT','TREAT_TIME','RESPONSE','AGE','SEX','ETHNICITY','LOCALIZATION']:
    for v in TEST2[lst]:
        TEST2['SYNONYMS'].setdefault(v,[v])
TEST2['SEM'] = {
    "cell_type":{"organ":{
        "thyrocytes":["thyroid"],
        "adrenocortical cells":["adrenal gland"],
        "trophoblasts":["placenta"],
        "syncytiotrophoblasts":["placenta"],
        "pituitary endocrine cells":["pituitary"],
        "somatotrophs":["pituitary"],
        "lactotrophs":["pituitary"],
        "parathyroid chief cells":["parathyroid"],
        "thymic epithelial cells":["thymus"]
    }},
    "disease":{"organ":{
        "papillary thyroid carcinoma":["thyroid"],
        "Hashimoto's thyroiditis":["thyroid"],
        "Cushing's syndrome":["adrenal gland"],
        "Addison's disease":["adrenal gland"],
        "preeclampsia":["placenta"],
        "pituitary adenoma":["pituitary"],
        "hyperparathyroidism":["parathyroid"],
        "thymoma":["thymus"],
        "adrenocortical carcinoma":["adrenal gland"]
    }},
    "cell_line":{"organ":{
        "Nthy-ori 3-1":["thyroid"],
        "TPC-1":["thyroid"],
        "NCI-H295R":["adrenal gland"],
        "SW-13":["adrenal gland"],
        "JEG-3":["placenta"],
        "BeWo":["placenta"],
        "HTR-8/SVneo":["placenta"],
        "GH3":["pituitary"],
        "HP75":["pituitary"],
        "TT":["thyroid"]
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
        "mitotane":["adrenocortical carcinoma"]
    }},
    "treatment_time":{"treatment":{
        "radioiodine ablation":TEST2['TREAT_TIME'],
        "levothyroxine":TEST2['TREAT_TIME'],
        "methimazole":TEST2['TREAT_TIME'],
        "transsphenoidal surgery":TEST2['TREAT_TIME'],
        "hydrocortisone therapy":TEST2['TREAT_TIME'],
        "ketoconazole therapy":TEST2['TREAT_TIME'],
        "thymectomy":TEST2['TREAT_TIME'],
        "parathyroidectomy":TEST2['TREAT_TIME'],
        "magnesium sulfate":TEST2['TREAT_TIME'],
        "antihypertensive therapy":TEST2['TREAT_TIME'],
        "mitotane":TEST2['TREAT_TIME']
    }},
    "response":{"treatment":{
        "radioiodine ablation":TEST2['RESPONSE'],
        "levothyroxine":TEST2['RESPONSE'],
        "methimazole":TEST2['RESPONSE'],
        "transsphenoidal surgery":TEST2['RESPONSE'],
        "hydrocortisone therapy":TEST2['RESPONSE'],
        "ketoconazole therapy":TEST2['RESPONSE'],
        "thymectomy":TEST2['RESPONSE'],
        "parathyroidectomy":TEST2['RESPONSE'],
        "magnesium sulfate":TEST2['RESPONSE'],
        "antihypertensive therapy":TEST2['RESPONSE'],
        "mitotane":TEST2['RESPONSE']
    }},
    "biopsy_site":{"organ":{
        "thyroid":["thyroid"],
        "adrenal gland":["adrenal gland"],
        "placenta":["placenta"],
        "pituitary":["pituitary"],
        "parathyroid":["parathyroid"],
        "thymus":["thymus"]
    }},
    "library_source":{"cell_type":{
        "single-cell":["thyrocytes","adrenocortical cells","trophoblasts","syncytiotrophoblasts","pituitary endocrine cells","somatotrophs","lactotrophs","parathyroid chief cells","thymic epithelial cells"],
        "bulk":["adrenocortical cells","trophoblasts","thyrocytes"],
        "spatial":["trophoblasts","syncytiotrophoblasts","thyrocytes","thymic epithelial cells"]
    }},
    "sex":{
        "disease":{"preeclampsia":["female"]},
        "organ":{"placenta":["male"]}
    }
}
TEST2['CANCER_DISEASES'] = {"papillary thyroid carcinoma","adrenocortical carcinoma","thymoma"}
TEST2['NOISE'] = TRAIN['NOISE']
TRAIN['LABELS']={'PRIMARY':'primary','METASTASIS':'metastasis','BLOOD':'blood'}
BASE_EVAL['LABELS']={'PRIMARY':'primary','METASTASIS':'metastasis','BLOOD':'blood'}
TEST2['LABELS']={'PRIMARY':'primary','METASTASIS':'metastasis','BLOOD':None}

def make_dataset(config, targets, data_path, expanded_path, out_csv, ctx_wrap=CTX_WRAP, attempts_cap=800000):
    CATS = config['CATEGORIES']
    syno = config['SYNONYMS']
    SEM  = config['SEM']
    NOISE= config.get('NOISE', [])
    DISEASES = config['DISEASE']
    ORG = config['ORGAN']
    BIOPSY_SITE = config['BIOPSY_SITE']
    SEQ_SRC = config['SEQ_SRC']
    LIB_SEL = config['LIB_SEL']
    LABELS = config.get('LABELS', {'PRIMARY':'primary','METASTASIS':'metastasis','BLOOD':'blood'})
    CANCER = set(config.get('CANCER_DISEASES', set()))
    NON_ONCO_LIMIT = set()
    if config is TRAIN:
        NON_ONCO_LIMIT = TRAIN['NON_ONCO_NON_INF']
    quota={cat: eq_quota(config[key], targets) for cat,key in [
        ('disease','DISEASE'),('organ','ORGAN'),('cell_type','CELL_TYPE'),('cell_line','CELL_LINE'),
        ('sequencing_source','SEQ_SRC'),('library_selection','LIB_SEL'),('biopsy_type','BIOPSY_TYPE'),
        ('biopsy_site','BIOPSY_SITE'),('treatment','TREATMENT'),('treatment_time','TREAT_TIME'),
        ('response','RESPONSE'),('age','AGE'),('sex','SEX'),('ethnicity','ETHNICITY'),
        ('localization','LOCALIZATION')
    ]}
    quota['is_cancer']={'true':targets//2,'false':targets-targets//2}
    remaining={k:Counter(v) for k,v in quota.items()}
    def pop_any(cat):
        pool=[v for v,c in remaining[cat].items() if c>0]
        if not pool: return None
        v=random.choice(pool); remaining[cat][v]-=1; return v
    def pop_from(cat, valid):
        pool=[v for v in valid if remaining[cat].get(v,0)>0]
        if pool:
            v=random.choice(pool); remaining[cat][v]-=1; return v
        return pop_any(cat)
    def normalize_is_cancer(d):
        return 'true' if d in CANCER else 'false'
    def allowed_sources_for_org(org):
        opts=list(SEQ_SRC)
        if org not in BIOPSY_SITE and 'spatial' in opts:
            opts=[x for x in opts if x!='spatial']
        if config is TRAIN and org in {'blood','bone marrow'} and 'spatial' in opts:
            opts=[x for x in opts if x!='spatial']
        return opts
    def allowed_libraries_for_source(src):
        if src in {'single-cell','spatial'}:
            if 'other' in LIB_SEL: return ['other']
            return [v for v in LIB_SEL if v!='inverse rRNA'] if 'polyA' in LIB_SEL else LIB_SEL
        if src=='bulk':
            return [v for v in LIB_SEL]
        return [v for v in LIB_SEL]
    def allowed_celltypes_for_org(org):
        return list(dict.fromkeys(SEM['cell_type']['organ'].get(org, [])))
    def allowed_celllines_for_org(org):
        cands=[cl for cl,orgs in SEM['cell_line']['organ'].items() if org in orgs]
        return cands if cands else list(config['CELL_LINE'])
    def primary_sites_for_org(org):
        sites=[site for site,orgs in SEM['biopsy_site']['organ'].items() if org in orgs]
        return sites if sites else ([org] if org in BIOPSY_SITE else [])
    def metastasis_sites_for_organ(org):
        pool=[s for s in BIOPSY_SITE if s!=org]
        return pool
    def allowed_treatments_for_disease(d):
        cands=[t for t,ds in SEM['treatment']['disease'].items() if d in ds and t in config['TREATMENT']]
        return cands if cands else [t for t in config['TREATMENT']]
    def allowed_tt_for_treatment(tr):
        return SEM['treatment_time']['treatment'].get(tr, config['TREAT_TIME'])
    def allowed_rr_for_treatment(tr, d):
        rr = SEM['response']['treatment'].get(tr, config['RESPONSE'])
        if config is TRAIN and d in NON_ONCO_LIMIT:
            rr = [x for x in rr if x in {'unknown','stable disease'}] or ['unknown','stable disease']
        return rr
    def allowed_src_for_celltype(ct):
        inv=[]
        for src,cts in SEM['library_source']['cell_type'].items():
            if ct in cts: inv.append(src)
        return inv or list(SEQ_SRC)
    with open(data_path) as f:
        templates=json.load(f)
    expanded=random.choices(templates,k=max(3*targets, 1200))
    with open(expanded_path,'w') as f:
        json.dump(expanded,f,indent=2)
    rows=[]
    attempts=0
    while len(rows)<targets and attempts<attempts_cap:
        attempts+=1
        d = pop_any('disease')
        if d is None: break
        isc = normalize_is_cancer(d)
        if remaining['is_cancer'][isc]==0:
            alt='false' if isc=='true' else 'true'
            if remaining['is_cancer'][alt]>0:
                isc=alt
        remaining['is_cancer'][isc]-=1
        organs = SEM['disease']['organ'].get(d, ORG) or ORG
        org = pop_from('organ', organs) or random.choice(ORG)
        ct = pop_from('cell_type', allowed_celltypes_for_org(org)) or random.choice(config['CELL_TYPE'])
        cl = pop_from('cell_line', allowed_celllines_for_org(org)) or random.choice(config['CELL_LINE'])
        src_allowed = list(set(allowed_src_for_celltype(ct)) & set(allowed_sources_for_org(org))) or allowed_sources_for_org(org)
        src = pop_from('sequencing_source', src_allowed) or random.choice(src_allowed)
        lib = pop_from('library_selection', allowed_libraries_for_source(src)) or random.choice(allowed_libraries_for_source(src))
        bt = pop_from('biopsy_type', config['BIOPSY_TYPE']) or random.choice(config['BIOPSY_TYPE'])
        bs = None
        if LABELS.get('BLOOD') and bt==LABELS['BLOOD']:
            bs = pop_from('biopsy_site', [LABELS['BLOOD']]) or LABELS['BLOOD']
        elif bt==LABELS['PRIMARY']:
            prim = primary_sites_for_org(org)
            if not prim:
                if LABELS.get('BLOOD') and LABELS['BLOOD'] in BIOPSY_SITE:
                    bt=LABELS['BLOOD']
                    bs = pop_from('biopsy_site', [LABELS['BLOOD']]) or LABELS['BLOOD']
                else:
                    bs = pop_from('biopsy_site', BIOPSY_SITE) or random.choice(BIOPSY_SITE)
            else:
                bs = pop_from('biopsy_site', prim) or random.choice(prim)
        else:
            if isc=='true' and d in CANCER:
                mets=metastasis_sites_for_organ(org)
                bs = pop_from('biopsy_site', mets) or random.choice(mets)
            else:
                bt=LABELS['PRIMARY']
                prim = primary_sites_for_org(org)
                if prim:
                    bs = pop_from('biopsy_site', prim) or random.choice(prim)
                else:
                    bs = pop_from('biopsy_site', BIOPSY_SITE) or random.choice(BIOPSY_SITE)
        if src=='spatial':
            if org in BIOPSY_SITE:
                if bs!=org and remaining['biopsy_site'].get(org,0)>0:
                    remaining['biopsy_site'][bs]+=1
                    remaining['biopsy_site'][org]-=1
                    bs = org
                bt=LABELS['PRIMARY']
            else:
                src = pop_from('sequencing_source', [x for x in allowed_sources_for_org(org) if x!='spatial']) or 'bulk'
                lib = pop_from('library_selection', allowed_libraries_for_source(src)) or lib
        if LABELS.get('BLOOD') and bs==LABELS['BLOOD'] and bt!=LABELS['BLOOD']:
            remaining['biopsy_type'][bt]+=1
            if remaining['biopsy_type'][LABELS['BLOOD']]>0:
                remaining['biopsy_type'][LABELS['BLOOD']]-=1
            bt=LABELS['BLOOD']
        tr = pop_from('treatment', allowed_treatments_for_disease(d)) or random.choice(config['TREATMENT'])
        tt_allowed = allowed_tt_for_treatment(tr)
        tt = pop_from('treatment_time', tt_allowed) or random.choice(tt_allowed)
        rr_allowed = allowed_rr_for_treatment(tr, d)
        rr = pop_from('response', rr_allowed) or random.choice(rr_allowed)
        sex_opts=[]
        if 'sex' in SEM:
            sx_by_d = SEM['sex'].get('disease',{}).get(d, [])
            sx_by_o = SEM['sex'].get('organ',{}).get(org, [])
            inter = list(set(sx_by_d) & set(sx_by_o))
            if inter: sex_opts=inter
            elif sx_by_d: sex_opts=sx_by_d
            elif sx_by_o: sex_opts=sx_by_o
        sex = pop_from('sex', sex_opts if sex_opts else config['SEX']) or random.choice(config['SEX'])
        age = pop_any('age') or random.choice(config['AGE'])
        eth = pop_any('ethnicity') or random.choice(config['ETHNICITY'])
        loc = pop_any('localization') or random.choice(config['LOCALIZATION'])
        rec={
            'library_selection':lib,
            'sequencing_source':src,
            'organ':org,
            'biopsy_site':bs,
            'biopsy_type':bt,
            'cell_line':cl,
            'cell_type':ct,
            'disease':d,
            'treatment':tr,
            'treatment_time':tt,
            'response':rr,
            'age':age,
            'sex':sex,
            'ethnicity':eth,
            'localization':loc,
            'is_cancer':isc
        }
        tpl = pick_template(expanded)
        if not isinstance(tpl,str): tpl=str(tpl)
        phr = phrase_with_context(tpl, rec, syno, NOISE, ctx_wrap)
        sets_keep = {
            'organ': set(), 'disease': set(), 'cell_line': set(), 'cell_type': set(),
            'treatment': set(), 'biopsy_site': set(), 'sequencing_source': set(), 'library_selection': set()
        }
        all_sets = {
            'organ': set(config['ORGAN']),
            'disease': set(config['DISEASE']),
            'cell_line': set([c for c in config['CELL_LINE'] if c not in {'not applicable','unknown'}]),
            'cell_type': set(config['CELL_TYPE']),
            'treatment': set(config['TREATMENT']),
            'biopsy_site': set(config['BIOPSY_SITE']),
            'sequencing_source': set(config['SEQ_SRC']),
            'library_selection': set(config['LIB_SEL']),
        }
        phr = sanitize_phrase(phr, rec, syno, sets_keep, all_sets)
        rec['phrase']=phr
        rows.append(rec)
    df=pd.DataFrame(rows, columns=CATS)
    df.to_csv(out_csv, index=False)

if __name__=='__main__':
    make_dataset(TRAIN, 1500, TRAIN_DATA_PATH, TRAIN_EXPANDED, TRAIN_CSV, ctx_wrap=CTX_WRAP)
    EVAL = {**BASE_EVAL}
    EVAL['SEM'] = json.loads(json.dumps(BASE_EVAL['SEM']))
    EVAL['SYNONYMS'] = json.loads(json.dumps(BASE_EVAL['SYNONYMS']))
    EVAL['NOISE']=BASE_EVAL['NOISE']
    EVAL['CANCER_DISEASES']=set(BASE_EVAL['CANCER_DISEASES'])
    EVAL['LABELS']=BASE_EVAL['LABELS']
    make_dataset(EVAL, 300, EVAL_DATA_PATH, EVAL_EXPANDED, EVAL_CSV, ctx_wrap=CTX_WRAP)
    TEST1 = {**BASE_EVAL}
    TEST1['SEM'] = json.loads(json.dumps(BASE_EVAL['SEM']))
    TEST1['SYNONYMS'] = json.loads(json.dumps(BASE_EVAL['SYNONYMS']))
    TEST1['NOISE']=BASE_EVAL['NOISE']
    TEST1['CANCER_DISEASES']=set(BASE_EVAL['CANCER_DISEASES'])
    TEST1['LABELS']=BASE_EVAL['LABELS']
    make_dataset(TEST1, 2500, TEST1_DATA_PATH, TEST1_EXPANDED, TEST1_CSV, ctx_wrap=CTX_WRAP)
    make_dataset(TEST2, 2500, TEST2_DATA_PATH, TEST2_EXPANDED, TEST2_CSV, ctx_wrap=CTX_WRAP)
