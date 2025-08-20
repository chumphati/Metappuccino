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

def sanitize_phrase(phrase, rec, syno, all_sets):
    allowed=set()
    for cat in all_sets.keys():
        v=rec.get(cat, None)
        if not v: continue
        allowed.update([v, *syno.get(v, [v])])
    forbidden=[]
    for cat, universe in all_sets.items():
        for val in universe:
            if val in allowed: continue
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
    extra_cats=['library_selection','sequencing_source','biopsy_type','treatment_time','age','sex','ethnicity','localization','is_cancer']
    random.shuffle(extra_cats)
    for cat in extra_cats:
        val=rec.get(cat,'')
        if not val or val in {'unknown','not applicable'}:
            continue
        if cat=='is_cancer':
            alt = random.choice(['oncologic case','cancer']) if val=='true' else random.choice(['non-oncologic','non-cancer'])
        else:
            alt = random.choice(syno.get(val,[val]))
        phrase=inject_value(phrase, alt)
    return phrase

TRAIN = {}
TRAIN['CATEGORIES'] = ['library_selection','sequencing_source','organ','biopsy_site','biopsy_type','cell_line','cell_type','disease','treatment','treatment_time','response','age','sex','ethnicity','localization','is_cancer','phrase']
TRAIN['LIB_SEL'] = ['polyA','inverse rRNA','hybrid selection','unknown']
TRAIN['SEQ_SRC'] = ['single-cell','bulk','spatial','unknown']
TRAIN['BIOPSY_SITE'] = ['blood','lung','heart','kidney','brain','stomach','intestine','colon','skin','liver','spleen','bone marrow','pancreas','prostate','breast','cervix','bone','cartilage','unknown']
TRAIN['BIOPSY_TYPE'] = ['primary','metastasis','blood']
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
TRAIN['SYNONYMS']['liver cells']+=['hepatic parenchyma','hepatocytes']
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
TRAIN['SYNONYMS']['leukemia']+=['blood cancer','hematologic malignancy','hematologic']
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
        'unknown':['liver','lung','heart','kidney','brain','muscle','spleen','pancreas','colon','stomach','intestine','skin','bone marrow','blood','prostate','breast','cervix','bone','cartilage','unknown']
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
        'unknown':['liver','lung','heart','kidney','brain','muscle','spleen','pancreas','colon','stomach','intestine','skin','bone marrow','blood','prostate','breast','cervix','bone','cartilage','unknown']
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
        'unknown':['liver','lung','heart','kidney','brain','muscle','spleen','pancreas','colon','stomach','intestine','skin','bone marrow','blood','prostate','breast','cervix','bone','cartilage','unknown'],
        'not applicable':['liver','lung','heart','kidney','brain','muscle','spleen','pancreas','colon','stomach','intestine','skin','bone marrow','blood','prostate','breast','cervix','bone','cartilage','unknown']
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
        'unknown':['lung cancer','hepatocellular carcinoma','breast cancer','leukemia','lymphoma','prostate cancer','pancreatic adenocarcinoma','glioblastoma','cirrhosis','chronic kidney disease',"Crohn's disease",'ulcerative colitis','myocardial infarction','diabetes mellitus','unknown']
    }},
    'treatment_time':{'treatment':{
        'cisplatin':['pre treatment','on treatment','post treatment','relapse','unknown','24 hours','48 hours','1 week','3 months','1 year'],
        'doxorubicin':['pre treatment','on treatment','post treatment','relapse','unknown','24 hours','48 hours','1 week','3 months','1 year'],
        'paclitaxel':['pre treatment','on treatment','post treatment','relapse','unknown','24 hours','48 hours','1 week','3 months','1 year'],
        'sorafenib':['pre treatment','on treatment','post treatment','relapse','unknown','24 hours','48 hours','1 week','3 months','1 year'],
        'imatinib':['pre treatment','on treatment','post treatment','relapse','unknown','24 hours','48 hours','1 week','3 months','1 year'],
        'erlotinib':['pre treatment','on treatment','post treatment','relapse','unknown','24 hours','48 hours','1 week','3 months','1 year'],
        'tamoxifen':['unknown'],
        'methotrexate':['pre treatment','on treatment','post treatment','relapse','unknown','24 hours','48 hours','1 week','3 months','1 year'],
        '5-fluorouracil':['pre treatment','on treatment','post treatment','relapse','unknown','24 hours','48 hours','1 week','3 months','1 year'],
        'amoxicillin':['pre treatment','on treatment','post treatment','relapse','unknown','24 hours','48 hours','1 week','3 months','1 year'],
        'gentamicin':['pre treatment','on treatment','post treatment','relapse','unknown','24 hours','48 hours','1 week','3 months','1 year'],
        'irradiation':['pre treatment','on treatment','post treatment','relapse','unknown','24 hours','48 hours','1 week','3 months','1 year'],
        'dexamethasone':['pre treatment','on treatment','post treatment','relapse','unknown','24 hours','48 hours','1 week','3 months','1 year'],
        'no treatment':['not applicable'],
        'unknown':['unknown']
    }},
    'response':{'treatment':{
        'cisplatin':['resistance','sensitivity','partial response','stable disease','progressive disease','unknown'],
        'doxorubicin':['resistance','sensitivity','partial response','stable disease','progressive disease','unknown'],
        'paclitaxel':['resistance','sensitivity','partial response','stable disease','progressive disease','unknown'],
        'sorafenib':['resistance','sensitivity','partial response','stable disease','progressive disease','unknown'],
        'imatinib':['resistance','sensitivity','partial response','stable disease','progressive disease','unknown'],
        'erlotinib':['resistance','sensitivity','partial response','stable disease','progressive disease','unknown'],
        'tamoxifen':['unknown'],
        'methotrexate':['resistance','sensitivity','partial response','stable disease','progressive disease','unknown'],
        '5-fluorouracil':['resistance','sensitivity','partial response','stable disease','progressive disease','unknown'],
        'amoxicillin':['resistance','sensitivity','partial response','stable disease','progressive disease','unknown'],
        'gentamicin':['resistance','sensitivity','partial response','stable disease','progressive disease','unknown'],
        'irradiation':['resistance','sensitivity','partial response','stable disease','progressive disease','unknown'],
        'dexamethasone':['resistance','sensitivity','partial response','stable disease','progressive disease','unknown'],
        'no treatment':['not applicable'],
        'unknown':['unknown']
    }},
    'biopsy_site':{'organ':{
        'blood':['blood','bone marrow'],
        'lung':['lung'],'heart':['heart'],'kidney':['kidney'],'brain':['brain'],'stomach':['stomach'],
        'intestine':['intestine','colon','stomach'],'colon':['colon','intestine'],'skin':['skin'],'liver':['liver'],
        'spleen':['spleen'],'bone marrow':['bone marrow','blood'],'pancreas':['pancreas'],'prostate':['prostate'],
        'breast':['breast'],'cervix':['cervix'],'bone':['bone','bone marrow'],'cartilage':['cartilage'],'unknown':['unknown']
    }},
    'library_source':{'cell_type':{
        'single-cell':['blood cells','nerve cells','liver cells','muscle cells','fibroblasts','bone cells','cartilage cells','connective cells','lung cells','kidney cells','migratory cells','primary tissue','stomach cells','specialized integrated cells'],
        'bulk':['blood cells','migratory cells','specialized integrated cells','fat cells','stomach cells','primary tissue','fibroblasts','liver cells','lung cells','kidney cells','muscle cells','bone cells','connective cells','nerve cells'],
        'spatial':['primary tissue','liver cells','lung cells','kidney cells','muscle cells','stomach cells','connective cells','fibroblasts','specialized integrated cells']
    }},
    'sex':{
        'organ':{'prostate':['male'],'cervix':['female']},
        'disease':{'prostate cancer':['male']}
    }
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
BASE_EVAL['SYNONYMS']['small RNA']+=['small-RNA','miRNA-focused','short RNA']
BASE_EVAL['SYNONYMS']['other']+=['unspecified','miscellaneous']
BASE_EVAL['SYNONYMS']['single-cell']+=['scRNA-seq','single cell']
BASE_EVAL['SYNONYMS']['bulk']+=['bulk RNA-seq']
BASE_EVAL['SYNONYMS']['spatial']+=['spatial transcriptomics']
BASE_EVAL['SYNONYMS']['blood']+=['peripheral blood']
BASE_EVAL['SYNONYMS']['eye']+=['ocular tissue']
BASE_EVAL['SYNONYMS']['ear']+=['auricular tissue']
BASE_EVAL['SYNONYMS']['ovaries']+=['ovarian tissue']
BASE_EVAL['SYNONYMS']['cartilage']+=['articular cartilage']
BASE_EVAL['SYNONYMS']['esophagus']+=['oesophagus']
BASE_EVAL['SYNONYMS']['primary']+=['primary site']
BASE_EVAL['SYNONYMS']['metastasis']+=['met secondary','met site']
BASE_EVAL['SYNONYMS']['KYSE-30']+=['KYSE30']
BASE_EVAL['SYNONYMS']['TE-1']+=['TE1']
BASE_EVAL['SYNONYMS']['ARPE-19']+=['ARPE19','RPE line']
BASE_EVAL['SYNONYMS']['HEI-OC1']+=['HEIOC1']
BASE_EVAL['SYNONYMS']['KGN']+=['KGN cell line']
BASE_EVAL['SYNONYMS']['NT2/D1']+=['NT2D1']
BASE_EVAL['SYNONYMS']['Ishikawa']+=['Ishikawa cells']
BASE_EVAL['SYNONYMS']['ECC-1']+=['ECC1']
BASE_EVAL['SYNONYMS']['SW1353']+=['SW-1353']
BASE_EVAL['SYNONYMS']['ATDC5']+=['ATDC-5']
BASE_EVAL['SYNONYMS']['cartilage cells']+=['cartilage chondrocytes']
BASE_EVAL['SYNONYMS']['chondrocytes']+=['cartilage cells']
BASE_EVAL['SYNONYMS']['esophageal epithelial cells']+=['oesophageal epithelium']
BASE_EVAL['SYNONYMS']['retinal ganglion cells']+=['RGCs']
BASE_EVAL['SYNONYMS']['photoreceptor cells']+=['photoreceptors']
BASE_EVAL['SYNONYMS']['retinal pigment epithelial cells']+=['RPE cells']
BASE_EVAL['SYNONYMS']['cochlear hair cells']+=['inner ear hair cells']
BASE_EVAL['SYNONYMS']['granulosa cells']+=['ovarian granulosa']
BASE_EVAL['SYNONYMS']['theca cells']+=['ovarian theca']
BASE_EVAL['SYNONYMS']['oocytes']+=['egg cells']
BASE_EVAL['SYNONYMS']['Leydig cells']+=['testicular Leydig']
BASE_EVAL['SYNONYMS']['Sertoli cells']+=['testicular Sertoli']
BASE_EVAL['SYNONYMS']['spermatogonia']+=['germline stem cells']
BASE_EVAL['SYNONYMS']['spermatocytes']+=['meiotic germ cells']
BASE_EVAL['SYNONYMS']['endometrial epithelial cells']+=['endometrial epithelium']
BASE_EVAL['SYNONYMS']['decidual cells']+=['decidua cells']
BASE_EVAL['SYNONYMS']["Barrett's esophagus"]+=["Barrett oesophagus","BE"]
BASE_EVAL['SYNONYMS']['esophageal squamous cell carcinoma']+=['ESCC','oesophageal SCC']
BASE_EVAL['SYNONYMS']['age-related macular degeneration']+=['AMD']
BASE_EVAL['SYNONYMS']['glaucoma']+=['optic neuropathy']
BASE_EVAL['SYNONYMS']['retinitis pigmentosa']+=['RP']
BASE_EVAL['SYNONYMS']['sensorineural hearing loss']+=['SNHL']
BASE_EVAL['SYNONYMS']['polycystic ovary syndrome']+=['PCOS']
BASE_EVAL['SYNONYMS']['testicular germ cell tumor']+=['TGCT']
BASE_EVAL['SYNONYMS']['endometriosis']+=['endometrial ectopia']
BASE_EVAL['SYNONYMS']['osteoarthritis']+=['OA']
BASE_EVAL['SYNONYMS']['esophagectomy']+=['oesophagectomy','esophageal resection']
BASE_EVAL['SYNONYMS']['photodynamic therapy']+=['PDT']
BASE_EVAL['SYNONYMS']['anti-VEGF intravitreal injection']+=['anti-VEGF IVI']
BASE_EVAL['SYNONYMS']['cochlear implant']+=['CI']
BASE_EVAL['SYNONYMS']['laparoscopic ovarian cystectomy']+=['LOC']
BASE_EVAL['SYNONYMS']['in vitro fertilization']+=['IVF']
BASE_EVAL['SYNONYMS']['testosterone replacement therapy']+=['TRT']
BASE_EVAL['SYNONYMS']['hysterectomy']+=['uterus removal']
BASE_EVAL['SYNONYMS']['arthroscopic debridement']+=['arthro debridement']
BASE_EVAL['SYNONYMS']['mesenchymal stem cell cartilage regeneration']+=['MSC cartilage therapy']
BASE_EVAL['SYNONYMS']['72 hours']+=['72h']
BASE_EVAL['SYNONYMS']['2 weeks']+=['fortnight']
BASE_EVAL['SYNONYMS']['6 months']+=['half-year']
BASE_EVAL['SYNONYMS']['relapse']+=['recurrence']
BASE_EVAL['SYNONYMS']['adverse event']+=['AE']
BASE_EVAL['SYNONYMS']['male']+=['man','male donor','M donor']
BASE_EVAL['SYNONYMS']['female']+=['woman','F donor']
BASE_EVAL['SYNONYMS']['American Indian']+=['Native American']
BASE_EVAL['SYNONYMS']['Pacific Islander']+=['Oceanian']
BASE_EVAL['SYNONYMS']['Mediterranean']+=['Med region']
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
        "mesenchymal stem cell cartilage regeneration":["72 hours","2 weeks","6 months"],
        "unknown":["unknown"]
    }},
    "response": {"treatment": {
        "esophagectomy":["relapse","adverse event","unknown"],
        "photodynamic therapy":["relapse","adverse event","unknown"],
        "anti-VEGF intravitreal injection":["relapse","adverse event","unknown"],
        "cochlear implant":["relapse","adverse event","unknown"],
        "laparoscopic ovarian cystectomy":["relapse","adverse event","unknown"],
        "in vitro fertilization":["relapse","adverse event","unknown"],
        "testosterone replacement therapy":["relapse","adverse event","unknown"],
        "hysterectomy":["relapse","adverse event","unknown"],
        "arthroscopic debridement":["relapse","adverse event","unknown"],
        "mesenchymal stem cell cartilage regeneration":["relapse","adverse event","unknown"],
        "unknown":["unknown"]
    }},
    "biopsy_site": {"organ": {
        "blood":["blood","ovaries","testes","uterus","cartilage","esophagus","eye","ear"],
        "eye":["eye"], "ear":["ear"], "ovaries":["ovaries"],
        "cartilage":["cartilage"], "esophagus":["esophagus"]
    }},
    "library_source": {"cell_type": {
        "single-cell": ["granulosa cells","photoreceptor cells","cochlear hair cells","endometrial epithelial cells","spermatogonia","retinal ganglion cells"],
        "bulk": ["cartilage cells","decidual cells","retinal pigment epithelial cells"],
        "spatial": ["Leydig cells","Sertoli cells","retinal ganglion cells","cartilage cells","esophageal epithelial cells"]
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
TEST2['CELL_LINE'] = ["Nthy-ori 3-1","TPC-1","NCI-H295R","SW-13","JEG-3","BeWo","HTR-8/SVneo","GH3","HP75","sHPT-1","TT"]
TEST2['CELL_TYPE'] = ["thyrocytes","adrenocortical cells","trophoblasts","syncytiotrophoblasts","pituitary endocrine cells","somatotrophs","lactotrophs","parathyroid chief cells","thymic epithelial cells"]
TEST2['ORGAN'] = ["thyroid","adrenal gland","placenta","pituitary","parathyroid","thymus"]
TEST2['DISEASE'] = ["papillary thyroid carcinoma","Hashimoto's thyroiditis","Cushing's syndrome","Addison's disease","preeclampsia","pituitary adenoma","hyperparathyroidism","thymoma","adrenocortical carcinoma"]
TEST2['TREATMENT'] = ["radioiodine ablation","levothyroxine","methimazole","transsphenoidal surgery","hydrocortisone therapy","ketoconazole therapy","thymectomy","parathyroidectomy","magnesium sulfate","antihypertensive therapy","mitotane"]
TEST2['TREAT_TIME'] = ["96 hours","5 days","10 days","4 weeks","8 weeks","9 months","18 months","unknown"]
TEST2['RESPONSE'] = ["complete response","no response","symptom resolution","dose-limiting toxicity","unknown"]
TEST2['AGE'] = ["28 years","33 years","45 years","52 years","63 years","71 years"]
TEST2['SEX'] = ["male","female"]
TEST2['ETHNICITY'] = ["Middle Eastern","Nordic","Eastern European","Sub-Saharan African","South Asian","Caribbean"]
TEST2['LOCALIZATION'] = ["Scandinavia","Iberian Peninsula","Balkan region","Baltic states","Andes","Southeast Asia"]
TEST2['SYNONYMS'] = {}
for lst in ['LIB_SEL','SEQ_SRC','BIOPSY_SITE','BIOPSY_TYPE','CELL_LINE','CELL_TYPE','ORGAN','DISEASE','TREATMENT','TREAT_TIME','RESPONSE','AGE','SEX','ETHNICITY','LOCALIZATION']:
    for v in TEST2[lst]:
        TEST2['SYNONYMS'].setdefault(v,[v])
TEST2['SYNONYMS']['small RNA']+=['small-RNA','miRNA-focused','short RNA']
TEST2['SYNONYMS']['other']+=['unspecified','miscellaneous']
TEST2['SYNONYMS']['single-cell']+=['scRNA-seq','single cell']
TEST2['SYNONYMS']['bulk']+=['bulk RNA-seq']
TEST2['SYNONYMS']['spatial']+=['spatial transcriptomics']
TEST2['SYNONYMS']['thyroid']+=['thyroid gland']
TEST2['SYNONYMS']['adrenal gland']+=['adrenals','suprarenal gland']
TEST2['SYNONYMS']['placenta']+=['placental tissue']
TEST2['SYNONYMS']['pituitary']+=['hypophysis','pituitary gland']
TEST2['SYNONYMS']['parathyroid']+=['parathyroid gland']
TEST2['SYNONYMS']['thymus']+=['thymic tissue']
TEST2['SYNONYMS']['primary']+=['primary site']
TEST2['SYNONYMS']['metastasis']+=['met secondary','met site']
TEST2['SYNONYMS']['Nthy-ori 3-1']+=['Nthy ori 3-1','Nthyori3-1']
TEST2['SYNONYMS']['TPC-1']+=['TPC1']
TEST2['SYNONYMS']['NCI-H295R']+=['H295R']
TEST2['SYNONYMS']['SW-13']+=['SW13']
TEST2['SYNONYMS']['JEG-3']+=['JEG3']
TEST2['SYNONYMS']['BeWo']+=['Be-Wo']
TEST2['SYNONYMS']['HTR-8/SVneo']+=['HTR8 SVneo','HTR-8 SVneo']
TEST2['SYNONYMS']['GH3']+=['GH-3']
TEST2['SYNONYMS']['HP75']+=['HP-75']
TEST2['SYNONYMS']['sHPT-1']+=['sHPT1']
TEST2['SYNONYMS']['TT']+=['TT cells']
TEST2['SYNONYMS']['thyrocytes']+=['thyroid epithelial cells']
TEST2['SYNONYMS']['adrenocortical cells']+=['adrenal cortex cells']
TEST2['SYNONYMS']['trophoblasts']+=['placental trophoblasts']
TEST2['SYNONYMS']['syncytiotrophoblasts']+=['syncytial trophoblasts']
TEST2['SYNONYMS']['pituitary endocrine cells']+=['pituitary hormone cells']
TEST2['SYNONYMS']['somatotrophs']+=['GH cells']
TEST2['SYNONYMS']['lactotrophs']+=['PRL cells']
TEST2['SYNONYMS']['parathyroid chief cells']+=['chief cells']
TEST2['SYNONYMS']['thymic epithelial cells']+=['TEC']
TEST2['SYNONYMS']['papillary thyroid carcinoma']+=['PTC']
TEST2['SYNONYMS']["Hashimoto's thyroiditis"]+=['Hashimoto disease']
TEST2['SYNONYMS']["Cushing's syndrome"]+=['hypercortisolism']
TEST2['SYNONYMS']["Addison's disease"]+=['primary adrenal insufficiency']
TEST2['SYNONYMS']['preeclampsia']+=['pre-eclampsia']
TEST2['SYNONYMS']['pituitary adenoma']+=['PA']
TEST2['SYNONYMS']['hyperparathyroidism']+=['HPT']
TEST2['SYNONYMS']['thymoma']+=['thymic tumor']
TEST2['SYNONYMS']['adrenocortical carcinoma']+=['ACC']
TEST2['SYNONYMS']['radioiodine ablation']+=['RAI ablation','I-131 ablation']
TEST2['SYNONYMS']['levothyroxine']+=['LT4']
TEST2['SYNONYMS']['methimazole']+=['MMI','thiamazole']
TEST2['SYNONYMS']['transsphenoidal surgery']+=['TSS']
TEST2['SYNONYMS']['hydrocortisone therapy']+=['HC therapy']
TEST2['SYNONYMS']['ketoconazole therapy']+=['KCZ therapy']
TEST2['SYNONYMS']['thymectomy']+=['thymus resection']
TEST2['SYNONYMS']['parathyroidectomy']+=['parathyroid resection']
TEST2['SYNONYMS']['magnesium sulfate']+=['MgSO4']
TEST2['SYNONYMS']['antihypertensive therapy']+=['AHT']
TEST2['SYNONYMS']['mitotane']+=['o,p\'-DDD']
TEST2['SYNONYMS']['96 hours']+=['96h']
TEST2['SYNONYMS']['5 days']+=['5d']
TEST2['SYNONYMS']['10 days']+=['10d']
TEST2['SYNONYMS']['4 weeks']+=['28 days']
TEST2['SYNONYMS']['8 weeks']+=['56 days']
TEST2['SYNONYMS']['9 months']+=['9 mos']
TEST2['SYNONYMS']['18 months']+=['1.5 years']
TEST2['SYNONYMS']['complete response']+=['CR']
TEST2['SYNONYMS']['no response']+=['NR']
TEST2['SYNONYMS']['symptom resolution']+=['symptoms resolved']
TEST2['SYNONYMS']['dose-limiting toxicity']+=['DLT']
TEST2['SYNONYMS']['male']+=['man','male donor','M donor']
TEST2['SYNONYMS']['female']+=['woman','F donor']
TEST2['SYNONYMS']['Middle Eastern']+=['MENA']
TEST2['SYNONYMS']['Nordic']+=['Scandinavian']
TEST2['SYNONYMS']['Eastern European']+=['East European']
TEST2['SYNONYMS']['Sub-Saharan African']+=['SSA']
TEST2['SYNONYMS']['South Asian']+=['Indian subcontinent']
TEST2['SYNONYMS']['Caribbean']+=['Caribbean region']
TEST2['SYNONYMS']['Scandinavia']+=['Nordic region']
TEST2['SYNONYMS']['Iberian Peninsula']+=['Iberia']
TEST2['SYNONYMS']['Balkan region']+=['Balkans']
TEST2['SYNONYMS']['Baltic states']+=['Baltics']
TEST2['SYNONYMS']['Andes']+=['Andean region']
TEST2['SYNONYMS']['Southeast Asia']+=['SEA']
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
        "sHPT-1":["parathyroid"],
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
        "mitotane":["adrenocortical carcinoma"],
        "unknown":["unknown"]
    }},
    "treatment_time":{"treatment":{
        "radioiodine ablation":["96 hours","5 days","10 days","4 weeks","8 weeks","9 months","18 months"],
        "levothyroxine":["96 hours","5 days","10 days","4 weeks","8 weeks","9 months","18 months"],
        "methimazole":["96 hours","5 days","10 days","4 weeks","8 weeks","9 months","18 months"],
        "transsphenoidal surgery":["96 hours","5 days","10 days","4 weeks","8 weeks","9 months","18 months"],
        "hydrocortisone therapy":["96 hours","5 days","10 days","4 weeks","8 weeks","9 months","18 months"],
        "ketoconazole therapy":["96 hours","5 days","10 days","4 weeks","8 weeks","9 months","18 months"],
        "thymectomy":["96 hours","5 days","10 days","4 weeks","8 weeks","9 months","18 months"],
        "parathyroidectomy":["96 hours","5 days","10 days","4 weeks","8 weeks","9 months","18 months"],
        "magnesium sulfate":["96 hours","5 days","10 days","4 weeks","8 weeks","9 months","18 months"],
        "antihypertensive therapy":["96 hours","5 days","10 days","4 weeks","8 weeks","9 months","18 months"],
        "mitotane":["96 hours","5 days","10 days","4 weeks","8 weeks","9 months","18 months"],
        "unknown":["unknown"]
    }},
    "response":{"treatment":{
        "radioiodine ablation":["complete response","no response","symptom resolution","dose-limiting toxicity"],
        "levothyroxine":["complete response","no response","symptom resolution","dose-limiting toxicity"],
        "methimazole":["complete response","no response","symptom resolution","dose-limiting toxicity"],
        "transsphenoidal surgery":["complete response","no response","symptom resolution","dose-limiting toxicity"],
        "hydrocortisone therapy":["complete response","no response","symptom resolution","dose-limiting toxicity"],
        "ketoconazole therapy":["complete response","no response","symptom resolution","dose-limiting toxicity"],
        "thymectomy":["complete response","no response","symptom resolution","dose-limiting toxicity"],
        "parathyroidectomy":["complete response","no response","symptom resolution","dose-limiting toxicity"],
        "magnesium sulfate":["complete response","no response","symptom resolution","dose-limiting toxicity"],
        "antihypertensive therapy":["complete response","no response","symptom resolution","dose-limiting toxicity"],
        "mitotane":["complete response","no response","symptom resolution","dose-limiting toxicity"],
        "unknown":["unknown"]
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
        "organ":{"placenta":["female"]}
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

    bt_values = [v for v in config['BIOPSY_TYPE'] if not (v=='blood' and not LABELS.get('BLOOD'))]
    quota={cat: eq_quota(config[key], targets) for cat,key in [
        ('disease','DISEASE'),('organ','ORGAN'),('cell_type','CELL_TYPE'),('cell_line','CELL_LINE'),
        ('sequencing_source','SEQ_SRC'),('library_selection','LIB_SEL'),('biopsy_type',None),
        ('biopsy_site','BIOPSY_SITE'),('treatment','TREATMENT'),('treatment_time','TREAT_TIME'),
        ('response','RESPONSE'),('age','AGE'),('sex','SEX'),('ethnicity','ETHNICITY'),
        ('localization','LOCALIZATION')
    ] if key}
    quota['biopsy_type']=eq_quota(bt_values, targets)
    meta_need = quota['biopsy_type'].get(LABELS.get('METASTASIS','metastasis'),0)
    isc_true = max(targets//2, meta_need)
    quota['is_cancer']={'true':isc_true,'false':targets-isc_true}
    remaining={k:Counter(v) for k,v in quota.items()}

    def pop_any(cat):
        pool=[v for v,c in remaining[cat].items() if c>0]
        if not pool: return None
        v=random.choice(pool); remaining[cat][v]-=1; return v

    def pop_best(cat, allowed):
        pool=[(v,remaining[cat].get(v,0)) for v in allowed if remaining[cat].get(v,0)>0]
        if pool:
            maxc=max(c for _,c in pool)
            cands=[v for v,c in pool if c==maxc]
            v=random.choice(cands); remaining[cat][v]-=1; return v
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
        vals=[ct for ct,orgs in SEM['cell_type']['organ'].items() if org in orgs]
        return list(dict.fromkeys(vals)) or list(config['CELL_TYPE'])

    def allowed_celllines_for_org(org):
        cands=[cl for cl,orgs in SEM['cell_line']['organ'].items() if org in orgs]
        return cands if cands else []

    def primary_sites_for_org(org):
        sites=[site for site,orgs in SEM['biopsy_site']['organ'].items() if org in orgs]
        return sites if sites else ([org] if org in BIOPSY_SITE else [])

    def metastasis_sites_for(disease, org):
        if 'DISEASE_MET_SITES' in config and isinstance(config['DISEASE_MET_SITES'], dict) and disease in config['DISEASE_MET_SITES']:
            s=config['DISEASE_MET_SITES'][disease]
            if not s: return []
            pool=[x for x in s if x in BIOPSY_SITE and x!=org]
            return pool
        return [s for s in BIOPSY_SITE if s!=org]

    def allowed_treatments_for_disease(d):
        cands=[t for t,ds in SEM['treatment']['disease'].items() if d in ds and t in config['TREATMENT']]
        if cands:
            return cands
        return ['unknown'] if 'unknown' in config['TREATMENT'] else []

    def allowed_tt_for_treatment(tr):
        if tr == 'unknown':
            return ['unknown']
        tt = SEM['treatment_time']['treatment'].get(tr, config['TREAT_TIME'])
        if tr not in {'no treatment','unknown','tamoxifen'}:
            tt=[x for x in tt if x!='not applicable']
        if tr=='tamoxifen':
            tt=['unknown']
        return tt

    def allowed_rr_for_treatment(tr, d):
        if tr == 'unknown':
            return ['unknown']
        rr = SEM['response']['treatment'].get(tr, config['RESPONSE'])
        if tr=='no treatment':
            rr=['not applicable']
        if tr=='tamoxifen':
            rr=['unknown']
        if config is TRAIN and d in NON_ONCO_LIMIT and tr!='no treatment':
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
        d_candidates=[v for v,c in remaining['disease'].items() if c>0]
        if not d_candidates: break
        d=random.choice(d_candidates)
        isc=normalize_is_cancer(d)
        if remaining['is_cancer'][isc]==0:
            alt=[x for x in d_candidates if normalize_is_cancer(x)!=isc]
            if not alt:
                continue
            d=random.choice(alt)
            isc=normalize_is_cancer(d)
        remaining['disease'][d]-=1
        remaining['is_cancer'][isc]-=1

        organs = SEM['disease']['organ'].get(d, ORG) or ORG
        org = pop_best('organ', organs) or random.choice(ORG)

        ct_allowed = allowed_celltypes_for_org(org) or list(config['CELL_TYPE'])
        ct = pop_best('cell_type', ct_allowed) or random.choice(ct_allowed)

        cl_allowed = allowed_celllines_for_org(org)
        if not cl_allowed:
            cl_allowed = [c for c in config['CELL_LINE']]
        cl = pop_best('cell_line', cl_allowed) or random.choice(cl_allowed)

        src_allowed = list(set(allowed_src_for_celltype(ct)) & set(allowed_sources_for_org(org))) or allowed_sources_for_org(org)
        src = pop_best('sequencing_source', src_allowed) or random.choice(src_allowed)
        lib = pop_best('library_selection', allowed_libraries_for_source(src)) or random.choice(allowed_libraries_for_source(src))

        if src == 'spatial':
            spatial_ok = set(config['SEM']['library_source']['cell_type'].get('spatial', []))
            if spatial_ok:
                spatial_ok_for_org = list(spatial_ok & set(allowed_celltypes_for_org(org))) or list(spatial_ok)
                if ct not in spatial_ok:
                    ct = pop_best('cell_type', spatial_ok_for_org) or random.choice(spatial_ok_for_org)

        bt_options=[LABELS['PRIMARY'],LABELS['METASTASIS']]+(([LABELS['BLOOD']] if LABELS.get('BLOOD') else []))
        if isc!='true' and LABELS.get('METASTASIS') in bt_options:
            bt_options=[x for x in bt_options if x!=LABELS['METASTASIS']]
        bt = pop_best('biopsy_type', bt_options) or random.choice(bt_options)

        if bt==LABELS.get('BLOOD'):
            if LABELS.get('BLOOD'):
                bs = pop_best('biopsy_site', [LABELS['BLOOD']]) or LABELS['BLOOD']
            else:
                bt = LABELS['PRIMARY']
                prim = primary_sites_for_org(org)
                bs = pop_best('biopsy_site', prim if prim else BIOPSY_SITE) or (prim[0] if prim else random.choice(BIOPSY_SITE))

        elif bt==LABELS['PRIMARY']:
            prim = primary_sites_for_org(org)
            if not prim:
                bs = pop_best('biopsy_site', BIOPSY_SITE) or random.choice(BIOPSY_SITE)
            else:
                bs = pop_best('biopsy_site', prim) or random.choice(prim)

        else:
            if isc=='true' and d in CANCER:
                mets=metastasis_sites_for(d, org)
                if mets:
                    bs = pop_best('biopsy_site', mets) or random.choice(mets)
                else:
                    bt=LABELS['PRIMARY']
                    prim = primary_sites_for_org(org)
                    bs = pop_best('biopsy_site', prim if prim else BIOPSY_SITE) or (prim[0] if prim else random.choice(BIOPSY_SITE))
            else:
                bt=LABELS['PRIMARY']
                prim = primary_sites_for_org(org)
                bs = pop_best('biopsy_site', prim if prim else BIOPSY_SITE) or (prim[0] if prim else random.choice(BIOPSY_SITE))

        if LABELS.get('BLOOD') and bs == LABELS['BLOOD']:
            bt = LABELS['BLOOD']

        if src=='spatial':
            if org in BIOPSY_SITE:
                if bs!=org and remaining['biopsy_site'].get(org,0)>-targets:
                    remaining['biopsy_site'][bs]+=1
                    remaining['biopsy_site'][org]-=1
                    bs = org
                bt=LABELS['PRIMARY']
                if 'not applicable' in config['CELL_LINE'] and org in SEM['cell_line']['organ'].get('not applicable', []):
                    cl = 'not applicable'
                else:
                    cl_allowed = allowed_celllines_for_org(org)
                    if cl not in cl_allowed and cl_allowed:
                        cl = pop_best('cell_line', cl_allowed) or random.choice(cl_allowed)
            else:
                alt_src=[x for x in allowed_sources_for_org(org) if x!='spatial'] or ['bulk']
                remaining['sequencing_source'][src]+=1
                src = pop_best('sequencing_source', alt_src) or 'bulk'
                lib = pop_best('library_selection', allowed_libraries_for_source(src)) or lib

        enforced_allowed_cls = allowed_celllines_for_org(org)
        if enforced_allowed_cls:
            if cl not in enforced_allowed_cls:
                cl = pop_best('cell_line', enforced_allowed_cls) or random.choice(enforced_allowed_cls)

        tr_candidates = allowed_treatments_for_disease(d)
        tr = pop_best('treatment', tr_candidates) if tr_candidates else None
        tr = tr or (random.choice(tr_candidates) if tr_candidates else 'unknown')

        if tr=='no treatment':
            tt = 'not applicable' if 'not applicable' in config['TREAT_TIME'] else random.choice(config['TREAT_TIME'])
            rr = 'not applicable' if 'not applicable' in config['RESPONSE'] else 'unknown'
            if 'treatment_time' in remaining and tt in remaining['treatment_time']: remaining['treatment_time'][tt]-=1
            if 'response' in remaining and rr in remaining['response']: remaining['response'][rr]-=1

        elif tr=='tamoxifen':
            tt='unknown'
            rr='unknown'
            if 'treatment_time' in remaining and 'unknown' in remaining['treatment_time']: remaining['treatment_time']['unknown']-=1
            if 'response' in remaining and 'unknown' in remaining['response']: remaining['response']['unknown']-=1

        elif tr=='unknown':
            tt='unknown'
            rr='unknown'
            if 'treatment_time' in remaining and 'unknown' in remaining['treatment_time']: remaining['treatment_time']['unknown']-=1
            if 'response' in remaining and 'unknown' in remaining['response']: remaining['response']['unknown']-=1

        else:
            tt_allowed = allowed_tt_for_treatment(tr)
            rr_allowed = allowed_rr_for_treatment(tr, d)
            tt = pop_best('treatment_time', tt_allowed) or random.choice(tt_allowed)
            rr = pop_best('response', rr_allowed) or random.choice(rr_allowed)

        sex_opts=[]
        if 'sex' in SEM:
            sx_by_d = SEM['sex'].get('disease',{}).get(d, [])
            sx_by_o = SEM['sex'].get('organ',{}).get(org, [])
            if not sx_by_o and org in {'prostate','testes'}: sx_by_o=['male']
            if not sx_by_o and org in {'cervix','ovaries','uterus'}: sx_by_o=['female']
            inter = list(set(sx_by_d) & set(sx_by_o))
            if inter: sex_opts=inter
            elif sx_by_d: sex_opts=sx_by_d
            elif sx_by_o: sex_opts=sx_by_o
        sex = pop_best('sex', sex_opts if sex_opts else config['SEX']) or random.choice(config['SEX'])

        age = pop_any('age') or random.choice(config['AGE'])
        eth = pop_any('ethnicity') or random.choice(config['ETHNICITY'])
        loc = pop_any('localization') or random.choice(config['LOCALIZATION'])

        if src != 'spatial' and ct == 'primary tissue' and cl != 'not applicable':
            if 'not applicable' in config['CELL_LINE'] and org in SEM['cell_line']['organ'].get('not applicable', []):
                cl = 'not applicable'

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

        all_sets = {
            'organ': set(config['ORGAN']),
            'disease': set(config['DISEASE']),
            'cell_line': set([c for c in config['CELL_LINE'] if c not in {'not applicable','unknown'}]),
            'cell_type': set(config['CELL_TYPE']),
            'treatment': set(config['TREATMENT']),
            'biopsy_site': set(config['BIOPSY_SITE']),
            'sequencing_source': set(config['SEQ_SRC']),
            'library_selection': set(config['LIB_SEL']),
            'treatment_time': set(config['TREAT_TIME']),
            'response': set(config['RESPONSE'])
        }
        phr = sanitize_phrase(phr, rec, syno, all_sets)

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
    make_dataset(TEST1, 1000, TEST1_DATA_PATH, TEST1_EXPANDED, TEST1_CSV, ctx_wrap=CTX_WRAP)
    make_dataset(TEST2, 1000, TEST2_DATA_PATH, TEST2_EXPANDED, TEST2_CSV, ctx_wrap=CTX_WRAP)
