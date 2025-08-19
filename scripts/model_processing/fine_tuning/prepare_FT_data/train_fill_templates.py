import os
import json
import random
import re
import pandas as pd
from collections import Counter, defaultdict, deque

DATA_PATH = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/data_templates_training/metadata_templates_train_withoutkeys.json'
EXPANDED_PATH = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/metadata_templates_train_expanded.json'
OUTPUT_CSV  = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/train_metadata_replaced_table.csv'

CATEGORIES = [
    'library_selection','sequencing_source','organ','biopsy_site','biopsy_type',
    'cell_line','cell_type','disease','treatment','treatment_time','response',
    'age','sex','ethnicity','localization','is_cancer','phrase'
]

LIB_SEL = ['polyA','inverse rRNA','hybrid selection','unknown']
SEQ_SRC = ['single-cell','bulk','spatial','unknown']
BIOPSY_SITE = ['blood','lung','heart','kidney','brain','stomach','intestine','colon','skin','liver','spleen','bone marrow','pancreas','prostate','breast','cervix','bone','cartilage','unknown']
BIOPSY_TYPE = ['primary','metastasis','blood','unknown']
CELL_LINE = ['HEK293','HeLa','HepG2','MCF7','A549','K562','U2OS','PC3','Jurkat','HCT116','SHSY5Y','C2C12','THP1','not applicable','unknown']
CELL_TYPE = ['muscle cells','liver cells','blood cells','kidney cells','nerve cells','connective cells','fat cells','bone cells','cartilage cells','specialized integrated cells','fibroblasts','migratory cells','stomach cells','lung cells','primary tissue','unknown']
ORGAN = ['liver','lung','heart','kidney','brain','muscle','spleen','pancreas','colon','stomach','intestine','skin','bone marrow','blood','prostate','breast','cervix','bone','cartilage','unknown']
DISEASE = ['lung cancer','hepatocellular carcinoma','breast cancer','leukemia','lymphoma','prostate cancer','pancreatic adenocarcinoma','glioblastoma','cirrhosis','chronic kidney disease',"Crohn's disease",'ulcerative colitis','myocardial infarction','diabetes mellitus','unknown']
TREATMENT = ['no treatment','cisplatin','doxorubicin','paclitaxel','sorafenib','imatinib','erlotinib','tamoxifen','methotrexate','5-fluorouracil','amoxicillin','gentamicin','irradiation','dexamethasone','unknown']
TREAT_TIME = ['pre treatment','on treatment','post treatment','relapse','unknown','24 hours','48 hours','1 week','3 months','1 year','not applicable']
RESPONSE = ['resistance','sensitivity','partial response','stable disease','progressive disease','unknown','not applicable']
AGE = ['34 years','62 years','47','born in 1982','55 years','19 years','44 years','67 years','41 years','unknown']
SEX = ['male','female','unknown']
ETHNICITY = ['Asian','Hispanic','European','African','American','unknown']
LOCALIZATION = ['North Africa','Paris','French','English','Australia','unknown']

SYNONYMS = {
    'muscle cells':['myocytes','contractile fibers','myofibers'],
    'liver cells':['hepatocytes','hepatic parenchyma'],
    'blood cells':['erythrocytes','leukocytes'],
    'kidney cells':['renal cells','tubular cells'],
    'nerve cells':['neurons','neural cells'],
    'connective cells':['stromal cells','mesenchymal cells'],
    'fat cells':['adipocytes'],
    'bone cells':['osteoblasts','osteocytes'],
    'cartilage cells':['chondrocytes'],
    'specialized integrated cells':['goblet cells','Paneth cells','myoepithelial cells'],
    'fibroblasts':['fibrocytes','stromal fibroblasts'],
    'migratory cells':['lymphocytes','natural killer cells'],
    'stomach cells':['parietal cells','chief cells'],
    'lung cells':['alveolar epithelial cells','bronchial epithelial cells'],
    'primary tissue':['primary sample','native tissue'],
    'liver':['hepatic organ','hepatobiliary organ','liver parenchyma','hepatic lobes'],
    'lung':['respiratory organ','pulmonary lobes'],
    'heart':['myocardium','cardiac organ','heart muscle'],
    'kidney':['renal organ','renal cortex','nephron organ'],
    'brain':['encephalon','cerebrum','cerebral organ'],
    'muscle':['muscle mass','contractile organ','skeletal muscle'],
    'spleen':['splenic organ','splenic tissue'],
    'pancreas':['pancreatic organ','pancreatic gland'],
    'colon':['colon mucosa'],
    'stomach':['gastric organ','stomach mucosa'],
    'intestine':['intestinal tract','gut','bowel'],
    'skin':['cutaneous tissue','epidermis','skin layer'],
    'bone marrow':['hematopoietic marrow','myeloid tissue','marrow tissue'],
    'blood':['bloodstream','vascular fluid','circulating fluid'],
    'prostate':['prostatic organ','prostate gland'],
    'breast':['mammary gland','breast tissue'],
    'cervix':['cervical tissue','uterine cervix'],
    'bone':['osseous tissue','skeletal bone'],
    'cartilage':['cartilaginous tissue'],
    'lung cancer':['pulmonary carcinoma','lung neoplasm'],
    'hepatocellular carcinoma':['HCC','hepatoma','liver cancer'],
    'breast cancer':['mammary carcinoma','breast neoplasm'],
    'leukemia':['blood cancer','hematologic malignancy'],
    'lymphoma':['lymphatic malignancy'],
    'prostate cancer':['prostatic carcinoma','prostate adenocarcinoma'],
    'pancreatic adenocarcinoma':['PDAC','pancreatic cancer'],
    'glioblastoma':['GBM','malignant glioma','astrocytoma grade IV'],
    'cirrhosis':['hepatic cirrhosis'],
    'chronic kidney disease':['CKD','renal disease'],
    "Crohn's disease":["regional enteritis","Crohn disease"],
    'ulcerative colitis':['UC','colitis ulcerosa'],
    'myocardial infarction':['heart attack','AMI'],
    'diabetes mellitus':['diabetes','DM'],
    'polyA':['polyA+','mRNA enrichment','oligo-dT capture'],
    'inverse rRNA':['rRNA depletion','RiboZero','ribominus'],
    'hybrid selection':['hybrid capture','RNA exome','exon capture'],
    'cisplatin':['CDDP'],
    'doxorubicin':['adriamycin','DOX'],
    'paclitaxel':['taxol','PTX'],
    'sorafenib':['Nexavar'],
    'imatinib':['Gleevec'],
    'erlotinib':['Tarceva'],
    'tamoxifen':['TAM'],
    'methotrexate':['MTX'],
    '5-fluorouracil':['5-FU'],
    'amoxicillin':['Amoxil'],
    'gentamicin':['Garamycin'],
    'irradiation':['radiation','gamma rays','X-ray'],
    'dexamethasone':['Decadron','DXM'],
    'pre treatment':['pretreatment','before treatment'],
    'on treatment':['during treatment','treatment-phase'],
    'post treatment':['after treatment','post-therapy'],
    'relapse':['recurrence','re-emergence'],
    'resistance':['refractory','non-responsive'],
    'sensitivity':['responsive','susceptible'],
    'partial response':['PR','partial remission'],
    'stable disease':['SD','disease stabilization'],
    'progressive disease':['PD','disease progression'],
    'male':['man','male donor','M donor'],
    'North Africa':['Maghreb','North-African'],
    'Paris':['Paris region'],
    'French':['France','French or'],
    'English':['UK origin','British'],
    'Australia':['AUS','Australia resident'],
    '34 years':['34-year-old','34yo','34 yrs'],
    '62 years':['62-year-old','62yo','62 yrs'],
    '47':['47-year-old','47yo','47 yrs'],
    '55 years':['55-year-old','55yo','55 yrs'],
    '19 years':['19-year-old','19yo','19 yrs'],
    '44 years':['44-year-old','44yo','44 yrs'],
    '67 years':['67-year-old','67yo','67 yrs'],
    '41 years':['41-year-old','41yo','41 yrs'],
}

SEM = {
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
        'unknown':ORGAN
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
        'unknown':ORGAN
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
        'unknown':ORGAN
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
        'unknown':DISEASE
    }},
    'treatment_time':{'treatment':{
        'cisplatin':TREAT_TIME,'doxorubicin':TREAT_TIME,'paclitaxel':TREAT_TIME,'sorafenib':TREAT_TIME,
        'imatinib':TREAT_TIME,'erlotinib':TREAT_TIME,'tamoxifen':TREAT_TIME,'methotrexate':TREAT_TIME,
        '5-fluorouracil':TREAT_TIME,'amoxicillin':TREAT_TIME,'gentamicin':TREAT_TIME,'irradiation':TREAT_TIME,
        'dexamethasone':TREAT_TIME,'no treatment':['not applicable'],'unknown':TREAT_TIME
    }},
    'response':{'treatment':{
        'cisplatin':RESPONSE,'doxorubicin':RESPONSE,'paclitaxel':RESPONSE,'sorafenib':RESPONSE,
        'imatinib':RESPONSE,'erlotinib':RESPONSE,'tamoxifen':RESPONSE,'methotrexate':RESPONSE,
        '5-fluorouracil':RESPONSE,'amoxicillin':RESPONSE,'gentamicin':RESPONSE,'irradiation':RESPONSE,
        'dexamethasone':RESPONSE,'no treatment':['not applicable'],
        'unknown':['unknown','partial response','stable disease','progressive disease']
    }},
    'biopsy_site':{'organ':{
        'blood':['blood','bone marrow'],
        'lung':['lung'],'heart':['heart'],'kidney':['kidney'],'brain':['brain'],'stomach':['stomach'],
        'intestine':['intestine','colon','stomach'],'colon':['colon','intestine'],'skin':['skin'],'liver':['liver'],
        'spleen':['spleen'],'bone marrow':['bone marrow','blood'],'pancreas':['pancreas'],'prostate':['prostate'],
        'breast':['breast'],'cervix':['cervix'],'bone':['bone','bone marrow'],'cartilage':['cartilage'],'unknown':BIOPSY_SITE
    }},
    'library_source':{'cell_type':{
        'single-cell':['blood cells','nerve cells','liver cells','muscle cells','fibroblasts','bone cells','cartilage cells','connective cells','lung cells','kidney cells','migratory cells'],
        'bulk':['blood cells','migratory cells','specialized integrated cells','fat cells','stomach cells','primary tissue','fibroblasts','liver cells','lung cells','kidney cells','muscle cells','bone cells'],
        'spatial':['primary tissue','liver cells','lung cells','kidney cells','muscle cells','stomach cells','connective cells','fibroblasts','specialized integrated cells']
    }}
}

CANCER_DISEASES = {'lung cancer','hepatocellular carcinoma','breast cancer','leukemia','lymphoma','prostate cancer','pancreatic adenocarcinoma','glioblastoma'}
HEMATO_DISEASES = {'leukemia','lymphoma'}
SOLID_DISEASES = {'lung cancer','hepatocellular carcinoma','breast cancer','prostate cancer','pancreatic adenocarcinoma','glioblastoma'}
NON_ONCO_NON_INF = {'cirrhosis','chronic kidney disease','myocardial infarction','diabetes mellitus'}
DISEASE_MET_SITES = {
    'pancreatic adenocarcinoma': {'liver','lung','bone marrow'},
    'lung cancer': {'brain','liver','bone marrow'},
    'breast cancer': {'bone marrow','lung','liver','brain'},
    'prostate cancer': {'bone marrow','liver','lung'},
    'hepatocellular carcinoma': {'lung','bone marrow'},
    'glioblastoma': set(),
    'lymphoma': set(),
    'leukemia': set(),
}

NOISE = [
    "sequencer lane rebalanced for duplex bias","insert size ~320bp","sample barcoded with UMIs","QC passed at RIN 8.1",
    "library molarity 7.5 nM","coverage uniformity within 1.3x","FASTQ trimmed (Phred≥30)","adapter clipping enabled",
    "lane bleedthrough negligible","5' bias not detected","batch randomization across plates","ERCC spike-ins present",
    "index-hopping mitigated","PCR duplicates removed","alignment on GRCh38","multi-mapping reads <2%","FRiP 0.21",
    "median insert 286","sequencing kit v3","freeze-thaw cycles=1","lane clustering density nominal","negative control clean",
    "RNase-free handling confirmed","unique fragments retained","flowcell ID anonymized"
]

def eq_quota(values, total):
    k=len(values); base=total//k; r=total-base*k
    lst=list(values); random.shuffle(lst)
    q={v:base for v in lst}
    for v in lst[:r]: q[v]+=1
    return q

TARGET_N=1500

with open(DATA_PATH) as f:
    templates=json.load(f)
expanded=random.choices(templates,k=max(6000,TARGET_N*3))
with open(EXPANDED_PATH,'w') as f:
    json.dump(expanded,f,indent=2)

quota={
    'disease':eq_quota(DISEASE,TARGET_N),
    'organ':eq_quota(ORGAN,TARGET_N),
    'cell_type':eq_quota(CELL_TYPE,TARGET_N),
    'cell_line':eq_quota([c for c in CELL_LINE if c!='not applicable']+['not applicable','unknown'],TARGET_N),
    'sequencing_source':eq_quota(SEQ_SRC,TARGET_N),
    'library_selection':eq_quota(LIB_SEL,TARGET_N),
    'biopsy_type':eq_quota(BIOPSY_TYPE,TARGET_N),
    'biopsy_site':eq_quota(BIOPSY_SITE,TARGET_N),
    'treatment':eq_quota(TREATMENT,TARGET_N),
    'treatment_time':eq_quota(TREAT_TIME,TARGET_N),
    'response':eq_quota(RESPONSE,TARGET_N),
    'age':eq_quota(AGE,TARGET_N),
    'sex':eq_quota(SEX,TARGET_N),
    'ethnicity':eq_quota(ETHNICITY,TARGET_N),
    'localization':eq_quota(LOCALIZATION,TARGET_N),
    'is_cancer':{'true':TARGET_N//2,'false':TARGET_N-TARGET_N//2}
}
remaining={k:Counter(v) for k,v in quota.items()}

def pop_any(cat):
    pool=[v for v,c in remaining[cat].items() if c>0]
    if not pool: return None
    v=random.choice(pool); remaining[cat][v]-=1; return v

def pop_from(cat, valid):
    pool=[v for v in valid if remaining[cat][v]>0]
    if pool: v=random.choice(pool); remaining[cat][v]-=1; return v
    return pop_any(cat)

def is_cancer_from_disease(d): return 'true' if d in CANCER_DISEASES else 'false'

def allowed_sources_for_org(org):
    opts=list(SEQ_SRC)
    if org in {'blood','bone marrow'}: opts=[x for x in opts if x!='spatial']
    return opts

def allowed_libraries_for_source(src):
    if src=='single-cell': return ['polyA','unknown']
    if src=='spatial': return ['polyA','hybrid selection','unknown']
    if src=='bulk': return ['polyA','inverse rRNA','unknown']
    return list(LIB_SEL)

def allowed_celltypes_for_org(org):
    return list(dict.fromkeys(SEM['cell_type']['organ'].get(org,[])+['primary tissue','unknown']))

def allowed_celllines_for_org_ct(org, ct):
    if ct=='primary tissue': return ['not applicable']
    opts=[cl for cl,orgs in SEM['cell_line']['organ'].items() if org in orgs]+['unknown']
    return list(dict.fromkeys([c for c in opts if c in remaining['cell_line']]))

def primary_sites_for_org(org):
    if org=='unknown': return ['unknown']
    return list(dict.fromkeys(SEM['biopsy_site']['organ'].get(org,[org])))

def metastasis_sites_for_disease(d, org):
    if d not in SOLID_DISEASES: return []
    sites=[x for x in DISEASE_MET_SITES.get(d,set()) if x!=org]
    return sites

def allowed_treatments_for_disease(d):
    feas=[]
    for t in TREATMENT:
        if t in {'unknown','no treatment'}: feas.append(t)
        elif d=='unknown': feas.append(t)
        elif d in SEM['treatment']['disease'].get(t,[]): feas.append(t)
    return feas

def fuzzy_token_patterns(values):
    pats=[]
    for v in values:
        base=re.sub(r'\s+',r'\\s*',re.escape(v))
        base=base.replace(r'\-',r'[-_ ]*').replace(r'_',r'[_ ]*')
        pats.append(r'(?i)(?<![A-Za-z0-9])'+base+r'(?![A-Za-z0-9])')
        if re.search(r'\d',v):
            letters=re.sub(r'[^A-Za-z]+','',v)
            digits=''.join(re.findall(r'\d+',v))
            if letters and digits:
                pats.append(r'(?i)(?<![A-Za-z0-9])'+re.escape(letters)+r'\s*[-_ ]*'+re.escape(digits)+r'(?![A-Za-z0-9])')
    return pats

def allowed_terms(value):
    return set([value]+SYNONYMS.get(value,[]))

def sanitize_phrase(phrase, rec):
    keep = {
        'organ': allowed_terms(rec['organ']) if rec['organ']!='unknown' else set(),
        'disease': allowed_terms(rec['disease']) if rec['disease']!='unknown' else set(),
        'cell_line': allowed_terms(rec['cell_line']) if rec['cell_line'] not in {'not applicable','unknown'} else set(),
        'cell_type': allowed_terms(rec['cell_type']) if rec['cell_type']!='unknown' else set(),
        'treatment': allowed_terms(rec['treatment']) if rec['treatment'] not in {'no treatment','unknown'} else set(),
        'biopsy_site': allowed_terms(rec['biopsy_site']) if rec['biopsy_site']!='unknown' else set()
    }
    forbidden=[]
    all_sets={
        'organ': set([x for x in ORGAN if x!='unknown']),
        'disease': set([x for x in DISEASE if x!='unknown']),
        'cell_line': set([c for c in CELL_LINE if c not in {'not applicable','unknown'}]),
        'cell_type': set([c for c in CELL_TYPE if c!='primary tissue' and c!='unknown']),
        'treatment': set([t for t in TREATMENT if t!='no treatment' and t!='unknown']),
        'biopsy_site': set([x for x in BIOPSY_SITE if x!='unknown'])
    }
    for cat, universe in all_sets.items():
        for val in universe:
            if val in keep.get(cat,set()): continue
            forbidden.extend([val]+SYNONYMS.get(val,[]))
    patterns=fuzzy_token_patterns(forbidden)
    for pat in patterns:
        phrase=re.sub(pat,'',phrase)
    phrase=re.sub(r'\s{2,}',' ',phrase).strip().strip(',').strip()
    return phrase

def inject_value(sentence, insertion):
    words=sentence.split()
    if not words: return insertion
    idx=random.randint(0,len(words))
    return ' '.join(words[:idx]+[insertion]+words[idx:])

def phrase_with_context(phrase, rec):
    cats=['biopsy_site','organ','response','treatment','disease','cell_type','cell_line']
    for cat in cats:
        val=rec[cat]
        if val in {'not applicable','unknown'} or (cat=='treatment' and val=='no treatment'): continue
        alt=random.choice(SYNONYMS.get(val,[val])) if random.random()<0.8 else val
        phrase=inject_value(phrase, alt)
    for _ in range(random.randint(3,5)):
        phrase=inject_value(phrase, random.choice(NOISE))
    return sanitize_phrase(phrase, rec)

rows=[]

for i in range(TARGET_N):
    d=pop_any('disease')
    if d is None: break
    isc=is_cancer_from_disease(d)
    if d=='unknown': isc='true' if remaining['is_cancer']['true']>remaining['is_cancer']['false'] else 'false'
    if remaining['is_cancer'][isc]==0: isc='false' if isc=='true' else 'true'
    remaining['is_cancer'][isc]-=1
    organs=SEM['disease']['organ'].get(d,ORGAN) or ORGAN
    o=pop_from('organ',organs)
    if o is None: o='unknown'
    ct=pop_from('cell_type',allowed_celltypes_for_org(o)) or 'unknown'
    if d not in CANCER_DISEASES and d!='unknown':
        if ct=='primary tissue':
            if remaining['cell_line']['not applicable']>0:
                remaining['cell_line']['not applicable']-=1
                cl='not applicable'
            else:
                cl=pop_from('cell_line',['not applicable']) or 'unknown'
        else:
            if remaining['cell_line']['unknown']>0:
                remaining['cell_line']['unknown']-=1
                cl='unknown'
            else:
                cl=pop_from('cell_line',['unknown']) or 'unknown'
    else:
        if ct=='primary tissue':
            if remaining['cell_line']['not applicable']>0:
                remaining['cell_line']['not applicable']-=1
                cl='not applicable'
            else:
                cl=pop_from('cell_line',[x for x in CELL_LINE if x!='not applicable']) or 'unknown'
        else:
            cl=pop_from('cell_line',allowed_celllines_for_org_ct(o,ct)) or 'unknown'
    src=pop_from('sequencing_source',allowed_sources_for_org(o)) or 'unknown'
    lib=pop_from('library_selection',allowed_libraries_for_source(src)) or 'unknown'
    bt = pop_from('biopsy_type', BIOPSY_TYPE) or 'unknown'
    if bt=='blood':
        if d in HEMATO_DISEASES or o in {'blood','bone marrow'}:
            if remaining['biopsy_site']['blood']>0: remaining['biopsy_site']['blood']-=1
            bs='blood'
        elif d in HEMATO_DISEASES and bt == 'primary':
            bs = pop_from('biopsy_site', ['blood', 'bone marrow', 'spleen']) or 'blood'
        else:
            bt='primary'
            bs=pop_from('biopsy_site',primary_sites_for_org(o)) or 'unknown'
    elif bt=='primary':
        bs=pop_from('biopsy_site',primary_sites_for_org(o)) or 'unknown'
    elif bt=='metastasis':
        if (d in SOLID_DISEASES) and (isc=='true'):
            mets=metastasis_sites_for_disease(d,o)
            if mets:
                bs=pop_from('biopsy_site',mets) or pop_from('biopsy_site',primary_sites_for_org(o)) or 'unknown'
            else:
                bt='primary'
                bs=pop_from('biopsy_site',primary_sites_for_org(o)) or 'unknown'
        else:
            bt='primary'
            bs=pop_from('biopsy_site',primary_sites_for_org(o)) or 'unknown'
    else:
        bs=pop_from('biopsy_site',['unknown']) or 'unknown'
    feas_t=allowed_treatments_for_disease(d)
    tr=pop_from('treatment',feas_t) or 'unknown'
    if tr=='no treatment':
        tt='not applicable'
        if remaining['treatment_time']['not applicable']>0: remaining['treatment_time']['not applicable']-=1
        rr='not applicable'
        if remaining['response']['not applicable']>0: remaining['response']['not applicable']-=1
        else: rr=pop_from('response',['not applicable']) or pop_any('response') or 'unknown'
    else:
        tt_allowed=[x for x in SEM['treatment_time']['treatment'].get(tr,TREAT_TIME) if x!='not applicable']
        tt=pop_from('treatment_time',tt_allowed) or pop_any('treatment_time') or 'unknown'
        rr_allowed=SEM['response']['treatment'].get(tr,RESPONSE)
        if d in NON_ONCO_NON_INF:
            rr_allowed=[x for x in rr_allowed if x in {'unknown','stable disease'}]
            if not rr_allowed: rr_allowed=['unknown','stable disease']
        rr=pop_from('response',rr_allowed) or pop_any('response') or 'unknown'
    age=pop_any('age') or 'unknown'
    sex=pop_any('sex') or 'unknown'
    eth=pop_any('ethnicity') or 'unknown'
    loc=pop_any('localization') or 'unknown'
    rec={
        'library_selection':lib,
        'sequencing_source':src,
        'organ':o,
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
    tpl=random.choice(expanded)
    if isinstance(tpl,dict): tpl=tpl.get('template', next(iter(tpl.values())))
    if isinstance(tpl,(list,tuple)): tpl=tpl[0]
    if not isinstance(tpl,str): tpl=str(tpl)
    rec['phrase']=phrase_with_context(tpl, rec)
    rows.append(rec)

df=pd.DataFrame(rows,columns=CATEGORIES)
df.to_csv(OUTPUT_CSV,index=False)
