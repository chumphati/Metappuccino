import pandas as pd
import random
import math
import os
import re

SUMMARY_CATS_PATH = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/old_logs/LAST_FINE_TUNING_TEST/summary_cats.csv"
OUTPUT_PATH = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/synthetic_balanced.csv"

BALANCE_CATS = [
    "organ","disease","host_phenotype",
    "library_selection","library_source",
    "treatment","treatment_time",
    "response","donor_information"
]

df = pd.read_csv(SUMMARY_CATS_PATH)
category_values = {}
counts = {}
for _, r in df.iterrows():
    cat = r["category"]
    parts = [p.strip() for p in r["all_values"].split(";") if p.strip()]
    vals, cnt = [], {}
    for p in parts:
        m = re.match(r"(.+?)\s*\((\d+)\)", p)
        v, c = (m.group(1).strip(), int(m.group(2))) if m else (p, 0)
        vals.append(v)
        cnt[v] = c
    category_values[cat] = vals
    counts[cat] = cnt

category_values.setdefault("treatment_time", [])
counts.setdefault("treatment_time", {})
for extra in ["post treatment", "no treatment", "nan"]:
    if extra not in category_values["treatment_time"]:
        category_values["treatment_time"].append(extra)
        counts["treatment_time"].setdefault(extra, 0)

max_counts = {c: max(counts[c].values() or [0]) for c in BALANCE_CATS}
target_counts = {c: math.ceil(0.9 * max_counts[c]) for c in BALANCE_CATS}

cell_types_by_organ_disease = {
    ('colon','colorectal cancer'):'Colonocyte',('lungs','lung cancer'):'monocyte',
    ('skin','skin melanoma'):'fibroblast',('brain','glioblastoma'):'astrocyte',
    ('brain','brain tumor'):'astrocyte',('breast','breast cancer'):'epithelial',
    ('liver','hepatitis'):'B cell',('lungs','pneumonia'):'monocyte'
}
cell_types_by_organ = {
    'colon':'Colonocyte','lungs':'monocyte','skin':'fibroblast','brain':'astrocyte',
    'breast':'epithelial','liver':'B cell','kidney':'fibroblast',
    'heart':'fibroblast','pancreas':'fibroblast'
}
tissue_types_by_organ = {
    'colon':'epithelial','lungs':'epithelial','skin':'connective','brain':'nervous',
    'breast':'epithelial','liver':'epithelial','kidney':'epithelial',
    'heart':'muscle','pancreas':'epithelial'
}
cell_lines_by_organ = {
    'colon':'DLD-1','lungs':'A549','skin':'Primary tissue','brain':'U87',
    'breast':'MCF7','liver':'Primary tissue','kidney':'Primary tissue',
    'heart':'Primary tissue','pancreas':'Primary tissue'
}

def weighted_choice(options, weights):
    total = sum(weights)
    if total <= 0:
        return random.choice(options)
    return random.choices(options, weights=weights, k=1)[0]

weight_maps = {
    'organ': {'heart':2, 'kidney':2, 'pancreas':2, 'brain':0.5, 'lungs':0.5, 'nan':0.2},
    'disease': {'normal':0.1},
    'host_phenotype': {'parental':0.75},
    'library_selection': {'nan':0.2},
    'treatment_time': {'no treatment':0.2, 'post treatment':1.0}
}

library_selections_context = {
    'polyA':['PolyA'],'inverse rRNA':['ribosomal RNA depletion'],
    'hybrid selection':['exome capture'],'small RNA':['microRNA'],
    'other':['random primers']
}
library_sources_context = {
    'single-cell':['10x genomics'],'bulk':['bulk RNA sequencing']
}
donor_clues = {
    'pregnant woman':['6 months before delivery','placental biopsy','prenatal donor'],
    'diabetic':['elevated fasting glucose','HbA1c 8.2%'],
    'alcohol consumer':['daily ethanol exposure','reported alcohol use'],
    'hypertensive':['hypertension managed with ACE inhibitors','140/90 mmHg BP']
}
cell_types_context = {
    'neuron':['neuron','neuronal','nerve cell','brain-derived'],
    'fibroblast':['fibroblast','skin-associated cell','connective tissue','muscle-related cell'],
    'CD8 T cell':['CD8 T cell','cytotoxic lymphocyte','immune-related cell'],
    'monocyte':['monocyte','blood-derived cell'],
    'NK cell':['NK cell','natural killer cell','immune effector'],
    'Colonocyte':['Colonocyte','intestinal epithelial cell','colon-related cell'],
    'B cell':['B cell','antibody-producing cell','humoral lymphocyte'],
    'astrocyte':['astrocyte','glial cell'],
    'epithelial':['epithelial cell','ductal epithelial'],
    'Primary tissue':['primary tissue','fresh tissue sample','uncultured primary cells']
}
noisy_sentences = [
    "Sample stored at −80°C.","Standard procedure was followed.",
    "Lab uses ISO9001 certification.","Operator noted batch variation in library prep.",
    "Serum potassium levels were inconclusive.","Participant consumed caffeine prior to sampling.",
    "Unexpected bands in gel electrophoresis.","pH adjusted to 7.4 before sequencing."
]
clinical_responses = ['Progressive Disease','Stable Disease']

def _ensure(cp, v, syn=None):
    if not v or v=='nan': return
    txt = ' '.join(cp).lower()
    if v.lower() in txt: return
    if syn:
        for alt in syn:
            if alt.lower() not in txt:
                cp.append(alt)
                return
    cp.append(v)

def generate_context(cols, donor, cell_line):
    cp=[]
    if cols['library_source'] in library_sources_context:
        cp.append(random.choice(library_sources_context[cols['library_source']]))
    if cols['cell_type'] in cell_types_context:
        cp.append(random.choice(cell_types_context[cols['cell_type']]))
    if cols['tissue_type'] not in ('nan',None):
        cp.append(f"{cols['tissue_type']} tissue")
    if cols['disease'] not in ('nan','normal'):
        cp.append(cols['disease'])
        if cols['disease'] in donor_clues:
            cp.append(random.choice(donor_clues[cols['disease']]))
    if cols['organ'] not in ('nan',None):
        cp.append(cols['organ'])
    if cols['library_selection'] in library_selections_context:
        cp.append(random.choice(library_selections_context[cols['library_selection']]))
    if cols['treatment'] not in ('nan','no treatment'):
        cp.append(cols['treatment'])
    cp.append(donor)
    if cell_line:
        cp.append(cell_line)
    for k in ['cell_type','tissue_type','organ','disease','library_selection','library_source','treatment']:
        _ensure(cp, cols[k], {
            'cell_type':cell_types_context.get(cols[k],[]),
            'library_selection':library_selections_context.get(cols[k],[]),
            'library_source':library_sources_context.get(cols[k],[])
        }.get(k))
    cp+=random.sample(noisy_sentences,k=random.randint(2,4))
    random.shuffle(cp)
    seps=[', ','; ',' | ',' ']
    return ''.join(p+(random.choice(seps) if i<len(cp)-1 else '') for i,p in enumerate(cp))

MAX_SYN = 6000
synthetic=[]
synth_counts={c:{v:0 for v in category_values[c]} for c in BALANCE_CATS}
uid=0
reverse_map={dis:org for (org,dis) in cell_types_by_organ_disease}

while uid<MAX_SYN and any(
    synth_counts[c][v]<target_counts[c]-counts[c].get(v,0)
    for c in BALANCE_CATS for v in category_values[c]
):
    uid+=1
    cols={}
    for c in BALANCE_CATS:
        if c=='disease':
            deficits={v:target_counts[c]-counts[c].get(v,0)-synth_counts[c][v] for v in category_values[c]}
            pos=[v for v,d in deficits.items() if d>0]
            opts=pos or category_values[c]
            wmap=weight_maps.get(c,{})
            weights=[deficits.get(v,1)*wmap.get(v,1) for v in opts]
            disease=weighted_choice(opts,weights)
            cols[c]=disease if disease not in (None,'nan') else 'normal'
        elif c=='organ':
            if cols.get('disease') in reverse_map:
                cols[c]=reverse_map[cols['disease']]
            else:
                deficits={v:target_counts[c]-counts[c].get(v,0)-synth_counts[c][v] for v in category_values[c]}
                pos=[v for v,d in deficits.items() if d>0]
                opts=pos or category_values[c]
                wmap=weight_maps.get(c,{})
                weights=[deficits.get(v,1)*wmap.get(v,1) for v in opts]
                cols[c]=weighted_choice(opts,weights)
        elif c=='treatment_time':
            if cols.get('disease')=='normal':
                cols[c]='no treatment'
            else:
                deficits={v:target_counts[c]-counts[c].get(v,0)-synth_counts[c][v] for v in category_values[c]}
                pos=[v for v,d in deficits.items() if d>0]
                opts=[v for v in category_values[c] if v not in ('nan','no treatment')]
                opts=pos or opts
                wmap=weight_maps.get(c,{})
                weights=[deficits.get(v,1)*wmap.get(v,1) for v in opts]
                cols[c]=weighted_choice(opts,weights)
        elif c=='host_phenotype':
            deficits={v:target_counts[c]-counts[c].get(v,0)-synth_counts[c][v] for v in category_values[c]}
            opts=category_values[c]
            wmap=weight_maps.get(c,{})
            weights=[deficits.get(v,1)*wmap.get(v,1) for v in opts]
            cols[c]=weighted_choice(opts,weights)
        elif c=='library_selection':
            deficits={v:target_counts[c]-counts[c].get(v,0)-synth_counts[c][v] for v in category_values[c]}
            pos=[v for v,d in deficits.items() if d>0]
            opts=pos or category_values[c]
            wmap=weight_maps.get(c,{})
            weights=[deficits.get(v,1)*wmap.get(v,1) for v in opts]
            cols[c]=weighted_choice(opts,weights)
        elif c=='treatment':
            if cols.get('disease')=='normal':
                cols[c]='no treatment'
            else:
                deficits={v:target_counts[c]-counts[c].get(v,0)-synth_counts[c][v] for v in category_values[c]}
                pos=[v for v,d in deficits.items() if d>0]
                opts=[v for v in category_values[c] if v!='no treatment']
                opts=pos or opts
                cols[c]=random.choice(opts)
        else:
            deficits={v:target_counts[c]-counts[c].get(v,0)-synth_counts[c][v] for v in category_values[c]}
            pos=[v for v,d in deficits.items() if d>0]
            cols[c]=random.choice(pos) if pos else random.choice(category_values[c])
    if cols['treatment'] in ('no treatment',None,'nan'):
        cols['treatment_time']='no treatment'
    organ=cols['organ']
    if cols['disease']=='normal':
        cl='Primary tissue'
        cols['response']='nan'
    else:
        cl=cell_lines_by_organ.get(organ,'Primary tissue')
        if cols.get('response') not in clinical_responses:
            cols['response']=random.choice(clinical_responses)
    cols['cell_line']=cl
    cols['cell_type']=('Primary tissue' if cl=='Primary tissue' else cell_types_by_organ_disease.get((organ,cols['disease']),cell_types_by_organ.get(organ,'nan')))
    cols['tissue_type']=tissue_types_by_organ.get(organ,'nan')
    ctx=generate_context(cols,cols['donor_information'],None if cl=='Primary tissue' else cl)
    synthetic.append({"run_accession":f"SYN{str(uid).zfill(6)}","context":ctx,**cols})
    for c in BALANCE_CATS:
        synth_counts[c][cols[c]]+=1

df_out=pd.DataFrame(synthetic)
os.makedirs(os.path.dirname(OUTPUT_PATH),exist_ok=True)
df_out.to_csv(OUTPUT_PATH,index=False)
