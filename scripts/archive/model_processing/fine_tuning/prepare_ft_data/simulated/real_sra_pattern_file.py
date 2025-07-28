import json
import random
import argparse
import csv
import uuid
import re

CELL_TYPE_TRAIN = ["muscle cells","liver cells","blood cells","kidney cells","nerve cells","connective cells","fat cells","bone cells","cartilage cells","specialized integrated cells","fibroblasts","migratory cells","stomach cells","lung cells"]
CELL_TYPE_VAL   = ["pancreatic cells","intestinal cells","neurons","lymphocytes","hepatocytes","chondrocytes"]

TISSUE_TYPE_TRAIN = ["epithelial tissue","connective tissue","muscle tissue","nervous tissue","blood tissue","lymphatic tissue","adipose tissue","cartilage tissue","bone tissue","glandular tissue","fibrous tissue","hematopoietic tissue","smooth muscle tissue","cardiac muscle tissue"]
TISSUE_TYPE_VAL   = ["skeletal muscle tissue","stratified epithelium","simple epithelium","columnar epithelium","cuboidal epithelium","glandular epithelium"]

CELL_LINE_TRAIN = ["Primary tissue","HEK293","HeLa","HepG2","MCF7","A549","K562","U2OS","PC3","Jurkat","HCT116","SHSY5Y","C2C12","THP1"]
CELL_LINE_VAL   = ["HL60","SKNSH","T47D","CHO","3T3","LNCaP"]

ORGAN_TRAIN = ["liver","lung","heart","kidney","brain","muscle","spleen","pancreas","colon","stomach","intestine","skin","bone marrow","blood"]
ORGAN_VAL   = ["testis","ovary","prostate","thymus","bladder","thyroid"]

DISEASE_TRAIN = ["lung cancer","hepatocellular carcinoma","breast cancer","leukemia","lymphoma","prostate cancer","pancreatic adenocarcinoma","glioblastoma","cirrhosis","chronic kidney disease","Crohn's disease","ulcerative colitis","myocardial infarction","diabetes mellitus"]
DISEASE_VAL   = ["Alzheimer's disease","Parkinson's disease","amyotrophic lateral sclerosis","tuberculosis","influenza","malaria"]

HOST_PHENOTYPE = ["parental","persistent"]

LIB_SELECTION_TRAIN = ["polyA","inverse rRNA","hybrid selection"]
LIB_SELECTION_VAL   = ["small RNA","other"]

LIB_SOURCE = ["single-cell","bulk"]

TREATMENT_TRAIN = ["no treatment","cisplatin","doxorubicin","paclitaxel","sorafenib","imatinib","erlotinib","tamoxifen","methotrexate","5-fluorouracil","amoxicillin","gentamicin","irradiation","dexamethasone"]
TREATMENT_VAL   = ["interferon alpha","aspirin","atorvastatin","hydrocortisone","cyclosporin A","rapamycin"]

TREATMENT_TIME_TRAIN = ["no treatment","pre treatment","on treatment","post treatment","relapse","undefined","24 hours","48 hours","1 week","3 months","1 year"]
TREATMENT_TIME_VAL   = ["72 hours","2 weeks","6 months"]

RESPONSE_TRAIN = ["nan","resistance","sensitivity","partial response","stable disease","progressive disease"]
RESPONSE_VAL   = ["relapse","adverse event"]

DONOR_INFO_TRAIN = ["male, 34 years","female, 62 years","age 47, North Africa","born in 1982, Paris","Asian, non-smoker","Hispanic, BMI 27","European, vegan","African American, 55 years","no information","19 years, runner","French, hypertensive","English, 44 years","67 years, Australia","smoker, 41 years"]
DONOR_INFO_VAL   = ["female, pregnant","Mediterranean, normal","39 years, immunosuppressed","American Indian, athlete","Pacific Islander, 36 years","male, ex-smoker","unknown"]

INSTRUMENT_PLATFORMS = ["Illumina NovaSeq 6000","Illumina HiSeq 4000","NextSeq 500","NextSeq 2000","MiSeq"]

VALUES_TRAIN = {
    "cell_type":CELL_TYPE_TRAIN,
    "tissue_type":TISSUE_TYPE_TRAIN,
    "cell_line":CELL_LINE_TRAIN,
    "organ":ORGAN_TRAIN,
    "disease":DISEASE_TRAIN,
    "host_phenotype":HOST_PHENOTYPE,
    "library_selection":LIB_SELECTION_TRAIN,
    "library_source":LIB_SOURCE,
    "treatment":TREATMENT_TRAIN,
    "treatment_time":TREATMENT_TIME_TRAIN,
    "response":RESPONSE_TRAIN,
    "donor_information":DONOR_INFO_TRAIN
}

VALUES_VAL = {
    "cell_type":CELL_TYPE_VAL,
    "tissue_type":TISSUE_TYPE_VAL,
    "cell_line":CELL_LINE_VAL,
    "organ":ORGAN_VAL,
    "disease":DISEASE_VAL,
    "host_phenotype":HOST_PHENOTYPE,
    "library_selection":LIB_SELECTION_VAL,
    "library_source":LIB_SOURCE,
    "treatment":TREATMENT_VAL,
    "treatment_time":TREATMENT_TIME_VAL,
    "response":RESPONSE_VAL,
    "donor_information":DONOR_INFO_VAL
}

SYNONYMS = {
    "muscle cells":["myocytes","contractile fibers","myofibrils","muscular cells","myofibers"],
    "liver cells":["hepatocytes","hepatic parenchyma","liver parenchyma cells","liver tissue cells","hepatic cells"],
    "blood cells":["hematocytes","corpuscles","erythrocytes","leukocytes","circulating cells"],
    "connective cells":["stromal cells","mesenchymal cells","support cells","CT cells","matrix cells"],
    "fat cells":["adipocytes","lipid cells","adipose cells","white adipose cells","brown adipose cells"],
    "bone cells":["osteoblasts","osteoclasts","osteocytes","skeletal cells","bone-forming cells"],
    "lung":["pulmonary tissue","pulmo","respiratory organ","lung lobes","bronchial tissue"],
    "lung cancer":["pulmonary carcinoma","bronchogenic carcinoma","lung neoplasm","pulmonary malignancy","airway cancer"],
    "breast cancer":["mammary carcinoma","ductal carcinoma","breast neoplasm","mammary gland cancer","breast malignancy"],
    "parental":["wild type","WT","baseline","native","untreated lineage"],
    "persistent":["persister","tolerant","drug-persistent","survivor cells","resistant population"],
    "polyA":["poly-A selected","oligo-dT capture","mRNA enrichment","polyA+","polyadenylation"],
    "inverse rRNA":["ribo-zero","rRNA depletion","ribominus","riboerase","ribozero"],
    "hybrid selection":["hybrid capture","exon capture","RNA exome","SeqCap","capture probe"],
    "small RNA":["sRNA","smallRNA isolation","tinyRNA","miRNA prep","shortRNA"],
    "no treatment":["untreated","vehicle","control","mock","NA"],
    "cisplatin":["CDDP","cis-Pt","cisplatinum","platin","cis diamminedichloroplatinum"],
    "doxorubicin":["adriamycin","DOX","anthracycline","rubidomycin","doxorubicine"],
    "irradiation":["radiation","ionizing radiation","gamma rays","X-ray","rad-exposure"],
    "24 hours":["24h","one day","day-1","24 hrs","t=24h"],
    "48 hours":["48h","two days","day-2","48 hrs","t=48h"],
    "resistance":["resistant","drug-resistant","non-responsive","refractory","tolerant"],
    "sensitivity":["sensitive","drug-sensitive","responsive","susceptible","high response"],
    "male, 34 years":["34yo male","male age 34","M34","man, 34","male (34)"],
    "female, 62 years":["62yo female","female age 62","F62","woman, 62","female (62)"],
    "Illumina NovaSeq 6000":["NovaSeq6000","Illumina NS6000","NS6000","Illumina NovaSeq","NovaSeq"],
    "Illumina HiSeq 4000":["HiSeq4000","Illumina HS4000","HS4000","Illumina HiSeq","HiSeq"]
}

TAGS = {c:f"__{c.upper()}__" for c in VALUES_TRAIN}
TAGS["instrument_platform"] = "__INSTRUMENT_PLATFORM__"

def maybe_syn(v):
    return random.choice(SYNONYMS[v]) if v in SYNONYMS and random.random()<0.5 else v

def pick(cat, split):
    pool = VALUES_TRAIN if split=="train" else VALUES_VAL
    return maybe_syn(random.choice(pool[cat]))

def pick_instrument():
    return random.choice(INSTRUMENT_PLATFORMS)

def dedup_instr(txt):
    return re.sub(r"(__INSTRUMENT_PLATFORM__[\s,]*){2,}", "__INSTRUMENT_PLATFORM__, ", txt)

ORG2DISEASE = {
    "lung":["lung cancer"],"liver":["hepatocellular carcinoma","cirrhosis"],
    "breast":["breast cancer"],"blood":["leukemia","lymphoma"],"pancreas":["pancreatic adenocarcinoma"],
    "prostate":["prostate cancer"]
}
CELL2TISSUE = {
    "muscle cells":"muscle tissue","liver cells":"epithelial tissue","blood cells":"blood tissue",
    "kidney cells":"epithelial tissue","nerve cells":"nervous tissue","lung cells":"epithelial tissue",
    "cartilage cells":"cartilage tissue","bone cells":"bone tissue"
}
LINE2CELL = {
    "HeLa":"epithelial tissue","A549":"lung cells","HepG2":"liver cells","K562":"blood cells","Jurkat":"blood cells","MCF7":"epithelial tissue"
}

def coherent_mapping(split):
    organ = pick("organ",split)
    disease = random.choice(ORG2DISEASE.get(organ, VALUES_TRAIN["disease"] if split=="train" else VALUES_VAL["disease"]))
    cell_type = random.choice(list(CELL2TISSUE.keys()))
    tissue_type = CELL2TISSUE[cell_type]
    cell_line = random.choice([ln for ln,ct in LINE2CELL.items() if ct==cell_type] or (VALUES_TRAIN["cell_line"] if split=="train" else VALUES_VAL["cell_line"]))
    return {"organ":maybe_syn(organ),"disease":maybe_syn(disease),"cell_type":maybe_syn(cell_type),"tissue_type":maybe_syn(tissue_type),"cell_line":cell_line}

def fill_template(tpl, split):
    tpl = dedup_instr(tpl)
    mapping = coherent_mapping(split)
    if TAGS["instrument_platform"] in tpl:
        tpl = tpl.replace(TAGS["instrument_platform"], pick_instrument(),1)
    for cat,val in mapping.items():
        tpl = tpl.replace(TAGS[cat], val)
    for cat,tag in TAGS.items():
        if cat=="instrument_platform": continue
        tpl = tpl.replace(tag, pick(cat,split))
    return tpl, mapping

PROMPT_STATIC = """For each row in the metadata line (the first line contains the column names), extract and format the following information concisely. For the 8 following categories, provide a single answer without redundancy. Each category **MUST** have one distinct and explicit answer, even if inferred. **Do not leave any category empty.** Do not repeat information already provided in previous categories. Remove redundant text.
cell_type - The type of cell in the sample. It can be among connective cells, fat cells, specialized integrated cells, migratory cells, kidney cells, muscle cells, bone cells, cartilage cells, stomach cells, sex cells, lung cells, pancreatic cells, liver cells (= hepatic cells), intestinal cells (= enterocytes), nerve cells, blood cells (= blood elements). First cite one of those huge categories, and if possible specify more specifically into each categorie (example: resident cells (fibroblasts, fibrocytes, tendinocytes, keratocytes), migratory cells (lymphocytes, histiocytes, melanocytes, natural killer cells), fat cells (mesenchymal cells, white adipocytes, brown adipocytes), specialized integrated cells (neuroepithelial cells, myoepithelial cells, goblet cells), kidney cells (podocytes, distal tubular cells, proximal tubular cells), muscle cells (smooth muscle cells, cardiac muscle cells, skeletal muscle cells), bone cells (osteoblasts, osteoclasts, osteocytes), cartilage cells (chondroblasts, hypertrophic chondrocytes), connective tissue cells (type A synoviocytes, type B synoviocytes), stomach cells (chief cells, parietal cells), sex cells (spermatozoa, spermatocytes, oocytes, sertoli cells, leydig cells), lung cells (type 1 pneumocytes, type 2 pneumocytes), pancreatic cells (alpha cells, beta cells, delta cells), liver cells (hepatocytes, kupffer cells), intestinal cells (enterocytes), nerve cells (neurons, neuroblasts, astrocytes, oligodendrocytes, schwann cells), blood cells (erythrocytes, thrombocytes, leukocytes, monocytes, eosinophils, basophils, neutrophils)). If not provided, deduce based on the tissue type and the rest of the context and state the inference. Use thee Cell Ontology terms terminology. WARNING: IF CELL LINE CAN'T BE DETERMINE AND IF IT IS A PRIMARY TISSUE THERE, mark this category as 'Primary tissue' too.
tissue_type – The tissue type from which the sample originates (e.g., epithelial tissue, connective tissue, muscle tissue, nervous tissue,). If not specified, deduce from context, from the organ or the type of cell it comes from.
cell_line – Specify the cell line which is a strict code, or state 'Primary tissue' if the sample is from a primary tissue and not a cell line.
organ - Provide me the organ(s) concerned by this study, in the UBERON GTEX terminology for the tissue type (e.g., lungs, liver, heart, etc ... name of the organ). If not specified, deduce from context, or search one related to the tissue.
disease – Return the Disease Ontology term corresponding to the disease associated with the sample in the format of the disease name. If the sample is explicitly described as 'normal' or 'healthy', or something similar do not infer any disease. In this case, do not search for disease-related information in the context. If the sample is not explicitly labeled as 'normal' or 'healthy' or 'no disease' etc, infer the disease from the context only if it is directly related to the sample (e.g., sample title, description, or metadata fields directly describing the sample). In case of cancer, something adjacent means that it's healthy. Non-disease conditions (e.g., pregnancy, aging, lifestyle factors) should be placed in the Donor information output column instead of the Disease Ontology Term field. DO NOT JUST STATE 'DISEASE' without inferring the type of disease. If nothing says there is a disease or any problem, state 'normal'.
host_phenotype - Based on the given context, determine if the phenotype classification is 'parental' (Refers to the original, untreated cell line or population, which has not been exposed to selective pressure (such as drug treatment). Typically represents the baseline phenotype.) or 'persistant' (Refers to cells or populations that have survived treatment and exhibit drug persistence or resistance, often through adaptive mechanisms rather than genetic mutations.). Choose between those two possibilities, use your knowledge if the answer is not clear.
library_selection - Based on the given context, determine the library selection fixed category by searching for specific keywords or synonyms that match one of the five strict categories: 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or 'other'. Assign 'polyA' if the context contains any of the following terms OR SIMIILAR MEANING THAT CAN BE INFERRED: 'PolyA', 'poly.A', 'oligo.dT', 'oligodT', 'truseq.mrna', 'truseq.stranded.mrna', 'truseq.standard.mrna', 'smarter.mRNA', 'stran ded.mRNA'. Assign 'inverse rRNA' if the context mentions depletion of ribosomal RNA with any of these terms OR SIMIILAR MEANING THAT CAN BE INFERRED: 'ribominus', 'ribodep', 'ribozero', 'ribo.zero', 'riboerase', 'ribogone', 'ribocop', 'ribo-dep', 'ribo-mi', 'ribo minus', 'depleted ribosom', 'remove ribosom', 'TruSeq.Stranded.Total', 'TruSeq.Total', 'SMARTer.Stranded.Total', 'SMARTer.Total'. Assign 'hybrid selection' if the context refers to hybrid capture or exon selection using any of these terms OR SIMIILAR MEANING THAT CAN BE INFERRED: 'Hybrid.Selection', 'Exon.capture', 'Exome.capture', 'RNA.Exome', 'geoMX'. Assign 'small RNA' if the context refers to small RNA isolation with keywords such as 'TruSeq.Small', 'size.fraction' OR SIMIILAR MEANING THAT CAN BE INFERRED. Assign 'other' if none of the above terms are found. Return only the exact category name: 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or 'other', with no additional text.
library_source - Based on the given context, determine the library source category by searching for specific keywords that match one of the two strict categories: 'single-cell' or 'bulk'. Assign 'single-cell' if the context contains any of the following terms: 'TRANSCRIPTOMIC SINGLE CELL', 'chromium', '10x', 'single.cell' OR SIMIILAR MEANING THAT CAN BE INFERRED. Assign 'bulk' if none of the above terms are found. Return only the exact category name: 'single-cell' or 'bulk', with no additional text.
treatment – If the sample received a treatment (e.g., drug, irradiation), specify it. Otherwise return 'no treatment'.
treatment_time – Specify the state of the current treatment if it can be found or inferred (on treatment, pre treatment, etc...). If no treatment, return 'no treatment'. Otherwise return 'nan'.
response – Indicate the response to treatment if available (e.g., resistance, sensitivity). Otherwise return 'nan'.
donor_information – Specify any relevant donor info such as age, sex, pregnancy, lifestyle, etc. Otherwise return 'nan'.

If any information is missing in the metadat can't be inferred for previous instruction, specify 'nan'. Don't double the answer. I want only one answer per category.
Strict output format (no additional text or special characters, no duplicated answers), ONLY print the answer. Do not elaborate.:
Output in this form, with the category and prediction SEPARATED BY DOUBLE POINTS LIKE: Cetegory: [single unique answer]

Respond with exactly one line. Do not elaborate. Only one word (or 3 max) is allowed after the "Category:".

(one line per category, no repetition):
Here is the output:
"""

def make_prompt(run,ctx):
    return f"Run accession: {run}\n            Metadata to analyze: {ctx}\n\n{PROMPT_STATIC}"


def build_output(mapping):
    order = [
        "cell_type",
        "tissue_type",
        "cell_line",
        "organ",
        "disease",
        "host_phenotype",
        "library_selection",
        "library_source",
        "treatment",
        "treatment_time",
        "response",
        "donor_information",
    ]
    return "\n".join(f"{c}: {mapping[c]}" for c in order)

def inject_categories(ctx: str, mapping: dict, split: str) -> (str, dict):
    parts = re.split(r',\s*', ctx.strip())
    missing = [c for c in VALUES_TRAIN if c not in mapping]
    random.shuffle(missing)
    for cat in missing:
        val = pick(cat, split)
        idx = random.randint(1, len(parts))
        parts.insert(idx, val)
        mapping[cat] = val
    return ", ".join(parts), mapping

def generate(split, templates, n, start):
    out = []
    idx = 0
    while len(out) < n:
        tpl, _ = templates[idx % len(templates)]
        raw_ctx, mapping = fill_template(tpl, split)
        full_ctx, full_map = inject_categories(raw_ctx, mapping, split)
        run = f"SRR{start + len(out):08d}"
        prompt = make_prompt(run, full_ctx)
        output = build_output(full_map)
        out.append({"prompt": prompt, "output": output})
        idx += 1
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--template_file",default="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/metadata_templates_1000_final.json")
    p.add_argument("--train_csv",default="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_metappuccino/FINE_TUNING/train_synth.csv")
    p.add_argument("--val_csv",default="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_metappuccino/FINE_TUNING/val_synth.csv")
    p.add_argument("--n_train",type=int,default=2000)
    p.add_argument("--n_val",type=int,default=500)
    args=p.parse_args()
    templates=json.load(open(args.template_file,encoding="utf-8"))
    train=generate("train",templates,args.n_train,1)
    val=generate("val",templates,args.n_val,args.n_train+1)
    with open(args.train_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["prompt","output"]);w.writeheader();w.writerows(train)
    with open(args.val_csv,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["prompt","output"]);w.writeheader();w.writerows(val)

if __name__=="__main__":
    main()
