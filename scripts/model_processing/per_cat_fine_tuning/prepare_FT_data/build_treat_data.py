from __future__ import annotations
import random
import pandas as pd
from collections import defaultdict

STATIC_PROMPT = "\n".join([
    "Run accession: {run_accession}",
    "Summary: {context}",
    "",
    "Categories and definitions:",
    "- treatment: treatment applied treatment applied = may be molecules or drug treatments or surgical operations, or something implanted, or events altering the state of the organism, etc. if not explicitly stated, infer from the context. DON'T STATE the disease, get info just from treatment",
    "",
    "For each category below:",
    "- Extract information from the summary if possible",
    "- If one value is impossible to extract, even by deducing it, return \"unknown\"",
    "",
    "BE CAREFUL: Sometimes the information concerns several samples from the same study. It is important to distinguish between them and semantically extract what applies to the current run, so everything must be consistent.",
    "FOR EACH CATEGORY SEVERAL ANSWERS CAN BE POSSIBLE, CITE THEM ALL WITH A ',' OR ';' SEPARATOR",
    "",
    "Respond strictly in a few words with valid JSON (double quotes around keys and values), no extra keys. ONLY THE CATEGORIES CITED UNDER 'Categories and definitions'",
    "",
    "Here is the output:",
])

CATALOG: dict[str, list[str]] = {
    "drug": [
        "cisplatin","doxorubicin","paclitaxel","carboplatin","gemcitabine","docetaxel",
        "cyclophosphamide","etoposide","methotrexate","5-fluorouracil","imatinib","dasatinib",
        "nilotinib","erlotinib","gefitinib","osimertinib","crizotinib","sorafenib","sunitinib",
        "bevacizumab","rituximab","trastuzumab","pembrolizumab","nivolumab","ipilimumab",
        "olaparib","tamoxifen","anastrozole","letrozole","fulvestrant","dexamethasone",
        "prednisone","remdesivir","oseltamivir","acyclovir","fluconazole","vancomycin",
        "gentamicin","amoxicillin","azithromycin","doxycycline","metformin","insulin glargine",
        "lisinopril","amlodipine","atorvastatin","warfarin","heparin","enoxaparin","apixaban",
        "clopidogrel","aspirin","ibuprofen","naproxen","morphine","propofol","levodopa",
        "sertraline","fluoxetine","valproate"
    ],
    "surgery": [
        "appendectomy","cholecystectomy","colectomy","hemicolectomy","mastectomy","lumpectomy",
        "nephrectomy","lobectomy","thyroidectomy","prostatectomy","hysterectomy","oophorectomy",
        "salpingectomy","craniotomy","laminectomy","coronary artery bypass grafting",
        "carotid endarterectomy","angioplasty","thrombectomy","embolectomy","esophagectomy",
        "gastrectomy","pancreatectomy","hepatectomy","splenectomy","hernia repair","cesarean section",
        "tonsillectomy","adenoidectomy","bariatric surgery","gastric bypass","sleeve gastrectomy",
        "cataract extraction","vitrectomy","trabeculectomy","laryngectomy","pneumonectomy",
        "tracheostomy","bronchoscopy with lavage","polypectomy","colonoscopy with biopsy",
        "lithotripsy","ureteroscopy","endometrial ablation","dilation and curettage",
        "joint arthroscopy","hip arthroplasty","knee arthroplasty","spinal fusion",
        "liver transplant","kidney transplant"
    ],
    "implant": [
        "pacemaker implantation","implantable cardioverter-defibrillator insertion",
        "coronary stent placement","ventricular shunt placement","cochlear implant",
        "deep brain stimulation","spinal cord stimulator implantation","insulin pump therapy",
        "intrauterine device placement","artificial hip implant","knee prosthesis implantation",
        "central venous catheter placement","peripherally inserted central catheter placement",
        "port-a-cath insertion","gastric band placement"
    ],
    "radiation": [
        "external beam radiotherapy","brachytherapy","stereotactic radiosurgery",
        "proton therapy","total body irradiation","radioiodine ablation"
    ],
    "gene_cell": [
        "CAR-T cell therapy","CRISPR gene editing","AAV gene therapy",
        "hematopoietic stem cell transplant","autologous stem cell transplant",
        "allogeneic stem cell transplant"
    ],
    "immunotherapy_vaccine": [
        "BCG instillation","desensitization immunotherapy","influenza vaccination",
        "COVID-19 vaccination","tetanus booster","HPV vaccination","hepatitis B vaccination"
    ],
    "behavioral": [
        "ketogenic diet","high-fat diet","caloric restriction","intermittent fasting",
        "sleep deprivation","cold exposure","heat exposure","endurance exercise training",
        "resistance training","immobilization","smoking cessation program",
        "mindfulness-based stress reduction","cognitive behavioral therapy"
    ],
    "supportive": [
        "hemodialysis","peritoneal dialysis","plasmapheresis","plasma exchange","blood transfusion",
        "platelet transfusion","intravenous immunoglobulin therapy","hyperbaric oxygen therapy",
        "mechanical ventilation","extracorporeal membrane oxygenation",
        "continuous positive airway pressure therapy","catheter ablation",
        "endoscopic sclerotherapy","transarterial chemoembolization","transcatheter aortic valve implantation",
        "endoscopic submucosal dissection","radiofrequency ablation","microwave ablation",
        "photodynamic therapy","laser trabeculoplasty"
    ],
}

DISEASES_BY_FAMILY = {
    "drug": [
        "lung cancer","breast cancer","colorectal cancer","leukemia","lymphoma","prostate cancer",
        "ovarian cancer","pancreatic adenocarcinoma","glioblastoma","melanoma","gastric cancer",
        "hepatocellular carcinoma","head and neck cancer","bladder cancer","cervical cancer",
        "influenza","COVID-19","pneumonia","bacterial sepsis","herpes zoster",
        "type 2 diabetes","hypertension","hyperlipidemia","Parkinson's disease","depression",
        "epilepsy","chronic pain"
    ],
    "surgery": [
        "appendicitis","cholelithiasis","colon cancer","breast cancer","renal tumor","lung cancer",
        "thyroid nodule","prostate cancer","uterine fibroids","ovarian cyst","cataract","glaucoma",
        "laryngeal cancer","lung abscess","airway obstruction","colonic polyps","kidney stones",
        "meniscal tear","hip fracture","knee osteoarthritis","spinal instability",
        "end-stage liver disease","end-stage renal disease"
    ],
    "implant": [
        "bradycardia","ventricular tachycardia","coronary artery disease","hydrocephalus",
        "sensorineural hearing loss","Parkinson's disease","chronic pain","type 1 diabetes",
        "contraception","hip osteoarthritis","knee osteoarthritis","venous access need",
        "obesity"
    ],
    "radiation": [
        "prostate cancer","cervical cancer","brain metastasis","head and neck cancer",
        "thyroid cancer","hematologic malignancy"
    ],
    "gene_cell": [
        "B-cell acute lymphoblastic leukemia","diffuse large B-cell lymphoma","retinal dystrophy",
        "spinal muscular atrophy","aplastic anemia"
    ],
    "immunotherapy_vaccine": [
        "non–muscle-invasive bladder cancer","allergic rhinitis","seasonal influenza",
        "COVID-19","tetanus prophylaxis","HPV prevention","hepatitis B prophylaxis"
    ],
    "behavioral": [
        "drug-resistant epilepsy","obesity","metabolic syndrome","insulin resistance",
        "sleep disorder","anxiety","nicotine dependence","deconditioning"
    ],
    "supportive": [
        "end-stage renal disease","Guillain–Barré syndrome","autoimmune hemolytic anemia",
        "severe anemia","decompression sickness","respiratory failure","cardiogenic shock",
        "obstructive sleep apnea","atrial fibrillation","esophageal varices","hepatocellular carcinoma",
        "aortic stenosis","early gastric cancer","liver tumor","glaucoma"
    ],
}

DISEASE_TO_ORGAN = {
    "lung cancer":"lung","breast cancer":"breast","colorectal cancer":"colon","leukemia":"blood",
    "lymphoma":"lymph nodes","prostate cancer":"prostate","ovarian cancer":"ovaries",
    "pancreatic adenocarcinoma":"pancreas","glioblastoma":"brain","melanoma":"skin",
    "gastric cancer":"stomach","hepatocellular carcinoma":"liver","head and neck cancer":"oropharynx",
    "bladder cancer":"bladder","cervical cancer":"cervix",
    "influenza":"lung","COVID-19":"lung","pneumonia":"lung","bacterial sepsis":"blood",
    "herpes zoster":"skin","type 2 diabetes":"pancreas","hypertension":"vascular system",
    "hyperlipidemia":"liver","Parkinson's disease":"basal ganglia","depression":"brain",
    "epilepsy":"brain","chronic pain":"nervous system",
    "appendicitis":"appendix","cholelithiasis":"gallbladder","colon cancer":"colon",
    "renal tumor":"kidney","thyroid nodule":"thyroid","uterine fibroids":"uterus",
    "ovarian cyst":"ovaries","cataract":"eye lens","glaucoma":"eye",
    "laryngeal cancer":"larynx","lung abscess":"lung","airway obstruction":"trachea",
    "colonic polyps":"colon","kidney stones":"kidney","meniscal tear":"knee",
    "hip fracture":"hip","knee osteoarthritis":"knee","spinal instability":"spine",
    "end-stage liver disease":"liver","end-stage renal disease":"kidney",
    "bradycardia":"heart","ventricular tachycardia":"heart","coronary artery disease":"coronary arteries",
    "hydrocephalus":"brain ventricles","sensorineural hearing loss":"cochlea","type 1 diabetes":"pancreas",
    "contraception":"uterus","hip osteoarthritis":"hip","venous access need":"subclavian vein",
    "obesity":"adipose tissue","brain metastasis":"brain","hematologic malignancy":"bone marrow",
    "non–muscle-invasive bladder cancer":"bladder","allergic rhinitis":"nasal mucosa",
    "seasonal influenza":"lung","tetanus prophylaxis":"systemic","HPV prevention":"cervix",
    "hepatitis B prophylaxis":"liver",
    "drug-resistant epilepsy":"brain","metabolic syndrome":"liver","insulin resistance":"skeletal muscle",
    "sleep disorder":"brain","anxiety":"brain","nicotine dependence":"brain reward system",
    "deconditioning":"skeletal muscle","Guillain–Barré syndrome":"peripheral nerves",
    "autoimmune hemolytic anemia":"blood","severe anemia":"blood","decompression sickness":"systemic",
    "respiratory failure":"lung","cardiogenic shock":"heart","obstructive sleep apnea":"upper airway",
    "atrial fibrillation":"atria","esophageal varices":"esophagus","aortic stenosis":"aortic valve",
    "early gastric cancer":"stomach","liver tumor":"liver"
}

NOISE_TECH = [
    "sequencer lane rebalanced for duplex bias","insert size ~320bp","sample barcoded with UMIs",
    "QC passed at RIN 8.1","library molarity 7.5 nM","coverage uniformity within 1.3x",
    "FASTQ trimmed (Phred≥30)","adapter clipping enabled","lane bleedthrough negligible",
    "5' bias not detected","batch randomization across plates","ERCC spike-ins present",
    "index-hopping mitigated","PCR duplicates removed","alignment on GRCh38",
    "multi-mapping reads <2%","FRiP 0.21","median insert 286","sequencing kit v3",
    "freeze-thaw cycles=1","lane clustering density nominal","negative control clean",
    "RNase-free handling confirmed","unique fragments retained","flowcell ID anonymized",
    "library preparation batch balanced","lane occupancy nominal","index cross-talk minimized",
    "blind duplicate included","positive control within range","spike-in normalization applied",
    "lanes pooled with staggered barcodes","kmer contamination check clean","index balance within ±5%"
]

NOISE_CLIN = [
    "vital signs stable at discharge","no perioperative complications recorded",
    "written informed consent obtained","blinded outcome assessment",
    "adverse events monitored per protocol","randomization block size=4",
    "follow-up at day 28 completed","biopsy adequacy confirmed by pathology",
    "imaging schedules adhered to","device integrity verified",
    "dose preparation double-checked","sterile technique maintained",
    "eligibility per inclusion criteria met","CRF completed without queries",
    "concomitant conditions documented","baseline labs within normal limits",
    "electrolytes repleted as needed","ECG unchanged from baseline",
    "oxygen saturation maintained >95%","pain scores recorded q4h",
    "premedication administered per protocol","infusion reactions not observed"
]

NOISE_ADMIN = [
    "shipment manifest reconciled","sample custody chain verified","data lock executed",
    "monitoring visit concluded","query resolution pending","site initiation completed",
    "inventory log updated","temperature excursion not observed","barcodes scanned successfully",
    "cold chain preserved","audit trail intact","versioned SOP followed",
    "delegation log signed","electronic consent archived","training records up to date"
]

NOISE_TIMING = [
    "day 1 cycle 1","day 8 cycle 2","week 6 assessment","month 3 follow-up","hour 24 sedation check",
    "timepoint T0 baseline","timepoint T1 post-procedure","pre-dose sampling complete","post-dose window met",
    "timepoint T2 late follow-up","cycle 3 day 15 window respected"
]

NOISE_MISC = [
    "subject preferred arm documented","nurse shift handover recorded","room ventilation within specs",
    "sterility indicators passed","calibration certificate valid","ultrasound guidance utilized",
    "contrast agent not required","allergy list updated","family notified of progress",
    "telemetry artifacts noted","meal timing adjusted","hydration encouraged",
    "sleep hygiene advice provided","discharge planning initiated",
    "identifier stickers aligned","typographical errors corrected","aux notes appended"
]

GLUE = [" | ", " /// ", " ⟶ ", " :: ", " ~ ", " — ", " <> ", " || ", " ↦ "]

def stratified_partition(labels_by_family: dict[str, list[str]], ratios=(0.5, 0.25, 0.25), seed=7):
    rng = random.Random(seed)
    train, val, test = [], [], []
    for fam, labels in labels_by_family.items():
        lab = labels[:]
        rng.shuffle(lab)
        n = len(lab)
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        n_test = max(0, n - n_train - n_val)
        train += [(x, fam) for x in lab[:n_train]]
        val   += [(x, fam) for x in lab[n_train:n_train+n_val]]
        test  += [(x, fam) for x in lab[n_train+n_val:]]
    return train, val, test

def pick_disease_and_organ(family: str, rng: random.Random):
    disease = rng.choice(DISEASES_BY_FAMILY[family])
    organ = DISEASE_TO_ORGAN.get(disease, rng.choice(list(DISEASE_TO_ORGAN.values())))
    return disease, organ

def build_noise(rng: random.Random, intensity: tuple[int,int,int,int,int]):
    a, b, c, d, e = intensity
    parts = []
    parts += rng.sample(NOISE_TECH, k=rng.randint(a, a+2))
    parts += rng.sample(NOISE_CLIN, k=rng.randint(b, b+2))
    parts += rng.sample(NOISE_ADMIN, k=rng.randint(c, c+1))
    parts += rng.sample(NOISE_TIMING, k=rng.randint(d, d+1))
    parts += rng.sample(NOISE_MISC, k=rng.randint(e, e+1))
    rng.shuffle(parts)
    return parts

def mutate_line(s: str, rng: random.Random):
    if rng.random() < 0.18:
        s = s.upper()
    if rng.random() < 0.18:
        s = "[" + s + "]"
    if rng.random() < 0.18:
        s = s + " ..."
    return s

def style_bullets(base: str, noise: list[str], rng: random.Random):
    lines = ["- " + mutate_line(base, rng)]
    for p in noise[:rng.randint(4,8)]:
        lines.append("- " + mutate_line(p, rng))
    return ",".join(lines)

def style_numbered(base: str, noise: list[str], rng: random.Random):
    idx = 1
    lines = [str(idx) + ") " + mutate_line(base, rng)]
    for p in noise[:rng.randint(4,8)]:
        idx += 1
        lines.append(str(idx) + ") " + mutate_line(p, rng))
    return ",".join(lines)

def style_kv(base: str, noise: list[str], rng: random.Random):
    n1 = noise[0] if noise else ""
    n2 = noise[1] if len(noise) > 1 else ""
    n3 = noise[2] if len(noise) > 2 else ""
    return ",".join([
        "Action: " + base,
        "Check: " + n1,
        "Note: " + n2,
        "Meta: " + n3
    ])

def style_jsonish(base: str, noise: list[str], rng: random.Random):
    n1 = noise[0] if noise else ""
    n2 = noise[1] if len(noise) > 1 else ""
    n3 = noise[2] if len(noise) > 2 else ""
    return "{action: \"" + base + "\", check: \"" + n1 + "\", note: \"" + n2 + "\", extra: \"" + n3 + "\"}"

def style_yamlish(base: str, noise: list[str], rng: random.Random):
    block = ["action: " + base, "noise:"]
    for p in noise[:rng.randint(4,8)]:
        block.append("- " + p)
    return ",".join(block)

def style_logs(base: str, noise: list[str], rng: random.Random):
    def ts(): return "2024-" + str(rng.randint(1,12)).zfill(2) + "-" + str(rng.randint(1,28)).zfill(2) + "T" + str(rng.randint(0,23)).zfill(2) + ":" + str(rng.randint(0,59)).zfill(2) + "Z"
    levels = ["INFO","WARN","DEBUG"]
    lines = [ts() + " " + rng.choice(levels) + " " + base]
    for p in noise[:rng.randint(5,10)]:
        lines.append(ts() + " " + rng.choice(levels) + " " + mutate_line(p, rng))
    return ",".join(lines)

def style_sections(base: str, noise: list[str], rng: random.Random):
    head = "## BLOCK-" + str(rng.randint(10,99))
    lines = [head, base, "--"]
    lines += noise[:rng.randint(5,9)]
    return ",".join(lines)

def style_tsv(base: str, noise: list[str], rng: random.Random):
    rows = ["action\t" + base]
    for p in noise[:rng.randint(4,7)]:
        rows.append("noise\t" + p)
    return ",".join(rows)

def style_md_table(base: str, noise: list[str], rng: random.Random):
    rows = ["| field | value |", "|---|---|", "| action | " + base + " |"]
    for p in noise[:rng.randint(4,7)]:
        rows.append("| note | " + p + " |")
    return ",".join(rows)

def style_csv(base: str, noise: list[str], rng: random.Random):
    rows = ["field,value", "action," + base]
    for p in noise[:rng.randint(4,7)]:
        rows.append("note," + p.replace(",", ";"))
    return ",".join(rows)

def style_pipe(base: str, noise: list[str], rng: random.Random):
    parts = [base] + noise[:rng.randint(5,9)]
    return (" | ").join(parts)

def style_ini(base: str, noise: list[str], rng: random.Random):
    rows = ["[block]", "action=" + base]
    for i, p in enumerate(noise[:rng.randint(4,7)], start=1):
        rows.append("note" + str(i) + "=" + p)
    return ",".join(rows)

def style_sql(base: str, noise: list[str], rng: random.Random):
    n1 = noise[0] if noise else ""
    n2 = noise[1] if len(noise) > 1 else ""
    return "INSERT INTO log(action,check,note) VALUES ('" + base.replace("'", "''") + "','" + n1.replace("'", "''") + "','" + n2.replace("'", "''") + "');"

def style_rfc822(base: str, noise: list[str], rng: random.Random):
    rows = ["Subject: ACTION " + str(rng.randint(1000,9999)), "X-Run: " + str(rng.randint(100000,999999)), "", base]
    rows += noise[:rng.randint(4,8)]
    return ",".join(rows)

def style_xmlish(base: str, noise: list[str], rng: random.Random):
    rows = ["<record>", "  <action>" + base + "</action>"]
    for p in noise[:rng.randint(5,9)]:
        rows.append("  <note>" + p + "</note>")
    rows.append("</record>")
    return ",".join(rows)

def style_latex(base: str, noise: list[str], rng: random.Random):
    rows = ["\\section*{Action}", base, "\\subsection*{Notes}"]
    for p in noise[:rng.randint(5,9)]:
        rows.append(p + " \\\\")
    return ",".join(rows)

def style_markdown_mix(base: str, noise: list[str], rng: random.Random):
    rows = ["* " + base, "> " + noise[0] if noise else "> note"]
    for p in noise[1:rng.randint(4,8)]:
        rows.append("`" + p + "`")
    return ",".join(rows)

def style_grid(base: str, noise: list[str], rng: random.Random):
    sep = "+" + "-"*30 + "+" + "-"*40 + "+"
    rows = [sep, "| FIELD                        | VALUE                                  |", sep, "| ACTION                       | " + base.ljust(38) + "|"]
    for p in noise[:rng.randint(5,9)]:
        rows.append("| NOTE                         | " + p[:38].ljust(38) + "|")
    rows.append(sep)
    return ",".join(rows)

def style_flow(base: str, noise: list[str], rng: random.Random):
    parts = [base] + noise[:rng.randint(5,9)]
    glue = rng.choice(GLUE)
    return glue.join(parts)

def style_multiheader(base: str, noise: list[str], rng: random.Random):
    rows = ["=== HEADER A ===", base, "=== HEADER B ==="]
    rows += noise[:rng.randint(5,9)]
    rows.append("=== END ===")
    return ",".join(rows)

def style_accordion(base: str, noise: list[str], rng: random.Random):
    rows = ["[[Action]]", base, "[[Notes]]"]
    rows += noise[:rng.randint(5,9)]
    return ",".join(rows)

STYLE_BUILDERS = {
    "bullets": style_bullets,
    "numbered": style_numbered,
    "kv": style_kv,
    "jsonish": style_jsonish,
    "yamlish": style_yamlish,
    "logs": style_logs,
    "sections": style_sections,
    "tsv": style_tsv,
    "md_table": style_md_table,
    "csv": style_csv,
    "pipe": style_pipe,
    "ini": style_ini,
    "sql": style_sql,
    "rfc822": style_rfc822,
    "xmlish": style_xmlish,
    "latex": style_latex,
    "markdown_mix": style_markdown_mix,
    "grid": style_grid,
    "flow": style_flow,
    "multiheader": style_multiheader,
    "accordion": style_accordion,
}

STYLE_SETS = {
    "train": ["bullets","numbered","kv","jsonish","yamlish","logs","sections","tsv","md_table"],
    "val":   ["csv","pipe","ini","sql","rfc822"],
    "test":  ["xmlish","latex","markdown_mix","grid","flow","multiheader","accordion"],
}

NOISE_INTENSITY = {
    "train": (5,4,2,1,2),
    "val":   (6,4,2,2,2),
    "test":  (7,5,2,2,2),
}

def enforce_single_treatment(text: str, treatment: str):
    count = text.count(treatment)
    if count > 1:
        first = text.find(treatment)
        after = first + len(treatment)
        tail = text[after:].replace(treatment, "the treatment")
        text = text[:after] + tail
    for fam_list in CATALOG.values():
        for other in fam_list:
            if other == treatment:
                continue
            if other in text:
                text = text.replace(other, "therapy")
    return text

def make_context(split: str, treatment: str, family: str, rng: random.Random):
    disease, organ = pick_disease_and_organ(family, rng)
    base_opts = [
        "{treatment} applied for {disease}; focus on {organ}; dosing per schedule; monitoring layered.",
        "Session flagged: {treatment} → context: {disease}; guard {organ}; protocol intact.",
        "Run note: {treatment} in setting {disease}; {organ} watched; traceability preserved.",
        "Applied: {treatment}; indication {disease}; {organ} metrics logged; cross-check complete."
    ]
    base = rng.choice(base_opts).format(treatment=treatment, disease=disease, organ=organ)
    noise = build_noise(rng, NOISE_INTENSITY[split])
    builder = STYLE_BUILDERS[rng.choice(STYLE_SETS[split])]
    ctx = builder(base, noise, rng)
    ctx = enforce_single_treatment(ctx, treatment)
    return ctx

def distribute_counts(n_labels: int, total: int, seed: int):
    rng = random.Random(seed)
    counts = [1] * n_labels
    remaining = total - n_labels
    while remaining > 0:
        i = rng.randrange(n_labels)
        counts[i] += 1
        remaining -= 1
    rng.shuffle(counts)
    return counts

def build_rows(split: str, label_pairs: list[tuple[str,str]], target_total: int, start_id: int, rng: random.Random):
    rows = []
    rid = start_id
    n_labels = len(label_pairs)
    counts = distribute_counts(n_labels, target_total, seed=int(rng.random()*1e9) or 1)
    for (treatment, family), k in zip(label_pairs, counts):
        for _ in range(k):
            rid += 1
            run_id = "SYNT" + str(rid).zfill(6)
            ctx = make_context(split, treatment, family, rng)
            prompt = STATIC_PROMPT.format(run_accession=run_id, context=ctx)
            rows.append({"prompt": prompt, "output": treatment})
    return rows, rid

if __name__ == "__main__":
    SEED = 42
    rng = random.Random(SEED)
    TARGET_TRAIN = 1000
    TARGET_VAL = 200
    TARGET_TEST = 700
    LABEL_RATIOS = (0.5, 0.25, 0.25)
    pairs = [(t, fam) for fam, lst in CATALOG.items() for t in lst]
    fam2labels = defaultdict(list)
    for t, fam in pairs:
        fam2labels[fam].append(t)
    train_labels, val_labels, test_labels = stratified_partition(fam2labels, ratios=LABEL_RATIOS, seed=SEED)
    assert set(t for t,_ in train_labels).isdisjoint(set(t for t,_ in val_labels))
    assert set(t for t,_ in train_labels).isdisjoint(set(t for t,_ in test_labels))
    assert set(t for t,_ in val_labels).isdisjoint(set(t for t,_ in test_labels))
    rows_train, cur = build_rows("train", train_labels, TARGET_TRAIN, 0, rng)
    rows_val,   cur = build_rows("val",   val_labels,   TARGET_VAL,   cur, rng)
    rows_test,  cur = build_rows("test",  test_labels,  TARGET_TEST,  cur, rng)
    df_train = pd.DataFrame(rows_train, columns=["prompt","output"])
    df_val   = pd.DataFrame(rows_val,   columns=["prompt","output"])
    df_test  = pd.DataFrame(rows_test,  columns=["prompt","output"])
    assert len(df_train) == TARGET_TRAIN
    assert len(df_val) == TARGET_VAL
    assert len(df_test) == TARGET_TEST
    df_train.to_csv("/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING15/finetune_data_train.csv", index=False)
    df_val.to_csv("/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING15/finetune_data_val.csv", index=False)
    df_test.to_csv("/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING15/finetune_data_test.csv", index=False)
    print("train:", len(df_train), "val:", len(df_val), "test:", len(df_test))
