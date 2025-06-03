##########################################################################################
# IMPORT
import pandas as pd
import numpy as np
import hashlib
from collections import Counter
from Levenshtein import distance as lev

file_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/finetune_data_all.csv"
df = pd.read_csv(file_path)

# test set on ft
test_runs = {'SYN001551', 'DRR201700', 'ERR10371253', 'SYN001060', 'SYN002870', 'ERR003346', 'ERR002362', 'DRR258670', 'SYN004714', 'ERR10747279', 'ERR000239', 'ERR10746115', 'SYN004860', 'ERR001602', 'ERR000166', 'SYN002322', 'ERR001261', 'ERR11007856', 'ERR002371', 'ERR10746098', 'ERR004001', 'SYN002355', 'ERR002831', 'ERR002491', 'DRR328242', 'DRR379808', 'SYN002682', 'ERR10747126', 'ERR10823397', 'ERR10083996', 'ERR004141', 'DRR381164', 'ERR10854542', 'ERR10452981', 'ERR004467', 'SYN003954', 'SYN004080', 'SYN000992', 'DRR527855', 'ERR10823161', 'ERR003973', 'SYN005713', 'ERR10232375', 'ERR003040', 'ERR10228765', 'SYN003069', 'ERR10746741', 'SYN002753', 'ERR10493857', 'ERR001580', 'SYN001488', 'ERR10228889', 'ERR001174', 'ERR10096061', 'ERR002067', 'DRR428615', 'SYN000353', 'SYN003648', 'ERR10746321', 'SYN001525', 'SYN000800', 'SYN000225', 'ERR002984', 'ERR11475015', 'SYN005931', 'ERR10317395', 'ERR10746575', 'DRR258720', 'SYN005674', 'ERR11834709', 'SYN002008', 'ERR10501139', 'ERR10639436', 'ERR10752871', 'ERR004018', 'SYN004811', 'ERR10823251', 'ERR003206', 'ERR10992938', 'ERR10501295', 'DRR308062', 'ERR11669611', 'ERR003536', 'SYN002536', 'SYN000140', 'SYN005247', 'SYN002633', 'SYN002495', 'SYN004979', 'SYN004806', 'SYN003194', 'SYN002514', 'ERR000604', 'SYN002965', 'ERR001563', 'ERR001162', 'SYN001605', 'SYN004194', 'ERR002974', 'SYN001878', 'SYN005294', 'SYN002585', 'SYN002003', 'SYN000378', 'SYN000295', 'SYN001069', 'ERR003712', 'ERR000758', 'SYN003437', 'SYN004302', 'SYN002768', 'SYN004805', 'SYN003696', 'SYN001684', 'ERR000810', 'ERR002515', 'ERR004548', 'SYN004827', 'SYN001318', 'SYN000441', 'ERR001777', 'SYN002537', 'ERR001679', 'SYN002704', 'SYN000207', 'SYN004089', 'ERR003785', 'SYN005303', 'SYN004476', 'ERR002856', 'ERR004594', 'ERR001592', 'SYN002625', 'SYN004058', 'SYN000829', 'ERR003208', 'SYN002872', 'ERR001642', 'SYN005799', 'SYN003754', 'ERR000234', 'SYN005053', 'ERR004763', 'ERR003078', 'SYN002190', 'SYN004152', 'SYN000261', 'SYN004697', 'SYN000272', 'SYN003643', 'SYN001253'}
val_runs = {"DRR056755","ERR000014","ERR000031","ERR000085","ERR000202","ERR000234","ERR000259","ERR000275","ERR000412","ERR000604","ERR000656","ERR000758","ERR000800","ERR000810","ERR000832","ERR001045","ERR001056","ERR001069","ERR001082","ERR001162","ERR001252","ERR001270","ERR001286","ERR001325","ERR001372","ERR001438","ERR001526","ERR001535","ERR001563","ERR001587","ERR001592","ERR001617","ERR001642","ERR001657","ERR001679","ERR001733","ERR001738","ERR001777","ERR001810","ERR001852","ERR002049","ERR002078","ERR002109","ERR002153","ERR002156","ERR002192","ERR002312","ERR002359","ERR002454","ERR002470","ERR002476","ERR002515","ERR002538","ERR002627","ERR002710","ERR002733","ERR002767","ERR002816","ERR002842","ERR002856","ERR002868","ERR002974","ERR002987","ERR002988","ERR003014","ERR003036","ERR003059","ERR003078","ERR003092","ERR003144","ERR003176","ERR003177","ERR003208","ERR003261","ERR003324","ERR003360","ERR003406","ERR003436","ERR003450","ERR003519","ERR003536","ERR003558","ERR003613","ERR003665","ERR003712","ERR003717","ERR003785","ERR003794","ERR003809","ERR003884","ERR003936","ERR003981","ERR004039","ERR004129","ERR004193","ERR004199","ERR004231","ERR004277","ERR004305","ERR004365","ERR004396","ERR004422","ERR004470","ERR004531","ERR004548","ERR004550","ERR004594","ERR004698","ERR004750","ERR004763","ERR004773","ERR004779","ERR004857","ERR004869","ERR004894","ERR004959","SYN000037","SYN000089","SYN000140","SYN000165","SYN000172","SYN000207","SYN000231","SYN000256","SYN000259","SYN000261","SYN000272","SYN000285","SYN000295","SYN000312","SYN000321","SYN000378","SYN000441","SYN000463","SYN000482","SYN000483","SYN000523","SYN000538","SYN000572","SYN000583","SYN000594","SYN000602","SYN000622","SYN000653","SYN000677","SYN000686","SYN000709","SYN000712","SYN000786","SYN000810","SYN000829","SYN000835","SYN000839","SYN000886","SYN000888","SYN000941","SYN000969","SYN000997","SYN001010","SYN001038","SYN001045","SYN001047","SYN001069","SYN001075","SYN001078","SYN001086","SYN001100","SYN001129","SYN001156","SYN001230","SYN001253","SYN001256","SYN001318","SYN001320","SYN001334","SYN001363","SYN001369","SYN001399","SYN001421","SYN001501","SYN001592","SYN001605","SYN001623","SYN001661","SYN001672","SYN001684","SYN001729","SYN001745","SYN001782","SYN001869","SYN001878","SYN001903","SYN001929","SYN001965","SYN001982","SYN001983","SYN002003","SYN002004","SYN002034","SYN002071","SYN002181","SYN002190","SYN002216","SYN002234","SYN002238","SYN002248","SYN002276","SYN002279","SYN002289","SYN002293","SYN002295","SYN002297","SYN002313","SYN002339","SYN002351","SYN002375","SYN002393","SYN002409","SYN002455","SYN002458","SYN002470","SYN002473","SYN002485","SYN002495","SYN002512","SYN002514","SYN002527","SYN002536","SYN002537","SYN002585","SYN002606","SYN002608","SYN002625","SYN002627","SYN002633","SYN002638","SYN002660","SYN002672","SYN002704","SYN002712","SYN002717","SYN002740","SYN002768","SYN002833","SYN002836","SYN002862","SYN002872","SYN002924","SYN002965","SYN003045","SYN003065","SYN003093","SYN003101","SYN003123","SYN003152","SYN003194","SYN003235","SYN003342","SYN003437","SYN003504","SYN003590","SYN003643","SYN003645","SYN003688","SYN003696","SYN003718","SYN003731","SYN003754","SYN003766","SYN003767","SYN003773","SYN003781","SYN003793","SYN003799","SYN003803","SYN003887","SYN003942","SYN004058","SYN004089","SYN004144","SYN004152","SYN004194","SYN004231","SYN004235","SYN004257","SYN004283","SYN004302","SYN004308","SYN004325","SYN004404","SYN004405","SYN004417","SYN004476","SYN004524","SYN004528","SYN004565","SYN004573","SYN004596","SYN004632","SYN004636","SYN004697","SYN004718","SYN004804","SYN004805","SYN004806","SYN004827","SYN004867","SYN004879","SYN004889","SYN004928","SYN004956","SYN004979","SYN005053","SYN005139","SYN005205","SYN005247","SYN005263","SYN005267","SYN005269","SYN005276","SYN005287","SYN005294","SYN005303","SYN005313","SYN005352","SYN005354","SYN005356","SYN005360","SYN005364","SYN005383","SYN005402","SYN005406","SYN005418","SYN005449","SYN005488","SYN005672","SYN005684","SYN005690","SYN005799","SYN005855","SYN005863","SYN005866","SYN005889","SYN005900","SYN005948","SYN005965","SYN005975","SYN005976"}

##########################################################################################
# GENERAL SIZE

def get_run(prompt):
    if isinstance(prompt, str) and "Run accession:" in prompt:
        return prompt.split("Run accession:")[1].split("\n")[0].strip()
    return None

df['run']   = df['prompt'].map(get_run)
df_test     = df[df['run'].isin(test_runs)].copy()
df_val      = df[df['run'].isin(val_runs)].copy()
df_train    = df[~df['run'].isin(test_runs | val_runs)].copy()

print(f"Train size: {len(df_train)} | Val size: {len(df_val)} | Test size: {len(df_test)}")

##########################################################################################
# EXTRACTION METADATA IN PROMPT

def get_meta(prompt):
    if isinstance(prompt, str) and "Metadata to analyze:" in prompt:
        return prompt.split("Metadata to analyze:")[1] \
                     .split("For each row in the metadata")[0] \
                     .strip()
    return ""
df_train['meta'] = df_train['prompt'].map(get_meta)
df_val['meta']   = df_val['prompt'].map(get_meta)
df_test['meta']  = df_test['prompt'].map(get_meta)

##########################################################################################
# DUPLICATION RATIO INTRA-TRAIN

# hash each bloc
df_train['meta_hash'] = df_train['meta'].map(lambda x: hashlib.md5(x.encode()).hexdigest())
counts = Counter(df_train['meta_hash'])
n_dups = sum(1 for h,c in counts.items() if c>1)
dup_ratio = n_dups / len(df_train)
print(f"▶ Duplication ratio in TRAIN: {dup_ratio:.2%} ({n_dups}/{len(df_train)})")

##########################################################################################
# LEAKAGE PROMPT/OUTPUT (entre sets)

def check_leakage(df1, df2, name1="Set1", name2="Set2"):
    prompts1 = set(df1['prompt'])
    prompts2 = set(df2['prompt'])
    outputs1 = set(df1['output'])
    outputs2 = set(df2['output'])
    prompt_overlap = prompts1 & prompts2
    output_overlap = outputs1 & outputs2
    print(f"\n▶ Data Leakage {name1} ↔ {name2} :")
    print(f"  - Prompt overlap: {len(prompt_overlap)} / {min(len(prompts1), len(prompts2))}")
    print(f"  - Output overlap: {len(output_overlap)} / {min(len(outputs1), len(outputs2))}")

check_leakage(df_train, df_val, "TRAIN", "VAL")
check_leakage(df_train, df_test, "TRAIN", "TEST")

##########################################################################################
# ΔL1 DISTANCES FOR EACH CATEGORIES: LABELS BALANCE BETWEEN SETS

# categories
categories = [
    'organ','disease','host_phenotype','library_selection','library_source',
    'treatment','treatment_time','response','donor_information',
    'cell_line','cell_type','tissue_type'
]

def extract_field(output, cat):
    if not isinstance(output, str): return np.nan
    for line in output.split("\n"):
        key, *rest = line.split(":",1)
        if key.strip().lower() == cat:
            return rest[0].strip().lower()
    return np.nan

for cat in categories:
    df_train[cat] = df_train['output'].apply(lambda out: extract_field(out, cat))
    df_val[cat]   = df_val['output'].apply(lambda out: extract_field(out, cat))
    df_test[cat]  = df_test['output'].apply(lambda out: extract_field(out, cat))

print("\n-----------------------------------------")
print("\n▶ Distribution shift train vs val:")
for cat in categories:
    train_dist = df_train[cat].value_counts(normalize=True)
    val_dist   = df_val[cat].value_counts(normalize=True)
    keys = set(train_dist.index) | set(val_dist.index)
    l1 = sum(abs(train_dist.get(k,0) - val_dist.get(k,0)) for k in keys)
    print(f"  {cat:15s}: ΔL1 = {l1:.3f}")

print("\n▶ Distribution shift train vs test:")
for cat in categories:
    train_dist = df_train[cat].value_counts(normalize=True)
    test_dist  = df_test[cat].value_counts(normalize=True)
    keys = set(train_dist.index) | set(test_dist.index)
    l1 = sum(abs(train_dist.get(k,0) - test_dist.get(k,0)) for k in keys)
    print(f"  {cat:15s}: ΔL1 = {l1:.3f}")

##########################################################################################
# SIGNAL PRESENCE RATIO

print("\n-----------------------------------------")
print("\n▶ Signal Presence Ratio (label in metadata bloc) - TRAIN:")
presence = {}
for cat in categories:
    ok = 0
    total = 0
    for meta, lbl in zip(df_train['meta'], df_train[cat]):
        if pd.isna(lbl):
            continue
        total += 1
        if lbl.lower() in meta.lower():
            ok += 1
    presence[cat] = ok / total if total>0 else np.nan
    print(f"  {cat:15s}: {presence[cat]*100:5.1f}% ({ok}/{total})")

print("\n▶ Signal Presence Ratio (label in metadata bloc) - VAL:")
for cat in categories:
    ok = 0; total = 0
    for meta, lbl in zip(df_val['meta'], df_val[cat]):
        if pd.isna(lbl): continue
        total += 1
        if lbl.lower() in meta.lower():
            ok += 1
    ratio = ok/total if total>0 else np.nan
    print(f"  {cat:15s}: {ratio*100:5.1f}% ({ok}/{total})")

print("\n▶ Signal Presence Ratio (label in metadata bloc) - TEST:")
for cat in categories:
    ok = 0; total = 0
    for meta, lbl in zip(df_test['meta'], df_test[cat]):
        if pd.isna(lbl): continue
        total += 1
        if lbl.lower() in meta.lower():
            ok += 1
    ratio = ok/total if total>0 else np.nan
    print(f"  {cat:15s}: {ratio*100:5.1f}% ({ok}/{total})")

##########################################################################################
# COVERAGE

def coverage(train_series, test_series, k=5):
    freq = train_series.value_counts()
    seen_labels = set(freq[freq >= k].index)
    test_labels = set(test_series.dropna().unique())
    return len(seen_labels & test_labels) / len(test_labels) if len(test_labels)>0 else np.nan

print("\n-----------------------------------------")
for cat in categories:
    cov_val = coverage(df_train[cat], df_val[cat], k=5)
    print(f"Coverage({cat}, val, k=5) = {cov_val:.2%}")
    cov_test = coverage(df_train[cat], df_test[cat], k=5)
    print(f"Coverage({cat}, test, k=5) = {cov_test:.2%}")

##########################################################################################
# LEVENSHTEIN DISTANCE

print("\n-----------------------------------------")
print("\n▶ Extraction Difficulty (Levenshtein distance) per category (TRAIN):")
for cat in categories:
    dists = []
    for meta, lbl in zip(df_train['meta'], df_train[cat]):
        if pd.isna(lbl) or not isinstance(meta, str):
            continue
        words = meta.split()
        dist = min(lev(lbl, w) for w in words)
        dists.append(dist)
    if not dists:
        print(f"  {cat:15s}: no data")
        continue
    arr = np.array(dists)
    print(f"  {cat:15s}: avg={arr.mean():.2f}, med={np.median(arr):.2f}, "
          f"0-dist%={(arr==0).mean()*100:.1f}%")
