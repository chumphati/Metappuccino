import pandas as pd
import random
import re

train_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/finetune_data_train_corrected.csv"
val_path   = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/finetune_data_val_corrected.csv"

df_train = pd.read_csv(train_path)
df_val = pd.read_csv(val_path)

def random_synonym(category, value):
    d = {
        ("library_source", "single-cell"): [
            "single-cell technology;", "individual cell assay;", "unicellular analysis;"
        ],
        ("library_source", "bulk"): [
            "bulk transcriptomics;", "population sample;", "non-single-cell approach;"
        ],
        ("response", "no response"): [
            "no observable therapeutic effect;", "unresponsive state;", "treatment ineffectiveness;"
        ],
        ("response", "no treatment"): [
            "untreated control;", "no intervention administered;", "baseline state;"
        ],
        ("treatment_time", "Pre-treatment"): [
            "before treatment;", "pre-exposure state;", "initial sampling;"
        ],
        ("treatment_time", "no treatment"): [
            "never treated;", "no therapy received;", "absence of intervention;"
        ],
        ("tissue_type", "connective"): [
            "connective matrix context;", "stromal environment;", "support tissue focus;"
        ],
        ("tissue_type", "nervous"): [
            "neuronal environment;", "nervous system sample;", "brain-derived tissue;"
        ],
    }
    return random.choice(d.get((category, value), [";"]))

def replace_and_add_context(df, category, old_val, new_val, context_col="prompt"):
    idxs = df[df["output"].str.contains(fr'{category}:\s*{old_val}')].index.tolist()
    for i in idxs:
        df.at[i, "output"] = re.sub(fr'{category}:\s*{old_val}', f'{category}: {new_val}', df.at[i, "output"])
        context_to_add = random_synonym(category, new_val)
        df.at[i, context_col] = re.sub(r'(Metadata to analyze:)', r'\1 ' + context_to_add, df.at[i, context_col], count=1, flags=re.IGNORECASE)
    return idxs

df_train["output"] = df_train["output"].apply(lambda x: re.sub(r'library_source:\s*nan', 'library_source: single-cell', x))
idxs_sc = df_train[df_train["output"].str.contains(r'library_source:\s*single-cell')].index.tolist()
for i in idxs_sc:
    context_to_add = random_synonym("library_source", "single-cell")
    df_train.at[i, "prompt"] = re.sub(r'(Metadata to analyze:)', r'\1 ' + context_to_add, df_train.at[i, "prompt"], count=1, flags=re.IGNORECASE)

df_val["output"] = df_val["output"].apply(lambda x: re.sub(r'library_source:\s*nan', 'library_source: bulk', x))
idxs_bulk = df_val[df_val["output"].str.contains(r'library_source:\s*bulk')].index.tolist()
for i in idxs_bulk:
    context_to_add = random_synonym("library_source", "bulk")
    df_val.at[i, "prompt"] = re.sub(r'(Metadata to analyze:)', r'\1 ' + context_to_add, df_val.at[i, "prompt"], count=1, flags=re.IGNORECASE)

df_train["output"] = df_train["output"].apply(lambda x: re.sub(r'Primary tissue tissue', 'Primary tissue', x))
df_val["output"] = df_val["output"].apply(lambda x: re.sub(r'Primary tissue tissue', 'Primary tissue', x))

idx_prog = df_val[df_val["output"].str.contains(r'response:\s*Progressive Disease')].sample(frac=0.2, random_state=42).index
for i in idx_prog:
    df_val.at[i, "output"] = re.sub(r'response:\s*Progressive Disease', 'response: no response', df_val.at[i, "output"])
    context_to_add = random_synonym("response", "no response")
    df_val.at[i, "prompt"] = re.sub(r'(Metadata to analyze:)', r'\1 ' + context_to_add, df_val.at[i, "prompt"], count=1, flags=re.IGNORECASE)

idx_stable = df_val[df_val["output"].str.contains(r'response:\s*Stable Disease')].sample(frac=0.3, random_state=42).index
for i in idx_stable:
    df_val.at[i, "output"] = re.sub(r'response:\s*Stable Disease', 'response: no treatment', df_val.at[i, "output"])
    context_to_add = random_synonym("response", "no treatment")
    df_val.at[i, "prompt"] = re.sub(r'(Metadata to analyze:)', r'\1 ' + context_to_add, df_val.at[i, "prompt"], count=1, flags=re.IGNORECASE)

tt_nan_idx = df_val[df_val["output"].str.contains(r'treatment_time:\s*nan')].index.tolist()
n_tt = len(tt_nan_idx)
pre_tt_idx = random.sample(tt_nan_idx, int(0.5 * n_tt))
remaining_tt = list(set(tt_nan_idx) - set(pre_tt_idx))
no_treat_tt_idx = random.sample(remaining_tt, int(0.2 * n_tt))
for i in pre_tt_idx:
    df_val.at[i, "output"] = re.sub(r'treatment_time:\s*nan', 'treatment_time: Pre-treatment', df_val.at[i, "output"])
    context_to_add = random_synonym("treatment_time", "Pre-treatment")
    df_val.at[i, "prompt"] = re.sub(r'(Metadata to analyze:)', r'\1 ' + context_to_add, df_val.at[i, "prompt"], count=1, flags=re.IGNORECASE)
for i in no_treat_tt_idx:
    df_val.at[i, "output"] = re.sub(r'treatment_time:\s*nan', 'treatment_time: no treatment', df_val.at[i, "output"])
    context_to_add = random_synonym("treatment_time", "no treatment")
    df_val.at[i, "prompt"] = re.sub(r'(Metadata to analyze:)', r'\1 ' + context_to_add, df_val.at[i, "prompt"], count=1, flags=re.IGNORECASE)

tt_nan_idx2 = df_val[df_val["output"].str.contains(r'tissue_type:\s*nan')].index.tolist()
n_tt2 = len(tt_nan_idx2)
conn_tt_idx = random.sample(tt_nan_idx2, int(0.9 * n_tt2))
remaining_tt2 = list(set(tt_nan_idx2) - set(conn_tt_idx))
nerv_tt_idx = random.sample(remaining_tt2, int(0.8 * n_tt2))
for i in conn_tt_idx:
    df_val.at[i, "output"] = re.sub(r'tissue_type:\s*nan', 'tissue_type: connective', df_val.at[i, "output"])
    context_to_add = random_synonym("tissue_type", "connective")
    df_val.at[i, "prompt"] = re.sub(r'(Metadata to analyze:)', r'\1 ' + context_to_add, df_val.at[i, "prompt"], count=1, flags=re.IGNORECASE)
for i in nerv_tt_idx:
    df_val.at[i, "output"] = re.sub(r'tissue_type:\s*nan', 'tissue_type: nervous', df_val.at[i, "output"])
    context_to_add = random_synonym("tissue_type", "nervous")
    df_val.at[i, "prompt"] = re.sub(r'(Metadata to analyze:)', r'\1 ' + context_to_add, df_val.at[i, "prompt"], count=1, flags=re.IGNORECASE)

df_train.to_csv(train_path, index=False)
df_val.to_csv(val_path, index=False)
