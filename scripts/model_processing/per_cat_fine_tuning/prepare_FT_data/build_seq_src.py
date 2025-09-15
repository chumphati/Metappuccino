import os, glob, random, re
import numpy as np
import pandas as pd
random.seed(42)
np.random.seed(42)

BASE_DIR = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/DATA_PER_CAT/sequencing_source"
CT_DIR = os.path.join(BASE_DIR, "clean_to_train")
TRAIN_IN = os.path.join(CT_DIR, "train.csv")
VAL_IN = os.path.join(CT_DIR, "val.csv")
TEST_IN = os.path.join(CT_DIR, "test.csv")

OUT_DIR = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING13"
os.makedirs(OUT_DIR, exist_ok=True)
TRAIN_OUT = os.path.join(OUT_DIR, "train.csv")
VAL_OUT = os.path.join(OUT_DIR, "val.csv")
TEST_OUT = os.path.join(OUT_DIR, "test.csv")

THREE = ["bulk", "spatial", "single cell"]
FOUR = ["bulk", "spatial", "single cell", "unknown"]

def read_csv_auto(p):
    return pd.read_csv(p, sep=None, engine="python", dtype=str).fillna("")

def write_csv(df, p):
    df.to_csv(p, index=False)

def canon_output(x):
    t = str(x).strip().lower()
    t = t.replace("-", " ").replace("_", " ")
    t = re.sub(r"\s+", " ", t)
    if t in {"single cell", "singlecell", "sc", "single cells"} or t.startswith("single cell "):
        return "single cell"
    if t in {"", "na", "n/a", "none", "unk", "unknown", "not applicable"} or t.startswith("unknown "):
        return "unknown"
    if t == "bulk":
        return "bulk"
    if t == "spatial":
        return "spatial"
    return t

def normalize_df(df):
    df = df.copy()
    df["prompt"] = df["prompt"].astype(str)
    df["output"] = df["output"].map(canon_output)
    return df

def dedup_on_prompt_keep_last(dfs_with_names):
    frames = []
    for name, df in dfs_with_names:
        tmp = df[["prompt","output"]].copy()
        tmp["_src"] = name
        frames.append(tmp)
    pool = pd.concat(frames, ignore_index=True)
    pool = pool.drop_duplicates(subset=["prompt"], keep="last").reset_index(drop=True)
    return pool

def stratified_sample(df, n, by="output", seed=42):
    if n <= 0 or len(df) == 0:
        return df.iloc[0:0].copy()
    counts = df[by].value_counts()
    total = counts.sum()
    base = (counts * n / total).apply(np.floor).astype(int)
    remainder = n - base.sum()
    if remainder > 0:
        fracs = (counts * n / total) - base
        take_more = fracs.sort_values(ascending=False).index.tolist()
        for lab in take_more:
            if remainder == 0:
                break
            cap = (df[by] == lab).sum()
            if base[lab] < cap:
                base[lab] += 1
                remainder -= 1
    for lab in base.index:
        cap = (df[by] == lab).sum()
        if base[lab] > cap:
            base[lab] = cap
    selected = []
    for lab, q in base.items():
        if q > 0:
            block = df[df[by] == lab].sample(n=q, random_state=seed, replace=False)
            selected.append(block)
    out = pd.concat(selected, ignore_index=True) if selected else df.iloc[0:0].copy()
    short = n - len(out)
    if short > 0:
        remain = df[~df["prompt"].isin(out["prompt"])]
        if short > 0 and len(remain) > 0:
            extra = remain.sample(n=min(short, len(remain)), random_state=seed, replace=False)
            out = pd.concat([out, extra], ignore_index=True)
    return out

def train_three_equal_plus_unknown(df, three_labels, reserve_for_val_test=700):
    sub3 = df[df["output"].isin(three_labels)]
    counts3 = sub3["output"].value_counts()
    m = min(int(counts3.get(l, 0)) for l in three_labels)
    if m == 0:
        raise ValueError("One of bulk/spatial/single cell has zero available samples.")
    parts = []
    for lab in three_labels:
        parts.append(sub3[sub3["output"] == lab].sample(n=m, random_state=42, replace=False))
    base = pd.concat(parts, ignore_index=True)
    remaining_pool = df[~df["prompt"].isin(base["prompt"])]
    unk_pool = remaining_pool[remaining_pool["output"] == "unknown"]
    max_unk = max(0, len(df) - len(base) - reserve_for_val_test)
    take_unk = min(len(unk_pool), max_unk)
    if take_unk == 0 and len(unk_pool) > 0 and (len(df) - len(base) - 1) >= reserve_for_val_test:
        take_unk = 1
    if take_unk > 0:
        unk_sel = unk_pool.sample(n=take_unk, random_state=42, replace=False)
        return pd.concat([base, unk_sel], ignore_index=True)
    return base

def assert_no_overlap(a, b, c):
    sa, sb, sc = set(a["prompt"]), set(b["prompt"]), set(c["prompt"])
    inter = (sa & sb) | (sa & sc) | (sb & sc)
    if inter:
        raise AssertionError(f"Cross-set prompt duplication detected: {len(inter)}")

def counts_dict(df):
    vc = df["output"].value_counts()
    return {str(k): int(v) for k, v in vc.items()}

def subset_counts(df, labels):
    return {lab: int((df["output"] == lab).sum()) for lab in labels}

meta_candidates = sorted([p for p in glob.glob(os.path.join(CT_DIR, "metadata*")) if os.path.isfile(p)])
if not meta_candidates:
    raise FileNotFoundError(f"No metadata* file found in {CT_DIR}")

meta_frames = [read_csv_auto(p)[["prompt","output"]] for p in meta_candidates]
meta = pd.concat(meta_frames, ignore_index=True)

for p in [TRAIN_IN, VAL_IN, TEST_IN]:
    if not os.path.exists(p):
        raise FileNotFoundError("Missing file: " + p)

train_old = read_csv_auto(TRAIN_IN)[["prompt","output"]]
val_old = read_csv_auto(VAL_IN)[["prompt","output"]]
test_old = read_csv_auto(TEST_IN)[["prompt","output"]]

meta = normalize_df(meta)
train_old = normalize_df(train_old)
val_old = normalize_df(val_old)
test_old = normalize_df(test_old)

pool = dedup_on_prompt_keep_last([
    ("train_old", train_old),
    ("val_old", val_old),
    ("test_old", test_old),
    ("metadata", meta),
])

pool = pool[pool["prompt"].str.len() > 0].reset_index(drop=True)

train_new = train_three_equal_plus_unknown(pool, THREE, reserve_for_val_test=700)
remain_after_train = pool[~pool["prompt"].isin(train_new["prompt"])].reset_index(drop=True)
val_new = stratified_sample(remain_after_train, 200, by="output", seed=42)
remain_after_val = remain_after_train[~remain_after_train["prompt"].isin(val_new["prompt"])].reset_index(drop=True)
test_new = stratified_sample(remain_after_val, 500, by="output", seed=42)

if len(val_new) < 200 or len(test_new) < 500:
    raise ValueError(f"Insufficient sizes: val={len(val_new)} test={len(test_new)}")

assert_no_overlap(train_new, val_new, test_new)
assert train_new["prompt"].is_unique
assert val_new["prompt"].is_unique
assert test_new["prompt"].is_unique

write_csv(train_new[["prompt","output"]], TRAIN_OUT)
write_csv(val_new[["prompt","output"]], VAL_OUT)
write_csv(test_new[["prompt","output"]], TEST_OUT)

print("TRAIN size:", len(train_new))
print("TRAIN distribution:", counts_dict(train_new))
print("TRAIN bulk/spatial/single cell:", subset_counts(train_new, THREE))
print("TRAIN unknown:", subset_counts(train_new, ["unknown"]))
print("VAL size:", len(val_new))
print("VAL distribution:", counts_dict(val_new))
print("VAL bulk/spatial/single cell:", subset_counts(val_new, THREE))
print("VAL unknown:", subset_counts(val_new, ["unknown"]))
print("TEST size:", len(test_new))
print("TEST distribution:", counts_dict(test_new))
print("TEST bulk/spatial/single cell:", subset_counts(test_new, THREE))
print("TEST unknown:", subset_counts(test_new, ["unknown"]))
