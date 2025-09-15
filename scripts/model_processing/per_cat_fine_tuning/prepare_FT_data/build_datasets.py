import os, glob, random, math
import pandas as pd
random.seed(42)

BASE_DIR = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/DATA_PER_CAT/sequencing_source"
CT_DIR = os.path.join(BASE_DIR, "clean_to_train")
TRAIN_PATH = os.path.join(CT_DIR, "train.csv")
VAL_PATH = os.path.join(CT_DIR, "val.csv")
TEST_PATH = os.path.join(CT_DIR, "test.csv")

def read_csv_auto(p):
    return pd.read_csv(p, sep=None, engine="python", dtype=str).fillna("")

def write_csv(df, p):
    df.to_csv(p, index=False)

def top_counts(df, k=5):
    vc = df["output"].value_counts()
    head = vc.head(k)
    return {str(idx): int(val) for idx, val in head.items()}

def assign_outputs_to_groups(df):
    total = len(df)
    t1 = int(total * 0.6)
    t2 = int(total * 0.2)
    t3 = total - (t1 + t2)
    targets = [t1, t2, t3]
    counts = df["output"].value_counts().sort_values(ascending=False)
    groups = {0: set(), 1: set(), 2: set()}
    sums = [0, 0, 0]
    for out, c in counts.items():
        best = None
        best_score = None
        for g in [0,1,2]:
            overshoot = max(0, (sums[g] + c) - targets[g])
            fill = sums[g] / (targets[g] + 1e-9)
            score = (overshoot, fill, sums[g])
            if best_score is None or score < best_score:
                best_score = score
                best = g
        groups[best].add(out)
        sums[best] += c
    return groups, sums, targets

def round_robin_indices_to_drop(df_old, n_drop):
    pools = {}
    for out, sub in df_old.groupby("output"):
        pools[out] = list(sub.index)
        random.shuffle(pools[out])
    outs = sorted(pools.keys(), key=lambda o: len(pools[o]), reverse=True)
    drop = []
    i = 0
    while len(drop) < n_drop and len(outs) > 0:
        o = outs[i % len(outs)]
        if pools[o]:
            drop.append(pools[o].pop())
        if not pools[o]:
            outs.remove(o)
            if len(outs) == 0:
                break
            i = i % len(outs)
        else:
            i += 1
    return drop

def select_balanced_indices(df_add, n_keep):
    if n_keep <= 0 or len(df_add) == 0:
        return []
    pools = {}
    for out, sub in df_add.groupby("output"):
        idxs = list(sub.index)
        random.shuffle(idxs)
        pools[out] = idxs
    outs = sorted(pools.keys(), key=lambda o: len(pools[o]), reverse=True)
    selected = []
    i = 0
    while len(selected) < n_keep and len(outs) > 0:
        o = outs[i % len(outs)]
        if pools[o]:
            selected.append(pools[o].pop())
        if not pools[o]:
            outs.remove(o)
            if len(outs) == 0:
                break
            i = i % len(outs)
        else:
            i += 1
    return selected

def summarize_removals(df_before, idx_removed):
    if not idx_removed:
        return {}
    removed = df_before.loc[idx_removed]
    return removed["output"].value_counts().to_dict()

meta_candidates = sorted([p for p in glob.glob(os.path.join(CT_DIR, "metadata*")) if os.path.isfile(p)])
META_PATH = meta_candidates[0]

meta = read_csv_auto(META_PATH)
meta = meta[["prompt","output"]].copy()

groups, sums, targets = assign_outputs_to_groups(meta)
g0_vals, g1_vals, g2_vals = groups[0], groups[1], groups[2]
g0 = meta[meta["output"].isin(g0_vals)].copy()
g1 = meta[meta["output"].isin(g1_vals)].copy()
g2 = meta[meta["output"].isin(g2_vals)].copy()

g0_path = os.path.join(CT_DIR, "group_metadata_60_train.csv")
g1_path = os.path.join(CT_DIR, "group_metadata_20_val.csv")
g2_path = os.path.join(CT_DIR, "group_metadata_20_test.csv")
write_csv(g0, g0_path)
write_csv(g1, g1_path)
write_csv(g2, g2_path)

total_meta = len(meta)
train_add = g0
val_add = g1
test_add = g2

for path_check in [TRAIN_PATH, VAL_PATH, TEST_PATH]:
    if not os.path.exists(path_check):
        raise FileNotFoundError("Fichier manquant : " + path_check)

train_old = read_csv_auto(TRAIN_PATH)[["prompt","output"]].copy()
val_old = read_csv_auto(VAL_PATH)[["prompt","output"]].copy()
test_old = read_csv_auto(TEST_PATH)[["prompt","output"]].copy()

def integrate_and_resize(old_df, add_df, target_n, name, out_path):
    before_len = len(old_df)
    max_new_allowed = target_n // 2
    if len(add_df) > max_new_allowed:
        idx_keep_new = select_balanced_indices(add_df, max_new_allowed)
        add_df_sel = add_df.loc[idx_keep_new]
        new_dropped_due_cap = len(add_df) - len(add_df_sel)
    else:
        add_df_sel = add_df
        new_dropped_due_cap = 0
    after_add = pd.concat([old_df.assign(_is_new=False), add_df_sel.assign(_is_new=True)], ignore_index=True)
    oversize = len(after_add) - target_n
    removed_old_summary = {}
    removed_new_summary = {}
    if oversize > 0:
        old_part = after_add[after_add["_is_new"] == False]
        new_part = after_add[after_add["_is_new"] == True]
        drop_from_old = min(oversize, len(old_part))
        idx_drop_old = round_robin_indices_to_drop(old_part, drop_from_old) if drop_from_old > 0 else []
        oversize_left = oversize - len(idx_drop_old)
        idx_drop_new = []
        if oversize_left > 0:
            idx_drop_new = round_robin_indices_to_drop(new_part, oversize_left)
        to_drop = idx_drop_old + idx_drop_new
        if idx_drop_old:
            removed_old_summary = summarize_removals(after_add, idx_drop_old)
        if idx_drop_new:
            removed_new_summary = summarize_removals(after_add, idx_drop_new)
        final_df = after_add.drop(index=to_drop)
    else:
        final_df = after_add
    final_df = final_df.drop(columns=["_is_new"])
    write_csv(final_df, out_path)
    if removed_old_summary:
        show_old = dict(sorted({str(k): int(v) for k, v in removed_old_summary.items()}.items(), key=lambda x: x[1], reverse=True)[:8])
    if removed_new_summary:
        show_new = dict(sorted({str(k): int(v) for k, v in removed_new_summary.items()}.items(), key=lambda x: x[1], reverse=True)[:8])
    new_ratio = round(len(add_df_sel) / max(1, len(final_df)) * 100, 2)
    return final_df

train_final = integrate_and_resize(train_old, train_add, 1000, "train", TRAIN_PATH)
val_final = integrate_and_resize(val_old, val_add, 200, "val", VAL_PATH)
test_final = integrate_and_resize(test_old, test_add, 500, "test", TEST_PATH)
