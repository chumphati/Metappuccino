import os
import glob
import re
import json
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize
from transformers import AutoTokenizer, AutoModel

# base_out = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/all_cat_sets"
base_out = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/all_cat_sets/new_organ_disease"
model_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/Mistral-7B-Instruct-v0.3"
synth_csv_ALL = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/all_cat_sets/create_synt_data/test_all.csv"
test_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/all_cat_sets/new_organ_disease/test"
text_col = "phrase"
os.makedirs(test_dir, exist_ok=True)

def discover_categories(root):
    cats = []
    for d in sorted(next(os.walk(root))[1]):
        has_train = glob.glob(os.path.join(root, d, "*_train_texts.csv")) or \
                    glob.glob(os.path.join(root, d, "train.csv"))
        if has_train:
            cats.append(d)
    return cats

CATEGORIES = discover_categories(base_out)

def extract_summary_from_prompt(prompt_text: str) -> str:
    if not isinstance(prompt_text, str):
        return ""
    m = re.search(r"Summary:\s*(.*)", prompt_text)
    return m.group(1).strip() if m else ""

def canonicalize_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.replace("\ufeff", "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def pick_text_series(df: pd.DataFrame) -> pd.Series:
    cols_low = [c.lower() for c in df.columns]
    if "summary" in cols_low:
        col = df.columns[cols_low.index("summary")]
        return df[col].astype(str).map(canonicalize_text)
    if "text" in cols_low:
        col = df.columns[cols_low.index("text")]
        return df[col].astype(str).map(canonicalize_text)
    if "prompt" in cols_low:
        col = df.columns[cols_low.index("prompt")]
        return df[col].astype(str).map(extract_summary_from_prompt).map(canonicalize_text)
    raise ValueError(f"Aucune colonne texte claire parmi: {list(df.columns)}")

def find_train_file_for_category(base_dir: str, cat: str) -> str:
    p1 = sorted(glob.glob(os.path.join(base_dir, cat, "*_train_texts.csv")))
    if p1: return p1[0]
    p2 = sorted(glob.glob(os.path.join(base_dir, cat, "train.csv")))
    if p2: return p2[0]
    raise RuntimeError(f"Aucun train trouvé pour '{cat}'.")

def load_category_train_texts(base_dir: str, cat: str):
    train_file = find_train_file_for_category(base_dir, cat)
    df = pd.read_csv(train_file, dtype=str, sep="\t")
    if "text" in df.columns:
        s = df["text"].astype(str).map(canonicalize_text)
    else:
        s = pick_text_series(df)
    s = s.dropna()
    s = s[s.astype(str).str.strip().astype(bool)].drop_duplicates()
    return s.tolist(), train_file

device = "cuda" if torch.cuda.is_available() else "cpu"
np.random.seed(42); torch.manual_seed(42)
print(f"[Init] Device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModel.from_pretrained(
    model_path,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None: tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
    model.config.pad_token_id = tokenizer.pad_token_id
model.eval()
print("[Init] Tokenizer/Model loaded.")

@torch.inference_mode()
def embed_texts(texts, batch_size=16, max_length=1024):
    texts = [t if isinstance(t, str) and t.strip() else " " for t in texts]
    out_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        hs = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        pooled = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = torch.nan_to_num(pooled, 0.0, 0.0, 0.0)
        out_list.append(pooled.detach().float().cpu())
    X = torch.cat(out_list, dim=0)
    X = normalize(X, p=2, dim=1)
    return X.numpy()

def cosine_sim(A, B):
    A = torch.tensor(A, dtype=torch.float32)
    B = torch.tensor(B, dtype=torch.float32)
    A = normalize(torch.nan_to_num(A, 0.0, 0.0, 0.0), p=2, dim=1)
    B = normalize(torch.nan_to_num(B, 0.0, 0.0, 0.0), p=2, dim=1)
    return (A @ B.T).cpu().numpy()

def proximity_scores_to_train(train_texts, synth_texts, k=5, alpha=0.5, beta=0.5, tr_emb=None, sy_emb=None):
    if tr_emb is None: tr_emb = embed_texts(train_texts)
    if sy_emb is None: sy_emb = embed_texts(synth_texts)
    tr_cent = normalize(torch.tensor(tr_emb, dtype=torch.float32).mean(dim=0, keepdim=True), p=2, dim=1).cpu().numpy()
    s_tr = cosine_sim(sy_emb, tr_cent).squeeze(1)
    S_tr = cosine_sim(sy_emb, tr_emb)
    topk_tr = np.sort(S_tr, axis=1)[:, -k:]
    m_tr = topk_tr.mean(axis=1)
    score = alpha * s_tr + beta * m_tr
    return score, s_tr, m_tr, tr_emb, sy_emb

def smart_three_way_split(scores, q_edges=(0.2, 0.8), min_gap=0.02, allow_two_way=True):
    s = pd.Series(scores)
    ql, qh = s.quantile(q_edges[0]), s.quantile(q_edges[1])
    if allow_two_way and (qh - ql < min_gap):
        med = s.median()
        return np.where(s >= med, "id", "ood")
    bins = np.array(["mid"] * len(s), dtype=object)
    bins[s <= ql] = "ood"
    bins[s >= qh] = "id"
    if allow_two_way and np.sum(bins == "mid") == 0:
        med = s.median()
        bins = np.where(s >= med, "id", "ood")
    return bins

def save_split(df_in, bins, out_base, scores_pack, cat_name, train_file, params):
    score, s_tr, m_tr = scores_pack[0], scores_pack[1], scores_pack[2]
    df = df_in.copy()
    df["_set"] = bins
    df["_score"] = score
    df["_sim_train_centroid"] = s_tr
    df["_knn_train_mean"] = m_tr

    id_df  = df[df["_set"] == "id"][["run_accession", text_col, "_score", "_sim_train_centroid", "_knn_train_mean"]]
    ood_df = df[df["_set"] == "ood"][["run_accession", text_col, "_score", "_sim_train_centroid", "_knn_train_mean"]]
    mid_df = df[df["_set"] == "mid"][["run_accession", text_col, "_score", "_sim_train_centroid", "_knn_train_mean"]]

    out_base = os.path.join(test_dir, cat_name)
    id_path  = f"{out_base}_id.csv"
    ood_path = f"{out_base}_ood.csv"
    diag_path = f"{out_base}_diagnostics.csv"

    id_df.rename(columns={text_col: "summary"}).to_csv(id_path, index=False)
    ood_df.rename(columns={text_col: "summary"}).to_csv(ood_path, index=False)
    mid_path = None
    if len(mid_df) > 0:
        mid_path = f"{out_base}_mid.csv"
        mid_df.rename(columns={text_col: "summary"}).to_csv(mid_path, index=False)

    df_diag = df[["run_accession", text_col, "_set", "_score", "_sim_train_centroid", "_knn_train_mean"]].rename(columns={text_col: "summary"})
    df_diag.to_csv(diag_path, index=False)

    manifest = {
        "category": cat_name,
        "train_file": train_file,
        "params": params,
        "counts": {"id": len(id_df), "mid": len(mid_df), "ood": len(ood_df)}
    }
    with open(f"{out_base}_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"[{cat_name}] ID : {id_path} (n={len(id_df)})")
    print(f"[{cat_name}] OOD: {ood_path} (n={len(ood_df)})")
    if mid_path: print(f"[{cat_name}] MID: {mid_path} (n={len(mid_df)})")
    else:        print(f"[{cat_name}] No MID generated.")
    print(f"[{cat_name}] DIAG: {diag_path} (n={len(df_diag)})")
    return {"id": len(id_df), "mid": len(mid_df), "ood": len(ood_df)}

def main():
    print("[Main] Loading test_all.csv...")
    df = pd.read_csv(synth_csv_ALL, dtype=str)
    if "run_accession" not in df.columns or text_col not in df.columns:
        raise RuntimeError("Missing columns run_accession or phrase in test_all.csv.")
    before = len(df)
    df[text_col] = df[text_col].astype(str).map(canonicalize_text)
    df = df.dropna(subset=[text_col])
    df = df[df[text_col].str.strip().astype(bool)].reset_index(drop=True)
    print(f"[Main] After drop/filter -> {len(df)} (was {before}).")
    synth_texts = df[text_col].tolist()
    print("[Main] Computing test embeddings once...")
    sy_emb = embed_texts(synth_texts)
    print(f"[Main] sy_emb shape: {sy_emb.shape}")
    summary_rows = []
    for cat in CATEGORIES:
        print(f"\n===== {cat} =====")
        tr_texts, tr_file = load_category_train_texts(base_out, cat)
        if len(tr_texts) == 0:
            print(f"[{cat}] Empty train texts. Skipping.")
            continue

        params = {"k": 5, "alpha": 0.5, "beta": 0.5, "q_edges": [0.2, 0.8], "min_gap": 0.02}
        scores_pack = proximity_scores_to_train(tr_texts, synth_texts,
                                                k=params["k"], alpha=params["alpha"], beta=params["beta"],
                                                sy_emb=sy_emb)
        bins = smart_three_way_split(scores_pack[0],
                                     q_edges=tuple(params["q_edges"]),
                                     min_gap=params["min_gap"], allow_two_way=True)
        counts = save_split(df, bins, os.path.join(test_dir, cat), scores_pack, cat, tr_file, params)
        summary_rows.append({"category": cat, **counts})

    sum_path = os.path.join(test_dir, "split_counts_summary_per_category.csv")
    pd.DataFrame(summary_rows).to_csv(sum_path, index=False)
    print(f"\n[Main] Global recap -> {sum_path}")
    print(pd.DataFrame(summary_rows))
    print("\n[Done] Pipeline finished.")

if __name__ == "__main__":
    main()
