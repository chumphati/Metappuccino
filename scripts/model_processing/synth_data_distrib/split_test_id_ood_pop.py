import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize
from transformers import AutoTokenizer, AutoModel

base_out = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/all_cat_sets"
model_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/Mistral-7B-Instruct-v0.3"
synth_csv_ALL = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/all_cat_sets/create_synt_data/test_all.csv"
test_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/all_cat_sets/test"
text_col = "phrase"
id_cols = ["run_accession", text_col]

cats_group1 = ["sex","biopsy_site","library_selection","biopsy_type"]
cats_group2 = ["sequencing_source","cell_line","cell_type","organ","disease","treatment","treatment_time","response","age","ethnicity","localization","is_cancer","phrase"]

os.makedirs(test_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Init] Device: {device}")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModel.from_pretrained(
    model_path,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None
)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
    model.config.pad_token_id = tokenizer.pad_token_id
model.eval()
print("[Init] Tokenizer/Model loaded.")

def _print_head(texts, n=3, title="Sample"):
    print(f"[Peek] {title} (n={len(texts)}):")
    for t in texts[:n]:
        if isinstance(t, str):
            t = t.replace("\n", " ")
        print("   ", (t[:120] + "…") if isinstance(t, str) and len(t) > 120 else t)

def _find_one_train_file_for_group(base_dir, categories):
    print(f"[Train Finder] Searching a single train file for the group. Categories: {categories}")
    for cat in categories:
        paths = sorted(glob.glob(os.path.join(base_dir, cat, "*_train_texts.csv")))
        if paths:
            print(f"[Train Finder] Train found for '{cat}': {paths[0]}")
            return paths[0]
    fallback = sorted(glob.glob(os.path.join(base_dir, "*", "*_train_texts.csv")))
    if fallback:
        print(f"[Train Finder] Using fallback train: {fallback[0]}")
        return fallback[0]
    raise RuntimeError(f"No *_train_texts.csv found for group {categories}.")

def load_one_group_train_texts(base_dir, categories):
    train_file = _find_one_train_file_for_group(base_dir, categories)
    print(f"[Train Loader] Reading: {train_file}")
    df = pd.read_csv(train_file, dtype=str)
    if "text" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "text"})
    df = df[["text"]].dropna().drop_duplicates()
    texts = df["text"].tolist()
    print(f"[Train Loader] Train texts: {len(texts)}")
    _print_head(texts, title="Train texts")
    return texts, train_file

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
    print(f"[Scores] Parameters -> k={k}, alpha={alpha}, beta={beta}")
    if tr_emb is None:
        print("[Scores] Train embeddings...")
        tr_emb = embed_texts(train_texts)
    if sy_emb is None:
        print("[Scores] Test embeddings...")
        sy_emb = embed_texts(synth_texts)
    tr_cent = normalize(torch.tensor(tr_emb, dtype=torch.float32).mean(dim=0, keepdim=True), p=2, dim=1).cpu().numpy()
    s_tr = cosine_sim(sy_emb, tr_cent).squeeze(1)
    S_tr = cosine_sim(sy_emb, tr_emb)
    topk_tr = np.sort(S_tr, axis=1)[:, -k:]
    m_tr = topk_tr.mean(axis=1)
    score = alpha * s_tr + beta * m_tr
    print(f"[Scores] Score stats: mean={score.mean():.4f} std={score.std():.4f} min={score.min():.4f} p20={np.quantile(score,0.2):.4f} median={np.quantile(score,0.5):.4f} p80={np.quantile(score,0.8):.4f} max={score.max():.4f}")
    return score, s_tr, m_tr, tr_emb, sy_emb

def smart_three_way_split(scores, q_edges=(0.2, 0.8), min_gap=0.02, allow_two_way=True):
    s = pd.Series(scores)
    ql, qh = s.quantile(q_edges[0]), s.quantile(q_edges[1])
    print(f"[Split] q{int(q_edges[0]*100)}={ql:.4f}, q{int(q_edges[1]*100)}={qh:.4f}, gap={qh-ql:.4f}")
    if allow_two_way and (qh - ql < min_gap):
        med = s.median()
        print(f"[Split] Using 2-way split around median={med:.4f} (id >= median).")
        bins = np.where(s >= med, "id", "ood")
        return bins
    low_mask = s <= ql
    high_mask = s >= qh
    bins = np.array(["mid"] * len(s), dtype=object)
    bins[low_mask.values] = "ood"
    bins[high_mask.values] = "id"
    mid_count = np.sum(bins == "mid")
    print(f"[Split] 3-way -> ood={np.sum(bins=='ood')} mid={mid_count} id={np.sum(bins=='id')}")
    if allow_two_way and mid_count == 0:
        med = s.median()
        print(f"[Split] No 'mid'. Fallback to 2-way around median={med:.4f}.")
        bins = np.where(s >= med, "id", "ood")
    return bins

def save_split(df_in, bins, out_base, scores_pack, save_mid=True):
    score, s_tr, m_tr = scores_pack[0], scores_pack[1], scores_pack[2]
    df = df_in.copy()
    df["_set"] = bins
    df["_score"] = score
    df["_sim_train_centroid"] = s_tr
    df["_knn_train_mean"] = m_tr
    id_df  = df[df["_set"] == "id"][["run_accession", text_col, "_score", "_sim_train_centroid", "_knn_train_mean"]]
    ood_df = df[df["_set"] == "ood"][["run_accession", text_col, "_score", "_sim_train_centroid", "_knn_train_mean"]]
    mid_df = df[df["_set"] == "mid"][["run_accession", text_col, "_score", "_sim_train_centroid", "_knn_train_mean"]]
    id_path  = f"{out_base}_id.csv"
    ood_path = f"{out_base}_ood.csv"
    diag_path = f"{out_base}_diagnostics.csv"
    id_df.rename(columns={text_col: "summary"}).to_csv(id_path, index=False)
    ood_df.rename(columns={text_col: "summary"}).to_csv(ood_path, index=False)
    mid_path = None
    if save_mid and len(mid_df) > 0:
        mid_path = f"{out_base}_mid.csv"
        mid_df.rename(columns={text_col: "summary"}).to_csv(mid_path, index=False)
        print(f"[Save] MID: {mid_path} (n={len(mid_df)})")
    else:
        print("[Save] No MID generated.")
    df_diag = df[["run_accession", text_col, "_set", "_score", "_sim_train_centroid", "_knn_train_mean"]].rename(columns={text_col: "summary"})
    df_diag.to_csv(diag_path, index=False)
    print(f"[Save] ID : {id_path} (n={len(id_df)})")
    print(f"[Save] OOD: {ood_path} (n={len(ood_df)})")
    print(f"[Save] DIAG: {diag_path} (n={len(df_diag)})")
    return {"id": len(id_df), "mid": len(mid_df), "ood": len(ood_df),
            "paths": {"id": id_path, "mid": mid_path, "ood": ood_path, "diag": diag_path}
            }

def main():
    print("[Main] Loading test_all.csv...")
    df = pd.read_csv(synth_csv_ALL, dtype=str)
    print(f"[Main] test_all rows: {len(df)}")
    if "run_accession" not in df.columns or text_col not in df.columns:
        raise RuntimeError("Missing columns run_accession or phrase in test_all.csv.")
    before = len(df)
    df = df.dropna(subset=[text_col]).reset_index(drop=True)
    print(f"[Main] After dropna({text_col}) -> {len(df)} (was {before}). No dedup applied.")
    _print_head(df[text_col].tolist(), title="Test previews")
    print("[Main] Computing test embeddings once...")
    synth_texts = df[text_col].tolist()
    sy_emb = embed_texts(synth_texts)
    print(f"[Main] sy_emb shape: {sy_emb.shape}")
    print("\n===== Group 1 =====")
    tr_g1, tr_file_g1 = load_one_group_train_texts(base_out, cats_group1)
    scores_g1 = proximity_scores_to_train(tr_g1, synth_texts, k=5, alpha=0.5, beta=0.5, sy_emb=sy_emb)
    score_g1 = scores_g1[0]
    bins_g1 = smart_three_way_split(score_g1, q_edges=(0.2, 0.8), min_gap=0.02, allow_two_way=True)
    base_g1 = os.path.join(test_dir, "group_sex_biopsysite_libraryselection_biopsytype")
    os.makedirs(test_dir, exist_ok=True)
    res_g1 = save_split(df, bins_g1, base_g1, scores_g1, save_mid=True)
    print(f"[G1 Summary] {res_g1}")
    for cat in cats_group1:
        out_base = os.path.join(test_dir, cat)
        print(f"[G1 Save by Cat] {cat} -> reusing the same bins. Shared train: {tr_file_g1}")
        save_split(df, bins_g1, out_base, scores_g1, save_mid=True)
    print("\n===== Group 2 =====")
    tr_g2, tr_file_g2 = load_one_group_train_texts(base_out, cats_group2)
    scores_g2 = proximity_scores_to_train(tr_g2, synth_texts, k=5, alpha=0.5, beta=0.5, sy_emb=sy_emb)
    score_g2 = scores_g2[0]
    bins_g2 = smart_three_way_split(score_g2, q_edges=(0.2, 0.8), min_gap=0.02, allow_two_way=True)
    base_g2 = os.path.join(test_dir, "group_other_categories")
    res_g2 = save_split(df, bins_g2, base_g2, scores_g2, save_mid=True)
    print(f"[G2 Summary] {res_g2}")
    for cat in cats_group2:
        out_base = os.path.join(test_dir, cat)
        print(f"[G2 Save by Cat] {cat} -> reusing the same bins. Shared train: {tr_file_g2}")
        save_split(df, bins_g2, out_base, scores_g2, save_mid=True)
    def _counts(bins):
        vals, counts = np.unique(bins, return_counts=True)
        d = dict(zip(vals, counts))
        return {"id": d.get("id", 0), "mid": d.get("mid", 0), "ood": d.get("ood", 0)}
    summary = []
    summary.append({"group": "group1", **_counts(bins_g1)})
    summary.append({"group": "group2", **_counts(bins_g2)})
    sum_path = os.path.join(test_dir, "split_counts_summary.csv")
    pd.DataFrame(summary).to_csv(sum_path, index=False)
    print(f"[Main] Global recap -> {sum_path}")
    print(pd.DataFrame(summary))
    print("\n[Done] Pipeline finished.")

if __name__ == "__main__":
    main()
