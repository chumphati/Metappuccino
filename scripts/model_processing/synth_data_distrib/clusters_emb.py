import os
import re
import json
import numpy as np
import pandas as pd
import torch
from torch.nn.functional import normalize
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score, silhouette_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

try:
    from umap import UMAP as UMAPCls
    UMAP_AVAILABLE = True
except Exception:
    try:
        from umap.umap_ import UMAP as UMAPCls
        UMAP_AVAILABLE = True
    except Exception:
        UMAP_AVAILABLE = False

try:
    from sklearn.neighbors import KernelDensity
    KDE_AVAILABLE = True
except Exception:
    KDE_AVAILABLE = False

base_out = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/all_cat_sets"
model_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/Mistral-7B-Instruct-v0.3"
cat_names = [
    "library_selection","sequencing_source","biopsy_site","biopsy_type","cell_line",
    "cell_type","organ","disease","treatment","treatment_time","response","age","sex","ethnicity","is_cancer"
]

real_paths = [
    "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/final_pop/id.csv",
    "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/final_pop/ood.csv",
    "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/final_pop/pop.csv",
]

palette = {
    "train":     "#000000",
    "val":       "#E69F00",
    "test_id":   "#0072B2",
    "test_ood":  "#D55E00",
    "test_mid":  "#CC79A7",
    "test_real": "#009E73"
}

markers = {
    "train": "o",
    "val": "s",
    "test_id": "^",
    "test_ood": "D",
    "test_mid": "P",
    "test_real": "X"
}

DRAW_ELLIPSES = True
ELLIPSE_STD = 1.5

def draw_confidence_ellipse(ax, X2, color, n_std=1.5, lw=1.25, alpha=0.15):
    if X2 is None or len(X2) < 5:
        return
    x = X2[:, 0]
    y = X2[:, 1]
    cov = np.cov(x, y)
    if not np.all(np.isfinite(cov)) or cov.shape != (2, 2):
        return
    vals, vecs = np.linalg.eigh(cov)
    if np.any(vals < 0) or not np.all(np.isfinite(vals)):
        return
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2 * n_std * np.sqrt(vals)
    mean = np.array([np.mean(x), np.mean(y)])
    ell = Ellipse(xy=mean, width=width, height=height, angle=theta,
                  edgecolor=color, facecolor=color, lw=lw, alpha=alpha)
    ax.add_patch(ell)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def extract_summary_from_prompt(prompt_text):
    if not isinstance(prompt_text, str):
        return ""
    m = re.search(r"Summary:\s*(.*)", prompt_text)
    return m.group(1).strip() if m else ""

def read_csv_flex(path):
    try:
        df = pd.read_csv(path, engine="python", sep=None, dtype=str, on_bad_lines="skip")
    except Exception:
        try:
            df = pd.read_csv(path, engine="python", sep="\t", dtype=str, on_bad_lines="skip")
        except Exception:
            df = pd.read_csv(path, engine="python", sep=",", dtype=str, on_bad_lines="skip")
    cols = [c.strip() if isinstance(c, str) else c for c in df.columns]
    df.columns = cols
    if "prompt" not in df.columns:
        if "\ufeffprompt" in df.columns:
            df = df.rename(columns={"\ufeffprompt": "prompt"})
        elif len(df.columns) >= 2 and "summary" not in df.columns:
            df = df.rename(columns={df.columns[0]: "prompt", df.columns[1]: "output"})
    return df

def read_train_val_for_category(cat_dir):
    train_csv = os.path.join(cat_dir, "train.csv")
    val_csv = os.path.join(cat_dir, "val.csv")
    df_train = read_csv_flex(train_csv)
    df_val = read_csv_flex(val_csv)
    if "prompt" not in df_train.columns or "prompt" not in df_val.columns:
        raise ValueError(f"Missing 'prompt' column in {cat_dir}")
    df_train["summary"] = df_train["prompt"].apply(extract_summary_from_prompt)
    df_val["summary"] = df_val["prompt"].apply(extract_summary_from_prompt)
    return df_train[["summary"]].rename(columns={"summary": "text"}), df_val[["summary"]].rename(columns={"summary": "text"})

def read_real_tests(paths):
    frames = []
    for p in paths:
        df = read_csv_flex(p)
        if "summary" in df.columns:
            col = "summary"
        elif len(df.columns) >= 2:
            col = df.columns[-1]
        else:
            raise ValueError(f"No 'summary' column found in {p}")
        frames.append(df[[col]].rename(columns={col: "text"}))
    df_all = pd.concat(frames, ignore_index=True)
    df_all.drop_duplicates(subset=["text"], inplace=True)
    return df_all

device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModel.from_pretrained(
    model_path,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
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

@torch.inference_mode()
def embed_texts(texts, batch_size=8, max_length=1024):
    texts = [t if isinstance(t, str) and len(t.strip()) > 0 else "N/A" for t in texts]
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model(**enc, output_hidden_states=True)
        hs = out.last_hidden_state
        mask = enc["attention_mask"].unsqueeze(-1)
        masked = hs * mask
        summed = masked.sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)
        mean_pooled = summed / lengths
        mean_pooled = torch.nan_to_num(mean_pooled, nan=0.0, posinf=0.0, neginf=0.0)
        embs.append(mean_pooled.detach().float().cpu())
    embs = torch.cat(embs, dim=0)
    embs = torch.nan_to_num(embs, nan=0.0, posinf=0.0, neginf=0.0)
    embs = normalize(embs, p=2, dim=1)
    embs = torch.nan_to_num(embs, nan=0.0, posinf=0.0, neginf=0.0)
    return embs.numpy()

def cosine_similarity(a, b):
    a = torch.tensor(a, dtype=torch.float32)
    b = torch.tensor(b, dtype=torch.float32)
    a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    b = torch.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)
    a = normalize(a, p=2, dim=1)
    b = normalize(b, p=2, dim=1)
    s = (a @ b.T).cpu().numpy()
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    return s

def auc_vs_train(train_emb, other_emb):
    te = torch.tensor(train_emb, dtype=torch.float32)
    oe = torch.tensor(other_emb, dtype=torch.float32)
    te = torch.nan_to_num(te, nan=0.0, posinf=0.0, neginf=0.0)
    oe = torch.nan_to_num(oe, nan=0.0, posinf=0.0, neginf=0.0)
    te = normalize(te, p=2, dim=1)
    oe = normalize(oe, p=2, dim=1)
    centroid = normalize(te.mean(dim=0, keepdim=True), p=2, dim=1)
    train_scores = (te @ centroid.T).squeeze(1).cpu().numpy()
    other_scores = (oe @ centroid.T).squeeze(1).cpu().numpy()
    train_scores = np.nan_to_num(train_scores, nan=0.0, posinf=0.0, neginf=0.0)
    other_scores = np.nan_to_num(other_scores, nan=0.0, posinf=0.0, neginf=0.0)
    y_true = np.concatenate([np.zeros_like(train_scores), np.ones_like(other_scores)])
    y_score = np.concatenate([train_scores, other_scores])
    y_score = np.nan_to_num(y_score, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        return float(roc_auc_score(y_true, y_score)), train_scores, other_scores
    except Exception:
        return 0.5, train_scores, other_scores

def knn_loo_accuracy(train_emb, other_emb):
    X = np.concatenate([train_emb, other_emb], axis=0)
    y = np.concatenate([np.zeros(len(train_emb), dtype=int), np.ones(len(other_emb), dtype=int)], axis=0)
    T = torch.tensor(X, dtype=torch.float32)
    T = torch.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)
    T = normalize(T, p=2, dim=1)
    S = (T @ T.T).cpu().numpy()
    np.fill_diagonal(S, -np.inf)
    nn_idx = S.argmax(axis=1)
    y_pred = y[nn_idx]
    acc = float((y_pred == y).mean()) if len(y) > 1 else 0.0
    return acc

def median_heuristic_sigma(X):
    if len(X) <= 2:
        return 1.0
    n = len(X)
    idx = np.random.RandomState(42).choice(n, size=min(2000, n), replace=False)
    Xs = X[idx]
    S = cosine_similarity(Xs, Xs)
    D = np.sqrt(np.clip(2.0 * (1.0 - S), 0.0, None))
    iu = np.triu_indices_from(D, k=1)
    vals = D[iu]
    med = np.median(vals[~np.isnan(vals)]) if np.any(~np.isnan(vals)) else 1.0
    return float(max(med, 1e-6))

def mmd_rbf(train_emb, other_emb, sigma=None):
    A = torch.tensor(train_emb, dtype=torch.float32)
    B = torch.tensor(other_emb, dtype=torch.float32)
    A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    B = torch.nan_to_num(B, nan=0.0, posinf=0.0, neginf=0.0)
    A = normalize(A, p=2, dim=1)
    B = normalize(B, p=2, dim=1)
    if sigma is None:
        sigma = median_heuristic_sigma(np.concatenate([A.numpy(), B.numpy()], axis=0))
    gamma = 1.0 / (2.0 * (sigma ** 2))

    def kxx(X):
        S = (X @ X.T)
        D2 = 2.0 * (1.0 - S)
        K = torch.exp(-gamma * D2)
        n = K.shape[0]
        mask = ~torch.eye(n, dtype=bool)
        return K[mask].mean()

    def kxy(X, Y):
        S = (X @ Y.T)
        D2 = 2.0 * (1.0 - S)
        K = torch.exp(-gamma * D2)
        return K.mean()

    mmd2 = kxx(A) + kxx(B) - 2.0 * kxy(A, B)
    return float(torch.clamp(mmd2, min=0.0).cpu().numpy())

def silhouette_two_groups(train_emb, other_emb):
    X = np.concatenate([train_emb, other_emb], axis=0)
    y = np.concatenate([np.zeros(len(train_emb), dtype=int), np.ones(len(other_emb), dtype=int)])
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 2:
        return None
    X = torch.tensor(X, dtype=torch.float32)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = normalize(X, p=2, dim=1).cpu().numpy()
    try:
        s = silhouette_score(X, y, metric="cosine")
        return float(s)
    except Exception:
        return None

def kde_scores(train_scores, other_scores, num=100):
    x_min = float(np.min([train_scores.min(), other_scores.min()]))
    x_max = float(np.max([train_scores.max(), other_scores.max()]))
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        x_min, x_max = 0.0, 1.0
    xs = np.linspace(x_min, x_max, num=num).reshape(-1, 1)

    def silverman(a):
        a = a[np.isfinite(a)]
        n = max(len(a), 1)
        std = np.std(a) if n > 1 else 1.0
        return 1.06 * std * (n ** (-1 / 5)) if std > 0 else 0.1

    if KDE_AVAILABLE:
        bw_t = max(silverman(train_scores), 1e-3)
        bw_o = max(silverman(other_scores), 1e-3)
        k_t = KernelDensity(kernel="gaussian", bandwidth=bw_t).fit(train_scores.reshape(-1, 1))
        k_o = KernelDensity(kernel="gaussian", bandwidth=bw_o).fit(other_scores.reshape(-1, 1))
        dt = np.exp(k_t.score_samples(xs))
        do = np.exp(k_o.score_samples(xs))
        dt = (dt / np.trapz(dt, xs.squeeze())).flatten().tolist()
        do = (do / np.trapz(do, xs.squeeze())).flatten().tolist()
    else:
        hist_t, bins = np.histogram(train_scores, bins=num, range=(x_min, x_max), density=True)
        hist_o, _ = np.histogram(other_scores, bins=num, range=(x_min, x_max), density=True)
        xs = 0.5 * (bins[:-1] + bins[1:])
        dt = hist_t.tolist()
        do = hist_o.tolist()
    return [float(x) for x in xs.squeeze()], [float(v) for v in dt], [float(v) for v in do]

def leakage_metrics(train_texts, val_texts, other_sets, train_emb, val_emb, other_embs):
    def exact_overlap(a, b):
        sa, sb = set(a), set(b)
        inter = sa & sb
        return {"count": len(inter), "ratio_vs_b": len(inter) / max(1, len(sb))}

    def near_dup(ae, be, thr=0.995):
        sims = cosine_similarity(ae, be)
        mx = sims.max(axis=0)
        return {"rate": float((mx >= thr).mean()), "max": float(np.nanmax(mx)), "mean": float(np.nanmean(mx))}

    def centroid_cos(ae, be):
        ae = torch.tensor(ae, dtype=torch.float32)
        be = torch.tensor(be, dtype=torch.float32)
        ae = torch.nan_to_num(ae, nan=0.0, posinf=0.0, neginf=0.0)
        be = torch.nan_to_num(be, nan=0.0, posinf=0.0, neginf=0.0)
        ca = normalize(ae.mean(dim=0, keepdim=True), p=2, dim=1)
        cb = normalize(be.mean(dim=0, keepdim=True), p=2, dim=1)
        v = (ca @ cb.T).squeeze().cpu().numpy().item()
        return float(v)

    def pack(ae, be, at, bt):
        auc, tr_s, ot_s = auc_vs_train(ae, be)
        xs, dt, do = kde_scores(tr_s, ot_s, num=100)
        return {
            "exact_overlap": exact_overlap(at, bt),
            "near_duplicate": near_dup(ae, be),
            "centroid_cosine": centroid_cos(ae, be),
            "auc_vs_train": auc,
            "knn_loo_accuracy": knn_loo_accuracy(ae, be),
            "mmd_rbf": mmd_rbf(ae, be),
            "silhouette": silhouette_two_groups(ae, be),
            "kde_scores": {"x": xs, "density_train": dt, "density_other": do},
        }

    metrics = {"val": pack(train_emb, val_emb, train_texts, val_texts)}
    for name, emb in other_embs.items():
        metrics[name] = pack(train_emb, emb, train_texts, other_sets[name])
    return metrics

def format_label(base):
    mapping = {
        "train": "Train",
        "val": "Val",
        "test_id": "Test ID",
        "test_ood": "Test OOD",
        "test_mid": "Test MID",
        "test_real": "Test Real",
    }
    return mapping.get(base, base.title().replace("-", " "))

ensure_dir(base_out)
fig, axes = plt.subplots(5, 3, figsize=(18, 25))
axes = axes.flatten()
all_metrics = {}

real_df_global = read_real_tests(real_paths)

for idx, cat in enumerate(cat_names):
    cat_dir = os.path.join(base_out, cat)
    out_dir = cat_dir
    ensure_dir(out_dir)

    df_train, df_val = read_train_val_for_category(cat_dir)

    test_id_csv = os.path.join(cat_dir, "test_id.csv")
    test_ood_csv = os.path.join(cat_dir, "test_ood.csv")
    test_mid_csv = os.path.join(cat_dir, "test_mid.csv")

    def read_synth(p):
        if not os.path.exists(p):
            return pd.DataFrame({"text": []})
        df = read_csv_flex(p)
        if "summary" in df.columns:
            col = "summary"
        elif len(df.columns) >= 2:
            col = df.columns[-1]
        else:
            raise ValueError(f"No 'summary' column found in {p}")
        return df[[col]].rename(columns={col: "text"})

    df_id = read_synth(test_id_csv)
    df_ood = read_synth(test_ood_csv)
    df_mid = read_synth(test_mid_csv)

    df_real = real_df_global.copy()

    texts = {
        "train": df_train["text"].fillna("").tolist(),
        "val": df_val["text"].fillna("").tolist(),
        "test_id": df_id["text"].fillna("").tolist(),
        "test_ood": df_ood["text"].fillna("").tolist(),
        "test_mid": df_mid["text"].fillna("").tolist(),
        "test_real": df_real["text"].fillna("").tolist(),
    }

    emb = {k: embed_texts(v) for k, v in texts.items()}

    for k in emb:
        np.save(os.path.join(out_dir, f"{cat}_{k}_embeddings.npy"), emb[k])
        pd.DataFrame({"text": texts[k]}).to_csv(os.path.join(out_dir, f"{cat}_{k}_texts.csv"), index=False)

    auc_far_vals = {}
    for subset in ["val", "test_id", "test_ood", "test_mid", "test_real"]:
        if len(emb[subset]) > 0:
            auc_raw, _, _ = auc_vs_train(emb["train"], emb[subset])
            auc_far_vals[subset] = 1.0 - auc_raw
        else:
            auc_far_vals[subset] = float("nan")

    other_sets = {k: texts[k] for k in ["test_id", "test_ood", "test_mid", "test_real"] if k in texts}
    other_embs = {k: emb[k] for k in other_sets.keys()}
    all_metrics[cat] = leakage_metrics(
        texts["train"], texts["val"], other_sets, emb["train"], emb["val"], other_embs
    )

    concat_list = [emb["train"], emb["val"], emb["test_id"], emb["test_ood"], emb["test_mid"], emb["test_real"]]
    sizes = [len(x) for x in concat_list]
    concat = np.concatenate(concat_list, axis=0)

    if UMAP_AVAILABLE:
        reducer = UMAPCls(n_neighbors=15, min_dist=0.05, metric="cosine", random_state=42)
        X2d = reducer.fit_transform(concat)
    else:
        X2d = TSNE(n_components=2, metric="cosine", init="random", random_state=42, perplexity=30).fit_transform(concat)

    n_tr, n_va, n_id, n_ood, n_mid, n_real = sizes
    s0 = 0
    s1 = s0 + n_tr
    s2 = s1 + n_va
    s3 = s2 + n_id
    s4 = s3 + n_ood
    s5 = s4 + n_mid

    X2d_train = X2d[s0:s1]
    X2d_val = X2d[s1:s2]
    X2d_id = X2d[s2:s3]
    X2d_ood = X2d[s3:s4]
    X2d_mid = X2d[s4:s5]
    X2d_real = X2d[s5:]

    ax = axes[idx]

    def _plot_subset(arr, label_key, size_pts):
        if len(arr):
            ax.scatter(arr[:, 0], arr[:, 1],
                       s=size_pts, alpha=0.85,
                       label=label_key, c=palette[label_key],
                       marker=markers[label_key], linewidths=0.3, edgecolors="none")
            if DRAW_ELLIPSES:
                draw_confidence_ellipse(ax, arr, color=palette[label_key], n_std=ELLIPSE_STD, lw=1.25, alpha=0.15)

    _plot_subset(X2d_train, "train", 8)
    _plot_subset(X2d_val, "val", 10)
    _plot_subset(X2d_id, "test_id", 10)
    _plot_subset(X2d_ood, "test_ood", 10)
    _plot_subset(X2d_mid, "test_mid", 10)
    _plot_subset(X2d_real, "test_real", 10)

    title = cat.replace("_", " ").replace("-", " ")
    title = title[:1].upper() + title[1:]
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    handles, labels = ax.get_legend_handles_labels()
    new_labels = []
    for lab in labels:
        pretty = format_label(lab)
        if lab in auc_far_vals and not np.isnan(auc_far_vals[lab]):
            pretty = f"{pretty} (AUC_far {auc_far_vals[lab]:.3f})"
        new_labels.append(pretty)
    if handles:
        leg = ax.legend(handles, new_labels, fontsize=8, loc="upper right", frameon=True)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_alpha(0.95)

for j in range(len(cat_names), len(axes)):
    axes[j].axis("off")

plt.tight_layout()
fig_path_png = os.path.join(base_out, "all_categories_embeddings.png")
fig_path_pdf = os.path.join(base_out, "all_categories_embeddings.pdf")
plt.savefig(fig_path_png, dpi=300)
plt.savefig(fig_path_pdf)

metrics_path = os.path.join(base_out, "leakage_metrics.json")
with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(all_metrics, f, indent=2, ensure_ascii=False)

print("Saved figure:", fig_path_png)
print("Saved figure:", fig_path_pdf)
print("Saved metrics:", metrics_path)
