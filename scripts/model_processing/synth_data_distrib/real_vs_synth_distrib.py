#!/usr/bin/env python3
import os, json, argparse, time, math, numpy as np, pandas as pd
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", f"/tmp/mpl_{os.getpid()}")
os.environ.setdefault("XDG_CACHE_HOME", f"/tmp/xdg_cache_{os.getpid()}")
os.environ.setdefault("XDG_CONFIG_HOME", f"/tmp/xdg_config_{os.getpid()}")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

np.random.seed(42)

class HFEmbedder:
    def __init__(self, model_path:str, device:str="auto", max_len:int=2048, batch_size:int=10, attn_impl:str="eager"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch=torch
        local = os.path.isdir(model_path) or os.path.isfile(os.path.join(model_path,"config.json"))
        if local: os.environ["TRANSFORMERS_OFFLINE"]="1"
        use_cuda = torch.cuda.is_available() and device in ("auto","cuda")
        self.device="cuda" if use_cuda else "cpu"
        self.max_len=int(max_len)
        self.batch_size=int(batch_size)
        dtype = torch.float16 if use_cuda else torch.bfloat16
        self.tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True, local_files_only=local)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, low_cpu_mem_usage=True,
                local_files_only=local, dtype=dtype, attn_implementation=attn_impl
            ).to(self.device).eval()
        except TypeError:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True, low_cpu_mem_usage=True,
                local_files_only=local, torch_dtype=dtype, attn_implementation=attn_impl
            ).to(self.device).eval()
        if hasattr(self.model,"config"):
            self.model.config.use_cache=False
            self.model.config.output_hidden_states=False
        if self.tok.pad_token is None:
            self.tok.pad_token=self.tok.eos_token
            self.tok.pad_token_id=self.tok.eos_token_id
        try:
            torch.backends.cuda.matmul.allow_tf32=True
        except Exception:
            pass
    def encode(self, texts, log_every=1000):
        vs=[]; tok=self.tok; torch=self.torch
        lengths=[len(tok.tokenize((t or "")[:4096])) for t in texts]
        order=np.argsort(lengths); total=len(order); done=0; t0=time.time()
        with torch.inference_mode():
            for i0 in range(0,total,self.batch_size):
                idx=order[i0:i0+self.batch_size]
                batch=[(texts[j] or "") for j in idx]
                enc=tok(batch, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
                model_core=getattr(self.model,"model",self.model)
                out=model_core(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"], use_cache=False, output_hidden_states=False, return_dict=True)
                h=out.last_hidden_state
                m=enc["attention_mask"].unsqueeze(-1)
                v=(h*m).sum(dim=1)/m.sum(dim=1).clamp(min=1)
                v=torch.nn.functional.normalize(v, dim=-1)
                vs.append(v.detach().cpu().numpy())
                done+=len(batch)
                if log_every>0 and (done==total or done%log_every==0):
                    dt=time.time()-t0; rate=done/max(dt,1e-6); eta=(total-done)/max(rate,1e-6)
                    print(f"[emb] {done}/{total} ({done/total*100:.1f}%) ~{rate:.1f} it/s ETA {eta/60:.1f} min", flush=True)
                if self.device=="cuda" and ((i0//self.batch_size)%8==0):
                    torch.cuda.synchronize(); torch.cuda.empty_cache()
        V=np.concatenate(vs,0) if vs else np.zeros((0,1),dtype=np.float32)
        return np.nan_to_num(V,0.0,0.0,0.0).astype(np.float32)

def read_table(path):
    try:
        df=pd.read_csv(path, sep=None, engine="python", dtype=str)
    except Exception:
        for sep in ("\t",","):
            try:
                df=pd.read_csv(path, sep=sep, dtype=str); break
            except Exception:
                df=None
    if df is None: raise RuntimeError(f"cannot read {path}")
    df=df.fillna("")
    low={c.lower():c for c in df.columns}
    id_col = next((low[k] for k in ["run_accession","accession","run","id","rid","srr","sra","gsm","gse"] if k in low), None)
    if id_col is None:
        df["run_accession"]=[f"ROW_{i:06d}" for i in range(len(df))]
        id_col="run_accession"
    sum_col = next((low[k] for k in ["summary","text"] if k in low), None)
    if sum_col is None:
        if df.shape[1]==1: sum_col=df.columns[0]
        else:
            rest=[c for c in df.columns if c!=id_col]
            df["summary"]=df[rest].astype(str).agg(" ".join,axis=1).str.replace(r"\s+"," ",regex=True).str.strip()
            sum_col="summary"
    return df[[id_col,sum_col]].rename(columns={id_col:"run_accession", sum_col:"summary"}).fillna("")

def cosine_normed(A,B):
    na=np.linalg.norm(A,axis=1,keepdims=True)+1e-9
    nb=np.linalg.norm(B,axis=1,keepdims=True)+1e-9
    return (A/na)@(B/nb).T

def auc_center_two_sets_embeddings(Xa, Xb):
    if len(Xa)==0 or len(Xb)==0: return float("nan")
    mu=np.vstack([Xa,Xb]).mean(axis=0, keepdims=True)
    sa=cosine_normed(Xa,mu).ravel(); sb=cosine_normed(Xb,mu).ravel()
    scores=np.concatenate([sa,sb],0); labels=np.concatenate([np.ones(len(sa)), np.zeros(len(sb))],0)
    order=np.argsort(scores); ranks=np.empty_like(order,dtype=float); ranks[order]=np.arange(1,len(scores)+1)
    pos=labels==1; n_pos=pos.sum(); n_neg=len(scores)-n_pos
    if n_pos==0 or n_neg==0: return float("nan")
    auc=(ranks[pos].sum()-n_pos*(n_pos+1)/2)/(n_pos*n_neg)
    return float(auc)

def js_divergence(p, q, eps=1e-12):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p = p / max(p.sum(), eps); q = q / max(q.sum(), eps)
    m = 0.5*(p+q)
    a = p + eps; b = m + eps
    kl_pm = np.sum(a*np.log(a/b))
    a = q + eps; b = m + eps
    kl_qm = np.sum(a*np.log(a/b))
    return float(0.5*(kl_pm + kl_qm))

def assign_to_centers(V, centers):
    if centers is None or len(centers)==0 or len(V)==0: return np.zeros(len(V),dtype=int)
    C = centers/(np.linalg.norm(centers,axis=1,keepdims=True)+1e-9)
    X = V/(np.linalg.norm(V,axis=1,keepdims=True)+1e-9)
    return np.argmax(X @ C.T, axis=1).astype(int)

def build_centers_from_labels(V_pop, labels_all):
    k = int(labels_all.max())+1 if labels_all.size>0 else 1
    centers = []
    for c in range(k):
        sel = (labels_all==c)
        if not np.any(sel):
            centers.append(np.zeros(V_pop.shape[1], dtype=np.float32))
        else:
            centers.append(V_pop[sel].mean(axis=0))
    C = np.stack(centers, axis=0).astype(np.float32)
    C = np.nan_to_num(C,0.0,0.0,0.0)
    return C

def ecdf(x):
    xs=np.sort(x); ys=np.arange(1,len(xs)+1)/len(xs)
    return xs,ys

def density_with_overlays_tsne(out_png, V_pop, idx_sets, caps):
    rng=np.random.default_rng(42)
    N=min(caps["pop_bg"], len(V_pop))
    bg_idx=rng.choice(len(V_pop), N, replace=False) if len(V_pop)>N else np.arange(len(V_pop))
    points=[V_pop[bg_idx]]
    tags=[np.zeros(len(bg_idx),dtype=int)]
    tag_map={"train":1,"val":2,"id":3,"ood":4,"pop":5}
    for name in ["train","val","id","ood","pop"]:
        ids = idx_sets.get(name, np.array([],int))
        if ids.size==0: continue
        take=min(caps.get(name, 1000), ids.size)
        sel = ids if ids.size<=take else rng.choice(ids, take, replace=False)
        points.append(V_pop[sel])
        tags.append(np.full(len(sel), tag_map[name], dtype=int))
    Z = np.vstack(points)
    lab = np.concatenate(tags)
    tsne = TSNE(n_components=2, perplexity=min(30, max(5, Z.shape[0]//1000)), init="pca", learning_rate="auto", n_iter=1000, random_state=42, verbose=0)
    T = tsne.fit_transform(Z)
    plt.figure(figsize=(11,8))
    mask_bg = lab==0
    plt.hexbin(T[mask_bg,0], T[mask_bg,1], gridsize=80, bins='log')
    for name, code, mk, sz in [("train",1,'o',20),("val",2,'s',18),("id",3,'^',18),("ood",4,'x',22),("pop",5,'.',12)]:
        m = lab==code
        if np.any(m):
            plt.scatter(T[m,0], T[m,1], marker=mk, s=sz, alpha=0.9, label=name)
    plt.legend(); plt.title("t-SNE density + overlays")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def distance_curves(out_png, V_pop, idx_sets, ref="train"):
    if idx_sets.get(ref, np.array([],int)).size<5: return
    mu = V_pop[idx_sets[ref]].mean(axis=0, keepdims=True)
    mu = mu/(np.linalg.norm(mu)+1e-9)
    plt.figure(figsize=(10,6))
    for name in ["train","id","val","pop","ood"]:
        ids = idx_sets.get(name, np.array([],int))
        if ids.size==0: continue
        X = V_pop[ids]
        s = cosine_normed(X, mu).ravel()
        hist,edges=np.histogram(s, bins=60, range=(-0.2,1.0), density=True)
        mids=(edges[:-1]+edges[1:])/2
        plt.plot(mids, hist, label=name, linewidth=2)
    plt.xlabel(f"cosine to {ref} centroid"); plt.ylabel("density"); plt.title(f"Distance-to-{ref}-centroid distributions")
    plt.legend()
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def qq_plot(out_png, a, b, name_a, name_b):
    q=np.linspace(0.01,0.99,99)
    qa=np.quantile(a,q); qb=np.quantile(b,q)
    lim=[min(qa.min(),qb.min()), max(qa.max(),qb.max())]
    plt.figure(figsize=(4,4))
    plt.scatter(qa,qb,s=10)
    plt.plot(lim,lim,'k--',linewidth=1)
    plt.xlabel(f"{name_a} quantiles"); plt.ylabel(f"{name_b} quantiles")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def knn_two_sample_embeddings(X,Y,k=5):
    Z=np.vstack([X,Y]); t=np.array([0]*len(X)+[1]*len(Y))
    nn=NearestNeighbors(n_neighbors=min(k+1,len(Z))).fit(Z)
    neigh=nn.kneighbors(return_distance=False)[:,1:]
    votes=(t[neigh]==t[:,None]).mean()
    return float(votes)

def c2st_auc_embeddings(X,Y,folds=3):
    Z=np.vstack([X,Y]); y=np.array([0]*len(X)+[1]*len(Y))
    cv=StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    aucs=[]
    from sklearn.metrics import roc_auc_score
    for tr,te in cv.split(Z,y):
        clf=LogisticRegression(max_iter=500, class_weight="balanced", solver="saga", n_jobs=-1)
        clf.fit(Z[tr],y[tr])
        p=clf.predict_proba(Z[te])[:,1]
        aucs.append(float(roc_auc_score(y[te],p)))
    return float(np.mean(aucs)), float(np.std(aucs))

def join_by_id(real_path, synth_path):
    dr=read_table(real_path); ds=read_table(synth_path)
    m=dr.merge(ds, on="run_accession", suffixes=("_real","_synth"))
    if len(m)==0: raise RuntimeError("no overlap between real and synth IDs")
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--real_train", required=True)
    ap.add_argument("--real_val", required=True)
    ap.add_argument("--synth_train", required=True)
    ap.add_argument("--synth_val", required=True)
    ap.add_argument("--id_path", required=True)
    ap.add_argument("--pop_path", required=True)
    ap.add_argument("--ood_path", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--max_len", type=int, default=2000)
    ap.add_argument("--pair_cos_thresh", type=float, default=0.65)
    ap.add_argument("--ks_med_tol", type=float, default=0.03)
    ap.add_argument("--auc_delta_tol", type=float, default=0.05)
    ap.add_argument("--pop_bg_cap", type=int, default=12000)
    ap.add_argument("--cap_train", type=int, default=3000)
    ap.add_argument("--cap_val", type=int, default=1200)
    ap.add_argument("--cap_id", type=int, default=2000)
    ap.add_argument("--cap_pop", type=int, default=2000)
    ap.add_argument("--cap_ood", type=int, default=2000)
    args=ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    emb_path=os.path.join(args.emb_dir,"pool_embeddings.npy")
    ids_path=os.path.join(args.emb_dir,"pool_ids.json")
    labels_path=os.path.join(args.emb_dir,"pool_labels.npy")

    V_pop=np.load(emb_path).astype(np.float32); V_pop=np.nan_to_num(V_pop,0.0,0.0,0.0)
    ids_all=json.load(open(ids_path))
    labels_all=np.load(labels_path).astype(int) if os.path.exists(labels_path) else np.zeros(len(V_pop),int)
    k_clusters=int(labels_all.max())+1 if labels_all.size>0 else 1
    pos={rid:i for i,rid in enumerate(ids_all)}

    df_train_real=read_table(args.real_train)
    df_val_real=read_table(args.real_val)
    df_train_synth=read_table(args.synth_train)
    df_val_synth=read_table(args.synth_val)
    df_id=read_table(args.id_path)
    df_pop=read_table(args.pop_path)
    df_ood=read_table(args.ood_path)

    join_tr=df_train_real.merge(df_train_synth, on="run_accession", suffixes=("_real","_synth"))
    join_va=df_val_real.merge(df_val_synth, on="run_accession", suffixes=("_real","_synth"))
    ids_tr=[i for i in join_tr["run_accession"].tolist() if i in pos]
    ids_va=[i for i in join_va["run_accession"].tolist() if i in pos]

    idx_train_real=np.array([pos[i] for i in ids_tr], int)
    idx_val_real=np.array([pos[i] for i in ids_va], int)
    idx_id=np.array([pos[i] for i in df_id["run_accession"].tolist() if i in pos], int)
    idx_pop=np.array([pos[i] for i in df_pop["run_accession"].tolist() if i in pos], int)
    idx_ood=np.array([pos[i] for i in df_ood["run_accession"].tolist() if i in pos], int)

    emb=HFEmbedder(args.model, device=args.device, max_len=args.max_len, batch_size=args.batch_size)
    S_tr = emb.encode(join_tr.set_index("run_accession").loc[ids_tr]["summary_synth"].tolist())
    S_va = emb.encode(join_va.set_index("run_accession").loc[ids_va]["summary_synth"].tolist())
    R_tr = V_pop[idx_train_real]
    R_va = V_pop[idx_val_real]

    pair_cos=(R_tr/(np.linalg.norm(R_tr,axis=1,keepdims=True)+1e-9) * S_tr/(np.linalg.norm(S_tr,axis=1,keepdims=True)+1e-9)).sum(axis=1)
    pair_med=float(np.median(pair_cos)); pair_q1=float(np.quantile(pair_cos,0.25)); pair_q3=float(np.quantile(pair_cos,0.75))

    mu_pop = V_pop.mean(axis=0, keepdims=True); mu_pop = mu_pop/(np.linalg.norm(mu_pop)+1e-9)
    cosR = cosine_normed(R_tr, mu_pop).ravel()
    cosS = cosine_normed(S_tr, mu_pop).ravel()
    ks = ks_2samp(cosR, cosS, alternative="two-sided", method="auto")
    ks_p=float(ks.pvalue); ks_stat=float(ks.statistic); ks_med_diff=float(np.median(cosR)-np.median(cosS))

    if (len(labels_all)==len(V_pop)) and (k_clusters>1):
        centers_raw = build_centers_from_labels(V_pop, labels_all)
        lab_real = labels_all[idx_train_real]
        lab_synth = assign_to_centers(S_tr, centers_raw)
        hr=np.bincount(lab_real, minlength=centers_raw.shape[0]).astype(float)
        hs=np.bincount(lab_synth, minlength=centers_raw.shape[0]).astype(float)
        jsd_clusters = js_divergence(hr, hs)
    else:
        jsd_clusters = float("nan")

    def aucs_block_ref(Xref, blocks):
        out={}
        for name, X in blocks.items():
            out[name]=auc_center_two_sets_embeddings(Xref, X)
        return out

    def cap_idx(a, cap):
        if a.size<=cap: return a
        rng=np.random.default_rng(42)
        return rng.choice(a, cap, replace=False)

    idx_caps_real={
        "train":cap_idx(idx_train_real, args.cap_train),
        "val":cap_idx(idx_val_real, args.cap_val),
        "id":cap_idx(idx_id, args.cap_id),
        "pop":cap_idx(idx_pop, args.cap_pop),
        "ood":cap_idx(idx_ood, args.cap_ood),
    }

    AUC_real = aucs_block_ref(V_pop[idx_caps_real["train"]], {
        "val":V_pop[idx_caps_real["val"]],
        "id":V_pop[idx_caps_real["id"]],
        "pop":V_pop[idx_caps_real["pop"]],
        "ood":V_pop[idx_caps_real["ood"]],
    })

    S_caps_val = S_va[:min(len(S_va), args.cap_val)]
    S_caps_tr  = S_tr[:min(len(S_tr), args.cap_train)]
    AUC_synth = aucs_block_ref(S_caps_tr, {
        "val":S_caps_val if len(S_caps_val)>0 else S_va,
        "id":V_pop[idx_caps_real["id"]],
        "pop":V_pop[idx_caps_real["pop"]],
        "ood":V_pop[idx_caps_real["ood"]],
    })

    delta_auc = {k:(AUC_synth.get(k, np.nan) - AUC_real.get(k, np.nan)) for k in ["val","id","pop","ood"]}

    pass_pair = bool(pair_med >= args.pair_cos_thresh)
    pass_ks   = bool(ks_p>0.05 and abs(ks_med_diff)<=args.ks_med_tol)
    pass_auc  = bool(all(abs(delta_auc[k])<=args.auc_delta_tol for k in delta_auc))
    verdict   = bool(pass_pair and pass_auc)

    print(f"pairs={len(pair_cos)} pair_cos_median={pair_med:.3f} q1={pair_q1:.3f} q3={pair_q3:.3f}", flush=True)
    print(f"KS_pop_cos p={ks_p:.4f} stat={ks_stat:.3f} med_diff={ks_med_diff:.3f}", flush=True)
    print("AUC real train vs {val,id,pop,ood}:", {k:round(v,3) for k,v in AUC_real.items()}, flush=True)
    print("AUC synth train vs {val,id,pop,ood}:", {k:round(v,3) for k,v in AUC_synth.items()}, flush=True)
    print("ΔAUC synth-real:", {k:round(v,3) for k,v in delta_auc.items()}, flush=True)
    if not math.isnan(jsd_clusters): print(f"JSD cluster(real vs synth train)={jsd_clusters:.3f}", flush=True)
    print("Verdict:", "REPRESENTATIVE" if verdict else "NOT REPRESENTATIVE", flush=True)

    out_json=os.path.join(args.out_dir,"representativity_report.json")
    json.dump({
        "n_pairs": int(len(pair_cos)),
        "pair_cos_median": pair_med,
        "pair_cos_q1": pair_q1,
        "pair_cos_q3": pair_q3,
        "ks_pop_cos_p": ks_p,
        "ks_pop_cos_stat": ks_stat,
        "ks_pop_cos_med_diff": ks_med_diff,
        "auc_real": AUC_real,
        "auc_synth": AUC_synth,
        "delta_auc": delta_auc,
        "jsd_clusters": jsd_clusters,
        "thresholds": {
            "pair_cos_thresh": args.pair_cos_thresh,
            "ks_med_tol": args.ks_med_tol,
            "auc_delta_tol": args.auc_delta_tol
        },
        "verdict": "REPRESENTATIVE" if verdict else "NOT REPRESENTATIVE"
    }, open(out_json,"w"), indent=2)

    caps={"pop_bg": args.pop_bg_cap, "train":args.cap_train, "val":args.cap_val, "id":args.cap_id, "pop":args.cap_pop, "ood":args.cap_ood}

    idx_sets_real = {k:v for k,v in idx_caps_real.items()}
    density_with_overlays_tsne(os.path.join(args.out_dir,"tsne_density_real.png"), V_pop, idx_sets_real, caps)
    distance_curves(os.path.join(args.out_dir,"distance_curves_real.png"), V_pop, idx_sets_real, ref="train")

    cos_tr = cosine_normed(V_pop[idx_caps_real["train"]], V_pop.mean(axis=0, keepdims=True)/(np.linalg.norm(V_pop.mean(axis=0))+1e-9)).ravel()
    cos_id = cosine_normed(V_pop[idx_caps_real["id"]], V_pop.mean(axis=0, keepdims=True)/(np.linalg.norm(V_pop.mean(axis=0))+1e-9)).ravel()
    qq_plot(os.path.join(args.out_dir,"qq_train_vs_id_real.png"), cos_tr, cos_id, "train_real", "id")

    idx_sets_synth = {
        "train": np.arange(len(S_caps_tr)),
        "val":   np.arange(len(S_caps_val)),
        "id":    idx_caps_real["id"],
        "pop":   idx_caps_real["pop"],
        "ood":   idx_caps_real["ood"],
    }

    def density_with_overlays_tsne_mix(out_png, S_tr, S_va, V_pop, idx_real_sets, caps):
        rng=np.random.default_rng(42)
        N=min(caps["pop_bg"], len(V_pop))
        bg_idx=rng.choice(len(V_pop), N, replace=False) if len(V_pop)>N else np.arange(len(V_pop))
        pts=[V_pop[bg_idx]]; lab=[np.zeros(len(bg_idx),int)]
        tag=1
        if len(S_tr)>0:
            take=min(caps.get("train",1000), len(S_tr))
            pts.append(S_tr[:take]); lab.append(np.full(min(take,len(S_tr)), tag, int)); tag+=1
        if len(S_va)>0:
            take=min(caps.get("val",800), len(S_va))
            pts.append(S_va[:take]); lab.append(np.full(min(take,len(S_va)), tag, int)); tag+=1
        for name in ["id","ood","pop"]:
            ids = idx_real_sets.get(name, np.array([],int))
            if ids.size==0: continue
            take=min(caps.get(name, 800), ids.size)
            sel = ids if ids.size<=take else rng.choice(ids, take, replace=False)
            pts.append(V_pop[sel]); lab.append(np.full(len(sel), tag, int)); tag+=1
        Z=np.vstack(pts); L=np.concatenate(lab)
        tsne=TSNE(n_components=2, perplexity=min(30, max(5, Z.shape[0]//1000)), init="pca", learning_rate="auto", n_iter=1000, random_state=42, verbose=0)
        T=tsne.fit_transform(Z)
        plt.figure(figsize=(11,8))
        mask_bg = L==0
        plt.hexbin(T[mask_bg,0], T[mask_bg,1], gridsize=80, bins='log')
        names=["train_synth","val_synth","id","ood","pop"]; code=1
        marks=['o','s','^','x','.']; sizes=[20,18,18,22,12]
        for nm,mk,sz in zip(names,marks,sizes):
            m=L==code
            if np.any(m): plt.scatter(T[m,0], T[m,1], marker=mk, s=sz, alpha=0.9, label=nm)
            code+=1
        plt.legend(); plt.title("t-SNE density + overlays (synth)")
        plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

    density_with_overlays_tsne_mix(os.path.join(args.out_dir,"tsne_density_synth.png"), S_caps_tr, S_caps_val, V_pop, idx_caps_real, caps)

    def distance_curves_mix(out_png, S_tr, S_va, V_pop, idx_real_sets):
        if len(S_tr)<5: return
        mu = S_tr.mean(axis=0, keepdims=True); mu=mu/(np.linalg.norm(mu)+1e-9)
        plt.figure(figsize=(10,6))
        for name, X in [("train_synth", S_tr), ("val_synth", S_va if len(S_va)>0 else S_tr[:1])]:
            s = cosine_normed(X, mu).ravel()
            hist,edges=np.histogram(s, bins=60, range=(-0.2,1.0), density=True)
            mids=(edges[:-1]+edges[1:])/2
            plt.plot(mids, hist, label=name, linewidth=2)
        for name in ["id","pop","ood"]:
            ids = idx_real_sets.get(name, np.array([],int))
            if ids.size==0: continue
            X=V_pop[ids]; s=cosine_normed(X, mu).ravel()
            hist,edges=np.histogram(s, bins=60, range=(-0.2,1.0), density=True)
            mids=(edges[:-1]+edges[1:])/2
            plt.plot(mids, hist, label=name, linewidth=2)
        plt.xlabel("cosine to train_synth centroid"); plt.ylabel("density"); plt.title("Distance-to-train_synth-centroid distributions")
        plt.legend(); plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

    distance_curves_mix(os.path.join(args.out_dir,"distance_curves_synth.png"), S_caps_tr, S_caps_val, V_pop, idx_caps_real)

    cosS_tr = cosine_normed(S_caps_tr, V_pop.mean(axis=0, keepdims=True)/(np.linalg.norm(V_pop.mean(axis=0))+1e-9)).ravel()
    cos_id  = cosine_normed(V_pop[idx_caps_real["id"]], V_pop.mean(axis=0, keepdims=True)/(np.linalg.norm(V_pop.mean(axis=0))+1e-9)).ravel()
    qq_plot(os.path.join(args.out_dir,"qq_trainSynth_vs_id.png"), cosS_tr, cos_id, "train_synth", "id")

    rep_tsv=os.path.join(args.out_dir,"representativity_report.tsv")
    rows=[["metric","value"],
          ["pairs", len(pair_cos)],
          ["pair_cos_median", pair_med],
          ["pair_cos_q1", pair_q1],
          ["pair_cos_q3", pair_q3],
          ["ks_pop_cos_p", ks_p],
          ["ks_pop_cos_stat", ks_stat],
          ["ks_pop_cos_med_diff", ks_med_diff],
          ["jsd_clusters", jsd_clusters],
          ["auc_real_val", AUC_real["val"]],
          ["auc_real_id",  AUC_real["id"]],
          ["auc_real_pop", AUC_real["pop"]],
          ["auc_real_ood", AUC_real["ood"]],
          ["auc_synth_val", AUC_synth["val"]],
          ["auc_synth_id",  AUC_synth["id"]],
          ["auc_synth_pop", AUC_synth["pop"]],
          ["auc_synth_ood", AUC_synth["ood"]],
          ["delta_auc_val", delta_auc["val"]],
          ["delta_auc_id",  delta_auc["id"]],
          ["delta_auc_pop", delta_auc["pop"]],
          ["delta_auc_ood", delta_auc["ood"]],
          ["verdict", "REPRESENTATIVE" if verdict else "NOT REPRESENTATIVE"]]
    pd.DataFrame(rows, columns=["metric","value"]).to_csv(rep_tsv, sep="\t", index=False)
    print("Wrote:", rep_tsv, flush=True)

if __name__=="__main__":
    main()
