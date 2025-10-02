import argparse, os, json, re, random
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

random.seed(42); np.random.seed(42)
SPEED = "fast"
WIN = 1800
STEP = 1200
LLM_TRIALS = 1

CATS = [
    "library_selection","sequencing_source","biopsy_site","biopsy_type",
    "cell_line","cell_type","organ","disease","is_cancer",
    "treatment","treatment_time","response","age","sex","ethnicity"
]

CAT2UP = {c: c.upper() for c in CATS}
LIBSEL_LABELS = ["polyA","inverse rRNA","hybrid selection","small RNA","other"]
ACCESSION_PAT = r"\b(?:ERR|ERS|SRR|SRX|SRP|SRA|SAM|SAME[A-Z]*|PRJ[EDN][A-Z]*|ERP|E-?MTAB|GSE|GSM)\w*\b"
GENERIC_CODE_PAT = r"\b[A-Za-z]{2,}\d{2,}[A-Za-z0-9-]*\b|\b\d{2,}[A-Za-z][A-Za-z0-9-]*\b|\b[A-Z]{2,}(?:-\d{1,4})\b"
MIXED_ALNUM_PAT = r"\b[A-Za-z]+\d+[A-Za-z0-9_.-]*\b|\b\d+[A-Za-z_.-]+\b"

SECTION_SPLIT = [
    (r"^\s*title[: ]", "title"),
    (r"^\s*abstract[: ]", "abstract"),
    (r"^\s*summary[: ]", "summary"),
    (r"^\s*introduction[: ]", "intro"),
    (r"^\s*methods?[: ]", "methods"),
    (r"^\s*materials?[: ]", "methods"),
    (r"^\s*results?[: ]", "results"),
    (r"^\s*discussion[: ]", "discussion"),
    (r"^\s*conclusion[: ]", "conclusion"),
    (r"^\s*availability|^\s*accession|^\s*data\s+availability", "availability"),
]

SECTION_WEIGHTS = {
    "title": 1.0, "abstract": 1.0, "summary": 1.0, "intro": 0.8,
    "methods": 0.6, "results": 1.0, "discussion": 0.8, "conclusion": 1.0,
    "availability": 0.2, "other": 0.8
}

class LLMInferencer:
    def infer(self, system:str, user:str, do_sample:bool, temperature:float, top_p:float, max_new_tokens:int)->str:
        raise NotImplementedError

class HFChatInferencer(LLMInferencer):
    def __init__(self, model_path:str, device:str="auto"):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.torch=torch
        is_local = os.path.isdir(model_path) or os.path.isfile(os.path.join(model_path,"config.json"))
        if is_local: os.environ["TRANSFORMERS_OFFLINE"]="1"
        use_cuda = torch.cuda.is_available() and device in ("auto","cuda")
        dtype = torch.float16 if use_cuda else torch.float32
        self.device="cuda" if use_cuda else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True, local_files_only=is_local)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=is_local).to(self.device).eval()
        if self.tok.pad_token is None:
            self.tok.pad_token=self.tok.eos_token; self.tok.pad_token_id=self.tok.eos_token_id
        self.has_chat_template = hasattr(self.tok,"apply_chat_template")
    def _build(self, system,user):
        if self.has_chat_template:
            return self.tok.apply_chat_template([
                {"role":"system","content":system},
                {"role":"user","content":user}
            ], tokenize=False, add_generation_prompt=True)
        return f"<s>[INST] <<SYS>>\n{system}\n<</SYS>>\n{user} [/INST]"
    def infer(self, system,user,do_sample,temperature,top_p,max_new_tokens):
        prompt=self._build(system,user)
        inputs=self.tok(prompt, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out=self.model.generate(
                **inputs,
                do_sample=bool(do_sample),
                temperature=float(temperature),
                top_p=float(top_p),
                max_new_tokens=int(max_new_tokens),
                pad_token_id=self.tok.pad_token_id,
                eos_token_id=self.tok.eos_token_id,
            )
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

class GGUFChatInferencer(LLMInferencer):
    def __init__(self, gguf_path:str, device:str="auto", n_ctx:int=8192, verbose:bool=False):
        from llama_cpp import Llama
        want_gpu = device in ("auto","cuda")
        n_gpu_layers = -1 if want_gpu else 0
        self.llm=Llama(
            model_path=gguf_path,
            n_ctx=n_ctx,
            n_threads=max(1,(os.cpu_count() or 8)),
            n_gpu_layers=n_gpu_layers,
            logits_all=False,
            use_mmap=True,
            use_mlock=False,
            verbose=verbose,
        )
    def infer(self, system,user,do_sample,temperature,top_p,max_new_tokens):
        params=dict(temperature=float(temperature if do_sample else 0.0), top_p=float(top_p), max_tokens=int(max_new_tokens))
        resp=self.llm.create_chat_completion(messages=[{"role":"system","content":system},{"role":"user","content":user}], **params)
        return resp["choices"][0]["message"]["content"].strip()

class HFEmbedder:
    def __init__(self, model_path:str, device:str="auto", max_len:int=512, batch_size:int=24):
        import torch
        from transformers import AutoTokenizer, AutoModel
        self.torch = torch
        self.max_len = max_len
        self.batch_size = batch_size
        is_local = os.path.isdir(model_path) or os.path.isfile(os.path.join(model_path, "config.json"))
        if is_local: os.environ["TRANSFORMERS_OFFLINE"] = "1"
        use_cuda = torch.cuda.is_available() and device in ("auto", "cuda")
        self.device = "cuda" if use_cuda else "cpu"
        dtype = torch.float16 if use_cuda else torch.float32
        self.tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True, local_files_only=is_local)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, low_cpu_mem_usage=True, trust_remote_code=True, local_files_only=is_local).to(self.device).eval()
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token; self.tok.pad_token_id = self.tok.eos_token_id
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
        except Exception:
            pass
    def encode(self, texts:List[str])->np.ndarray:
        if not texts: return np.zeros((0, 1), dtype=np.float32)
        chunks = []
        bs = max(1, int(self.batch_size))
        with self.torch.inference_mode():
            for i in range(0, len(texts), bs):
                batch = texts[i:i+bs]
                enc = self.tok(batch, padding=True, truncation=True, max_length=self.max_len, return_tensors="pt").to(self.device)
                out = self.model(**enc)
                h = out.last_hidden_state
                mask = enc["attention_mask"].unsqueeze(-1)
                summed = (h * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1)
                v = summed / counts
                v = self.torch.nn.functional.normalize(v, dim=-1)
                chunks.append(v.detach().cpu().numpy())
        return np.concatenate(chunks, axis=0)

def cosine(a:np.ndarray,b:np.ndarray)->np.ndarray:
    if b.ndim==1: b=b[None,:]
    a=a/(np.linalg.norm(a,axis=1,keepdims=True)+1e-9)
    b=b/(np.linalg.norm(b,axis=1,keepdims=True)+1e-9)
    return (a*b).sum(axis=1)

def mann_whitney_auc(scores:np.ndarray, labels:np.ndarray)->float:
    order=np.argsort(scores); ranks=np.empty_like(order,dtype=float); ranks[order]=np.arange(1,len(scores)+1)
    pos=labels==1; n_pos=pos.sum(); n_neg=len(scores)-n_pos
    if n_pos==0 or n_neg==0: return float("nan")
    auc=(ranks[pos].sum()-n_pos*(n_pos+1)/2)/(n_pos*n_neg)
    return float(auc)

def read_real_file(path:str)->pd.DataFrame:
    if not os.path.exists(path): raise FileNotFoundError(path)
    base, ext = os.path.splitext(path)
    if ext.lower() in [".tsv",".csv"]:
        sep = "\t" if ext.lower()==".tsv" else ","
        df = pd.read_csv(path, sep=sep, dtype=str)
        cols = {c.lower():c for c in df.columns}
        sm = cols.get("summary")
        if sm is None:
            if df.shape[1]==1: df = df.rename(columns={df.columns[0]:"summary"})
            else: raise ValueError("TSV/CSV must contain a 'summary' column or a single column with the text.")
        else:
            df = df[[sm]].rename(columns={sm:"summary"})
    else:
        rows=[]
        with open(path,"r",encoding="utf-8") as f:
            for line in f:
                t=line.strip()
                if t: rows.append({"summary":t})
        df=pd.DataFrame(rows)
    df["run_accession"]= [f"real_{i:06d}" for i in range(len(df))]
    return df[["run_accession","summary"]]


SPAN_SYS = (
"You perform span tagging.\n"
"Task: Given TEXT, find every substring that expresses any of these categories and output JSON with 'spans'.\n"
"Output strictly JSON: {\"spans\":[{\"start\":int,\"end\":int,\"category\":str}, ...]}.\n"
"General rules:\n"
"- Work only from TEXT; offsets are over original TEXT; do not hallucinate.\n"
"- For closed categories, classify but do NOT output the value (we only need the category tag).\n"
"- Categories:\n"
"  library_selection: pick among ['polyA','inverse rRNA','hybrid selection','small RNA','other'] using context.\n"
"  sequencing_source: one of ['spatial','bulk','single cell'].\n"
"  biopsy_site: organ/body part/fluid WHERE TISSUE WAS SAMPLED; xenograft mention here.\n"
"  biopsy_type: exactly one of ['metastasis','blood','primary'] with explicit evidence.\n"
"  cell_line: standardized line names; if primary tissue, still tag the span as 'cell_line' only if a line is present, otherwise ignore.\n"
"  cell_type: biological cell type; deduce if possible.\n"
"  organ: organ/tissue concerned (not the disease).\n"
"  disease: disease or 'healthy/normal/adjacent'.\n"
"  treatment: drug/surgery/event; include planned interventions.\n"
"  treatment_time: time/phase relative to treatment.\n"
"  response: one of ['no treatment','unknown','stable','progressive','success'].\n"
"  age, sex, ethnicity, is_cancer.\n"
"Hints for library_selection classification: PolyA/oligo-dT/TruSeq.mRNA/... => polyA; rRNA depletion variants => inverse rRNA; hybrid/exome/GeoMX => hybrid selection; small-RNA/size fraction => small RNA; else other.\n"
)

LIBSEL_SYNS = {
    "polyA": [r"poly\s*[-\.]?\s*a\b", r"oligo\s*[-\.]?\s*d\s*?t\b", r"truseq[^a-zA-Z0-9]{0,3}mrna", r"stranded[^a-z]*mrna", r"smarter[^a-z]*mrna", r"standard[^a-z]*mrna"],
    "inverse rRNA": [r"ribo\s*(minus|dep|zero|erase|gone|cop|-?dep|-?mi)\b", r"deplet\w+\s+ribo", r"remove\w*\s+ribo", r"truseq\s*stranded\s*total", r"truseq\s*total", r"smarter\s*(stranded\s*)?total"],
    "hybrid selection": [r"hybrid\s*selection", r"exon\s*capture", r"exome\s*capture", r"rna\s*exome", r"\bgeomx\b"],
    "small RNA": [r"\bsmall\s*rna\b", r"size\s*fraction", r"truseq\s*small"],
}
SPATIAL_HINTS = [r"\bvisium\b", r"\bgeomx\b", r"\bcosmx\b", r"slide[-\s]?seq", r"merfish", r"seqfish", r"st\s*omics", r"spatial\s*trans"]
SC_HINTS = [r"\b10x\b", r"\bsn[-\s]?rna[-\s]?seq\b", r"\bscrna[-\s]?seq\b", r"\bsmart[-\s]?seq\b", r"drop[-\s]?seq", r"single[-\s]?cell"]

BIOPSY_TYPE_META = {
    "metastasis": [r"\bmetasta\w+\b", r"\bmet\.\b"],
    "blood": [r"\bblood\b", r"\bplasma\b", r"\bserum\b", r"\bpbmc\b"],
}

AGE_PAT = r"\b(\d{1,2})\s*(?:yo|years?\s*old|ans?)\b|\b(adult|child|teen(ager)?|senior|elderly)\b"
SEX_PAT = r"\b(male|female|man|woman|f\b|m\b|masculin|féminin)\b"
ETH_PAT = r"\b(caucasian|white|asian|black|african|latino|hispanic|japanese|chinese|european)\b"

DISEASE_HINTS = [
    r"\b\w+oma\b", r"\b\w+itis\b", r"\b\w+emia\b", r"\bcarcinoma\b", r"\bmelanoma\b", r"\bneoplasm\b", r"\bgbm\b", r"\bhealthy\b", r"\bnormal\b", r"\badjacent\b"
]
CANCER_CUE = r"\b(cancer|carcinoma|sarcoma|lymphoma|leukemia|tumou?r|neoplasm|glioblastoma|melanoma)\b"

NEGATION_CUES = r"\b(no|without|absence of|not|none|negative for)\b"

TREATMENT_PATTERNS = [r"treated with\s+[^.;,()]+", r"therapy\s+with\s+[^.;,()]+", r"administered\s+[^.;,()]+", r"received\s+[^.;,()]+", r"on\s+treatment"]
BIOPSY_PATTERNS = [r"sample from\s+[^.;,()]+", r"biopsy\s+of\s+[^.;,()]+", r"tissue from\s+[^.;,()]+", r"tumou?r from\s+[^.;,()]+"]

def split_sections(text:str)->List[Tuple[str,str]]:
    lines = text.splitlines()
    out=[]; cur_name="other"; cur=[]
    for ln in lines:
        name=None
        for pat, nm in SECTION_SPLIT:
            if re.match(pat, ln, flags=re.I):
                name=nm; break
        if name is not None:
            if cur:
                out.append((cur_name, "\n".join(cur)))
                cur=[]
            cur_name=name; cur.append(ln)
        else:
            cur.append(ln)
    if cur:
        out.append((cur_name, "\n".join(cur)))
    if not out:
        out=[("other", text)]
    return out

ACRO_PAT = re.compile(r"\b([A-Z]{2,})\b\s*(?:\(|=)\s*([^()=]{3,80})\)?")
def resolve_acronyms(text:str)->Dict[str,str]:
    mapp={}
    for m in ACRO_PAT.finditer(text):
        acro = m.group(1).strip()
        long = m.group(2).strip()
        if len(acro)>=2 and len(long.split())>=2:
            mapp[acro]=long
    return mapp

def filter_spans_by_negation(text:str, spans:List[Dict[str,Any]], window:int=40)->List[Dict[str,Any]]:
    neg = [m.span() for m in re.finditer(NEGATION_CUES, text, flags=re.I)]
    if not neg: return spans
    kept=[]
    for s in spans:
        a,b=int(s['start']), int(s['end'])
        bad=False
        for (x,y) in neg:
            if 0 <= a - y <= window:
                bad=True; break
        if not bad:
            kept.append(s)
    return kept

def pretag_rules(text:str)->List[Dict[str,Any]]:
    spans=[]; t=text
    for pats in LIBSEL_SYNS.values():
        for p in pats:
            for m in re.finditer(p, t, flags=re.I):
                spans.append({"start":m.start(),"end":m.end(),"category":"library_selection"})

    if any(re.search(p,t,re.I) for p in SPATIAL_HINTS):
        for m in re.finditer("|".join(SPATIAL_HINTS), t, flags=re.I):
            spans.append({"start":m.start(),"end":m.end(),"category":"sequencing_source"})
    if any(re.search(p,t,re.I) for p in SC_HINTS):
        for m in re.finditer("|".join(SC_HINTS), t, flags=re.I):
            spans.append({"start":m.start(),"end":m.end(),"category":"sequencing_source"})

    for lab, pats in BIOPSY_TYPE_META.items():
        for p in pats:
            for m in re.finditer(p, t, flags=re.I):
                spans.append({"start":m.start(),"end":m.end(),"category":"biopsy_type"})

    for p in DISEASE_HINTS:
        for m in re.finditer(p, t, flags=re.I):
            spans.append({"start":m.start(),"end":m.end(),"category":"disease"})

    for p in TREATMENT_PATTERNS:
        for m in re.finditer(p, t, flags=re.I):
            spans.append({"start":m.start(),"end":m.end(),"category":"treatment"})
    for p in BIOPSY_PATTERNS:
        for m in re.finditer(p, t, flags=re.I):
            spans.append({"start":m.start(),"end":m.end(),"category":"biopsy_site"})

    for m in re.finditer(AGE_PAT, t, flags=re.I):      spans.append({"start":m.start(),"end":m.end(),"category":"age"})
    for m in re.finditer(SEX_PAT, t, flags=re.I):      spans.append({"start":m.start(),"end":m.end(),"category":"sex"})
    for m in re.finditer(ETH_PAT, t, flags=re.I):      spans.append({"start":m.start(),"end":m.end(),"category":"ethnicity"})

    for m in re.finditer(CANCER_CUE, t, flags=re.I):   spans.append({"start":m.start(),"end":m.end(),"category":"is_cancer"})

    return spans

def _coerce_spans(spans:List[Dict[str,Any]])->List[Dict[str,Any]]:
    out=[]
    for s in spans or []:
        try:
            a = int(s.get("start"))
            b = int(s.get("end"))
            if b <= a: continue
            cat = str(s.get("category",""))
            if not cat: continue
            out.append({"start":a, "end":b, "category":cat})
        except Exception:
            continue
    return out

def _llm_tag_once(llm:LLMInferencer, text:str)->List[Dict[str,Any]]:
    if llm is None: return []
    user = json.dumps({"TEXT": text}, ensure_ascii=False)
    trials=[dict(do_sample=False, temperature=0.0, top_p=1.0, max_new_tokens=400)]
    if LLM_TRIALS >= 2:
        trials.append(dict(do_sample=True,  temperature=0.4, top_p=0.95, max_new_tokens=400))
    for tr in trials:
        raw=llm.infer(SPAN_SYS, user, **tr) or ""
        try:
            js=json.loads(raw)
            spans=js.get("spans", [])
            ok=_coerce_spans(spans)
            if ok:
                return ok
        except Exception:
            continue
    return []

def tag_spans_with_llm(llm:LLMInferencer, text:str)->List[Dict[str,Any]]:
    if len(text) <= WIN:
        return _llm_tag_once(llm, text)
    spans=[]; win=WIN; step=STEP
    for a in range(0, len(text), step):
        b=min(len(text), a+win)
        sub=text[a:b]
        sub_spans=_llm_tag_once(llm, sub)
        for s in sub_spans:
            s={**s}
            s["start"]=int(s["start"]) + a
            s["end"]=int(s["end"]) + a
            spans.append(s)
        if b==len(text): break
    return spans

def resolve_overlaps(spans:List[Dict[str,Any]])->List[Dict[str,Any]]:
    def pri(s):
        return 0 if s.get("category")!="CHUNK" else 1
    spans=_coerce_spans(spans)
    spans=sorted(spans, key=lambda s: (int(s['start']), -(int(s['end'])-int(s['start'])), pri(s)))
    out=[]; last_end=-1
    for s in spans:
        a=int(s['start']); b=int(s['end'])
        if a>=last_end:
            out.append(s); last_end=b
        else:
            prev=out[-1]
            pa=int(prev['start']); pb=int(prev['end'])
            if prev.get("category")=="CHUNK" and s.get("category")!="CHUNK" and (b-a)>=(pb-pa):
                out[-1]=s; last_end=b
            else:
                continue
    return out

def enrich_with_regex(text:str, spans:List[Dict[str,Any]])->List[Dict[str,Any]]:
    taken=[(int(s['start']), int(s['end'])) for s in _coerce_spans(spans)]
    def overlaps(a,b,c,d): return max(a,c) < min(b,d)
    for pat in (ACCESSION_PAT, GENERIC_CODE_PAT, MIXED_ALNUM_PAT):
        for m in re.finditer(pat, text):
            a,b=m.span()
            if any(overlaps(a,b,x,y) for x,y in taken):
                continue
            spans.append({"start":a,"end":b,"category":"CHUNK"})
            taken.append((a,b))
    return spans

def build_synthesis(text:str, spans:List[Dict[str,Any]])->str:
    spans=sorted(_coerce_spans(spans), key=lambda s: s['start'])
    pieces=[]; cur=0; idx=0
    for s in spans:
        a,b=int(s['start']), int(s['end'])
        if a>cur:
            pieces.append(f"[{idx}]"+text[cur:a]); idx+=1
        cat=s.get("category")
        token = CAT2UP.get(cat, cat.upper() if cat else "")
        if token:
            pieces.append(token)
        cur=b
    if cur < len(text):
        pieces.append(f"[{idx}]"+text[cur:])
    syn=" ".join(re.sub(r"\s+"," ",p.strip()) for p in pieces if p.strip())
    syn=re.sub(r"\s+([,;:.!?])", r"\1", syn)
    return syn.strip()

def weight_spans_by_section(text:str, spans:List[Dict[str,Any]])->List[Dict[str,Any]]:
    secs = split_sections(text)
    idx_map=[]
    pos=0
    for name, content in secs:
        L=len(content)
        idx_map.append((pos, pos+L, SECTION_WEIGHTS.get(name, SECTION_WEIGHTS["other"])))
        pos+=L+1
    kept=[]
    for s in spans:
        a=int(s['start'])
        ok=True
        for (x,y,w) in idx_map:
            if x<=a<y:
                if w<0.5: ok=False
                break
        if ok: kept.append(s)
    return kept

def apply_acronyms_guard(text:str, spans:List[Dict[str,Any]])->List[Dict[str,Any]]:
    acr = resolve_acronyms(text)
    if not acr: return spans
    seen=set()
    out=[]
    for s in spans:
        frag=text[int(s['start']):int(s['end'])]
        key=(s['category'], frag.upper())
        if key in seen:
            continue
        seen.add(key); out.append(s)
    return out

def tag_spans_with_llm_and_rules(llm, text:str)->List[Dict[str,Any]]:
    spans_rules = pretag_rules(text)
    spans_llm   = tag_spans_with_llm(llm, text)
    spans = spans_rules + spans_llm
    spans = weight_spans_by_section(text, spans)
    spans = filter_spans_by_negation(text, spans)
    spans = apply_acronyms_guard(text, spans)
    spans = resolve_overlaps(spans)
    return spans

def token_set_fuzzy(a,b):
    A=set(re.findall(r"[A-Za-z0-9+-]+", (a or "").lower()))
    B=set(re.findall(r"[A-Za-z0-9+-]+", (b or "").lower()))
    if not A or not B: return 0.0
    inter=len(A&B); return 2*inter/(len(A)+len(B))

def kendall_tau_order(chunks_real, chunks_syn):
    try:
        idx_r=[int(x.strip("[]")) for x in chunks_real]
        idx_s=[int(x.strip("[]")) for x in chunks_syn]
        if idx_r==idx_s: return 1.0
        n=len(idx_r)
        if n<2: return 1.0
        inversions=0
        for i in range(n):
            for j in range(i+1,n):
                a=idx_s.index(idx_r[i]); b=idx_s.index(idx_r[j])
                if a>b: inversions+=1
        total=n*(n-1)/2
        return 1.0 - inversions/total
    except Exception:
        return 0.0

def spans_to_category_set(spans:List[Dict[str,Any]])->set:
    return {s.get("category") for s in spans if s.get("category") and s.get("category")!="CHUNK"}

def compute_representation_scores_per_sample(df:pd.DataFrame, spans_all:List[List[Dict[str,Any]]], emb_model:str, device:str)->pd.DataFrame:
    enc=HFEmbedder(emb_model, device=device)

    texts_seed = df["seed_summary"].tolist()
    texts_syn  = df["summary"].tolist()
    V_seed = enc.encode(texts_seed) if texts_seed else np.zeros((0,1))
    V_syn  = enc.encode(texts_syn)  if texts_syn  else np.zeros((0,1))

    sims=[]; covs=[]; ords=[]; rmss=[]

    syn_like_real=[]
    for i, r in enumerate(df.itertuples(index=False), start=0):
        text=r.seed_summary
        spans = spans_all[i]
        if not spans:
            syn_real = "[0]"+text
        else:
            spans_sorted = sorted(_coerce_spans(spans), key=lambda s: s['start'])
            pieces=[]; cur=0; idx=0
            for s in spans_sorted:
                a,b=int(s['start']), int(s['end'])
                if a>cur:
                    pieces.append(f"[{idx}]"+text[cur:a]); idx+=1
                pieces.append(text[a:b])
                cur=b
            if cur < len(text):
                pieces.append(f"[{idx}]"+text[cur:])
            syn_real = " ".join(re.sub(r"\s+"," ",p.strip()) for p in pieces if p.strip())
            syn_real = re.sub(r"\s+([,;:.!?])", r"\1", syn_real).strip()
        syn_like_real.append(syn_real)

    for i in range(len(df)):
        vg = V_seed[i]
        vp = V_syn[i]
        sim = float(np.dot(vg, vp))
        sims.append(sim)

        gold_cats = spans_to_category_set(spans_all[i])
        pred_cats = set()
        for c in CATS:
            if CAT2UP[c] in texts_syn[i]:
                pred_cats.add(c)
        TP = len(gold_cats & pred_cats)
        P = len(pred_cats) if pred_cats else 1
        R = len(gold_cats) if gold_cats else 1
        prec = TP / P
        rec  = TP / R
        cov  = (2*prec*rec/(prec+rec)) if (prec+rec)>0 else 0.0
        covs.append(cov)

        r_chunks = re.findall(r"\[\d+\]", syn_like_real[i])
        s_chunks = re.findall(r"\[\d+\]", texts_syn[i])
        ord_score = kendall_tau_order(r_chunks, s_chunks)
        ords.append(ord_score)

        rms = 0.45*sim + 0.35*cov + 0.20*ord_score
        rmss.append(rms)

    out = pd.DataFrame({"sim":sims,"cov":covs,"ord":ords,"RMS":rmss})
    return out

def process_split(df:pd.DataFrame, llm:LLMInferencer)->Tuple[pd.DataFrame, List[List[Dict[str,Any]]]]:
    rows=[]; all_spans=[]
    for i, r in enumerate(df.itertuples(index=False), start=0):
        seed=r.summary
        spans=tag_spans_with_llm_and_rules(llm, seed)
        spans=enrich_with_regex(seed, spans)
        spans=resolve_overlaps(spans)
        masked=build_synthesis(seed, spans)
        rows.append({"run_accession":r.run_accession, "seed_summary":seed, "summary":masked})
        all_spans.append(spans)
        if (i+1)%50==0 or (i+1)==len(df):
            print(f"processed {i+1}/{len(df)}", flush=True)
    return pd.DataFrame(rows), all_spans

def compute_auc_and_plot(df_train:pd.DataFrame, df_val:pd.DataFrame, emb_model:str, device:str, out_dir:str)->Dict[str,float]:
    import matplotlib.pyplot as plt
    emb=HFEmbedder(emb_model, device=device)
    def clean(xs): return [re.sub(r"\s+", " ", x).strip() for x in xs]
    s_tr=clean(df_train["seed_summary"].tolist()); p_tr=clean(df_train["summary"].tolist())
    s_va=clean(df_val["seed_summary"].tolist()); p_va=clean(df_val["summary"].tolist())
    V_st=emb.encode(s_tr) if s_tr else np.zeros((0,1)); V_pt=emb.encode(p_tr) if p_tr else np.zeros((0,1))
    V_sv=emb.encode(s_va) if s_va else np.zeros((0,1)); V_pv=emb.encode(p_va) if p_va else np.zeros((0,1))
    def auc_split(Vo,Vp):
        if Vo.size==0 or Vp.size==0: return float('nan')
        mu=Vo.mean(axis=0, keepdims=True)
        s_o=cosine(Vo, mu[0]); s_p=cosine(Vp, mu[0])
        scores=np.concatenate([s_o,s_p],0)
        labels=np.concatenate([np.ones(len(s_o)), np.zeros(len(s_p))],0)
        return mann_whitney_auc(scores, labels)
    auc_train=auc_split(V_st, V_pt); auc_val=auc_split(V_sv, V_pv)
    try:
        from sklearn.decomposition import PCA
        pca=PCA(n_components=2)
        X=np.vstack([V_st, V_pt, V_sv, V_pv]) if V_st.size and V_pt.size and V_sv.size and V_pv.size else np.zeros((0,2))
        if X.size:
            Z=pca.fit_transform(X)
            n1=len(V_st); n2=len(V_pt); n3=len(V_sv); n4=len(V_pv)
            a=0; b=n1; c=b+n2; d=c+n3; e=d+n4
            plt.figure()
            plt.scatter(Z[a:b,0], Z[a:b,1], label='train_original')
            plt.scatter(Z[b:c,0], Z[b:c,1], label='train_masked')
            plt.scatter(Z[c:d,0], Z[c:d,1], label='val_original')
            plt.scatter(Z[d:e,0], Z[d:e,1], label='val_masked')
            plt.legend()
            plot_path=os.path.join(out_dir, "embeddings_pca.png")
            plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    except Exception as e:
        print(f"PCA plot skipped: {e}")
    return {"auc_train": float(auc_train), "auc_val": float(auc_val)}

def build_llm(path:str, device:str, n_ctx:int=16384)->LLMInferencer:
    return GGUFChatInferencer(path, device=device, n_ctx=n_ctx, verbose=False) if path.endswith(".gguf") else HFChatInferencer(path, device=device)

def build_llm_auto(path:str, device:str, n_ctx:int):
    return build_llm(path, device=device, n_ctx=n_ctx)

def main():
    global SPEED, WIN, STEP, LLM_TRIALS
    ap=argparse.ArgumentParser()
    ap.add_argument("--real_train", required=True)
    ap.add_argument("--real_val", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--model", required=True)
    ap.add_argument("--emb_model", required=True)
    ap.add_argument("--n_ctx", type=int, default=16384)
    ap.add_argument("--rms_accept", type=float, default=0.80)
    ap.add_argument("--sim_min", type=float, default=0.75)
    ap.add_argument("--cov_min", type=float, default=0.70)
    ap.add_argument("--speed", choices=["fast","balanced"], default="fast")
    ap.add_argument("--win", type=int, default=1800)
    ap.add_argument("--step", type=int, default=1200)
    ap.add_argument("--llm_trials", type=int, default=1, help="1=greedy seul, 2=+sampling")
    args=ap.parse_args()

    SPEED = args.speed
    WIN = max(600, int(args.win)) if SPEED=="balanced" else max(2000, int(args.win))
    STEP = max(400, int(args.step)) if SPEED=="balanced" else max(1800, int(args.step))
    LLM_TRIALS = max(1, int(args.llm_trials)) if SPEED=="balanced" else 1

    os.makedirs(args.out_dir, exist_ok=True)

    train_df=read_real_file(args.real_train)
    val_df=read_real_file(args.real_val)

    llm=build_llm_auto(args.model, device=args.device, n_ctx=args.n_ctx)

    df_train, spans_train=process_split(train_df, llm)
    df_val, spans_val=process_split(val_df, llm)

    out_train=os.path.join(args.out_dir, "train_masked.tsv")
    out_val=os.path.join(args.out_dir, "val_masked.tsv")
    df_train.to_csv(out_train, sep='\t', index=False)
    df_val.to_csv(out_val, sep='\t', index=False)
    metrics=compute_auc_and_plot(df_train, df_val, args.emb_model, args.device, args.out_dir)
    train_scores = compute_representation_scores_per_sample(df_train[["seed_summary","summary"]], spans_train, args.emb_model, args.device)
    val_scores   = compute_representation_scores_per_sample(df_val[["seed_summary","summary"]], spans_val, args.emb_model, args.device)

    train_scores_path=os.path.join(args.out_dir, "per_sample_scores_train.tsv")
    val_scores_path  =os.path.join(args.out_dir, "per_sample_scores_val.tsv")
    train_scores.to_csv(train_scores_path, sep='\t', index=False)
    val_scores.to_csv(val_scores_path, sep='\t', index=False)

    train_accept = float((((train_scores["RMS"] >= args.rms_accept) & (train_scores["sim"] >= args.sim_min) & (train_scores["cov"] >= args.cov_min)).mean())) if len(train_scores) else float('nan')
    val_accept   = float((((val_scores["RMS"]   >= args.rms_accept) & (val_scores["sim"]   >= args.sim_min) & (val_scores["cov"]   >= args.cov_min)).mean()))   if len(val_scores)   else float('nan')

    meta={
        "files":{
            "train_tsv":out_train,
            "val_tsv":out_val,
            "plot_pca":os.path.join(args.out_dir,"embeddings_pca.png"),
            "per_sample_scores_train": train_scores_path,
            "per_sample_scores_val": val_scores_path,
        },
        "auc_train":metrics["auc_train"],
        "auc_val":metrics["auc_val"],
        "rms_thresholds": {"rms_accept": args.rms_accept, "sim_min": args.sim_min, "cov_min": args.cov_min},
        "accept_rate_train": train_accept,
        "accept_rate_val": val_accept,
    }
    json.dump(meta, open(os.path.join(args.out_dir, "masking_meta.json"), "w"), indent=2)
    print(f"AUC train={metrics['auc_train']:.4f} | AUC val={metrics['auc_val']:.4f}", flush=True)
    print(f"Accept(train)={train_accept:.3f} | Accept(val)={val_accept:.3f}", flush=True)

if __name__=="__main__":
    main()
