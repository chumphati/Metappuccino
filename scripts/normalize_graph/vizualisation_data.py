########################################################################################################################
#IMPORT
import argparse, csv, math, hashlib, re, os
from pathlib import Path
from collections import defaultdict, Counter
from statistics import mean, median
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.image as mpimg
import numpy as np

NLL_PREFIX = "nll_"
PPL_PREFIX = "ppl_"
DEFAULT_METRIC_PREFIX = PPL_PREFIX

MISSING_TOKENS = {"", "none", "null", "na", "n/a", "nan", "unk", "unknown",
                  "not applicable", "not_applicable"}
IMPORTANT_FIELDS = [
    "disease", "treatment", "organ", "biopsy_site", "biopsy_type",
    "cell_type", "cell_line", "sex", "ethnicity",
    "instrument_platform", "library_strategy", "sequencing_source",
    "organ_uberon_code", "bs_uberon_code", "do_code",
    "response", "is_cancer", "treatment_time"
]

########################################################################################################################
#FUNCTIONS
def safe_float(x):
    s = str(x).strip() if x is not None else ""
    if not s or s.lower() in MISSING_TOKENS:
        return None
    try:
        return float(s)
    except ValueError:
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        return float(m.group()) if m else None

def normalize_token(x):
    s = str(x).strip() if x is not None else ""
    return "" if s.lower() in MISSING_TOKENS else s

def bucket_nll(v):
    if v is None or math.isnan(v): return "missing", "missing"
    if 0 <= v <= 1:  return "green",  "confident"
    if 1 < v <= 2:   return "orange", "medium"
    if v > 2:        return "red",    "uncertain"
    return "missing", "missing"

def bucket_ppl(v):
    if v is None or math.isnan(v): return "missing", "missing"
    if 0 <= v <= 3:  return "green",  "confident"
    if 3 < v <= 6:   return "orange", "medium"
    if v > 6:        return "red",    "uncertain"
    return "missing", "missing"

def ensure_dir(p): p.mkdir(parents=True, exist_ok=True)

def slugify(text, max_len=80):
    text = re.sub(r"[^\w\-]+", "_", text)
    text = re.sub(r"_{2,}", "_", text).strip("_")
    if len(text) <= max_len:
        return text or "unnamed"
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{text[:max_len-9]}_{h}"

def save_hist(values, title, outpath, xlabel="Value"):
    if not values: return
    vmin, vmax = min(values), max(values)
    bins = 50 if vmin == vmax else max(20, min(80, int( (vmax - vmin) / max(1e-6, (vmax - vmin)/50 ) )))
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=bins)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

def save_bar(labels, counts, title, outpath, rotation=45):
    if not labels: return
    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts)
    plt.title(title); plt.ylabel("Count")
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

def save_stacked(bucket_counts, title, outpath, rotation=45):
    if not bucket_counts: return
    labels = list(bucket_counts.keys())
    g = [bucket_counts[k].get("green",   0) for k in labels]
    o = [bucket_counts[k].get("orange",  0) for k in labels]
    r = [bucket_counts[k].get("red",     0) for k in labels]
    m = [bucket_counts[k].get("missing", 0) for k in labels]
    plt.figure(figsize=(11, 7))
    plt.bar(labels, g)
    plt.bar(labels, o, bottom=g)
    br  = [a+b for a, b in zip(g, o)]
    plt.bar(labels, r, bottom=br)
    br2 = [a+b for a, b in zip(br, r)]
    plt.bar(labels, m, bottom=br2)
    plt.title(title); plt.ylabel("Count")
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

def numeric_hist(values, title, outpath, log=False, bins=50, xlabel="Value"):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals: return
    plt.figure(figsize=(8, 5))
    plt.hist(vals, bins=bins)
    if log: plt.yscale("log")
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("Count")
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

def metric_summary_boxplot(values_by_col, label_map, ylabel, out_png):
    cols = list(values_by_col.keys())
    data = [values_by_col[c] for c in cols]
    if not data: return
    plt.figure(figsize=(max(6, len(cols)*0.6), 6))
    plt.boxplot(data, labels=[label_map(c) for c in cols], vert=True, showmeans=True)
    plt.xticks(rotation=60, ha="right"); plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

def metric_stats_bar(stats_summary, ylabel, out_png):
    if not stats_summary: return
    bases = list(stats_summary.keys())
    means = [stats_summary[b]["mean"] for b in bases]
    plt.figure(figsize=(max(6, len(bases)*0.6), 5))
    plt.bar(bases, means)
    plt.xticks(rotation=60, ha="right"); plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150); plt.close()

def combine_category_images(cat_slug, cat_dir):
    imgs, files = [], []
    for p in sorted(cat_dir.glob(f"{cat_slug}_*.png")):
        if p.name.endswith("_combined.png"):
            continue
        imgs.append(mpimg.imread(p))
        files.append(p)
    if not imgs:
        return

    fig, axs = plt.subplots(len(imgs), 1, figsize=(max(i.shape[1] for i in imgs)/100, 5*len(imgs)))
    if len(imgs) == 1:
        axs = [axs]
    for ax, img in zip(axs, imgs):
        ax.imshow(img); ax.axis("off")
    plt.tight_layout()
    out = cat_dir / f"{cat_slug}_combined.png"
    plt.savefig(out, dpi=150)
    plt.close()

    for p in files:
        try:
            p.unlink()
        except OSError:
            pass

def write_html(header, rows, base2metric_idx, out_html, max_rows=None):
    non_metric_cols = [c for c in header if not c.startswith(NLL_PREFIX) and not c.startswith(PPL_PREFIX)]
    idxs = [header.index(c) for c in non_metric_cols]
    css = ("<style>body{font-family:Arial,Helvetica,sans-serif;}table{border-collapse:collapse;width:100%;}"
           "th,td{border:1px solid #ddd;padding:4px 6px;font-size:12px;}th{position:sticky;top:0;background:#f2f2f2;z-index=2;}"
           "tr:nth-child(even){background:#fafafa;}.conf-green{background:#d9f2d9;}.conf-orange{background:#ffe5cc;}"
           ".conf-red{background:#ffd6d6;}.conf-missing{background:#eeeeee;}.nowrap{white-space:nowrap;}</style>")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write("<!DOCTYPE html><html><head><meta charset='utf-8'><title>Colored table</title>"
                + css + "</head><body><table><thead><tr>")
        for col in non_metric_cols: f.write(f"<th class='nowrap'>{col}</th>")
        f.write("</tr></thead><tbody>")
        for i, row in enumerate(rows):
            if max_rows is not None and i >= max_rows: break
            f.write("<tr>")
            for idx, col in zip(idxs, non_metric_cols):
                val = row[idx]; cls = ""; title = ""
                if col in base2metric_idx:
                    v = safe_float(row[base2metric_idx[col]])
                    if DEFAULT_METRIC_PREFIX == PPL_PREFIX:
                        bucket, _ = bucket_ppl(v)
                        title = f" title='ppl={v if v is not None else 'NA'}'"
                    else:
                        bucket, _ = bucket_nll(v)
                        title = f" title='nll={v if v is not None else 'NA'}'"
                    cls = f" conf-{bucket}"
                f.write(f"<td class='nowrap{cls}'{title}>{'' if val is None else val}</td>")
            f.write("</tr>")
        f.write("</tbody></table></body></html>")

########################################################################################################################
#MAIN
def main():
    p = argparse.ArgumentParser(description="Build graphs from metadata inference")
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--topn", type=int, default=10)
    p.add_argument("--html_max_rows", type=int)
    p.add_argument("--base_path", type=str, required=True, help="Base path to Metappuccino")
    p.add_argument("--verbose", action="store_true", help="Verbose output")
    args = p.parse_args()

    in_path  = Path(args.input)
    out_dir  = Path(args.outdir)
    base_path = args.base_path
    FLAG_FILE = os.path.join(base_path, "STEP4_2.flag")
    VERBOSE = args.verbose
    vprint = print if VERBOSE else (lambda *a, **k: None)
    for d in ("model_confidence", "categories", "tables", "stats"):
        ensure_dir(out_dir/d)

    with open(in_path, "r", encoding="utf-8") as fh:
        first_line = fh.readline()
    delimiter = "\t" if "\t" in first_line else ","

    with open(in_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        header = next(reader)
        col2idx = {c: i for i, c in enumerate(header)}

        nll_cols = [c for c in header if c.startswith(NLL_PREFIX)]
        ppl_cols = [c for c in header if c.startswith(PPL_PREFIX)]
        base_cols = [c for c in header if not c.startswith(NLL_PREFIX) and not c.startswith(PPL_PREFIX)]

        if DEFAULT_METRIC_PREFIX == PPL_PREFIX:
            metric_cols = ppl_cols
            base2metric_idx = {c[len(PPL_PREFIX):]: col2idx[c]
                               for c in ppl_cols if c[len(PPL_PREFIX):] in col2idx}
        else:
            metric_cols = nll_cols
            base2metric_idx = {c[len(NLL_PREFIX):]: col2idx[c]
                               for c in nll_cols if c[len(NLL_PREFIX):] in col2idx}

        rows = []
        nll_values_by_col = defaultdict(list)
        ppl_values_by_col = defaultdict(list)
        bucket_counts_ppl = defaultdict(Counter)
        bucket_counts_nll = defaultdict(Counter)
        value_counts  = defaultdict(Counter)
        pair_counts   = Counter()
        numeric       = {"age": [], "base_count": []}

        for row in reader:
            rows.append(row)

            for nc in nll_cols:
                v = safe_float(row[col2idx[nc]])
                if v is not None:
                    nll_values_by_col[nc].append(v)
            for pc in ppl_cols:
                v = safe_float(row[col2idx[pc]])
                if v is not None:
                    ppl_values_by_col[pc].append(v)

            for base in base_cols:
                val = normalize_token(row[col2idx[base]])
                value_counts[base][val] += 1

                ppl_col = f"{PPL_PREFIX}{base}"
                if ppl_col in col2idx:
                    vp = safe_float(row[col2idx[ppl_col]])
                    b, _ = bucket_ppl(vp)
                    bucket_counts_ppl[base][b] += 1

                nll_col = f"{NLL_PREFIX}{base}"
                if nll_col in col2idx:
                    vn = safe_float(row[col2idx[nll_col]])
                    b, _ = bucket_nll(vn)
                    bucket_counts_nll[base][b] += 1

            d = normalize_token(row[col2idx.get("disease", -1)]) if "disease" in col2idx else ""
            t = normalize_token(row[col2idx.get("treatment", -1)]) if "treatment" in col2idx else ""
            if d or t: pair_counts[f"{d}|{t}"] += 1
            if "age" in col2idx: numeric["age"].append(safe_float(row[col2idx["age"]]))
            if "base_count" in col2idx: numeric["base_count"].append(safe_float(row[col2idx["base_count"]]))

    dir_ent, dir_cat = out_dir/"model_confidence", out_dir/"categories"
    for nc, vals in nll_values_by_col.items():
        base = nc[len(NLL_PREFIX):]
        slug = slugify(base)
        save_hist(vals, f"NLL distribution: {base}", dir_ent/f"nll_{slug}.png", xlabel="NLL")

    for pc, vals in ppl_values_by_col.items():
        base = pc[len(PPL_PREFIX):]
        slug = slugify(base)
        save_hist(vals, f"PPL distribution: {base}", dir_ent/f"ppl_{slug}.png", xlabel="Perplexity")

    def write_stats(values_by_col, prefix, out_txt):
        stats_summary = {}
        with open(out_txt, "w") as f:
            for ec, vals in sorted(values_by_col.items()):
                base = ec[len(prefix):]
                vals = sorted(vals)
                if not vals: continue
                m, med = mean(vals), median(vals)
                p10, p90 = vals[int(.1*(len(vals)-1))], vals[int(.9*(len(vals)-1))]
                stats_summary[base] = {"mean": m, "median": med, "p10": p10, "p90": p90, "n": len(vals)}
                f.write(f"{base}\tn={len(vals)}\tmean={m:.4f}\tmedian={med:.4f}\tp10={p10:.4f}\tp90={p90:.4f}\n")
        return stats_summary

    stats_nll = write_stats(nll_values_by_col, NLL_PREFIX, out_dir/"stats"/"nll_stats.txt")
    stats_ppl = write_stats(ppl_values_by_col, PPL_PREFIX, out_dir/"stats"/"ppl_stats.txt")
    metric_summary_boxplot(nll_values_by_col, lambda c: c[len(NLL_PREFIX):], "NLL", out_dir/"stats"/"nll_summary.png")
    metric_summary_boxplot(ppl_values_by_col, lambda c: c[len(PPL_PREFIX):], "Perplexity", out_dir/"stats"/"ppl_summary.png")
    metric_stats_bar(stats_nll, "Mean NLL", out_dir/"stats"/"nll_stats_means.png")
    metric_stats_bar(stats_ppl, "Mean Perplexity", out_dir/"stats"/"ppl_stats_means.png")

    topn = max(5, args.topn)
    for base, cnt in bucket_counts_ppl.items():
        slug = slugify(base)
        save_bar(["confident","medium","uncertain","missing"],
                 [cnt.get("green",0), cnt.get("orange",0),
                  cnt.get("red",0),   cnt.get("missing",0)],
                 f"Confidence levels (PPL): {base}",
                 dir_cat/f"{slug}_ppl.png", rotation=0)

    for base, cnt in bucket_counts_nll.items():
        slug = slugify(base)
        save_bar(["confident","medium","uncertain","missing"],
                 [cnt.get("green",0), cnt.get("orange",0),
                  cnt.get("red",0),   cnt.get("missing",0)],
                 f"Confidence levels (NLL): {base}",
                 dir_cat/f"{slug}_nll.png", rotation=0)

    for base, cnt in value_counts.items():
        items  = sorted(cnt.items(), key=lambda kv: kv[1], reverse=True)[:topn]
        labels = ["(empty)" if k=="" else k for k,_ in items]
        counts = [v for _, v in items]
        slug   = slugify(base)
        save_bar(labels, counts, f"Top {topn} values: {base}",
                 dir_cat/f"{slug}_top{topn}.png")

    if numeric["age"]:
        numeric_hist(numeric["age"], "Age distribution",
                     dir_cat/"age_hist.png", bins=40, xlabel="Age")
    if numeric["base_count"]:
        numeric_hist(numeric["base_count"], "base_count distribution",
                     dir_cat/"base_count_hist_lin.png", bins=60, xlabel="base_count")
        numeric_hist(numeric["base_count"], "base_count distribution (log Y)",
                     dir_cat/"base_count_hist_log.png", log=True, bins=60, xlabel="base_count")

    if pair_counts:
        items  = sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True)[:topn]
        labels = ["(empty)" if k=="" else k.replace("|"," | ") for k,_ in items]
        counts = [v for _, v in items]
        save_bar(labels, counts, f"Top {topn} disease|treatment pairs",
                 dir_cat/"disease_treatment_pairs_top{topn}.png")

    write_html(header, rows, base2metric_idx,
               out_dir/"tables"/"completed_metadata_confidence.html", args.html_max_rows)

    for base in base_cols:
        combine_category_images(slugify(base), dir_cat)

    with open(out_dir/"README_VISUALISATION.txt", "w") as f:
        f.write("Metappuccino visualization outputs\n")
        f.write("model_confidence/: confidence information\n")
        f.write("categories/: one combined file per category showing confidence and top values\n")
        f.write("tables/table_colored.html: full table with cells coloured by entropy\n")
        f.write("stats/: numeric stats summary per category\n")

    open(FLAG_FILE, 'w').close()

if __name__ == "__main__":
    main()
