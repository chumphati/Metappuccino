########################################################################################################################
# IMPORT LIB
import pandas as pd
import re
import glob
import os
import argparse

########################################################################################################################
#PATHS
parser = argparse.ArgumentParser(description="Fetch information with Cellosaurus")
parser.add_argument("--base_path", type=str, required=True, help="Base path to Metappuccino")
args = parser.parse_args()

base_path = args.base_path
xml_dir = os.path.join(base_path, "metadata")
cell_df_file = os.path.join(base_path, "CELLOSAURUS_CLEAN.csv")
dot_file = os.path.join(base_path, "DOT_TABLE_CLEAN.csv")
uberon_file = os.path.join(base_path, "UBERON_TABLE_CLEAN.csv")
input_file = os.path.join(base_path, "metadata_sra.txt")
output_file_df = os.path.join(base_path, "database_metadata_curated.csv")
FLAG_FILE = os.path.join(base_path, "STEP2_1.flag")

# xml_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/store/Metappuccino_store/results_ft_patron_semi/tmp/metadata"
# cell_df_file = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/CELLOSAURUS_CLEAN.csv"
# input_file = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/store/Metappuccino_store/results_ft_patron_semi/tmp/metadata_sra.txt"
# output_file_df = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/store/Metappuccino_store/results_ft_final_templates/output_metadata_curated.tsv"

invalid_entries = {"unknown", "not applicable", "missing", "n/a", "na", "none", ""}

########################################################################################################################

def load_syn(csv, sc, nc, cc):
    df = pd.read_csv(csv, dtype=str, on_bad_lines='skip').fillna('')
    sd, cd = {}, {}
    for _, r in df.iterrows():
        name = r[nc].strip()
        if not name: continue
        syns = [s.strip() for s in r[sc].split(';')] if r[sc] else []
        for s in syns + [name]:
            sd[s.lower()] = name
        if cc and r[cc].strip():
            cd[name] = r[cc].strip()
    return sd, cd

def extract_tag(ctx, tag):
    m = re.search(rf"<TAG>{tag}</TAG><VALUE>([^<]+)</VALUE>", ctx)
    return m.group(1).strip() if m else ""

def normalize(v, d):
    return d.get(v.strip().lower(), v.strip()) if isinstance(v, str) and v.strip() else ""

def extract_libsel(ctx):
    if re.search(r"poly[.\s-]?a|oligo[.\s-]?d[.\s-]?t|truseq[.\s-]?mrna|smarter[.\s-]?mrna|polyadenylated|polyadenylation|nebnext poly[.\s-]?a|magnetic poly[.\s-]?a",ctx,re.IGNORECASE):return"polyA"
    if re.search(r"ribo[.\s-]?(minus|dep|zero)|deplete[.\s-]?ribosom|rrna[.\s-]?deple|rrna[.\s-]?minus|rrna removal|rrna depletion kit|ribominus|ribodepletion",ctx,re.IGNORECASE):return"inverse rRNA"
    if re.search(r"hybrid[.\s-]?selection|exon[.\s-]?capture|exome[.\s-]?capture|rna[.\s-]?exome|bait|capture-based|targeted transcriptome|myBaits|SureSelect",ctx,re.IGNORECASE):return"hybrid selection"
    if re.search(r"truseq[.\s-]?small|size[.\s-]?fraction|small[.\s-]?rna|mirna|micro[.\s-]?rna|small RNA-Seq|smallRNA|miRNA-Seq|small RNA library",ctx,re.IGNORECASE):return"small RNA"
    return""

def extract_src(ctx):
    if re.search(r"(spatial.{0,10}(transcriptomic|sequencing|rna.?seq|visium|geomx|merfish|hdst|slide[-\s]?seq|nanostring|st-seq))|(visium|geomx|merfish|hdst|slide[-\s]?seq|nanostring|spatial transcriptomics)",ctx,re.IGNORECASE):return"spatial"
    if re.search(r"single[\s\-]?cell|scrna|10x|drop[-\s]?seq|smart[-\s]?seq|fluidigm|inDrop|seq[-\s]?well|c1 platform|smartseq|smart-seq|microwell|scRNA(-seq)?|single[-\s]?nucleus",ctx,re.IGNORECASE):return"single cell"
    if re.search(r"bulk(\s*rna)?(\s*-?\s*seq)?|conventional rna[\s\-]?seq|whole[\s\-]?transcriptome|wta|standard rna[\s\-]?seq|total rna[\s\-]?seq|rna[\s\-]?seq",ctx,re.IGNORECASE):return"bulk"
    return""


disease_syn, disease_code = load_syn(
    dot_file,
    "synonym", "name", "code_dot"
)
organ_syn, organ_code = load_syn(
    uberon_file,
    "synonym", "name", "code_uberon"
)

cell_df = pd.read_csv(cell_df_file, dtype=str, on_bad_lines='skip').fillna('')
cell_syn = {}
for _, r in cell_df.iterrows():
    name = r['name'].strip()
    if not name: continue
    syns = [s.strip() for s in r['synonym'].split(';')] if r['synonym'] else []
    for s in syns + [name]:
        cell_syn[s.lower()] = name

cols = [
    "run_accession", "library_selection", "sequencing_source",
    "biopsy_site", "bs_uberon_code", "biopsy_type", "cell_line", "cell_type",
    "organ", "organ_uberon_code", "disease", "do_code",
    "treatment", "treatment_time", "response",
    "age", "sex", "ethnicity", "localization"
]

stopwords = r"\b(?:for|to|and|in|with|via|on|of|the)\b"

out = []
for path in glob.glob(os.path.join(xml_dir, "*_metadata.xml")):
    run = os.path.basename(path).split("_metadata.xml")[0]
    ctx = open(path, "r", encoding="utf-8", errors="ignore").read()
    o = {c: "" for c in cols}
    o["run_accession"] = run
    raw_cl = extract_tag(ctx, "cell_line")
    if raw_cl and raw_cl.lower() not in invalid_entries:
        mapped = cell_syn.get(raw_cl.lower(), raw_cl)
        o["cell_line"] = mapped
        if mapped in cell_df["name"].values:
            row = cell_df[cell_df["name"] == mapped].iloc[0]
            for f in ["disease", "age", "sex", "ethnicity", "localization", "biopsy_type", "biopsy_site", "uberon_code", "cell_type"]:
                v = row.get(f, "")
                if v and v.strip():
                    if f == "uberon_code":
                        o["bs_uberon_code"] = v.strip()
                    else:
                        o[f] = v.strip()
    for tag in ["sex", "treatment", "treatment_time", "response", "age", "ethnicity", "localization", "biopsy_site", "biopsy_type", "organ", "disease"]:
        if o[tag]: continue
        val = extract_tag(ctx, tag)
        if val and val.lower() not in invalid_entries:
            o[tag] = val
    if not o["library_selection"]:
        o["library_selection"] = extract_libsel(ctx)
    if not o["sequencing_source"]:
        o["sequencing_source"] = extract_src(ctx)
    if o["treatment"]:
        times = re.findall(r"\b\d+\s*(?:h|hr|hrs|hours?|d|day|days?)\b", o["treatment"], re.IGNORECASE)
        if times:
            for t in times:
                if o["treatment_time"]:
                    o["treatment_time"] += "; " + t
                else:
                    o["treatment_time"] = t
            o["treatment"] = re.sub(r"\b\d+\s*(?:h|hr|hrs|hours?|d|day|days?)\b", "", o["treatment"], flags=re.IGNORECASE).strip(" ,;")
        o["treatment"] = re.sub(stopwords, "", o["treatment"], flags=re.IGNORECASE).strip()
        o["treatment"] = re.sub(r"\s{2,}", " ", o["treatment"])
    o["disease"] = normalize(o["disease"], disease_syn)
    o["do_code"] = disease_code.get(o["disease"], "")
    o["organ"] = normalize(o["organ"], organ_syn)
    o["organ_uberon_code"] = organ_code.get(o["organ"], "")
    o["biopsy_site"] = normalize(o["biopsy_site"], organ_syn)
    o["bs_uberon_code"] = organ_code.get(o["biopsy_site"], o["bs_uberon_code"])
    out.append(o)

df = pd.DataFrame(out, columns=cols)

def fmt_codes(x):
    if not isinstance(x, str) or not x:
        return x
    return x.replace('_', ':').replace('+', ';')

df["bs_uberon_code"] = df["bs_uberon_code"].apply(fmt_codes)
df["organ_uberon_code"] = df["organ_uberon_code"].apply(fmt_codes)

#get ncbi consistent data
sra_df = pd.read_csv(input_file, sep='\t', dtype=str, on_bad_lines='skip').fillna('')
cols_to_extract = ['run_accession', 'base_count', 'library_strategy', 'instrument_platform', 'study_accession']
cols_exist = [c for c in cols_to_extract if c in sra_df.columns]
sra_df = sra_df[cols_exist]
df = df.merge(sra_df, on='run_accession', how='left')
for c in ['base_count', 'library_strategy', 'instrument_platform', 'study_accession']:
    if c in df.columns:
        df[c] = df[c].fillna('')

df['is_cancer'] = ''

for i, row in df.iterrows():
    if not row['biopsy_type']:
        ctx_path = os.path.join(xml_dir, f"{row['run_accession']}_metadata.xml")
        if not os.path.exists(ctx_path):
            continue
        ctx = open(ctx_path, "r", encoding="utf-8", errors="ignore").read()

        if re.search(r"\bmeta(?:stasis|static|stases|)\b", ctx, re.IGNORECASE):
            df.at[i, 'biopsy_type'] = 'metastasis'
        elif re.search(r"\bblood|plasma|serum|buffy coat|venous|capillary|whole[-\s]?blood\b", ctx, re.IGNORECASE):
            df.at[i, 'biopsy_type'] = 'blood'

df.to_csv(output_file_df, sep="\t", index=False)

open(FLAG_FILE, 'w').close()