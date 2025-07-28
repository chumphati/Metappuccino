import csv
import re
import numpy as np
from tqdm import tqdm
import spacy
from spacy.cli import download
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    nlp = spacy.load('en_core_web_md')
except OSError:
    # download('en_core_web_md')
    nlp = spacy.load('en_core_web_md')

INPUT_ORIG = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_metappuccino/tmp/metadata_sra.txt"
INPUT_SUMM = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_ft_final_templates/metadata_sra_summarized.txt"

def build_ctx(fields):
    raw = " ".join(f for f in fields
                   if not any(x in f.lower() for x in ["run accession", "study accession", "experiment accession", "sample accession"]))
    return re.sub(r'\S+?\.fastq\.gz', '', raw).strip()

orig = {}
with open(INPUT_ORIG, 'r', encoding='utf-8') as f:
    r = csv.reader(f, delimiter='\t')
    next(r)
    for row in r:
        orig[row[0]] = build_ctx(row[1:])

summ = {}
with open(INPUT_SUMM, 'r', encoding='utf-8') as f:
    r = csv.reader(f, delimiter='\t')
    next(r)
    for row in r:
        summ[row[0]] = row[1]

semantic_sims = []
term_overlaps = []
missing_details = []

for run, otext in tqdm(orig.items()):
    stext = summ.get(run, "")
    doc_o = nlp(otext)
    doc_s = nlp(stext)
    semantic_sims.append(doc_o.similarity(doc_s))
    vect = TfidfVectorizer(stop_words='english', token_pattern=r'(?u)\b[^\d\W]+\b')
    mat = vect.fit_transform([otext, stext])
    fnames = np.array(vect.get_feature_names_out())
    tfidf_o, tfidf_s = mat.toarray()
    top_o = set(fnames[np.argsort(-tfidf_o)[:20]])
    top_s = set(fnames[np.argsort(-tfidf_s)[:20]])
    overlap = len(top_o & top_s) / len(top_o) if top_o else 1.0
    term_overlaps.append(overlap)
    if overlap < 0.5:
        missing = sorted(top_o - top_s)
        missing_details.append((run, overlap, missing))

print(f"Average semantic similarity: {np.mean(semantic_sims):.3f}")
print(f"Average top-term overlap:    {np.mean(term_overlaps)*100:.1f}%")
print(f"Runs with low overlap (<50%): {len(missing_details)} / {len(orig)}")
for run, ov, miss in missing_details:
    print(f"{run}\toverlap={ov:.2f}\tmissing={','.join(miss)}")
