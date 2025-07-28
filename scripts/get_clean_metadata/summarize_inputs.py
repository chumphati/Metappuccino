########################################################################################################################
# IMPORT
import csv
import os
import argparse
import re
import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from spacy.cli import download

########################################################################################################################
#DOWNLOADS
nltk.download('punkt_tab', quiet=True)
download('en_core_web_md')
nlp = spacy.load('en_core_web_md')

########################################################################################################################
#PATHS
parser = argparse.ArgumentParser(description="Fetch information with Cellosaurus")
parser.add_argument("--base_path", type=str, required=True, help="Base path to Metappuccino")
args = parser.parse_args()

base_path = args.base_path
INPUT_FILE = os.path.join(base_path, "metadata_sra.txt")
OUTPUT_FILE = os.path.join(base_path, "metadata_sra_summarized.txt")
FLAG_FILE = os.path.join(base_path, "STEP2_2.flag")

# INPUT_FILE = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_metappuccino/tmp/metadata_sra.txt"
# OUTPUT_FILE = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_ft_final_templates/metadata_sra_summarized.txt"

MAX_WORDS = 1200
TOKEN_THRESHOLD = 1200
CHUNK_SIZE = 100
CATEGORY_KEYWORDS = [
    'cell type', 'tissue type', 'cell line', 'organ', 'disease',
    'host phenotype', 'library selection', 'library source',
    'treatment', 'treatment time', 'response', 'donor information', 'instrument platform'
]
SEMANTIC_THRESHOLD = 0.4
RELEVANCE_BOOST = 2.0

category_docs = [nlp(cat) for cat in CATEGORY_KEYWORDS]

########################################################################################################################
#FUNCTIONS
def extract_clauses(text):
    tokens = word_tokenize(text)
    chunks = []
    i = 0
    n = len(tokens)
    while i < n:
        j = min(i + CHUNK_SIZE, n)
        k = j
        while k > i and tokens[k-1] not in '.?!;,' :
            k -= 1
        if k == i:
            k = j
        chunks.append({'text': ' '.join(tokens[i:k]), 'orig_idx': i})
        i = k
    return chunks

def score_clauses(clauses):
    texts = [c['text'] for c in clauses]
    vect = TfidfVectorizer(stop_words='english').fit(texts)
    X = vect.transform(texts)
    tfidf_scores = np.array(X.sum(axis=1)).ravel()
    cat_scores = []
    for t in texts:
        doc = nlp(t)
        score = 0.0
        for cdoc in category_docs:
            sim = doc.similarity(cdoc)
            if sim > SEMANTIC_THRESHOLD:
                score += sim
        cat_scores.append(score)
    combined = tfidf_scores + RELEVANCE_BOOST * np.array(cat_scores)
    return combined, vect

def summarize_by_clauses(text):
    clauses = extract_clauses(text)
    if not clauses:
        return text.strip()
    scores, _ = score_clauses(clauses)
    order = np.argsort(-scores)
    summary = []
    total = 0
    for i in order:
        words = clauses[i]['text'].split()
        if total + len(words) <= MAX_WORDS:
            summary.append((clauses[i]['orig_idx'], clauses[i]['text']))
            total += len(words)
        if total >= MAX_WORDS:
            break
    summary.sort(key=lambda x: x[0])
    texts = [s for _, s in summary]
    return ' '.join(t if t.endswith(('.', '?', '!')) else t + '.' for t in texts)

########################################################################################################################
#MAIN
with open(INPUT_FILE, 'r', encoding='utf-8') as fin, open(OUTPUT_FILE, 'w', encoding='utf-8') as fout:
    rdr = csv.reader(fin, delimiter='\t')
    wtr = csv.writer(fout, delimiter='\t')
    hdr = next(rdr)
    wtr.writerow([hdr[0], 'summary'])
    for row in tqdm(rdr, desc='Processing rows'):
        run_acc = row[0]
        raw = ' '.join(f for f in row[1:]
                       if not any(x in f.lower() for x in ['run accession', 'study accession', 'experiment accession', 'sample accession']))
        ctx = re.sub(r'\S+?\.fastq\.gz', '', raw).strip()
        if len(ctx.split()) > TOKEN_THRESHOLD:
            summ = summarize_by_clauses(ctx)
        else:
            summ = ctx
        wtr.writerow([run_acc, summ])

open(FLAG_FILE, 'w').close()