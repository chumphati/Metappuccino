#IMPORT
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

##########################################################################################
#PATHS
with open('/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/SPECIFIC_RUN_ANALYSIS/final_llm_sample_analysis.csv', 'r', encoding='utf-8') as f:
    output_lines = f.readlines()

with open('/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/data/processed_ref_metadata.csv', 'r', encoding='utf-16') as f:
    reference_lines = f.readlines()

##########################################################################################
#COLUMNS TO COMPARE
ref = 'UBERON term'
llm = 'Gtex'

##########################################################################################
#MAIN
output_headers = output_lines[0].strip().split('\t')
output_data = [dict(zip(output_headers, line.strip().split('\t'))) for line in output_lines[1:]]
reference_headers = reference_lines[0].strip().split(',')
reference_data = [dict(zip(reference_headers, line.strip().split(','))) for line in reference_lines[1:]]
output_df = pd.DataFrame(output_data)
reference_df = pd.DataFrame(reference_data)

output_accessions = set(output_df['Run accession number'])
reference_accessions = set(reference_df['run_accession'])

common_accessions = output_accessions.intersection(reference_accessions)
print(len(common_accessions))
merged = output_df.merge(reference_df, left_on='Run accession number', right_on='run_accession', how='inner')

merged['Run accession'] = merged['Run accession number']

#final col to compare
merged[llm] = merged[llm].fillna('NA').str.replace('[\\[\\]\']', '', regex=True)
merged[ref] = merged[ref].fillna('NA').str.replace('[\\[\\]\']', '', regex=True)


#normalize
def preprocess_terms(terms):
    terms = [term.strip().lower() for term in terms.split(',') if term.strip() != 'na']
    return ' '.join(terms)


merged['LLM processed'] = merged[llm].apply(preprocess_terms)
merged['Ref processed'] = merged[ref].apply(preprocess_terms)

corpus_llm = merged['LLM processed'].tolist()
corpus_ref = merged['Ref processed'].tolist()
# print(corpus_llm[1])
# print(corpus_ref[1])

## TF-IDF + Cosine Similarity
#transform tex into vector
vectorizer = TfidfVectorizer()
tfidf_llm = vectorizer.fit_transform(corpus_llm)
tfidf_ref = vectorizer.transform(corpus_ref)
#calcul similarity cosinus line by line
cosine_similarities = [cosine_similarity(tfidf_llm[i], tfidf_ref[i])[0][0] for i in range(len(merged))]
merged['TF-IDF Cosine Similarity'] = cosine_similarities

## BERT + Cosine Similarity
#BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')
#compute embeddings for semantic similarity
bert_embeddings_llm = model.encode(corpus_llm)
bert_embeddings_ref = model.encode(corpus_ref)
#calculate cosine similarity line by line for BERT embeddings
bert_cosine_similarities = [cosine_similarity([bert_embeddings_llm[i]], [bert_embeddings_ref[i]])[0][0] for i in range(len(merged))]
merged['BERT Cosine Similarity'] = bert_cosine_similarities

## Jaccard score
#display score
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

#jaccard similarity line by line
jaccard_similarities = []
for i, row in merged.iterrows():
    set_llm = set(row['LLM processed'].split())
    set_ref = set(row['Ref processed'].split())
    jaccard_similarities.append(jaccard_similarity(set_llm, set_ref))
merged['Jaccard Similarity'] = jaccard_similarities

## PRINT ALL
table_to_display = merged[['Run accession', llm, ref, 'TF-IDF Cosine Similarity', 'BERT Cosine Similarity', 'Jaccard Similarity']]
table_to_display.to_csv("/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/SPECIFIC_RUN_ANALYSIS/output_similarity_table.csv", index=False, encoding='utf-8')

## Accuracy
accuracy_cumulative = {}
thresholds = np.linspace(0, 1, 20)
for threshold in thresholds:
    merged['predicted_label'] = (merged['BERT Cosine Similarity'] >= threshold).astype(int)
    merged['true_label'] = merged.apply(
        lambda row: int(bool(set(row['LLM processed'].split()) & set(row['Ref processed'].split()))), axis=1
    )
    accuracy = accuracy_score(merged['true_label'], merged['predicted_label'])
    accuracy_cumulative[threshold] = accuracy

threshold = 0.42
merged['predicted_label'] = (merged['BERT Cosine Similarity'] >= threshold).astype(int)
merged['true_label'] = merged.apply(
    lambda row: int(bool(set(row['LLM processed'].split()) & set(row['Ref processed'].split()))), axis=1
)
final_accuracy = accuracy_score(merged['true_label'], merged['predicted_label'])
print(f"Accuracy: {final_accuracy:.2f}")

# Plots
plt.figure(figsize=(8, 6))
plt.plot(list(accuracy_cumulative.keys()), list(accuracy_cumulative.values()), marker='o')
plt.title("Cumulative accuracy as a function of threshold (BERT Cosine Similarity)")
plt.xlabel("Threshold")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()
plt.figure(figsize=(18, 6))
plt.subplot(1, 3, 1)
sns.histplot(merged['TF-IDF Cosine Similarity'], kde=True, bins=30, color='blue', alpha=0.7)
plt.title("Distribution of TF-IDF Cosine Similarity")
plt.xlabel("TF-IDF Cosine Similarity")
plt.ylabel("Frequency")
plt.subplot(1, 3, 2)
sns.histplot(merged['BERT Cosine Similarity'], kde=True, bins=30, color='green', alpha=0.7)
plt.title("Distribution of BERT Cosine Similarity")
plt.xlabel("BERT Cosine Similarity")
plt.ylabel("Frequency")
plt.subplot(1, 3, 3)
sns.histplot(merged['Jaccard Similarity'], kde=True, bins=30, color='orange', alpha=0.7)
plt.title("Distribution of Jaccard Similarity")
plt.xlabel("Jaccard Similarity")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

## STATS
mean_bert = merged['BERT Cosine Similarity'].mean()
median_bert = merged['BERT Cosine Similarity'].median()
std_bert = merged['BERT Cosine Similarity'].std()
print(f"BERT Cosine Similarity: Mean = {mean_bert:.2f}, Median = {median_bert:.2f}, Std = {std_bert:.2f}")
