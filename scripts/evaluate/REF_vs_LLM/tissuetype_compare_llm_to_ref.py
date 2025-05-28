import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

with open('/Users/fionahak/Documents/phd/phd_code/final_results/ccle_deepseek.csv', 'r', encoding='utf-8') as f:
    output_lines = f.readlines()

with open('/Users/fionahak/Documents/phd/phd_code/MetaMap/data/ccle_ref_cell.txt', 'r', encoding='utf-8') as f:
    reference_lines = f.readlines()


output_headers = output_lines[0].strip().split('\t')
output_data = [dict(zip(output_headers, line.strip().split('\t'))) for line in output_lines[1:]]
reference_headers = reference_lines[0].strip().split(',')
reference_data = [dict(zip(reference_headers, line.strip().split(','))) for line in reference_lines[1:]]
output_df = pd.DataFrame(output_data)
reference_df = pd.DataFrame(reference_data)

output_accessions = set(output_df['Run accession number'])
reference_accessions = set(reference_df['Run accession number ref'])

common_accessions = output_accessions.intersection(reference_accessions)
# print(len(common_accessions))
merged = output_df.merge(reference_df, left_on='Run accession number', right_on='Run accession number ref', how='inner')

#final col to compare
merged['Tissue type'] = merged['Tissue type'].fillna('NA').str.replace('[\\[\\]\']', '', regex=True)
merged['Tissue type ref'] = merged['Tissue type ref'].fillna('NA').str.replace('[\\[\\]\']', '', regex=True)


#normalize
def preprocess_terms(terms):
    terms = [term.strip().lower() for term in terms.split(',') if term.strip() != 'na']
    return ' '.join(terms)


merged['Tissue type processed'] = merged['Tissue type'].apply(preprocess_terms)
merged['Tissue type ref processed'] = merged['Tissue type ref'].apply(preprocess_terms)

corpus_uberon = merged['Tissue type processed'].tolist()
corpus_gtex = merged['Tissue type ref processed'].tolist()
# print(corpus_uberon[1])
# print(corpus_gtex[1])

## TF-IDF + Cosine Similarity
#transform tex into vector
vectorizer = TfidfVectorizer()
tfidf_uberon = vectorizer.fit_transform(corpus_uberon)
tfidf_gtex = vectorizer.transform(corpus_gtex)
#calcul similarity cosinus line by line
cosine_similarities = [cosine_similarity(tfidf_uberon[i], tfidf_gtex[i])[0][0] for i in range(len(merged))]
merged['TF-IDF Cosine Similarity'] = cosine_similarities

## BERT + Cosine Similarity
#BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')
#compute embeddings for semantic similarity
bert_embeddings_uberon = model.encode(corpus_uberon)
bert_embeddings_gtex = model.encode(corpus_gtex)
#calculate cosine similarity line by line for BERT embeddings
bert_cosine_similarities = [cosine_similarity([bert_embeddings_uberon[i]], [bert_embeddings_gtex[i]])[0][0] for i in range(len(merged))]
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
    set_uberon = set(row['Tissue type processed'].split())
    set_gtex = set(row['Tissue type ref processed'].split())
    jaccard_similarities.append(jaccard_similarity(set_uberon, set_gtex))
merged['Jaccard Similarity'] = jaccard_similarities

## PRINT ALL
table_to_display = merged[['Tissue type', 'Tissue type ref', 'TF-IDF Cosine Similarity', 'BERT Cosine Similarity', 'Jaccard Similarity']]
table_to_display.to_csv("/Users/fionahak/Documents/phd/phd_code/MetaMap/ccle_results/SPECIFIC_RUN_ANALYSIS/output_similarity_table.csv", index=False, encoding='utf-8')

## Accuracy
accuracy_cumulative = {}
thresholds = np.linspace(0, 1, 20)
for threshold in thresholds:
    merged['predicted_label'] = (merged['BERT Cosine Similarity'] >= threshold).astype(int)
    merged['true_label'] = merged.apply(
        lambda row: int(bool(set(row['Tissue type processed'].split()) & set(row['Tissue type ref processed'].split()))), axis=1
    )
    accuracy = accuracy_score(merged['true_label'], merged['predicted_label'])
    accuracy_cumulative[threshold] = accuracy

threshold = 0.56
merged['predicted_label'] = (merged['BERT Cosine Similarity'] >= threshold).astype(int)
merged['true_label'] = merged.apply(
    lambda row: int(bool(set(row['Tissue type processed'].split()) & set(row['Tissue type ref processed'].split()))), axis=1
)
final_accuracy = accuracy_score(merged['true_label'], merged['predicted_label'])
print(f"Accuracy: {final_accuracy:.2f}")

## Plots
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
