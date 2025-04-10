##########################################################################################
#IMPORT
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModel, AutoTokenizer
import matplotlib.pyplot as plt
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sentence_transformers import SentenceTransformer

##########################################################################################
#PATHS
ref_path = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/ref_sort_output.csv'
original_path = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/llm_original_sort_output.csv'
ft_path = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/llm_FT_sort_output.csv'
table_metrics = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/metrics_results.tex'
roc_curves = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/roc_curves.png'  # (non utilisé ici)
hist_cs = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/cosine_similarity_histogram.png'
final_manual_compt = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/comparison_results.csv'

runs_to_compare = [
    'SRR1424691','SRR19095764','SRR16013060','SRR22847384','SRR25387888','SRR4785829',
    'ERR3549178','SRR7094291','SRR20695355','SRR18363185','SRR6937757','SRR18363171',
    'SRR22966269','SRR9007553','SRR22925398','SRR7094282','SRR1424662','SRR15013557',
    'SRR22283244','SRR20695347','SRR25387901','SRR24709842','SRR22283281','SRR7430742',
    'SRR6937753','SRR13572929','SRR19256827','SRR22283291','SRR22966271','SRR18363179',
    'SRR915768','SRR22532373','SRR25387886','SRR8518134','SRR4785835','SRR22925192',
    'SRR19666460','SRR6937800','SRR25387924','SRR7167724','SRR8518355','SRR1424692',
    'SRR4785815','SRR4785838','SRR23920437','SRR23630185','SRR13485950','SRR15013650',
    'SRR15013485','SRR22301776','SRR15013571','ERR5320490','ERR1514452','SRR22283234',
    'SRR22925347','SRR15013491','SRR25098099','SRR11547383','SRR22532386','SRR13518181',
    'SRR5259639','SRR15013474','SRR20140291','SRR12817269','SRR7430738','SRR15013470',
    'SRR1603664','ERR3549198','SRR13518178','SRR8518278','SRR16013057','ERR5285553',
    'SRR11547421','ERR1883116','SRR7767519','SRR11049435','SRR24709855','SRR22283241',
    'SRR15013559','SRR15013496','SRR22283249','SRR18363169','SRR25098111','SRR15013462',
    'SRR8932009','SRR13485967','ERR5285538','SRR13572928','SRR1721309','SRR5591607',
    'SRR9007567','SRR16013068','SRR15013513','SRR7430749','SRR1424687','SRR15013627',
    'SRR15013562','SRR8518148','SRR3393497','SRR15013486','SRR22283222','SRR7012369',
    'SRR7094294','SRR26436583','SRR14027942','SRR15013514','SRR20695353','SRR5259643',
    'SRR15013677','SRR7094278','SRR3393521','SRR15013524','SRR3703021','SRR25387875',
    'SRR16013087','SRR22925215','SRR4240761','SRR6937766','SRR22301777','SRR1721301',
    'SRR15013575','SRR22532397','SRR22301788','SRR15013615','SRR14362387','SRR19243478',
    'SRR26436585','SRR1424670','DRR326900','SRR16212321','SRR7430734','SRR14027938',
    'SRR1603663','SRR1424672','SRR22925281','SRR9007537','SRR15013598','SRR23630255',
    'SRR14362390','SRR15013606','SRR23630234','SRR19779072','SRR25387899','SRR22532395',
    'SRR13485933','ERR1993159'
]

#modèle
model_name = 'sentence-transformers/all-mpnet-base-v2'

##########################################################################################
#FUNCTIONS


def filter_df(df):
    return df[df['run_accession'].isin(runs_to_compare)].set_index('run_accession')


def clean_text(text):
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = re.sub(r'Organe:\s*.*', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    return text


def weighted_embed(texts, model):
    cleaned_texts = [clean_text(t) for t in texts]
    embeddings = model.encode(cleaned_texts, convert_to_numpy=True)
    return embeddings

##########################################################################################
# MAIN


ref_df = pd.read_csv(ref_path)
original_df = pd.read_csv(original_path)
ft_df = pd.read_csv(ft_path)

#keep only common run
ref_df, original_df, ft_df = map(filter_df, [ref_df, original_df, ft_df])
common_runs = ref_df.index.intersection(original_df.index).intersection(ft_df.index)
ref_df = ref_df.loc[common_runs]
original_df = original_df.loc[common_runs]
ft_df = ft_df.loc[common_runs]

#get embeddings
model = SentenceTransformer(model_name)
ref_embeddings = weighted_embed(ref_df['output'].tolist(), model)
original_embeddings = weighted_embed(original_df['output'].tolist(), model)
ft_embeddings = weighted_embed(ft_df['output'].tolist(), model)

#sim cos
cos_original = cosine_similarity(ref_embeddings, original_embeddings).diagonal()
cos_ft = cosine_similarity(ref_embeddings, ft_embeddings).diagonal()

#accuracy, true is > 0.35
threshold = 0.35
acc_original = np.mean(cos_original > threshold)
acc_ft = np.mean(cos_ft > threshold)

print(f"ACCURACY ORIGINAL: {acc_original * 100:.2f} %", flush=True)
print(f"ACCURACY FT: {acc_ft * 100:.2f} %", flush=True)

#plot cos som
plt.figure(figsize=(8, 6))
plt.hist(cos_original, bins=20, alpha=0.6, label='Original')
plt.hist(cos_ft, bins=20, alpha=0.6, label='Fine-tuned')
plt.xlabel('Cosine Similarity')
plt.ylabel('Frequency')
plt.title('CS Distribution of Mistral-7B-Instruct Models')
plt.legend()
plt.grid(True)
plt.savefig(hist_cs)

comparison_df = pd.DataFrame({
    'run_accession': common_runs,
    'reference_output': ref_df['output'].values,
    'original_model_output': original_df['output'].values,
    'finetuned_model_output': ft_df['output'].values,
    'cosine_similarity_original': cos_original,
    'cosine_similarity_finetuned': cos_ft
})
comparison_df.to_csv(final_manual_compt, index=False)
