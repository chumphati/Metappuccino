####################################################################################
# IMPORT

import os
import re
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns

####################################################################################
#PATHS AND SETTINGS

input_dir = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/final_MetaMap_LLM_results/all_compare'
reference_path = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/data/mela_sample.tsv'
pred_and_acc = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/final_MetaMap_LLM_results/output_analysis-v2-1500/predictions_and_accuracy.csv'
metrics_summary_path = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/final_MetaMap_LLM_results/output_analysis-v2-1500/metrics_summary.csv'
accuracy_plot_path = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/final_MetaMap_LLM_results/output_analysis-v2-1500/plots/accuracy_plots.png'
prf_plot_path = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/final_MetaMap_LLM_results/output_analysis-v2-1500/plots/precision_recall_f1.png'

reference_df = pd.read_csv(reference_path, sep='\t')
reference_runs = set(reference_df['run_accession_number'])
embed_model = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')
normal_accuracy_categories = {'cell_line', 'phenotype', 'library_selection_fixed', 'library_source', 'treatment_time'}

detail_output = []
metrics_summary = []

model_name_map = {
    'deepseekv2original4M': ('DeepSeek-V2', 'Q4_M'),
    'deepseekv2original': ('DeepSeek-V2', 'FP32/Q8_0'),
    'llama70b4M': ('LLaMA-70B', 'Q4_M'),
    'llama70b80': ('LLaMA-70B', 'FP32/Q8_0'),
    'llama8b4M': ('LLaMA-8B', 'Q4_M'),
    'llama8boriginal': ('LLaMA-8B', 'FP32/Q8_0'),
    'mistral7b4M': ('Mistral-7B', 'Q4_M'),
    'mistral7boriginal': ('Mistral-7B', 'FP32/Q8_0'),
    'mistral7BFT4M': ('Fine-tuned Mistral-7B', 'Q4_M'),
    'mistral7BFToriginal': ('Fine-tuned Mistral-7B', 'FP32/Q8_0')
}

category_rename = {
    'uberon_term': 'Organ',
    'dot_term': 'Disease'
}

####################################################################################
#FILTER RUNS

clean_pred = lambda x: re.sub(r'\(e=.*?\)', '', str(x)).strip()
threshold = 0.45


def normalize(x):
    return re.sub(r'[-_]', ' ', str(x)).strip().lower()


for filename in os.listdir(input_dir):
    if filename.endswith('_final_llm_sample_analysis.csv'):
        model_key = filename.replace('_final_llm_sample_analysis.csv', '')
        model_name, precision_type = model_name_map[model_key]
        file_path = os.path.join(input_dir, filename)
        df = pd.read_csv(file_path, sep='\t')
        df = df[df['run_accession_number'].isin(reference_runs)]

        for category in df.columns:
            if category in ['run_accession_number', 'study_accession', 'number_base_pairs']:
                continue
            if category not in reference_df.columns:
                continue

            df[f'{category}_clean'] = df[category].apply(clean_pred)
            reference_df[f'{category}_clean'] = reference_df[category].apply(clean_pred)
            preds = df[f'{category}_clean'].values
            refs = reference_df.set_index('run_accession_number').loc[df['run_accession_number']][f'{category}_clean'].values
            accuracy_type = 'Normal' if category in normal_accuracy_categories else 'Semantic'

            if accuracy_type == 'Normal':
                refs_norm = [normalize(x) for x in refs]
                preds_norm = [normalize(x) for x in preds]
                accuracy = accuracy_score(refs_norm, preds_norm)
                similarities = [np.nan] * len(preds)
                mean_similarity = np.nan
                # Standard metrics
                precision, recall, f1, _ = precision_recall_fscore_support(refs, preds, average='weighted', zero_division=0)
            else:
                embeddings_preds = embed_model.encode(preds)
                embeddings_refs = embed_model.encode(refs)
                similarities = np.diag(cosine_similarity(embeddings_preds, embeddings_refs))
                similarities = np.clip(similarities, 0, 1)
                accuracy = np.mean(similarities > threshold)
                mean_similarity = float(np.mean(similarities))
                true_positives = np.sum(similarities > threshold)
                soft_precision = true_positives / len(preds) if len(preds) > 0 else 0
                soft_recall = true_positives / len(refs) if len(refs) > 0 else 0
                soft_f1 = 2 * (soft_precision * soft_recall) / (soft_precision + soft_recall + 1e-10) if (soft_precision + soft_recall) > 0 else 0
                precision, recall, f1 = soft_precision, soft_recall, soft_f1

            metrics_summary.append({
                'Model': model_name,
                'Precision Type': precision_type,
                'Category': category,
                'Accuracy': accuracy,
                'Accuracy Type': accuracy_type,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'Mean Similarity': mean_similarity
            })

            for run, pred, ref, sim in zip(df['run_accession_number'], preds, refs, similarities):
                sim = np.clip(sim, 0, 1)
                detail_output.append({
                    'Run Accession': run,
                    'Model': model_name,
                    'Precision Type': precision_type,
                    'Category': category,
                    'Prediction': pred,
                    'Reference': ref,
                    'Prediction_normalized': normalize(pred) if accuracy_type == 'Normal' else '',
                    'Reference_normalized': normalize(ref) if accuracy_type == 'Normal' else '',
                    'Semantic Accuracy': sim if accuracy_type == 'Semantic' else np.nan
                })

detail_df = pd.DataFrame(detail_output)
detail_df.to_csv(pred_and_acc, index=False)
metrics_df = pd.DataFrame(metrics_summary)
metrics_df.to_csv(metrics_summary_path, index=False)

####################################################################################
#MAKE PLOT

categories = metrics_df['Category'].unique()
num_plots = len(categories)
rows = int(np.ceil(num_plots / 3))
order_models = ['DeepSeek-V2', 'LLaMA-70B', 'LLaMA-8B', 'Mistral-7B', 'Fine-tuned Mistral-7B']

semantic_categories = [c for c in categories if c not in normal_accuracy_categories]
mean_semantic_acc = (metrics_df[metrics_df['Category'].isin(semantic_categories)]
                     .groupby(['Model', 'Precision Type'])['Accuracy']
                     .mean()
                     .reset_index()
                     .rename(columns={'Accuracy': 'Mean Semantic Accuracy'}))

fig, axes = plt.subplots(rows, 3, figsize=(18, 5 * rows))
sns.set_style('whitegrid')

for idx, category in enumerate(categories):
    ax = axes.flatten()[idx]
    plot_data = metrics_df[metrics_df['Category'] == category].copy()
    plot_data['Model'] = pd.Categorical(plot_data['Model'], categories=order_models, ordered=True)
    plot_data['Precision Type'] = pd.Categorical(plot_data['Precision Type'], categories=['FP32/Q8_0', 'Q4_M'],
                                                 ordered=True)
    plot_data = plot_data.sort_values(['Model', 'Precision Type'])

    sns.barplot(
        data=plot_data,
        x='Model', y='Accuracy', hue='Precision Type', ax=ax,
        palette='Greys', edgecolor='black'
    )
    ax.legend(title=None)
    category_title = category_rename.get(category, category.replace('_', ' ').capitalize())
    ax.set_title(category_title)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(order_models)))
    ax.set_xticklabels(order_models, rotation=45, ha='right')

    if category in semantic_categories:
        for i, (model, precision_type) in enumerate(plot_data[['Model', 'Precision Type']].drop_duplicates().values):
            mean_val = mean_semantic_acc[
                (mean_semantic_acc['Model'] == model) &
                (mean_semantic_acc['Precision Type'] == precision_type)
                ]['Mean Semantic Accuracy']
            if not mean_val.empty:
                ax.axhline(mean_val.values[0], color='red', linestyle='--', linewidth=2,
                           xmin=(i + 0.1) / len(order_models), xmax=(i + 0.9) / len(order_models),
                           label='Mean Semantic Accuracy' if i == 0 else None)
        handles, labels = ax.get_legend_handles_labels()
        if 'Mean Semantic Accuracy' not in labels:
            handles.append(plt.Line2D([], [], color='red', linestyle='--', linewidth=2))
            labels.append('Mean Semantic Accuracy')
        ax.legend(handles, labels, loc='upper right', fontsize='small')
    else:
        ax.legend(title=None)

for idx in range(len(categories), len(axes.flatten())):
    fig.delaxes(axes.flatten()[idx])

plt.tight_layout()
plt.savefig(accuracy_plot_path, dpi=300, bbox_inches='tight')
plt.close()

####################################################################################
# PLOT PRECISION/RECALL/F1

fig2, axes2 = plt.subplots(rows, 3, figsize=(18, 5 * rows))

for idx, category in enumerate(categories):
    ax = axes2.flatten()[idx]
    plot_data = metrics_df[metrics_df['Category'] == category].copy()
    plot_data['Model'] = pd.Categorical(plot_data['Model'], categories=order_models, ordered=True)
    plot_data['Precision Type'] = pd.Categorical(plot_data['Precision Type'], categories=['FP32/Q8_0', 'Q4_M'],
                                                 ordered=True)
    plot_data = plot_data.sort_values(['Model', 'Precision Type'])

    melted = plot_data.melt(
        id_vars=['Model', 'Precision Type'],
        value_vars=['Precision', 'Recall', 'F1'],
        var_name='Metric', value_name='Score'
    )
    sns.barplot(
        data=melted,
        x='Model', y='Score', hue='Metric', ax=ax,
        palette='Set2', edgecolor='black'
    )
    ax.set_title(category_rename.get(category, category.replace('_', ' ').capitalize()))
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(order_models)))
    ax.set_xticklabels(order_models, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize='small')

for idx in range(len(categories), len(axes2.flatten())):
    fig2.delaxes(axes2.flatten()[idx])

plt.tight_layout()
plt.savefig(prf_plot_path, dpi=300, bbox_inches='tight')
plt.close()
