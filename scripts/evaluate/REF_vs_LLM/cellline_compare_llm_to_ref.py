import pandas as pd
import matplotlib.pyplot as plt

with open('/Users/fionahak/Documents/phd/phd_code/final_results/ccle_llama.csv', 'r', encoding='utf-8') as f:
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
merged = output_df.merge(reference_df, left_on='Run accession number', right_on='Run accession number ref', how='inner')


def preprocess_terms(terms):
    import re
    terms = re.sub(r'[^a-zA-Z0-9 ]', '', terms.lower().strip())
    return terms

merged['Cell line processed'] = merged['Cell line'].apply(preprocess_terms)
merged['Cell line ref processed'] = merged['Cell line ref'].apply(preprocess_terms)

corpus_uberon = merged['Cell line processed'].tolist()
corpus_gtex = merged['Cell line ref processed'].tolist()

print(corpus_uberon[:10])
print(corpus_gtex[:10])

matches = []
for uberon, gtex in zip(corpus_uberon, corpus_gtex):
    reference_words = set(gtex.split())
    llm_words = set(uberon.split())
    matches.append(1 if reference_words.issubset(llm_words) else 0)
merged['Exact Match'] = matches
accuracy = sum(matches) / len(matches)
print(f"Accuracy: {accuracy:.2%} ({sum(matches)} matches out of {len(matches)})")

## PRINT ALL
merged['Match'] = matches
table_to_display = merged[['Cell line', 'Cell line ref', 'Match']]
table_to_display.to_csv("/Users/fionahak/Documents/phd/phd_code/MetaMap/ccle_results/SPECIFIC_RUN_ANALYSIS/cell_line_table.csv", index=False, encoding='utf-8')

labels = ['Exact Matches', 'Non-Matches']
sizes = [sum(matches), len(matches) - sum(matches)]
colors = ['#A2D9CE', '#F5B7B1']
explode = (0.1, 0)
plt.figure(figsize=(8, 8))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, explode=explode, shadow=False)
plt.title("Correspondence of cell line codes between LLM output and reference")
plt.ylabel("Run number")
plt.show()
