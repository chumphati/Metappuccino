import pandas as pd
import numpy as np

sort_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/store/ORGAN_MISTRAL_FINE_TUNING/llm_FT_sort_output.csv'
ena_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/ena_results.tsv'

sort_df = pd.read_csv(sort_file)
ena_df = pd.read_csv(ena_file, sep='\t')
ena_df.set_index('run_accession', inplace=True)

for idx, row in sort_df.iterrows():
    run = row['run_accession']
    output_value = row['output']
    if run in ena_df.index:
        ena_df.at[run, 'disease'] = output_value
    else:
        new_row = {col: '' for col in ena_df.columns}
        new_row['disease'] = output_value
        ena_df.loc[run] = new_row

ena_df.reset_index(inplace=True)
modified_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/ena_results_modified.tsv'
ena_df.to_csv(modified_file, sep='\t', index=False)

formatted_data = []

all_columns = ena_df.columns.tolist()
for idx, row in ena_df.iterrows():
    run = row['run_accession']
    info_lines = []
    for col in all_columns:
        if col == 'run_accession':
            continue
        val = row[col]
        if pd.isna(val) or val == '':
            val_str = 'nan'
        else:
            val_str = str(val)
        info_lines.append(f"{col}: {val_str}")
    info_text = "\n".join(info_lines)
    formatted_data.append({'run_accession': run, 'info': info_text})

formatted_df = pd.DataFrame(formatted_data)
formatted_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/ena_results_formatted.csv'
formatted_df.to_csv(formatted_file, index=False)
