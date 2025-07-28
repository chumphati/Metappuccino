import pandas as pd
import csv

input_path = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_metappuccino/FINE_TUNING/finetune_data.csv'
df = pd.read_csv(input_path, quoting=csv.QUOTE_ALL)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

train_frac = 0.75
n_train = int(len(df) * train_frac)
train_df = df.iloc[:n_train]
val_df   = df.iloc[n_train:]

train_path = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_metappuccino/FINE_TUNING/train_finetune_data.csv'
val_path   = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_metappuccino/FINE_TUNING/val_finetune_data.csv'
train_df.to_csv(train_path, index=False, quoting=csv.QUOTE_ALL)
val_df.to_csv(val_path,   index=False, quoting=csv.QUOTE_ALL)

