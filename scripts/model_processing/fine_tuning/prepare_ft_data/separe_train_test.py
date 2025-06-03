import pandas as pd

df = pd.read_csv("/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/mistral7B_FINE_TUNING_v1/finetune_data_short.csv")
disease_keywords = ["pneumonia", "lung cancer", "colorectal cancer"]
organ_keywords = ["lungs", "colon"]

def contains_target_keywords(output):
    if not isinstance(output, str):
        return False
    for kw in disease_keywords + organ_keywords:
        if kw.lower() in output.lower():
            return True
    return False

df_target = df[df["output"].apply(contains_target_keywords)].reset_index(drop=True)
df_rest = df[~df["output"].apply(contains_target_keywords)].reset_index(drop=True)
df_target.to_csv("/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/finetune_data_val.csv", index=False)
df_rest.to_csv("/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/finetune_data_train.csv", index=False)
