import pandas as pd

input_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/finetune_data_train.csv"
output_path = input_path.replace(".csv", "_corrected.csv")
df = pd.read_csv(input_path)

# disease_to_organ = {
#     "pneumonia": "lungs",
#     "lung cancer": "lungs",
#     "colorectal cancer": "colon"
# }

disease_to_organ = {
    "glioblastoma": "brain",
    "breast cancer": "breast",
    "brain tumor": "brain",
    "hepatitis": "liver",
    "skin melanoma": "skin"
}

def correct_output_block(output: str) -> str:
    if not isinstance(output, str):
        return output
    lines = output.strip().splitlines()
    disease = None
    for line in lines:
        if line.lower().startswith("disease:"):
            disease = line.split(":", 1)[1].strip().lower()
            break
    corrected = []
    for line in lines:
        if line.lower().startswith("organ:") and disease in disease_to_organ:
            corrected.append(f"organ: {disease_to_organ[disease]}")
        else:
            corrected.append(line)
    return "\n".join(corrected)

df["output"] = df["output"].apply(correct_output_block)
df.to_csv(output_path, index=False)
