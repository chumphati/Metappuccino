import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

csv_input = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/finetune_data_val_corrected.csv"
output_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING"
summary_csv_path = os.path.join(output_dir, "summary_cats_final.csv")
os.makedirs(output_dir, exist_ok=True)

CATEGORIES = [
    "cell_type", "tissue_type", "cell_line", "organ", "disease", "host_phenotype",
    "library_selection", "library_source", "treatment", "treatment_time",
    "response", "donor_information"
]

df = pd.read_csv(csv_input)

def extract_categories(output):
    result = {}
    lines = output.strip().splitlines()
    for line in lines:
        match = re.match(r"^\s*([a-zA-Z_]+)\s*:\s*(.+)$", line)
        if match:
            cat, val = match.group(1).strip(), match.group(2).strip()
            result[cat] = val
    return result

parsed_outputs = df["output"].fillna("").apply(extract_categories)
df_cats = pd.DataFrame(parsed_outputs.tolist())

summary = []

pdf_path = os.path.join(output_dir, "val_categories_final.pdf")
with PdfPages(pdf_path) as pdf:
    for cat in CATEGORIES:
        if cat not in df_cats.columns:
            summary.append({
                "category": cat,
                "non_null": 0,
                "distinct_values": 0,
                "all_values": ""
            })
            continue

        non_null_values = df_cats[cat].dropna()
        value_counts = Counter(non_null_values)
        all_items = value_counts.most_common()

        summary.append({
            "category": cat,
            "non_null": len(non_null_values),
            "distinct_values": len(value_counts),
            "all_values": "; ".join([f"{val} ({count})" for val, count in all_items])
        })

        labels, counts = zip(*all_items)

        fig, ax = plt.subplots(figsize=(10, max(4, len(labels) * 0.3)))
        if len(labels) > 10:
            ax.barh(labels, counts)
            ax.set_xlabel("Fréquence")
            ax.set_title(f"Valeurs pour {cat}")
        else:
            ax.bar(labels, counts)
            ax.set_ylabel("Fréquence")
            ax.set_title(f"Valeurs pour {cat}")
            plt.xticks(rotation=45, ha="right")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

summary_df = pd.DataFrame(summary)
summary_df.to_csv(summary_csv_path, index=False)
