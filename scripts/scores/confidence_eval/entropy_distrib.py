import os
import re
import pandas as pd
import matplotlib.pyplot as plt

directory = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/store/results_3fev2025/SPECIFIC_RUN_ANALYSIS/INFO_BIO_LLM"
output_directory = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/store/results_3fev2025/ENTROPY_DISTRIBUTION"

if not os.path.exists(output_directory):
    os.makedirs(output_directory)

entropy_patterns = {
    "Cell type Entropy": r"Cell type Entropy:\s*([\d\.]+)",
    "UBERON organ and code Entropy": r"UBERON organ and code Entropy:\s*([\d\.]+)",
    "Disease Ontology Term Entropy": r"Disease Ontology Term Entropy:\s*([\d\.]+)",
    "Tissue type Entropy": r"Tissue type Entropy:\s*([\d\.]+)",
    "Cell line Entropy": r"Cell line Entropy:\s*([\d\.]+)"
}

entropy_data = {key: [] for key in entropy_patterns}

if os.path.exists(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

                for category, pattern in entropy_patterns.items():
                    match = re.search(pattern, content)
                    if match:
                        entropy_data[category].append(float(match.group(1)))
                    else:
                        entropy_data[category].append(None)

    df_entropy = pd.DataFrame(entropy_data)

    for category in entropy_patterns.keys():
        if df_entropy[category].dropna().empty:
            continue

        plt.figure(figsize=(8, 5))
        plt.hist(df_entropy[category].dropna(), bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Entropy')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {category}')
        plt.grid(True)
        plt.savefig(os.path.join(output_directory, f"{category.replace(' ', '_')}_distribution.png"))
        plt.close()
else:
    print("Error: folder doesn't exist")
