import requests
import pandas as pd
import random
from io import StringIO

base_url = "https://www.ebi.ac.uk/ena/portal/api/search"
base_query = (
    'tax_eq(9606) AND library_strategy="RNA-seq" '
    "AND first_public>='2012-01-01' AND first_public<'2025-01-01' "
    'AND instrument_platform="ILLUMINA" AND read_count>=10000000'
)
categories = {
    "Cell type": "cell_type",
    "UBERON term": "host_body_site",
    "Tissue type": "tissue_type",
    "Cell line": "cell_line",
    "DOT term": "disease",
    "Phenotype": "host_phenotype",
    "Library selection fixed": "library_selection",
    "Library source": "library_source"
}
all_fields = list(set(categories.values()))
all_fields_str = "run_accession," + ",".join(all_fields)
output_rows = []
num_samples_per_category = 10

for cat_label, field in categories.items():
    query = f'{base_query} AND {field}:*'
    params = {
        "result": "read_run",
        "query": query,
        "fields": all_fields_str,
        "limit": 1000,
        "format": "tsv"
    }
    print(f"Get category '{cat_label}' (field {field})...")
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        print(f"Error: '{cat_label}': HTTP {response.status_code}")
        continue
    data = pd.read_csv(StringIO(response.text), sep="\t")
    data = data.dropna(subset=[field])
    if len(data) > num_samples_per_category:
        data = data.sample(n=num_samples_per_category, random_state=42)
    for _, row in data.iterrows():
        run_accession = row["run_accession"]
        infos = []
        for cat, f in categories.items():
            val = row.get(f)
            if pd.notna(val) and str(val).strip() != "":
                infos.append(f"{cat}: {val}")
        info_str = "; ".join(infos)
        output_rows.append({"Run Accession": run_accession, "Info": info_str})
output_df = pd.DataFrame(output_rows)
print(output_df.head(20))
output_df.to_csv("/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/ena_random_runs.csv", index=False)
