import pandas as pd
import random

# ---------------------------
# 1. Reference dictionaries
# ---------------------------
cell_types_by_organ_disease = {
    ('colon', 'colorectal cancer'): 'Colonocyte',
    ('lungs', 'lung cancer'): 'monocyte',
    ('skin', 'skin melanoma'): 'fibroblast',
    ('brain', 'glioblastoma'): 'astrocyte',
    ('brain', 'brain tumor'): 'astrocyte',
    ('breast', 'breast cancer'): 'epithelial',
    ('liver', 'hepatitis'): 'B cell',
    ('lungs', 'pneumonia'): 'monocyte',
}

cell_types_by_organ = {
    'colon': 'Colonocyte',
    'lungs': 'monocyte',
    'skin': 'fibroblast',
    'brain': 'astrocyte',
    'breast': 'epithelial',
    'liver': 'B cell',
    'kidney': 'fibroblast',
    'heart': 'fibroblast',
    'pancreas': 'fibroblast',
}

cell_types_context = {
    'neuron':       ['neuron', 'neuronal', 'nerve cell', 'brain-derived'],
    'fibroblast':   ['fibroblast', 'skin-associated cell', 'connective tissue', 'muscle-related cell'],
    'CD8 T cell':   ['CD8 T cell', 'cytotoxic lymphocyte', 'immune-related cell'],
    'monocyte':     ['monocyte', 'blood-derived cell'],
    'NK cell':      ['NK cell', 'natural killer cell', 'immune effector'],
    'Colonocyte':   ['Colonocyte', 'intestinal epithelial cell', 'colon-related cell'],
    'B cell':       ['B cell', 'antibody-producing cell', 'humoral lymphocyte'],
    'astrocyte':    ['astrocyte', 'glial cell'],
    'epithelial':   ['epithelial cell', 'ductal epithelial'],
    # Added mapping for primary tissue samples:
    'Primary tissue': ['primary tissue', 'fresh tissue sample', 'uncultured primary cells'],
}

tissue_types_by_organ = {
    'colon': 'epithelial',
    'lungs': 'epithelial',
    'skin': 'connective',
    'brain': 'nervous',
    'breast': 'epithelial',
    'liver': 'epithelial',
    'kidney': 'epithelial',
    'heart': 'muscle',
    'pancreas': 'epithelial',
}

cell_lines_by_organ = {
    'colon': 'DLD-1',
    'lungs': 'A549',
    'skin': 'Primary tissue',
    'brain': 'U87',
    'breast': 'MCF7',
    'liver': 'Primary tissue',
    'kidney': 'Primary tissue',
    'heart': 'Primary tissue',
    'pancreas': 'Primary tissue',
}

Diseases_context = [
    'colorectal cancer', 'lung cancer', 'skin melanoma', 'brain tumor', 'pneumonia',
    'breast cancer', 'glioblastoma', 'hepatitis', 'normal',
]

Disease_to_organ = {
    'colorectal cancer': 'colon',
    'lung cancer': 'lungs',
    'skin melanoma': 'skin',
    'brain tumor': 'brain',
    'breast cancer': 'breast',
    'glioblastoma': 'brain',
    'hepatitis': 'liver',
    'pneumonia': 'lungs',
}

disease_markers = {
    'skin melanoma': ['BRAF V600E mutation'],
    'lung cancer': ['EGFR exon 19 deletion'],
    'colorectal cancer': ['KRAS G12D mutation'],
    'glioblastoma': ['IDH1 R132H mutation'],
    'breast cancer': ['HER2 overexpression'],
    'hepatitis': ['HBV surface antigen'],
    'pneumonia': ['elevated CRP'],
}

library_selections_context = {
    'polyA': ['PolyA'],
    'inverse rRNA': ['ribosomal RNA depletion'],
    'hybrid selection': ['exome capture'],
    'small RNA': ['microRNA'],
    'other': ['random primers'],
}

library_sources_context = {
    'single-cell': ['10x genomics'],
    'bulk': ['bulk RNA sequencing'],
}

treatments_cancer = ['Nivolumab', 'Cisplatin']
treatments_infection = ['Amoxicillin']
treatments_other = ['gene therapy']

clinical_responses = ['Progressive Disease', 'Stable Disease']

donor_clues = {
    'pregnant woman': ['6 months before delivery', 'placental biopsy', 'prenatal donor'],
    'diabetic': ['elevated fasting glucose', 'HbA1c 8.2%'],
    'alcohol consumer': ['daily ethanol exposure', 'reported alcohol use'],
    'hypertensive': ['hypertension managed with ACE inhibitors', '140/90 mmHg BP'],
}

noisy_sentences = [
    "Sample stored at −80°C.",
    "Standard procedure was followed.",
    "Lab uses ISO9001 certification.",
    "Operator noted batch variation in library prep.",
    "Serum potassium levels were inconclusive.",
    "Participant consumed caffeine prior to sampling.",
    "Unexpected bands in gel electrophoresis.",
    "pH adjusted to 7.4 before sequencing.",
]

# ---------------------------
# 2. Context generator
# ---------------------------

def _ensure_present(context_parts, value, synonyms=None):
    if value in ('nan', '', None):
        return
    if any(value.lower() in p.lower() for p in context_parts):
        return
    if synonyms:
        for alt in synonyms:
            if alt.lower() not in ' '.join(context_parts).lower():
                context_parts.append(alt)
                return
    else:
        context_parts.append(value)


def generate_context(columns, donor_label, explicit_cell_line):
    """
    Build context with at least one cue per non-nan column,
    omitting cues for columns set to 'nan'.
    """
    context_parts = []

    # Insert only for non-nan columns
    if columns['library_source'] != 'nan':
        context_parts.append(random.choice(library_sources_context[columns['library_source']]))
    if columns['cell_type'] != 'nan':
        context_parts.append(random.choice(cell_types_context[columns['cell_type']]))
    if columns['tissue_type'] != 'nan':
        context_parts.append(f"{columns['tissue_type']} tissue")
    if columns['disease'] not in ('nan', 'normal'):
        context_parts.append(columns['disease'])
        if columns['disease'] in disease_markers:
            context_parts.append(random.choice(disease_markers[columns['disease']]))
    if columns['organ'] != 'nan':
        context_parts.append(columns['organ'])
    if columns['library_selection'] != 'nan':
        context_parts.append(random.choice(library_selections_context[columns['library_selection']]))
    if columns['treatment'] != 'no treatment':
        context_parts.append(columns['treatment'])
    # donor info always
    context_parts.append(donor_label)
    # explicit cell line if provided
    if explicit_cell_line:
        context_parts.append(explicit_cell_line)

    # Ensure each non-nan column covered
    for key in ['cell_type', 'tissue_type', 'organ', 'disease', 'library_selection', 'library_source', 'treatment']:
        _ensure_present(context_parts, columns[key], {
            'cell_type': cell_types_context.get(columns[key], []),
            'library_selection': library_selections_context.get(columns[key], []),
            'library_source': library_sources_context.get(columns[key], [])
        }.get(key, None))

    # Add random noise
    context_parts += random.sample(noisy_sentences, k=random.randint(2, 4))
    random.shuffle(context_parts)

    # Mix separators
    separators = [', ', '; ', ' | ', ' ']
    context = ''
    for i, part in enumerate(context_parts):
        sep = random.choice(separators) if i < len(context_parts) - 1 else ''
        context += part + sep
    return context.strip()

# ---------------------------
# 3. Dataset generation
# ---------------------------

data = []
for i in range(5000):
    disease = random.choice(Diseases_context)
    organ = Disease_to_organ.get(disease, random.choice(list(cell_lines_by_organ.keys())))
    default_cell_line = cell_lines_by_organ[organ]
    explicit_cell_line = default_cell_line if default_cell_line != 'Primary tissue' and random.random() < 0.5 else None
    cell_line = default_cell_line if explicit_cell_line else 'Primary tissue'

    # Ensure cell_type matches cell_line when Primary tissue
    if cell_line == 'Primary tissue':
        cell_type = 'Primary tissue'
    else:
        cell_type = cell_types_by_organ_disease.get((organ, disease), cell_types_by_organ[organ])

    tissue_type = tissue_types_by_organ[organ]

    # Randomized nan introduction
    library_selection = random.choice(list(library_selections_context.keys()) + ['nan', 'nan'])
    library_source = random.choice(list(library_sources_context.keys()) + ['nan', 'nan'])

    # Treatment & phenotype
    if disease == 'normal':
        treatment = 'no treatment'
        host_phenotype = 'parental'
    else:
        treatment = random.choice(
            treatments_cancer + treatments_other if 'cancer' in disease or 'tumor' in disease
            else treatments_infection if disease in ['pneumonia', 'hepatitis']
            else ['no treatment']
        )
        host_phenotype = 'persistent' if treatment != 'no treatment' else 'parental'

    treatment_time = 'Pre-treatment' if treatment != 'no treatment' else 'no treatment'
    response = random.choice(clinical_responses + ['nan', 'nan']) if treatment != 'no treatment' else 'nan'
    donor_info = random.choice(list(donor_clues.keys()) + ['male, 45 years', 'female, 60 years'])

    columns = {
        'cell_type': cell_type if random.random() > 0.2 else 'nan',
        'tissue_type': tissue_type if random.random() > 0.2 else 'nan',
        'cell_line': cell_line,
        'organ': organ if random.random() > 0.2 else 'nan',
        'disease': disease if random.random() > 0.2 else 'nan',
        'host_phenotype': host_phenotype,
        'library_selection': library_selection,
        'library_source': library_source,
        'treatment': treatment
    }

    context = generate_context(columns, donor_info, explicit_cell_line)
    run_accession = f"ERR{str(i+1).zfill(6)}"
    data.append({
        'run_accession': run_accession,
        'context': context,
        'cell_type': columns['cell_type'],
        'tissue_type': columns['tissue_type'],
        'cell_line': cell_line,
        'organ': columns['organ'],
        'disease': columns['disease'],
        'host_phenotype': columns['host_phenotype'],
        'library_selection': columns['library_selection'],
        'library_source': columns['library_source'],
        'treatment': columns['treatment'],
        'treatment_time': treatment_time,
        'response': response,
        'donor_information': donor_info
    })

# save
df = pd.DataFrame(data)
output_path = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/simulated_metadata.csv'
df.to_csv(output_path, index=False)
