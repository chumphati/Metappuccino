import csv

# Chargement du premier fichier pour obtenir la correspondance
metadata_dict = {}
with open('/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap2/results/METADATA/cleaned_metadata_sra.txt', 'r', newline='') as metadata_file:
    reader = csv.DictReader(metadata_file, delimiter='\t')
    for row in reader:
        metadata_dict[row['run_accession']] = row['study_accession']

# Chargement du deuxième fichier et ajout de la colonne 'study accession'
with open('/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap2/results/SPECIFIC_RUN_ANALYSIS/10k_llm_metadata_analysis.csv', 'r', newline='') as analysis_file:
    reader = csv.reader(analysis_file, delimiter='\t')
    header = next(reader)
    data = list(reader)

# Ajout de la nouvelle colonne en deuxième position
header.insert(1, 'study accession')

# Correspondance et insertion des valeurs
for row in data:
    run_accession_number = row[0]
    row.insert(1, metadata_dict.get(run_accession_number, ''))

# Sauvegarde du résultat dans un nouveau fichier
with open('/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap2/results/SPECIFIC_RUN_ANALYSIS/10k_llm_metadata_analysis_with_study.csv', 'w', newline='') as output_file:
    writer = csv.writer(output_file)
    writer.writerow(header)
    writer.writerows(data)

print("Le fichier a été créé avec succès.")
