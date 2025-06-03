import csv

metadata_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/synthetic_balanced.csv'
output_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/balanced_simulated_finetune_data.csv'


def create_prompt(run_accession, context):
    prompt = f"""Run accession: {run_accession}
            Metadata to analyze: {context}

            For each row in the metadata line (the first line contains the column names), extract and format the following information concisely. For the 8 following categories, provide a single answer without redundancy. Each category **MUST** have one distinct and explicit answer, even if inferred. **Do not leave any category empty.** Do not repeat information already provided in previous categories. Remove redundant text.
            cell_type - The type of cell in the sample. It can be among connective cells, fat cells, specialized integrated cells, migratory cells, kidney cells, muscle cells, bone cells, cartilage cells, stomach cells, sex cells, lung cells, pancreatic cells, liver cells (= hepatic cells), intestinal cells (= enterocytes), nerve cells, blood cells (= blood elements). First cite one of those huge categories, and if possible specify more specifically into each categorie (example: resident cells (fibroblasts, fibrocytes, tendinocytes, keratocytes), migratory cells (lymphocytes, histiocytes, melanocytes, natural killer cells), fat cells (mesenchymal cells, white adipocytes, brown adipocytes), specialized integrated cells (neuroepithelial cells, myoepithelial cells, goblet cells), kidney cells (podocytes, distal tubular cells, proximal tubular cells), muscle cells (smooth muscle cells, cardiac muscle cells, skeletal muscle cells), bone cells (osteoblasts, osteoclasts, osteocytes), cartilage cells (chondroblasts, hypertrophic chondrocytes), connective tissue cells (type A synoviocytes, type B synoviocytes), stomach cells (chief cells, parietal cells), sex cells (spermatozoa, spermatocytes, oocytes, sertoli cells, leydig cells), lung cells (type 1 pneumocytes, type 2 pneumocytes), pancreatic cells (alpha cells, beta cells, delta cells), liver cells (hepatocytes, kupffer cells), intestinal cells (enterocytes), nerve cells (neurons, neuroblasts, astrocytes, oligodendrocytes, schwann cells), blood cells (erythrocytes, thrombocytes, leukocytes, monocytes, eosinophils, basophils, neutrophils)). If not provided, deduce based on the tissue type and the rest of the context and state the inference. Use thee Cell Ontology terms terminology. WARNING: IF CELL LINE CAN'T BE DETERMINE AND IF IT IS A PRIMARY TISSUE THERE, mark this category as 'Primary tissue' too.
            tissue_type – The tissue type from which the sample originates (e.g., epithelial tissue, connective tissue, muscle tissue, nervous tissue,). If not specified, deduce from context, from the organ or the type of cell it comes from.
            cell_line – Specify the cell line which is a strict code, or state 'Primary tissue' if the sample is from a primary tissue and not a cell line.
            organ - Provide me the organ(s) concerned by this study, in the UBERON GTEX terminology for the tissue type (e.g., lungs, liver, heart, etc ... name of the organ). If not specified, deduce from context, or search one related to the tissue.
            disease – Return the Disease Ontology term corresponding to the disease associated with the sample in the format of the disease name. If the sample is explicitly described as 'normal' or 'healthy', or something similar do not infer any disease. In this case, do not search for disease-related information in the context. If the sample is not explicitly labeled as 'normal' or 'healthy' or 'no disease' etc, infer the disease from the context only if it is directly related to the sample (e.g., sample title, description, or metadata fields directly describing the sample). In case of cancer, something adjacent means that it's healthy. Non-disease conditions (e.g., pregnancy, aging, lifestyle factors) should be placed in the Donor information output column instead of the Disease Ontology Term field. DO NOT JUST STATE 'DISEASE' without inferring the type of disease. If nothing says there is a disease or any problem, state 'normal'.
            host_phenotype - Based on the given context, determine if the phenotype classification is 'parental' (Refers to the original, untreated cell line or population, which has not been exposed to selective pressure (such as drug treatment). Typically represents the baseline phenotype.) or 'persistant' (Refers to cells or populations that have survived treatment and exhibit drug persistence or resistance, often through adaptive mechanisms rather than genetic mutations.). Choose between those two possibilities, use your knowledge if the answer is not clear.
            library_selection - Based on the given context, determine the library selection fixed category by searching for specific keywords or synonyms that match one of the five strict categories: 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or 'other'. Assign 'polyA' if the context contains any of the following terms OR SIMIILAR MEANING THAT CAN BE INFERRED: 'PolyA', 'poly.A', 'oligo.dT', 'oligodT', 'truseq.mrna', 'truseq.stranded.mrna', 'truseq.standard.mrna', 'smarter.mRNA', 'stran ded.mRNA'. Assign 'inverse rRNA' if the context mentions depletion of ribosomal RNA with any of these terms OR SIMIILAR MEANING THAT CAN BE INFERRED: 'ribominus', 'ribodep', 'ribozero', 'ribo.zero', 'riboerase', 'ribogone', 'ribocop', 'ribo-dep', 'ribo-mi', 'ribo minus', 'depleted ribosom', 'remove ribosom', 'TruSeq.Stranded.Total', 'TruSeq.Total', 'SMARTer.Stranded.Total', 'SMARTer.Total'. Assign 'hybrid selection' if the context refers to hybrid capture or exon selection using any of these terms OR SIMIILAR MEANING THAT CAN BE INFERRED: 'Hybrid.Selection', 'Exon.capture', 'Exome.capture', 'RNA.Exome', 'geoMX'. Assign 'small RNA' if the context refers to small RNA isolation with keywords such as 'TruSeq.Small', 'size.fraction' OR SIMIILAR MEANING THAT CAN BE INFERRED. Assign 'other' if none of the above terms are found. Return only the exact category name: 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or 'other', with no additional text.
            library_source - Based on the given context, determine the library source category by searching for specific keywords that match one of the two strict categories: 'single-cell' or 'bulk'. Assign 'single-cell' if the context contains any of the following terms: 'TRANSCRIPTOMIC SINGLE CELL', 'chromium', '10x', 'single.cell' OR SIMIILAR MEANING THAT CAN BE INFERRED. Assign 'bulk' if none of the above terms are found. Return only the exact category name: 'single-cell' or 'bulk', with no additional text.
            treatment – If the sample received a treatment (e.g., drug, irradiation), specify it. Otherwise return 'no treatment'.
            treatment_time – Specify the state of the current treatment if it can be found or inferred (on treatment, pre treatment, etc...). If no treatment, return 'no treatment'. Otherwise return 'nan'.
            response – Indicate the response to treatment if available (e.g., resistance, sensitivity). Otherwise return 'nan'.
            donor_information – Specify any relevant donor info such as age, sex, pregnancy, lifestyle, etc. Otherwise return 'nan'.

            If any information is missing in the metadat can't be inferred for previous instruction, specify 'nan'. Don't double the answer. I want only one answer per category.
            Strict output format (no additional text or special characters, no duplicated answers), ONLY print the answer. Do not elaborate.:
            Output in this form, with the category and prediction SEPARATED BY DOUBLE POINTS LIKE: Cetegory: [single unique answer]

            Respond with exactly one line. Do not elaborate. Only one word (or 3 max) is allowed after the "Category:".

            (one line per category, no repetition):
            Here is the output:
            """
    return prompt


with open(metadata_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    reader = csv.DictReader(infile)
    writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)
    writer.writerow(['prompt', 'output'])

    for row in reader:
        run_accession = row.get("run_accession", "").strip()
        context = row.get("context", "").strip()
        prompt = create_prompt(run_accession, context)

        output_lines = []
        for key, value in row.items():
            if key in ("run_accession", "context"):
                continue
            value_cleaned = value.strip() if value else 'nan'
            output_lines.append(f"{key}: {value_cleaned}")

        output_text = "\n".join(output_lines)
        writer.writerow([prompt, output_text])
