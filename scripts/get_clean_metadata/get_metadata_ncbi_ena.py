##########################################################################################
# IMPORT
import os
import csv
import requests
import xml.etree.ElementTree as ET
import subprocess
from multiprocessing import Pool
import argparse

########################################################################################################################
#PATHS
parser = argparse.ArgumentParser(description="Download metadata from NCBI Ensembl")
parser.add_argument("--base_path", type=str, required=True, help="Base path to MetaMap")
args = parser.parse_args()

base_path = args.base_path
RAW_CSV = os.path.join(base_path, "annotated_totalRNA.csv")
RUNS_TSV = os.path.join(base_path, "runs.tsv")
METADATA_DIR = os.path.join(base_path, "metadata")
OUTPUT_FILE = os.path.join(base_path, "metadata_sra.txt")
FLAG_FILE = os.path.join(base_path, "STEP1_1.flag")
FIELDS = "study_accession,first_public,study_title,project_name,run_accession,sample_accession,sample_title,sample_description,library_name,library_selection,library_source,library_strategy,library_construction_protocol,library_layout,rna_integrity_num,instrument_platform,rt_prep_protocol,cell_line,cell_type,tissue_lib,tissue_type,host_phenotype,isolate,age,host_body_site,sampling_site,base_count,description"

########################################################################################################################
#FUNCTIONS

#from annotated total rna extract runs
def extract_run_accessions_from_file():
    os.makedirs(METADATA_DIR, exist_ok=True)
    with open(RAW_CSV, 'r', encoding='ISO-8859-1') as infile, open(RUNS_TSV, 'w', encoding='utf-8') as outfile:
        reader = csv.reader(infile, delimiter=';')
        writer = csv.writer(outfile, delimiter='\t')
        for row in reader:
            if len(row) > 4 and row[4].strip():
                writer.writerow([row[4]])


#from project accession EBI ENA
def get_run_accessions(project_id):
    os.makedirs(METADATA_DIR, exist_ok=True)
    url = f"https://www.ebi.ac.uk/ena/portal/api/search"
    query = {
        "result": "read_run",
        "query": f"study_accession={project_id} AND library_strategy=RNA",
        "fields": "run_accession",
        "format": "tsv"
    }
    response = requests.get(url, params=query, timeout=30)
    if response.status_code == 200:
        run_accessions = response.text.strip().split("\n")[1:]
        with open(RUNS_TSV, 'w', encoding='utf-8') as outfile:
            outfile.write("\n".join(run_accessions))
        return run_accessions
    else:
        print(f"Error: can't fetch runs for project {project_id}: {response.status_code}")
        return []


# Replace download_metadata with Bash script execution
def execute_bash_download_metadata():
    bash_script = f"""#!/bin/bash
                if [ -d "{METADATA_DIR}" ]; then
                    echo "{METADATA_DIR} already downloaded."
                else
                    /usr/bin/mkdir -p "{METADATA_DIR}"
                    /usr/bin/tail -n +2 "{RUNS_TSV}" | while IFS=$'\t' read -r RUN_ACCESSION; do
                        OUTPUT_FILE="{METADATA_DIR}/${{RUN_ACCESSION}}_metadata.xml"
                        if [ ! -f "$OUTPUT_FILE" ]; then
                            echo "Download metadonn\u00e9es pour $RUN_ACCESSION"
                            /usr/bin/curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=sra&id=${{RUN_ACCESSION}}&retmode=text" \
                                 -o "$OUTPUT_FILE"
                        fi
                    done
                fi
                """
    with open("download_metadata.sh", "w") as bash_file:
        bash_file.write(bash_script)

    os.chmod("download_metadata.sh", 0o755)
    subprocess.run(["./download_metadata.sh"], check=True)


#get metadata from xml
def extract_and_save_metadata(run_accession):
    xml_file = os.path.join(METADATA_DIR, f"{run_accession}_metadata.xml")
    try:
        #get xml
        tree = ET.parse(xml_file)
        root = tree.getroot()
        sample_metadata = "".join(root.findall(".//SAMPLE")[0].itertext()).replace('\n', ' ')
        study_metadata = "".join(root.findall(".//STUDY")[0].itertext()).replace('\n', ' ')
        #search in sample and study
        query = f"result=read_run&query=run_accession%3D{run_accession}&format=tsv&fields={FIELDS}&limit=1"
        ena_url = "https://www.ebi.ac.uk/ena/portal/api/search"
        response = requests.post(ena_url, data=query)
        if response.status_code == 200:
            ena_data = response.text.strip().split("\n")[-1]
            #first line
            with open(OUTPUT_FILE, 'a') as f_out:
                f_out.write(f"{ena_data}\t{sample_metadata}\t{study_metadata}\n")
        else:
            print(f"Can't get {run_accession} from API.")
    except Exception as e:
        print(f"Error {run_accession}: {e}")

########################################################################################################################
#MAIN FUNCTION
def main():
    #run extraction from file (columns 1)
    # extract_run_accessions()

    #run extraction from project accession
    get_run_accessions("PRJNA523380")

    with open(RUNS_TSV, 'r') as file:
        run_accessions = [line.strip() for line in file]
        if not run_accessions:
            print("Error: no run accessions found")
            return

    # Execute Bash script to download metadata
    execute_bash_download_metadata()

    #get API structured metadata from API and study/sample extraction from xml
    with Pool(20) as pool:  # multiprocess on 20 CPUs
        pool.map(extract_and_save_metadata, run_accessions)

    #create flag end process before cleaning
    open(FLAG_FILE, 'w').close()


if __name__ == "__main__":
    main()
