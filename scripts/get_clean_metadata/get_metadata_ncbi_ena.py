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
RAW_CSV = os.path.join(base_path, "mela-select.tsv")
RUNS_TSV = os.path.join(base_path, "runs.tsv")
METADATA_DIR = os.path.join(base_path, "metadata")
OUTPUT_FILE = os.path.join(base_path, "metadata_sra.txt")
FLAG_FILE = os.path.join(base_path, "STEP1_1.flag")
# FIELDS = "study_accession,first_public,study_title,project_name,run_accession,sample_accession,sample_title,sample_description,library_name,library_selection,library_source,library_strategy,library_construction_protocol,library_layout,rna_integrity_num,instrument_platform,rt_prep_protocol,cell_line,cell_type,tissue_lib,tissue_type,host_phenotype,isolate,age,host_body_site,sampling_site,base_count,description"
# HEADER_LINE = "run_accession\tfirst_public\tstudy_title\tproject_name\tstudy_accession\tsample_accession\tsample_title\tsample_description\tlibrary_name\tlibrary_selection\tlibrary_source\tlibrary_strategy\tlibrary_construction_protocol\tlibrary_layout\trna_integrity_num\tinstrument_platform\trt_prep_protocol\tcell_line\tcell_type\ttissue_lib\ttissue_type\thost_phenotype\tisolate\tage\thost_body_site\tsampling_site\tbase_count\tdescription\tsample_metadata_ncbi\tstudy_metadata_ncbi\n"

FIELDS = "study_accession,first_public,study_title,project_name,run_accession,sample_accession,sample_title,sample_description,library_name,library_selection,library_source,library_strategy,library_construction_protocol,library_layout,rna_integrity_num,instrument_platform,rt_prep_protocol,cell_line,cell_type,tissue_lib,tissue_type,host_phenotype,isolate,age,host_body_site,sampling_site,base_count,description,host_sex,sex,submitted_host_sex,disease,host_status"
HEADER_LINE = "run_accession\tfirst_public\tstudy_title\tproject_name\tstudy_accession\tsample_accession\tsample_title\tsample_description\tlibrary_name\tlibrary_selection\tlibrary_source\tlibrary_strategy\tlibrary_construction_protocol\tlibrary_layout\trna_integrity_num\tinstrument_platform\trt_prep_protocol\tcell_line\tcell_type\ttissue_lib\ttissue_type\thost_phenotype\tisolate\tage\thost_body_site\tsampling_site\tbase_count\tdescription\thost_sex\tsex\tsubmitted_host_sex\tdisease\thost_status\tsample_metadata_ncbi\tstudy_metadata_ncbi\n"

########################################################################################################################
#FUNCTIONS


#from annotated total rna extract runs
def extract_run_accessions_from_file():
    with open(RAW_CSV, 'r', encoding='ISO-8859-1') as infile, open(RUNS_TSV, 'w', encoding='utf-8') as outfile:
        reader = csv.reader(infile, delimiter='\t')
        writer = csv.writer(outfile, delimiter='\t')
        for row in reader:
            if len(row) > 0 and row[0].strip():
                writer.writerow([row[0]])


#get header for out file
def ensure_output_file_header():
    if not os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'w') as f_out:
            f_out.write(HEADER_LINE)


#from project accession EBI ENA
def get_run_accessions(project_id):
    url = f"https://www.ebi.ac.uk/ena/portal/api/search"
    query = {
        "result": "read_run",
        "query": f"study_accession={project_id} AND library_strategy=RNA",
        "fields": "run_accession",
        "format": "tsv"
    }
    response = requests.get(url, params=query, timeout=30)
    run_accessions = response.text.strip().split("\n")[1:]
    with open(RUNS_TSV, 'w', encoding='utf-8') as outfile:
        outfile.write("\n".join(run_accessions))
    return run_accessions


#download metadata
def execute_bash_download_metadata():
    bash_script = f"""#!/bin/bash
                if [ -d "{METADATA_DIR}" ]; then
                    echo "{METADATA_DIR} already downloaded."
                else
                    /usr/bin/mkdir -p "{METADATA_DIR}"
                    /usr/bin/tail -n +2 "{RUNS_TSV}" | while IFS=$'\t' read -r RUN_ACCESSION; do
                        RUN_ACCESSION=$(echo "$RUN_ACCESSION" | tr -d '\r' | tr -d '\n' | tr -d ' ')
                        OUTPUT_FILE="{METADATA_DIR}/${{RUN_ACCESSION}}_metadata.xml"
                        if [ ! -f "$OUTPUT_FILE" ]; then
                            echo "Download metadata for $RUN_ACCESSION"
                            /usr/bin/curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=sra&id=${{RUN_ACCESSION}}&retmode=text" \
                                 -o "$OUTPUT_FILE"
                        fi
                    done
                fi
                """
    subprocess.run(bash_script, shell=True, check=True, executable="/bin/bash")


#get specific metadata from xml
def extract_and_save_metadata(run_accession):
    xml_file = os.path.join(METADATA_DIR, f"{run_accession}_metadata.xml")
    print(run_accession)
    try:
        #get xml
        tree = ET.parse(xml_file)
        root = tree.getroot()
        sample_metadata = "".join(root.findall(".//SAMPLE")[0].itertext()).replace('\n', ' ')
        study_metadata = "".join(root.findall(".//STUDY")[0].itertext()).replace('\n', ' ')
        print(sample_metadata)

        #search in sample and study
        curl_command = [
            "/usr/bin/curl", "-s", "-X", "POST", "-H", "Content-Type: application/x-www-form-urlencoded",
            "-d", f"result=read_run&query=run_accession%3D{run_accession}&format=tsv&fields={FIELDS}&limit=1",
            "https://www.ebi.ac.uk/ena/portal/api/search"
        ]

        response = subprocess.run(curl_command, capture_output=True, text=True)
        ena_data = response.stdout.strip().split("\n")[-1]

        #first line
        with open(OUTPUT_FILE, 'a') as f_out:
            f_out.write(f"{ena_data}\t{sample_metadata}\t{study_metadata}\n")

    except Exception as e:
        print(f"Error {run_accession}: {e}")

########################################################################################################################
#MAIN FUNCTION
def main():
    ensure_output_file_header()
    #run extraction from file (columns 1)
    extract_run_accessions_from_file()

    #run extraction from project accession
    # get_run_accessions("PRJNA523380")

    with open(RUNS_TSV, 'r') as file:
        next(file)
        run_accessions = [line.strip() for line in file if line.strip()]
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
