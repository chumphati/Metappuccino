########################################################################################################################
#IMPORT LIB
import os
import csv
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
from multiprocessing import Pool
from time import sleep
from tqdm import tqdm

########################################################################################################################
#PATHS
BASE_PATH = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results"
RAW_CSV = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/data/raw/annotated_totalRNA.csv"
RUNS_TSV = os.path.join(BASE_PATH, "runs.tsv")
METADATA_DIR = os.path.join(BASE_PATH, "metadata")
OUTPUT_FILE = os.path.join(BASE_PATH, "METADATA_SRA.txt")
FLAG_FILE = os.path.join(BASE_PATH, "logs/STEP1_1.flag")
FIELDS = "study_accession,first_public,study_title,project_name,run_accession,sample_accession,sample_title,sample_description,library_name,library_selection,library_source,library_strategy,library_construction_protocol,library_layout,rna_integrity_num,instrument_platform,rt_prep_protocol,cell_line,cell_type,tissue_lib,tissue_type,host_phenotype,isolate,age,host_body_site,sampling_site,base_count,description"

########################################################################################################################
#FUNCTIONS
#from annotated total rna extract runs
def extract_run_accessions():
    os.makedirs(METADATA_DIR, exist_ok=True)
    with open(RAW_CSV, 'r', encoding='ISO-8859-1') as infile, open(RUNS_TSV, 'w', encoding='utf-8') as outfile:
        reader = csv.reader(infile, delimiter=';')
        writer = csv.writer(outfile, delimiter='\t')
        for row in reader:
            if len(row) > 4 and row[4].strip():
                writer.writerow([row[4]])


#download metadata from ncbi
def download_metadata(run_accession):
    xml_file = os.path.join(METADATA_DIR, f"{run_accession}_metadata.xml")
    if not os.path.exists(xml_file):
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=sra&id={run_accession}&retmode=text"
        try:
            time.sleep(0.34)
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(xml_file, 'wb') as file:
                    file.write(response.content)
            else:
                print(f"Error {run_accession}: {response.status_code}")
        except Exception as e:
            print(f"Error {run_accession}: {e}")


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
    extract_run_accessions()
    with open(RUNS_TSV, 'r') as file:
        run_accessions = [line.strip() for line in file]

    #download all metadata file of the extracted runs from ncbi in parallel
    with Pool(5) as pool:
        list(tqdm(pool.imap(download_metadata, run_accessions), total=len(run_accessions)))

    #get API structured metadata from API and study/sample extraction from xml
    with Pool(20) as pool:  # multiprocess on 20 CPUs
        pool.map(extract_and_save_metadata, run_accessions)

    #create flag end process before cleaning
    open(FLAG_FILE, 'w').close()


if __name__ == "__main__":
    main()
