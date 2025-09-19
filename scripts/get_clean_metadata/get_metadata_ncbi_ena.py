##########################################################################################
# IMPORT
import os
import csv
import requests
import xml.etree.ElementTree as ET
import subprocess
from multiprocessing import Pool
import argparse
import time
import shutil

########################################################################################################################
#PATHS
parser = argparse.ArgumentParser(description="Download metadata from NCBI Ensembl")
parser.add_argument("--base_path", type=str, required=True, help="Base path to Metappuccino")
parser.add_argument("--verbose", action="store_true", help="Verbose output")
args = parser.parse_args()

base_path = args.base_path
# RAW_CSV = os.path.join(base_path, "annotated_totalRNA.csv")
# RAW_CSV = os.path.join(base_path, "ena_results.tsv")
RUNS_TSV = os.path.join(base_path, "runs.txt")
METADATA_DIR = os.path.join(base_path, "metadata")
OUTPUT_FILE = os.path.join(base_path, "metadata_sra.txt")
FLAG_FILE = os.path.join(base_path, "STEP1_1.flag")
FIELDS = "study_accession,first_public,study_title,project_name,run_accession,sample_accession,sample_title,sample_description,library_name,library_selection,library_source,library_strategy,library_construction_protocol,library_layout,rna_integrity_num,instrument_platform,rt_prep_protocol,cell_line,cell_type,tissue_lib,tissue_type,host_phenotype,isolate,age,host_body_site,sampling_site,base_count,description"
HEADER_LINE = "run_accession\tfirst_public\tstudy_title\tproject_name\tstudy_accession\tsample_accession\tsample_title\tsample_description\tlibrary_name\tlibrary_selection\tlibrary_source\tlibrary_strategy\tlibrary_construction_protocol\tlibrary_layout\trna_integrity_num\tinstrument_platform\trt_prep_protocol\tcell_line\tcell_type\ttissue_lib\ttissue_type\thost_phenotype\tisolate\tage\thost_body_site\tsampling_site\tbase_count\tdescription\tsample_metadata_ncbi\tstudy_metadata_ncbi\n"

# FIELDS = "study_accession,first_public,study_title,project_name,run_accession,sample_accession,sample_title,sample_description,library_name,library_selection,library_source,library_strategy,library_construction_protocol,library_layout,rna_integrity_num,instrument_platform,rt_prep_protocol,cell_line,cell_type,tissue_lib,tissue_type,host_phenotype,isolate,age,host_body_site,sampling_site,base_count,description,host_sex,sex,submitted_host_sex,disease,host_status"
# HEADER_LINE = "run_accession\tfirst_public\tstudy_title\tproject_name\tstudy_accession\tsample_accession\tsample_title\tsample_description\tlibrary_name\tlibrary_selection\tlibrary_source\tlibrary_strategy\tlibrary_construction_protocol\tlibrary_layout\trna_integrity_num\tinstrument_platform\trt_prep_protocol\tcell_line\tcell_type\ttissue_lib\ttissue_type\thost_phenotype\tisolate\tage\thost_body_site\tsampling_site\tbase_count\tdescription\thost_sex\tsex\tsubmitted_host_sex\tdisease\thost_status\tsample_metadata_ncbi\tstudy_metadata_ncbi\n"

VERBOSE = args.verbose
vprint = print if VERBOSE else (lambda *a, **k: None)

def _find_cmd(name, extra_paths=None):
    p = shutil.which(name)
    if p:
        return p
    for path in (extra_paths or [f"/usr/bin/{name}", f"/bin/{name}", f"/usr/local/bin/{name}"]):
        if os.path.exists(path):
            return path
    return name

BASH_PATH = _find_cmd("bash", ["/bin/bash", "/usr/bin/bash", "/usr/local/bin/bash"])
CURL_PATH = _find_cmd("curl", ["/usr/bin/curl", "/bin/curl", "/usr/local/bin/curl"])
MKDIR_PATH = _find_cmd("mkdir")
CAT_PATH = _find_cmd("cat")
GREP_PATH = _find_cmd("grep")

########################################################################################################################
#FUNCTIONS


#from annotated total rna extract runs
#annotated_totalRNA.csv
# def extract_run_accessions_from_file():
#     with open(RAW_CSV, 'r', encoding='ISO-8859-1') as infile, open(RUNS_TSV, 'w', encoding='utf-8') as outfile:
#         reader = csv.reader(infile, delimiter=';')
#         writer = csv.writer(outfile, delimiter=';')
#         for row in reader:
#             if len(row) > 4 and row[4].strip():
#                 writer.writerow([row[4]])

#mela-select.tsv
# def extract_run_accessions_from_file():
#     with open(RAW_CSV, 'r', encoding='ISO-8859-1') as infile, open(RUNS_TSV, 'w', encoding='utf-8') as outfile:
#         reader = csv.reader(infile, delimiter='\t')
#         writer = csv.writer(outfile, delimiter='\t')
#         for row in reader:
#             if len(row) > 0 and row[0].strip():
#                 writer.writerow([row[0]])

#ena_results.tsv
# def extract_run_accessions_from_file():
#     with open(RAW_CSV, 'r', encoding='ISO-8859-1') as infile, open(RUNS_TSV, 'w', encoding='utf-8') as outfile:
#         reader = csv.reader(infile, delimiter='\t')
#         writer = csv.writer(outfile, delimiter='\t')
#         for row in reader:
#             if len(row) > 0 and row[0].strip():
#                 writer.writerow([row[0]])


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
                MKDIR=\"{MKDIR_PATH}\"
                CAT=\"{CAT_PATH}\"
                GREP=\"{GREP_PATH}\"
                CURL=\"{CURL_PATH}\"
                if [ -d \"{METADATA_DIR}\" ]; then
                    echo \"{METADATA_DIR} already downloaded.\"
                else
                    \"$MKDIR\" -p \"{METADATA_DIR}\"
                    \"$CAT\" \"{RUNS_TSV}\" | \"$GREP\" -v '^\\s*$' | while IFS=$'\\t' read -r RUN_ACCESSION; do
                        RUN_ACCESSION=$(echo \"$RUN_ACCESSION\" | tr -d '\\r' | tr -d '\\n' | tr -d ' ')
                        OUTPUT_FILE=\"{METADATA_DIR}/${{RUN_ACCESSION}}_metadata.xml\"
                        TMP_FILE=\"${{OUTPUT_FILE}}.tmp\"
                        if [ ! -f \"$OUTPUT_FILE\" ]; then
                            echo \"Download metadata for $RUN_ACCESSION\"
                            \"$CURL\" -SsfL --retry 5 --retry-delay 2 --retry-all-errors --connect-timeout 10 --max-time 120 \\
                                 \"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=sra&id=${{RUN_ACCESSION}}&retmode=xml\" \\
                                 -o \"$TMP_FILE\" && \\
                            if head -c 1 \"$TMP_FILE\" | grep -q '<'; then
                                if grep -q '<\\(STUDY\\|SAMPLE\\)\\b' \"$TMP_FILE\"; then
                                    mv \"$TMP_FILE\" \"$OUTPUT_FILE\"
                                else
                                    rm -f \"$TMP_FILE\"
                                fi
                            else
                                rm -f \"$TMP_FILE\"
                            fi
                            sleep 1
                        fi
                    done
                fi
                """
    subprocess.run(bash_script, shell=True, check=True, executable=BASH_PATH)


def _looks_like_xml(path):
    try:
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            return False
        with open(path, "rb") as f:
            start = f.read(4096)
        if not start.lstrip().startswith(b"<"):
            return False
        txt = start.decode("utf-8", errors="ignore")
        if ("<STUDY" not in txt) and ("<SAMPLE" not in txt):
            if os.path.getsize(path) <= 2_000_000:
                with open(path, "r", encoding="utf-8", errors="ignore") as f2:
                    alltxt = f2.read()
                if ("<STUDY" not in alltxt) and ("<SAMPLE" not in alltxt):
                    return False
        return True
    except Exception:
        return False

def _download_xml_ncbi(run_accession, dest, max_retries=3):
    headers = {"User-Agent": "metappuccino/1.0 (contact: none)"}
    params = {"db": "sra", "id": run_accession, "retmode": "xml"}
    tmp = dest + ".tmp"
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params=params, headers=headers, timeout=60,
            )
            if r.status_code == 200 and r.text.strip().startswith("<") and (("<STUDY" in r.text) or ("<SAMPLE" in r.text)):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(tmp, "w", encoding="utf-8") as fxml:
                    fxml.write(r.text)
                os.replace(tmp, dest)
                time.sleep(1.0)
                return True
        except Exception:
            pass
        time.sleep(1 * attempt)
    return False

def _parse_xml_with_retry(xml_file, max_retries=2):
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            tree = ET.parse(xml_file)
            return tree.getroot()
        except ET.ParseError as e:
            last_exc = e
            time.sleep(1 + attempt)
            if attempt < max_retries:
                _download_xml_ncbi(os.path.basename(xml_file).split("_metadata.xml")[0], xml_file, max_retries=2)
    if last_exc:
        raise last_exc
    return None


#get specific metadata from xml
def extract_and_save_metadata(run_accession):
    xml_file = os.path.join(METADATA_DIR, f"{run_accession}_metadata.xml")

    if not os.path.exists(xml_file):
        try:
            ok = _download_xml_ncbi(run_accession, xml_file, max_retries=3)
            if not ok:
                pass
        except Exception:
            pass

    try:
        #get xml
        if not _looks_like_xml(xml_file):
            _download_xml_ncbi(run_accession, xml_file, max_retries=2)
        root = _parse_xml_with_retry(xml_file)
        sample_metadata = " ".join(root.findall(".//SAMPLE")[0].itertext()).replace('\n', ' ')
        study_metadata = " ".join(root.findall(".//STUDY")[0].itertext()).replace('\n', ' ')

        time.sleep(4)

        #search in sample and study
        curl_command = [
            CURL_PATH, "-s", "-X", "POST", "-H", "Content-Type: application/x-www-form-urlencoded",
            "-d", f"result=read_run&query=run_accession%3D{run_accession}&format=tsv&fields={FIELDS}&limit=1",
            "https://www.ebi.ac.uk/ena/portal/api/search"
        ]

        response = subprocess.run(curl_command, capture_output=True, text=True)
        vprint("RUN: ", run_accession)
        vprint("RAW ENA RESPONSE:", response.stdout)
        ena_data = response.stdout.strip().split("\n")[-1]
        vprint("ENA DATA: ", ena_data)

        if not response.stdout.strip():
            vprint(f"Warning: No data returned from ENA API for run accession {run_accession}.")
        if response.returncode != 0:
            vprint(f"Error: curl command failed with return code {response.returncode}.")
            vprint("stderr:", response.stderr)

        #first line
        with open(OUTPUT_FILE, 'a') as f_out:
            f_out.write(f"{ena_data}\t{sample_metadata}\t{study_metadata}\n")
            vprint("STUDY DATA: ", study_metadata)

    except FileNotFoundError as e:
        vprint(f"Error: XML file not found {e}")
        try:
            with open(xml_file, 'r', encoding='utf-8') as xf:
                xml_raw = xf.read().replace('\n', ' ')
        except FileNotFoundError:
            xml_raw = ""
        curl_command = [
            CURL_PATH, "-s", "-X", "POST", "-H", "Content-Type: application/x-www-form-urlencoded",
            "-d", f"result=read_run&query=run_accession%3D{run_accession}&format=tsv&fields={FIELDS}&limit=1",
            "https://www.ebi.ac.uk/ena/portal/api/search"
        ]
        proc = subprocess.run(curl_command, capture_output=True, text=True)
        ena_lines = proc.stdout.strip().split("\n") if proc.stdout else []
        ena_data = ena_lines[-1] if len(ena_lines) > 1 else (ena_lines[0] if ena_lines else "")
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
            f_out.write(f"{ena_data}\terror\terror\n")
    except ET.ParseError as e:
        vprint(f"Error: Failed to parse XML {e}")
        try:
            with open(xml_file, 'r', encoding='utf-8') as xf:
                xml_raw = xf.read().replace('\n', ' ')
        except FileNotFoundError:
            xml_raw = ""
        curl_command = [
            CURL_PATH, "-s", "-X", "POST", "-H", "Content-Type: application/x-www-form-urlencoded",
            "-d", f"result=read_run&query=run_accession%3D{run_accession}&format=tsv&fields={FIELDS}&limit=1",
            "https://www.ebi.ac.uk/ena/portal/api/search"
        ]
        proc = subprocess.run(curl_command, capture_output=True, text=True)
        ena_lines = proc.stdout.strip().split("\n") if proc.stdout else []
        ena_data = ena_lines[-1] if len(ena_lines) > 1 else (ena_lines[0] if ena_lines else "")
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
            f_out.write(f"{ena_data}\terror\terror\n")
    except subprocess.SubprocessError as e:
        vprint(f"Error: Subprocess execution failed {e}")
        try:
            with open(xml_file, 'r', encoding='utf-8') as xf:
                xml_raw = xf.read().replace('\n', ' ')
        except FileNotFoundError:
            xml_raw = ""
        curl_command = [
            CURL_PATH, "-s", "-X", "POST", "-H", "Content-Type: application/x-www-form-urlencoded",
            "-d", f"result=read_run&query=run_accession%3D{run_accession}&format=tsv&fields={FIELDS}&limit=1",
            "https://www.ebi.ac.uk/ena/portal/api/search"
        ]
        proc = subprocess.run(curl_command, capture_output=True, text=True)
        ena_lines = proc.stdout.strip().split("\n") if proc.stdout else []
        ena_data = ena_lines[-1] if len(ena_lines) > 1 else (ena_lines[0] if ena_lines else "")
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
            f_out.write(f"{ena_data}\terror\terror\n")
    except Exception as e:
        vprint(f"Unexpected error {run_accession}: {e}")

        try:
            with open(xml_file, 'r', encoding='utf-8') as xf:
                xml_raw = xf.read().replace('\n', ' ')
        except FileNotFoundError:
            xml_raw = ""

        curl_command = [
            CURL_PATH, "-s", "-X", "POST", "-H", "Content-Type: application/x-www-form-urlencoded",
            "-d", f"result=read_run&query=run_accession%3D{run_accession}&format=tsv&fields={FIELDS}&limit=1",
            "https://www.ebi.ac.uk/ena/portal/api/search"
        ]
        proc = subprocess.run(curl_command, capture_output=True, text=True)
        ena_lines = proc.stdout.strip().split("\n") if proc.stdout else []
        ena_data = ena_lines[-1] if len(ena_lines) > 1 else (ena_lines[0] if ena_lines else "")

        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
            f_out.write(f"{ena_data}\terror\terror\n")

########################################################################################################################
#MAIN FUNCTION
def main():
    ensure_output_file_header()
    #run extraction from file (columns 1)
    # extract_run_accessions_from_file()

    #run extraction from project accession
    # get_run_accessions("PRJNA523380")

    with open(RUNS_TSV, 'r') as file:
        # next(file)
        run_accessions = [line.strip() for line in file if line.strip()]
        vprint(run_accessions, flush=True)
        if not run_accessions:
            vprint("Error: no run accessions found")
            return

    #download metadata
    execute_bash_download_metadata()

    #get API structured metadata from API and study/sample extraction from xml
    with Pool(4) as pool:  #multiprocess 4 CPU
        pool.map(extract_and_save_metadata, run_accessions)

    #create flag end process before cleaning
    open(FLAG_FILE, 'w').close()


if __name__ == "__main__":
    main()
