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
import sys
import random

########################################################################################################################
#PATHS
parser = argparse.ArgumentParser(description="Download metadata from NCBI Ensembl")
parser.add_argument("--base_path", type=str, required=True, help="Base path to Metappuccino")
parser.add_argument("--verbose", action="store_true", help="Verbose output")
parser.add_argument("--cpu_number", type=int, default=os.cpu_count(), help="Number of CPU used to parallelize the metadata downloads")
parser.add_argument("--ncbi_api_key", type=str, default=None, help="NCBI E-utilities API key")
parser.add_argument("--ncbi_email", type=str, default=None, help="Contact email for NCBI E-utilities")
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

cpu_number = args.cpu_number

# FIELDS = "study_accession,first_public,study_title,project_name,run_accession,sample_accession,sample_title,sample_description,library_name,library_selection,library_source,library_strategy,library_construction_protocol,library_layout,rna_integrity_num,instrument_platform,rt_prep_protocol,cell_line,cell_type,tissue_lib,tissue_type,host_phenotype,isolate,age,host_body_site,sampling_site,base_count,description,host_sex,sex,submitted_host_sex,disease,host_status"
# HEADER_LINE = "run_accession\tfirst_public\tstudy_title\tproject_name\tstudy_accession\tsample_accession\tsample_title\tsample_description\tlibrary_name\tlibrary_selection\tlibrary_source\tlibrary_strategy\tlibrary_construction_protocol\tlibrary_layout\trna_integrity_num\tinstrument_platform\trt_prep_protocol\tcell_line\tcell_type\ttissue_lib\ttissue_type\thost_phenotype\tisolate\tage\thost_body_site\tsampling_site\tbase_count\tdescription\thost_sex\tsex\tsubmitted_host_sex\tdisease\thost_status\tsample_metadata_ncbi\tstudy_metadata_ncbi\n"

VERBOSE = args.verbose
vprint = print if VERBOSE else (lambda *a, **k: None)
ERR_LOG = os.path.join(base_path, "download_metadata.err")

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

orig_meta_dir = os.path.join(base_path, "metadata")
if os.path.isdir(orig_meta_dir):
    METADATA_DIR = orig_meta_dir

os.makedirs(METADATA_DIR, exist_ok=True)

NCBI_API_KEY = args.ncbi_api_key
NCBI_EMAIL = args.ncbi_email

threshold = 16 if NCBI_API_KEY else 8
effective_cpu = max(1, min(cpu_number, threshold, (os.cpu_count() or 1)))

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
    vprint(f"Begin executing bash download metadata")
    bash_script = f"""#!/bin/bash
                MKDIR=\"{MKDIR_PATH}\"
                CAT=\"{CAT_PATH}\"
                GREP=\"{GREP_PATH}\"
                CURL=\"{CURL_PATH}\"
                \"$MKDIR\" -p \"{METADATA_DIR}\"
                \"$CAT\" \"{RUNS_TSV}\" | \"$GREP\" -v '^\\s*$' | while IFS=$'\\t' read -r RUN_ACCESSION; do
                    RUN_ACCESSION=$(echo "$RUN_ACCESSION" | tr -d '\r' | tr -d '\n' | tr -d ' ')
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

def _download_xml_ncbi(run_accession, dest, max_retries=5, api_key=NCBI_API_KEY, email=NCBI_EMAIL):
    headers = {"User-Agent": "metappuccino/1.0 (contact: none)"}
    params = {"db": "sra", "id": run_accession, "retmode": "xml"}
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email
    tmp = dest + ".tmp"
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params=params, headers=headers, timeout=45,
            )
            txt = (r.text or "").lstrip()
            if r.status_code == 200 and txt.startswith("<") and (("<STUDY" in txt) or ("<SAMPLE" in txt)):
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with open(tmp, "w", encoding="utf-8") as fxml:
                    fxml.write(txt)
                os.replace(tmp, dest)
                return True
            if r.status_code != 200:
                try:
                    with open(ERR_LOG, "a") as ef:
                        ef.write(f"[{run_accession}] http {r.status_code} url={r.url}\n")
                except Exception:
                    pass
            if r.status_code in (429, 500, 502, 503, 504):
                try:
                    with open(ERR_LOG, "a") as ef:
                        ef.write(f"[{run_accession}] transient http {r.status_code}, retry attempt {attempt} after {delay:.2f}s\n")
                except Exception:
                    pass
                vprint(f"Retrying SRA XML for {run_accession}: attempt {attempt} after {delay:.2f}s")
                time.sleep(delay + random.uniform(0, 0.25))
                delay = min(delay * 2, 16)
            else:
                break
        except Exception as ex:
            try:
                with open(ERR_LOG, "a") as ef:
                    ef.write(f"[{run_accession}] download error attempt {attempt}: {ex}\n")
            except Exception:
                pass
            vprint(f"Retrying SRA XML for {run_accession}: attempt {attempt} after {delay:.2f}s")
            time.sleep(delay + random.uniform(0, 0.25))
            delay = min(delay * 2, 16)
    return False

def _parse_xml_with_retry(xml_file, max_retries=4):
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            tree = ET.parse(xml_file)
            return tree.getroot()
        except ET.ParseError as e:
            last_exc = e
            time.sleep(1 + attempt)
            if attempt < max_retries:
                _download_xml_ncbi(os.path.basename(xml_file).split("_metadata.xml")[0], xml_file, max_retries=5)
    if last_exc:
        raise last_exc
    return None

def _fetch_biosample_xml(samn_accession: str, dest_path: str, max_retries: int = 5, api_key=NCBI_API_KEY, email=NCBI_EMAIL) -> bool:
    headers = {"User-Agent": "metappuccino/1.0 (contact: none)"}
    params = {"db": "biosample", "id": samn_accession, "retmode": "xml"}
    if api_key:
        params["api_key"] = api_key
    if email:
        params["email"] = email
    tmp = dest_path + ".tmp"
    delay = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                params=params, headers=headers, timeout=45,
            )
            txt = (r.text or "").lstrip()
            if r.status_code == 200 and txt.startswith("<"):
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                with open(tmp, "w", encoding="utf-8") as fxml:
                    fxml.write(txt)
                os.replace(tmp, dest_path)
                return True
            if r.status_code != 200:
                try:
                    with open(ERR_LOG, "a") as ef:
                        ef.write(f"[{samn_accession}] biosample http {r.status_code} url={r.url}\n")
                except Exception:
                    pass
            if r.status_code in (429, 500, 502, 503, 504):
                try:
                    with open(ERR_LOG, "a") as ef:
                        ef.write(f"[{samn_accession}] biosample transient http {r.status_code}, retry attempt {attempt} after {delay:.2f}s\n")
                except Exception:
                    pass
                vprint(f"Retrying BioSample XML for {samn_accession}: attempt {attempt} after {delay:.2f}s")
                time.sleep(delay + random.uniform(0, 0.25))
                delay = min(delay * 2, 16)
            else:
                break
        except Exception as ex:
            try:
                with open(ERR_LOG, "a") as ef:
                    ef.write(f"[{samn_accession}] biosample download error attempt {attempt}: {ex}\n")
            except Exception:
                pass
            vprint(f"Retrying BioSample XML for {samn_accession}: attempt {attempt} after {delay:.2f}s")
            time.sleep(delay + random.uniform(0, 0.25))
            delay = min(delay * 2, 16)
    return False

def _extract_biosample_attributes_from_file(biosample_xml_path: str) -> str:
    try:
        root = ET.parse(biosample_xml_path).getroot()
    except Exception:
        return ""
    attrs = []
    for a in root.findall(".//BioSample/Attributes/Attribute"):
        name = a.get("attribute_name") or a.get("harmonized_name") or ""
        val = (a.text or "").strip()
        if name and val:
            name = " ".join(name.split())
            val = " ".join(val.split())
            attrs.append(f"{name}: {val}")
    return "; ".join(attrs)

def _extract_sra_sample_attributes_as_fallback(root) -> str:
    pairs = []
    for sa in root.findall(".//SAMPLE/SAMPLE_ATTRIBUTES/SAMPLE_ATTRIBUTE"):
        tag = (sa.findtext("TAG") or "").strip()
        val = (sa.findtext("VALUE") or "").strip()
        if tag and val:
            pairs.append(f"{tag}: {val}")
    return "; ".join(pairs)

def _find_samn_in_sra_root(root) -> str:
    def _local(tag):
        return tag.rsplit('}', 1)[-1] if '}' in tag else tag
    for x in root.iter():
        if _local(getattr(x, 'tag', '')) != "XREF_LINK":
            continue
        db = None
        idv = None
        for child in x:
            ln = _local(getattr(child, 'tag', ''))
            if ln == "DB":
                db = (child.text or "").strip() if child.text else ""
            elif ln == "ID":
                idv = (child.text or "").strip() if child.text else ""
        if (db or "").lower() == "biosample" and idv and idv.startswith("SAMN"):
            return idv
    for ext in root.iter():
        if _local(getattr(ext, 'tag', '')) == "EXTERNAL_ID":
            ns = (ext.get("namespace") or "").strip().lower()
            val = (ext.text or "").strip()
            if "biosample" in ns and val.startswith("SAMN"):
                return val
    import re
    sampat = re.compile(r"\bSAMN\d+\b")
    for node in root.iter():
        txt = (getattr(node, 'text', None) or "").strip()
        if txt:
            m = sampat.search(txt)
            if m:
                return m.group(0)
    return ""

def _find_samn_anywhere_from_file(xml_path: str) -> str:
    try:
        with open(xml_path, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
    except Exception:
        return ""
    import re
    m = re.search(r"\bSAMN\d+\b", txt)
    return m.group(0) if m else ""

def _get_biosample_acc_from_ena(run_accession: str) -> str:
    try:
        payload = (
            "result=read_run"
            f"&query=run_accession%3D{run_accession}"
            "&format=tsv"
            "&fields=run_accession,biosample_accession,secondary_sample_accession,sample_accession"
            "&limit=1"
        )
        resp = requests.post(
            "https://www.ebi.ac.uk/ena/portal/api/search",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=payload,
            timeout=30
        )
        lines = resp.text.strip().splitlines()
        if len(lines) >= 2:
            header = lines[0].split("\t")
            row = lines[1].split("\t")
            idx = {h:i for i,h in enumerate(header)}
            for key in ("biosample_accession", "secondary_sample_accession", "sample_accession"):
                if key in idx and len(row) > idx[key]:
                    v = row[idx[key]].strip()
                    if v.startswith("SAMN"):
                        return v
    except Exception as ex:
        with open(ERR_LOG, "a") as ef: ef.write(f"[{run_accession}] ENA biosample lookup error: {ex}\n")
    return ""

#get specific metadata from xml
def extract_and_save_metadata(run_accession):
    xml_file = os.path.join(METADATA_DIR, f"{run_accession}_metadata.xml")

    if not os.path.exists(xml_file):
        try:
            ok = _download_xml_ncbi(run_accession, xml_file, max_retries=5)
            if not ok:
                with open(ERR_LOG, "a") as ef: ef.write(f"[{run_accession}] failed to download SRA XML\n")
        except Exception as ex:
            with open(ERR_LOG, "a") as ef: ef.write(f"[{run_accession}] download exception: {ex}\n")

    try:
        #get xml
        if not _looks_like_xml(xml_file):
            _download_xml_ncbi(run_accession, xml_file, max_retries=5)
        root = _parse_xml_with_retry(xml_file, max_retries=4)
        if root is None:
            raise ET.ParseError("empty root")

        sample_metadata = ""

        samn = _find_samn_in_sra_root(root)
        if not samn:
            samn = _find_samn_anywhere_from_file(xml_file)
        if not samn:
            samn = _get_biosample_acc_from_ena(run_accession)
        if samn:
            biosample_xml_path = os.path.join(METADATA_DIR, f"{run_accession}_biosample.xml")
            if (not os.path.exists(biosample_xml_path)) or os.path.getsize(biosample_xml_path) == 0:
                _fetch_biosample_xml(samn, biosample_xml_path, max_retries=5)
            biosample_attrs = _extract_biosample_attributes_from_file(biosample_xml_path)
            if biosample_attrs:
                sample_metadata = biosample_attrs

        if not sample_metadata:
            sra_attrs = _extract_sra_sample_attributes_as_fallback(root)
            if sra_attrs:
                sample_metadata = sra_attrs

        if not sample_metadata:
            sample_metadata = " ".join(root.findall(".//SAMPLE")[0].itertext()).replace('\n', ' ')

        study_metadata = " ".join(root.findall(".//STUDY")[0].itertext()).replace('\n', ' ')

        #search in sample and study
        curl_command = [
            CURL_PATH, "-S", "-s", "-X", "POST", "-H", "Content-Type: application/x-www-form-urlencoded",
            "-d", f"result=read_run&query=run_accession%3D{run_accession}&format=tsv&fields={FIELDS}&limit=1",
            "https://www.ebi.ac.uk/ena/portal/api/search"
        ]

        response = subprocess.run(curl_command, capture_output=True, text=True)
        vprint("--------------------------")
        vprint("RUN: ", run_accession)
        vprint("RAW ENA RESPONSE:", response.stdout)
        ena_data = response.stdout.strip().split("\n")[-1] if response.stdout.strip() else ""
        vprint("ENA DATA: ", ena_data)

        if not response.stdout.strip():
            msg = f"Warning: No data returned from ENA API for run accession {run_accession}."
            vprint(msg)
            with open(ERR_LOG, "a") as ef: ef.write(msg + "\n")
        if response.returncode != 0:
            with open(ERR_LOG, "a") as ef: ef.write(f"[{run_accession}] curl return code {response.returncode}. stderr: {response.stderr}\n")
            print(f"[{run_accession}] curl error: {response.stderr}", file=sys.stderr, flush=True)

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
            CURL_PATH, "-S", "-s", "-X", "POST", "-H", "Content-Type: application/x-www-form-urlencoded",
            "-d", f"result=read_run&query=run_accession%3D{run_accession}&format=tsv&fields={FIELDS}&limit=1",
            "https://www.ebi.ac.uk/ena/portal/api/search"
        ]
        proc = subprocess.run(curl_command, capture_output=True, text=True)
        if proc.returncode != 0:
            with open(ERR_LOG, "a") as ef: ef.write(f"[{run_accession}] curl return code {proc.returncode}. stderr: {proc.stderr}\n")
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
            CURL_PATH, "-S", "-s", "-X", "POST", "-H", "Content-Type: application/x-www-form-urlencoded",
            "-d", f"result=read_run&query=run_accession%3D{run_accession}&format=tsv&fields={FIELDS}&limit=1",
            "https://www.ebi.ac.uk/ena/portal/api/search"
        ]
        proc = subprocess.run(curl_command, capture_output=True, text=True)
        if proc.returncode != 0:
            with open(ERR_LOG, "a") as ef: ef.write(f"[{run_accession}] curl return code {proc.returncode}. stderr: {proc.stderr}\n")
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
            CURL_PATH, "-S", "-s", "-X", "POST", "-H", "Content-Type: application/x-www-form-urlencoded",
            "-d", f"result=read_run&query=run_accession%3D{run_accession}&format=tsv&fields={FIELDS}&limit=1",
            "https://www.ebi.ac.uk/ena/portal/api/search"
        ]
        proc = subprocess.run(curl_command, capture_output=True, text=True)
        if proc.returncode != 0:
            with open(ERR_LOG, "a") as ef: ef.write(f"[{run_accession}] curl return code {proc.returncode}. stderr: {proc.stderr}\n")
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
            CURL_PATH, "-S", "-s", "-X", "POST", "-H", "Content-Type: application/x-www-form-urlencoded",
            "-d", f"result=read_run&query=run_accession%3D{run_accession}&format=tsv&fields={FIELDS}&limit=1",
            "https://www.ebi.ac.uk/ena/portal/api/search"
        ]
        proc = subprocess.run(curl_command, capture_output=True, text=True)
        if proc.returncode != 0:
            with open(ERR_LOG, "a") as ef: ef.write(f"[{run_accession}] curl return code {proc.returncode}. stderr: {proc.stderr}\n")
        ena_lines = proc.stdout.strip().split("\n") if proc.stdout else []
        ena_data = ena_lines[-1] if len(ena_lines) > 1 else (ena_lines[0] if ena_lines else "")

        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
            f_out.write(f"{ena_data}\terror\terror\n")
        with open(ERR_LOG, "a") as ef: ef.write(f"[{run_accession}] unexpected exception: {e}\n")

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
        # vprint(run_accessions, flush=True)
        if not run_accessions:
            vprint("Error: no run accessions found")
            return

    already = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                parts = line.rstrip("\n").split("\t")
                if parts and parts[0]:
                    already.add(parts[0])
    run_accessions = [r for r in run_accessions if r not in already]
    if not run_accessions:
        vprint("All runs already processed.")
        open(FLAG_FILE, 'w').close()
        return

    #download metadata
    # execute_bash_download_metadata()

    def chunks(seq, n):
        for i in range(0, len(seq), n):
            yield seq[i:i+n]

    max_passes = 3
    batch_size = 10000
    total_runs = len(run_accessions)
    processed_counter = 0
    for batch_index, batch in enumerate(chunks(run_accessions, batch_size), start=1):
        vprint(f"Starting batch {batch_index} with {len(batch)} runs")
        pending = batch[:]
        backoff = 20
        for attempt in range(1, max_passes + 1):
            vprint(f"Batch {batch_index}: pass {attempt} on {len(pending)} runs using {effective_cpu} processes")
            for sub in chunks(pending, 500):
                with Pool(effective_cpu) as pool:
                    pool.map(extract_and_save_metadata, sub)
                processed_counter += len(sub)
                print(f"Progress: {processed_counter}/{total_runs}")
            still_missing = []
            for r in pending:
                p = os.path.join(METADATA_DIR, f"{r}_metadata.xml")
                if not os.path.exists(p) or not _looks_like_xml(p):
                    still_missing.append(r)
            if not still_missing:
                vprint(f"Batch {batch_index}: pass {attempt} completed with no missing XML")
                break
            vprint(f"Batch {batch_index}: {len(still_missing)} runs still missing XML after pass {attempt}")
            try:
                with open(ERR_LOG, "a") as ef:
                    ef.write(f"[batch {batch_index}] {len(still_missing)} runs pending after pass {attempt}\n")
            except Exception:
                pass
            pause = min(backoff, 120)
            print(f"Batch {batch_index}: sleeping {pause}s before next pass")
            time.sleep(pause)
            backoff = min(backoff * 2, 300)
            pending = still_missing

    def _collect_failed_runs_from_output(out_path):
        failed = []
        if not os.path.exists(out_path):
            return failed
        with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 30:
                    run = parts[0].strip()
                    sample_meta = parts[-2].strip()
                    if sample_meta == "error" and run:
                        failed.append(run)
        return failed

    retry_runs = _collect_failed_runs_from_output(OUTPUT_FILE)
    print(f"Post-processing: {len(retry_runs)} runs flagged for retry based on output errors")
    if retry_runs:
        print("Post-processing: retry run list:")
        try:
            print(", ".join(retry_runs))
        except Exception:
            pass
        retry_counter = 0
        for sub in chunks(retry_runs, 100):
            with Pool(effective_cpu) as pool:
                pool.map(extract_and_save_metadata, sub)
            retry_counter += len(sub)
            print(f"Retry progress: {retry_counter}/{len(retry_runs)}")
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
            header = lines[0] if lines else HEADER_LINE
            latest = {}
            for i, line in enumerate(lines[1:], start=1):
                parts = line.rstrip("\n").split("\t")
                if parts:
                    latest[parts[0]] = line
            with open(OUTPUT_FILE, "w", encoding="utf-8") as fw:
                fw.write(header if header.startswith("run_accession") else HEADER_LINE)
                for run, line in latest.items():
                    fw.write(line if line.endswith("\n") else line + "\n")
        print("Post-processing: metadata_sra.txt has been rebuilt to keep latest entries")

    open(FLAG_FILE, 'w').close()


if __name__ == "__main__":
    main()
