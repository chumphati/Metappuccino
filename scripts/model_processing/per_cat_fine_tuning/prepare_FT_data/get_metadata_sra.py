import os
import re
import glob
import xml.etree.ElementTree as ET

METADATA_DIR = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/DATA_PER_CAT/treatment_time"
OUTPUT_FILE  = os.path.join(METADATA_DIR, "metadata_sra.txt")

def clean(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", s).strip()

def extract_block_text(root, tag):
    node = root.find(f".//{tag}")
    if node is None:
        return ""
    return clean(" ".join(node.itertext()))

def main():
    xml_files = sorted(glob.glob(os.path.join(METADATA_DIR, "xml_known/", "*_metadata.xml")))
    if not xml_files:
        raise SystemExit(f"No XML files found in {METADATA_DIR}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("run_accession\tsample\tstudy\n")
        for xml_path in xml_files:
            run_accession = os.path.basename(xml_path).replace("_metadata.xml", "")
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                sample_txt = extract_block_text(root, "SAMPLE")
                study_txt  = extract_block_text(root, "STUDY")
            except ET.ParseError:
                sample_txt = ""
                study_txt  = ""
            out.write(f"{run_accession}\t{sample_txt}\t{study_txt}\n")

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()

    header = lines[0] if lines else "run_accession\tsample\tstudy"
    rows = lines[1:] if len(lines) > 1 else []
    filtered = []
    for row in rows:
        parts = row.split("\t")
        if len(parts) == 3 and all(p.strip() for p in parts):
            filtered.append(row)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(header + "\n")
        for row in filtered:
            f.write(row + "\n")

if __name__ == "__main__":
    main()
