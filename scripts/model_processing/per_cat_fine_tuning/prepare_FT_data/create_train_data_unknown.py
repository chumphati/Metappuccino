#!/usr/bin/env python3
import os
import sys
import csv
import re

STATIC_PROMPT = """Run accession: {run_accession}
Summary: {context}

Categories and definitions:
- cell_line: exact cell line code, ie standardized names of cultured/immortalized laboratory cell populations

For each category below:
- Extract information from the summary if possible
- If one value is impossible to extract, even by deducing it, return "unknown"

BE CAREFUL: Sometimes the information concerns several samples from the same study. It is important to distinguish between them and semantically extract what applies to the current run, so everything must be consistent.
FOR EACH CATEGORY SEVERAL ANSWERS CAN BE POSSIBLE, CITE THEM ALL WITH A ',' OR ';' SEPARATOR

Respond strictly in a few words with valid JSON (double quotes around keys and values), no extra keys. ONLY THE CATEGORIES CITED UNDER 'Categories and definitions'

Here is the output:
"""

INPUT_FILE_DEFAULT = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/DATA_PER_CAT/cell_line/metadata_sra_summarized.txt"

def safe_prompt(s):
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    return re.sub(r"[ \t]+", " ", s)

def process_input_file(in_path):
    base = os.path.splitext(os.path.basename(in_path))[0]
    out_path = os.path.join(os.path.dirname(in_path), base + "_nocl_prompt_output.tsv")
    with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8", newline="") as fout:
        r = csv.reader(fin, delimiter="\t")
        w = csv.writer(fout, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
        cols = next(r, [])
        col_idx = {name: i for i, name in enumerate(cols)}
        ra_i = col_idx.get("run_accession", 0)
        su_i = col_idx.get("summary", 1)
        w.writerow(["prompt", "output"])
        for row in r:
            if not row or len(row) <= max(ra_i, su_i):
                continue
            run_accession = row[ra_i].strip()
            summary = "\t".join(row[su_i:]).strip()
            prompt = STATIC_PROMPT.format(run_accession=run_accession, context=summary)
            prompt = safe_prompt(prompt)
            w.writerow([prompt, "unknown"])

def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else INPUT_FILE_DEFAULT
    if os.path.exists(in_path):
        process_input_file(in_path)

if __name__ == "__main__":
    main()
