import pandas as pd
import numpy as np
import re


def detect_library_selection(text):
    txt = text.lower()
    exact_matches = {
        "polya": "polyA",
        "inverse rna": "inverse rRNA",
        "inverse rrna": "inverse rRNA",
        "hybrid selection": "hybrid selection",
        "small rna": "small RNA",
        "cdna": "cDNA"
    }

    for key, value in exact_matches.items():
        if key in txt:
            return value

    polyA_patterns = [r"polya", r"poly[\.\s]?a", r"oligo[\.-]?dt", r"truseq\..*mrna", r"smarter[\.-]?mrna"]
    inverse_patterns = [r"ribominus", r"ribodep", r"ribozero", r"ribo[\. -]?zero", r"riboerase", r"ribogone", r"ribocop", r"ribo[\.-]?dep", r"depleted ribosom", r"remove ribosom", r"truseq\..*total", r"smarter\..*total"]
    hybrid_patterns = [r"hybrid[\.-]?selection", r"exon[\. -]?capture", r"exome[\. -]?capture", r"rna\.exome", r"geomx"]
    small_patterns = [r"truseq\..*small", r"size[\. -]?fraction"]

    for pat in polyA_patterns:
        if re.search(pat, txt):
            return 'polyA'
    for pat in inverse_patterns:
        if re.search(pat, txt):
            return 'inverse rRNA'
    for pat in hybrid_patterns:
        if re.search(pat, txt):
            return 'hybrid selection'
    for pat in small_patterns:
        if re.search(pat, txt):
            return 'small RNA'
    if re.search(r"cdna", txt):
        return 'cDNA'

    if re.search(r"unspecified|other|nan|none", txt):
        return 'nan'
    return 'other'



def detect_library_source(text):
    txt = text.lower()
    single_patterns = [r"transcriptomic single cell", r"chromium", r"10x", r"single[\.\s]?cell"]
    for pat in single_patterns:
        if re.search(pat, txt):
            return 'single-cell'
    if re.search(r"transcriptom|bulk", txt):
        return 'bulk'
    if re.search(r"unspecified|other|nan|none", txt):
        return 'nan'
    return 'bulk'


def clean_output_column(df):
    def clean_text(text):
        lines = text.strip().split('\n')
        cleaned_lines = []
        for line in lines:
            if line.lower().startswith("library_selection:"):
                corrected = detect_library_selection(text)
                cleaned_lines.append(f"library_selection: {corrected}")
            elif line.lower().startswith("library_source:"):
                corrected = detect_library_source(text)
                cleaned_lines.append(f"library_source: {corrected}")
            else:
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)

    df['output'] = df['output'].apply(clean_text)
    return df


def main():
    df = pd.read_csv("/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/finetune_data_tmp.csv")
    df_clean = clean_output_column(df)
    df_clean.to_csv("/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/FINE_TUNING/ttfinetune_data.csv", index=False)

if __name__ == '__main__':
    main()
