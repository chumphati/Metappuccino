import pandas as pd
import re

dot_file = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/raw/DOT_TABLE_CLEAN.csv"
uberon_file = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/raw/UBERON_TABLE_CLEAN.csv"
cell_file = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/raw/CELLOSAURUS_HUMANS_CLEAN.csv"

dot_file_out = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/DOT_TABLE_CLEAN.csv"
uberon_file_out = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/UBERON_TABLE_CLEAN.csv"
cell_file_out = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/CELLOSAURUS_CLEAN.csv"

def has_word_number_pattern(s):
    parts = s.split()
    return len(parts) == 2 and parts[0].isalpha() and parts[1].isdigit()

def clean_synonyms(df, synonym_col, name_col):
    names = set(df[name_col].dropna().str.strip().str.lower())
    def filter_synonyms(s):
        if not isinstance(s, str):
            return ''
        syns = [x.strip() for x in s.split(";") if x.strip()]
        filtered = []
        for x in syns:
            if has_word_number_pattern(x):
                continue
            if len(x.split()) > 2:
                filtered.append(x)
            elif x.lower() not in names and not re.fullmatch(r"[a-zA-Z]+", x):
                filtered.append(x)
        return ";".join(filtered)
    df[synonym_col] = df[synonym_col].apply(filter_synonyms)
    return df

def clean_cellosaurus(df, synonym_col):
    def keep(s):
        if not isinstance(s, str):
            return ''
        syns = [x.strip() for x in s.split(";") if x.strip()]
        filtered = []
        for x in syns:
            if has_word_number_pattern(x):
                continue
            if len(x.split()) > 2:
                filtered.append(x)
            elif any(c.isdigit() for c in x):
                filtered.append(x)
        return ";".join(filtered)
    df[synonym_col] = df[synonym_col].apply(keep)
    return df

dot_df = pd.read_csv(dot_file, dtype=str, on_bad_lines='skip').fillna("")
uberon_df = pd.read_csv(uberon_file, dtype=str, on_bad_lines='skip').fillna("")
cell_df = pd.read_csv(cell_file, dtype=str, on_bad_lines='skip').fillna("")

dot_df = clean_synonyms(dot_df, "synonym", "name")
uberon_df = clean_synonyms(uberon_df, "synonym", "name")
cell_df = clean_cellosaurus(cell_df, "synonym")

dot_df.to_csv(dot_file_out, index=False)
uberon_df.to_csv(uberon_file_out, index=False)
cell_df.to_csv(cell_file_out, index=False)
