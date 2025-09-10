#!/bin/bash
set -euo pipefail

FILE="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/DATA_PER_CAT/cell_line/metadata_sra_summarized.txt"

TMP="$(mktemp)"
cp "$FILE" "${FILE}.bak"
grep -vi -E 'cell[[:space:]-]*lines?' "$FILE" > "$TMP"
mv "$TMP" "$FILE"