#get nb line per category ena_results
awk -F'\t' 'BEGIN{cols["cell_type"]=0;cols["host_body_site"]=0;cols["tissue_type"]=0;cols["cell_line"]=0;cols["disease"]=0;cols["host_phenotype"]=0;cols["library_selection"]=0;cols["library_source"]=0} NR==1{for(i=1;i<=NF;i++){if($i in cols){col_idx[$i]=i}};next} {for(col in col_idx){val=$(col_idx[col]);if(val!=""&&val!="NA"&&val!="null"){cols[col]++}}} END{for(col in cols){printf "%s: %d\n",col,cols[col]}}' /store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/ena_results.tsv

#count lines with some categories
python3 - <<'PYCODE'
import csv, sys
cnt = 0
path = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/complete_finetune_data.csv'
with open(path, newline='') as f:
    for row in csv.DictReader(f):
        lines = [L for L in row['output'].splitlines() if L.strip()]
        if len(lines) == 2 and lines[0] == 'library_selection: other' and lines[1].startswith('library_source: '):
            cnt += 1
print(cnt)
PYCODE

#shuffle ft data
python3 -c "import pandas as pd; df=pd.read_csv('/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/mistral7B_FINE_TUNING_v1/finetune_data.csv'); df.sample(n=min(2000, len(df)), random_state=42).to_csv('/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/mistral7B_FINE_TUNING_v1/finetune_data_short.csv', index=False)"

#get all run accession list from ft data
cut -d, -f1 /store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/finetune_data_val_corrected.csv | grep -oP 'Run accession:\s*\K\S+' | sort | uniq | awk '{print "\"" $0 "\""}' | pas^C -sd,

#merge files
head -n 1 /store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/finetune_data_train_corrected.csv > /store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/finetune_data_all.csv
tail -n +2 /store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/finetune_data_train_corrected.csv >> /store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/finetune_data_all.csv
tail -n +2 /store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/finetune_data_val_corrected.csv >> /store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/finetune_data_all.csv

head -n 1 f1 > f2
tail -n +2 f1 >> f2
tail -n +2 f3 >> f2

head -n 1 /store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_metappuccino/FINE_TUNING/sample_finetune_data.csv > /store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_metappuccino/FINE_TUNING/finetune_data.csv
tail -n +2 /store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_metappuccino/FINE_TUNING/sample_finetune_data.csv >> /store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_metappuccino/FINE_TUNING/finetune_data.csv
tail -n +2 /store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_metappuccino/FINE_TUNING/study_finetune_data.csv >> /store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results_metappuccino/FINE_TUNING/finetune_data.csv

sed -E 's/[Rr][Nn][Aa]//g; s/[Hh][Uu][Mm][Aa][Nn]//g; s/\b[[:alnum:]_]*[Ss][Ee][Qq]\b//g' /store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/data_templates_training/metadata_templates_train_withoutkeys.json > /store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/data/data_templates_training/metadata_templates_train_cleaned.json

tail -n +2 /store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/finetune_data_train2_corrected.csv >> /store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/finetune_data_train_corrected.csv

python3 -c "import pandas as pd, re, sys
df = pd.read_csv(sys.argv[1], dtype=str)
df['output'] = df['output'].apply(lambda s: re.sub(r':\s*nan\\b', ': \"unknown\"', s, flags=re.IGNORECASE) if isinstance(s, str) else s)
df.to_csv(sys.argv[1], index=False)
" /store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING/all_raw/finetune_data_test_corrected.csv

#are files same
cmp -s "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/METADATA_LLM_INFERENCE/synt_summary.txt" "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/utils_scripts/METADATA_LLM_INFERENCE/synt_summary2.txt" \
  && echo "identiques" || echo "différents"

#ft mv
for d in /store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/FINE_TUNING*/; do
  [ -d "$d" ] || continue
  mkdir -p "$d/archive"
  shopt -s nullglob
  for item in "$d"*; do
    base=$(basename "$item")
    [ "$base" = "finetune_data_test.csv" ] && continue
    [ "$base" = "archive" ] && continue
    mv -v -- "$item" "$d/archive/"
  done
done
