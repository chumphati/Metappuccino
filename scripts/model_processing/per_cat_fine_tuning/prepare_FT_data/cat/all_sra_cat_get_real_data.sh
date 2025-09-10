#!/bin/bash
set -euo pipefail

BASE_DIR="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/results/DATA_PER_CAT/all_categories"
LOG_DIR="${BASE_DIR}/logs"
mkdir -p "${BASE_DIR}" "${LOG_DIR}"

K_KNOWN=1700
K_UNKNOWN=300
MAX_PER_STUDY_KNOWN=5
MAX_PER_STUDY_UNKNOWN=3
MAX_PER_SAMPLE=1
MAX_PER_VALUE=40

CATS=(library_selection sequencing_source biopsy_site biopsy_type cell_type organ disease is_cancer treatment treatment_time response age sex ethnicity localization)

FIELDS="run_accession,study_accession,sample_accession,cell_line,cell_type,tissue_type,disease,library_selection,library_source,library_strategy,description,sample_description,sample_title,sex,host_sex,local_environmental_context,country,age,lat,lon,location,location_start,location_end,host_body_site,experiment_title,experiment_alias,study_title,first_public,read_count"

QUERY="first_public%3E%3D1900-01-01"

for CAT in "${CATS[@]}"; do
  OUT_DIR="${BASE_DIR}/${CAT}"
  mkdir -p "${OUT_DIR}"
  RUNS_ALL_TSV="${OUT_DIR}/runs_all.tsv"
  KNOWN_RAND_TSV="${OUT_DIR}/known_rand.tsv"
  UNKNOWN_RAND_TSV="${OUT_DIR}/unknown_rand.tsv"
  KNOWN_SORTED_TSV="${OUT_DIR}/known_rand_sorted.tsv"
  UNKNOWN_SORTED_TSV="${OUT_DIR}/unknown_rand_sorted.tsv"
  KNOWN_OUT="${OUT_DIR}/known_${CAT}.tsv"
  UNKNOWN_OUT="${OUT_DIR}/unknown_${CAT}.tsv"
  FINAL_RUNS="${OUT_DIR}/final_runs_${CAT}.txt"
  REPORT="${OUT_DIR}/report_${CAT}.txt"
  : > "${KNOWN_RAND_TSV}"
  : > "${UNKNOWN_RAND_TSV}"

  /usr/bin/curl -s -X POST -H "Content-Type: application/x-www-form-urlencoded" -d "result=read_run&query=${QUERY}&format=tsv&fields=${FIELDS}&limit=100000" "https://www.ebi.ac.uk/ena/portal/api/search" > "${RUNS_ALL_TSV}"

  /usr/bin/awk -F'\t' -v OFS='\t' -v cat="${CAT}" '
    function g(n){return ((n in h)?$(h[n]):"")}
    function lc(s){return tolower(s)}
    BEGIN{srand()}
    NR==1{for(i=1;i<=NF;i++)h[$i]=i; next}
    {
      for(i=1;i<=NF;i++){sub(/\r$/,"",$i)}
      run=g("run_accession"); study=g("study_accession"); sample=g("sample_accession")
      cell_type=lc(g("cell_type")); tissue_type=lc(g("tissue_type")); disease=lc(g("disease"))
      libsel=lc(g("library_selection")); libsrc=lc(g("library_source")); libstrat=lc(g("library_strategy"))
      desc=lc(g("description")); sampdesc=lc(g("sample_description")); samp_title=lc(g("sample_title"))
      sex=lc(g("sex")); hsex=lc(g("host_sex"))
      lectx=lc(g("local_environmental_context")); country=lc(g("country")); age=lc(g("age"))
      lat=g("lat"); lon=g("lon"); loc=lc(g("location")); locs=lc(g("location_start")); loce=lc(g("location_end"))
      hsite=lc(g("host_body_site")); et=lc(g("experiment_title")); ea=lc(g("experiment_alias")); st=lc(g("study_title"))
      text=desc " " sampdesc " " samp_title " " et " " ea " " st " " tissue_type " " hsite " " libsel " " libsrc " " libstrat
      value=""
      if(cat=="library_selection"){
        ctx=text
        if(ctx ~ /(polya|poly[-\.\ ]?a|oligo[\.\ -]?dt|oligodt|truseq(\.| )?stranded(\.| )?mrna|truseq(\.| )?mrna|standard(\.| )?mrna|smarter(\.| )?mrna|stranded(\.| )?mrna)/){value="polyA"}
        else if(ctx ~ /(ribominus|ribo[-\ ]?dep|ribo[-\ ]?zero|ribozero|riboerase|ribogone|ribocop|ribo[-\ ]?mi|deplet[^ ]* ribosom|remove ribosom|truseq(\.| )?stranded(\.| )?total|truseq(\.| )?total|smarter(\.| )?stranded(\.| )?total|smarter(\.| )?total|total rna)/){value="inverse rRNA"}
        else if(ctx ~ /(hybrid(\.| )?selection|exon(\.| )?capture|exome(\.| )?capture|rna(\.| )?exome|geomx|hybrid capture|capture rna|bait)/){value="hybrid selection"}
        else if(ctx ~ /(truseq(\.| )?small|small[ \-]?rna|size(\.| )?fraction|mirna|pi[-\ ]?rna)/){value="small RNA"}
        else if(libsel!=""){value="other"}
      } else if(cat=="sequencing_source"){
        if(libstrat ~ /(scrna|snrna|smart[-\ ]?seq|single[-\ ]?cell|single[-\ ]?nucleus|cite[-\ ]?seq|drop[-\ ]?seq|10x|10xgenomics|cell[-\ ]?ranger)/ || text ~ /(scrna|snrna|smart[-\ ]?seq|single[-\ ]?cell|single[-\ ]?nucleus|cite[-\ ]?seq|drop[-\ ]?seq|10x|10xgenomics|cell[-\ ]?ranger)/){value="single_cell"}
        else if(text ~ /(spatial|visium|slide[-\ ]?seq|stereo[-\ ]?seq|merfish|seqfish|cosmx|xenium|geomx)/){value="spatial"}
        else if(libstrat!=""){value="bulk"}
      } else if(cat=="biopsy_site"){
        if(tissue_type!=""){value=tissue_type}
      } else if(cat=="biopsy_type"){
        if(text ~ /\bmetastasis\b|\bmetastatic\b|(^|[^a-z])mets?\b/){value="metastasis"}
        else if(text ~ /\bpbmc\b|\bperipheral blood\b|\bwhole blood\b|\bblood\b|\bplasma\b|\bserum\b/){value="blood"}
        else if(text ~ /\bprimary\b|\bprimary tumor\b/){value="primary"}
      } else if(cat=="cell_type"){
        if(cell_type!=""){value=cell_type}
      } else if(cat=="organ"){
        cand=tissue_type" "hsite" "text
        if(cand ~ /\blung\b/){value="lung"}
        else if(cand ~ /\bliver\b/){value="liver"}
        else if(cand ~ /\bbreast\b/){value="breast"}
        else if(cand ~ /\bprostate\b/){value="prostate"}
        else if(cand ~ /\bcolon|colorectal\b/){value="colon"}
        else if(cand ~ /\bkidney|renal\b/){value="kidney"}
        else if(cand ~ /\bbrain|glioblastoma|cortex|cerebellum\b/){value="brain"}
        else if(cand ~ /\bpancreas|pancreatic\b/){value="pancreas"}
        else if(cand ~ /\bovary|ovarian\b/){value="ovary"}
        else if(cand ~ /\bstomach|gastric\b/){value="stomach"}
        else if(cand ~ /\bskin|dermal|melanoma\b/){value="skin"}
        else if(cand ~ /\bheart|cardiac\b/){value="heart"}
        else if(cand ~ /\bmuscle|myocard\b/){value="muscle"}
        else if(cand ~ /\bspleen\b/){value="spleen"}
        else if(cand ~ /\bthyroid\b/){value="thyroid"}
        else if(cand ~ /\besophagus|esophageal\b/){value="esophagus"}
        else if(cand ~ /\buterus|endometrium|endometrial\b/){value="uterus"}
        else if(cand ~ /\bcervix|cervical\b/){value="cervix"}
        else if(cand ~ /\bbladder|urothelial\b/){value="bladder"}
        else if(cand ~ /\bbone marrow\b/){value="bone_marrow"}
        else if(cand ~ /\bblood\b/){value="blood"}
      } else if(cat=="disease"){
        if(disease!=""){value=disease}
        else if(text ~ /\bhealthy\b|\bnormal\b|\bcontrol\b|\bwild[-\ ]?type\b/){value="control"}
        else if(text ~ /(carcinoma|cancer|leukemia|lymphoma|myeloma|melanoma|sarcoma|glioblastoma|tumou?r|neoplasm|adenocarcinoma|oma\b)/){value="cancer"}
      } else if(cat=="is_cancer"){
        if(disease!=""){
          if(disease ~ /(carcinoma|cancer|leukemia|lymphoma|myeloma|melanoma|sarcoma|glioblastoma|tumou?r|neoplasm|adenocarcinoma|oma\b)/ && disease !~ /(non[-\ ]?cancer|benign)/){value="true"} else {value="false"}
        } else {
          if(text ~ /(carcinoma|cancer|leukemia|lymphoma|myeloma|melanoma|sarcoma|glioblastoma|tumou?r|neoplasm|adenocarcinoma|oma\b)/){value="true"}
          else if(text ~ /\bhealthy\b|\bnormal\b|\bcontrol\b|\bwild[-\ ]?type\b/){value="false"}
        }
      } else if(cat=="treatment"){
        if(text ~ /\buntreated\b|\bvehicle\b|\bcontrol\b|\bdmso\b/){value="control"}
        else if(text ~ /(treated with|treatment with|administered|exposure to|received)[^\.]{0,60}/){value="treated"}
        else if(text ~ /[0-9]+[ ]*(d|h|w|mo|day|hour|week|month)s?[^a-z]*treat/){value="treated"}
      } else if(cat=="treatment_time"){
        if(match(text,/[0-9]+[ ]*(d|h|w|mo|day|hour|week|month)s?/)){value=substr(text,RSTART,RLENGTH)}
      } else if(cat=="response"){
        if(text ~ /\bstable disease\b|\bsd\b/){value="stable_disease"}
        else if(text ~ /\bpartial response\b|\bpr\b/){value="partial_response"}
        else if(text ~ /\bcomplete response\b|\bcr\b/){value="complete_response"}
        else if(text ~ /\bprogression\b|\bprogressive disease\b|\bpd\b/){value="progressive_disease"}
        else if(text ~ /\bnon[-\ ]?responder\b|\bnr\b/){value="non_responder"}
        else if(text ~ /\bresponder\b/){value="responder"}
      } else if(cat=="age"){
        if(age!=""){value=age}
        else if(match(text,/(age[^0-9]{0,6})?([0-9]{1,2})[ ]*(y|yo|yrs?|years?|ans)/)){value=substr(text,RSTART,RLENGTH)}
      } else if(cat=="sex"){
        if(sex!=""){value=sex}
        else if(hsex!=""){value=hsex}
        else if(text ~ /\bfemale\b|\bf\b/){value="female"}
        else if(text ~ /\bmale\b|\bm\b/){value="male"}
      } else if(cat=="ethnicity"){
        if(country!=""){value=country}
        else if(lectx ~ /(caucasian|european|african|asian|hispanic|latino|han|yoruba|ashkenazi)/){value=lectx}
      } else if(cat=="localization"){
        if(lat!="" && lon!=""){value=lat","lon}
        else if(loc!=""){value=loc}
        else if(locs!=""){value=locs}
        else if(loce!=""){value=loce}
        else if(lectx!=""){value=lectx}
      }
      r=sprintf("%.12f", rand())
      if(value!=""){print r,run,study,sample,value >> "'"${KNOWN_RAND_TSV}"'"} else {print r,run,study,sample,"" >> "'"${UNKNOWN_RAND_TSV}"'"}
    }
  ' "${RUNS_ALL_TSV}"

  /usr/bin/sort -g -k1,1 "${KNOWN_RAND_TSV}" -o "${KNOWN_SORTED_TSV}" || true
  /usr/bin/sort -g -k1,1 "${UNKNOWN_RAND_TSV}" -o "${UNKNOWN_SORTED_TSV}" || true

  if [[ "${CAT}" == "is_cancer" ]]; then
    PART_A="${OUT_DIR}/known_true.tsv"
    PART_B="${OUT_DIR}/known_false.tsv"
    awk -F'\t' '$5=="true"{print $0}' "${KNOWN_SORTED_TSV}" > "${PART_A}" || true
    awk -F'\t' '$5=="false"{print $0}' "${KNOWN_SORTED_TSV}" > "${PART_B}" || true
    HALF=$((K_KNOWN/2))
    awk -F'\t' -v OFS='\t' -v K="${HALF}" -v MPS="${MAX_PER_STUDY_KNOWN}" -v MPA="${MAX_PER_SAMPLE}" 'BEGIN{sel=0}{run=$2;study=$3;sample=$4;if(sel>=K)next;if(seen[run])next;if(study_c[study]>=MPS)next;if(sample_c[sample]>=MPA)next;seen[run]=1;study_c[study]++;sample_c[sample]++;print $2"\t"$3"\t"$4"\t"$5;sel++}' "${PART_A}" > "${OUT_DIR}/_true_sel.tsv" || true
    awk -F'\t' -v OFS='\t' -v K="${HALF}" -v MPS="${MAX_PER_STUDY_KNOWN}" -v MPA="${MAX_PER_SAMPLE}" 'BEGIN{sel=0}{run=$2;study=$3;sample=$4;if(sel>=K)next;if(seen[run])next;if(study_c[study]>=MPS)next;if(sample_c[sample]>=MPA)next;seen[run]=1;study_c[study]++;sample_c[sample]++;print $2"\t"$3"\t"$4"\t"$5;sel++}' "${PART_B}" > "${OUT_DIR}/_false_sel.tsv" || true
    printf "run_accession\tstudy_accession\tsample_accession\tvalue\n" > "${KNOWN_OUT}"
    cat "${OUT_DIR}/_true_sel.tsv" "${OUT_DIR}/_false_sel.tsv" >> "${KNOWN_OUT}" || true
  elif [[ "${CAT}" == "sequencing_source" ]]; then
    SC="${OUT_DIR}/known_sc.tsv"; SP="${OUT_DIR}/known_sp.tsv"; BK="${OUT_DIR}/known_bulk.tsv"
    awk -F'\t' '$5=="single_cell"{print $0}' "${KNOWN_SORTED_TSV}" > "${SC}" || true
    awk -F'\t' '$5=="spatial"{print $0}' "${KNOWN_SORTED_TSV}" > "${SP}" || true
    awk -F'\t' '$5=="bulk"{print $0}' "${KNOWN_SORTED_TSV}" > "${BK}" || true
    NBUCKETS=0; [[ -s "${SC}" ]] && NBUCKETS=$((NBUCKETS+1)); [[ -s "${SP}" ]] && NBUCKETS=$((NBUCKETS+1)); [[ -s "${BK}" ]] && NBUCKETS=$((NBUCKETS+1))
    [[ $NBUCKETS -eq 0 ]] && NBUCKETS=1
    TARGET=$((K_KNOWN/NBUCKETS))
    awk -F'\t' -v OFS='\t' -v K="${TARGET}" -v MPS="${MAX_PER_STUDY_KNOWN}" -v MPA="${MAX_PER_SAMPLE}" 'BEGIN{sel=0}{run=$2;study=$3;sample=$4;if(sel>=K)next;if(seen[run])next;if(study_c[study]>=MPS)next;if(sample_c[sample]>=MPA)next;seen[run]=1;study_c[study]++;sample_c[sample]++;print $2"\t"$3"\t"$4"\t"$5;sel++}' "${SC}" > "${OUT_DIR}/_sc_sel.tsv" || true
    awk -F'\t' -v OFS='\t' -v K="${TARGET}" -v MPS="${MAX_PER_STUDY_KNOWN}" -v MPA="${MAX_PER_SAMPLE}" 'BEGIN{sel=0}{run=$2;study=$3;sample=$4;if(sel>=K)next;if(seen[run])next;if(study_c[study]>=MPS)next;if(sample_c[sample]>=MPA)next;seen[run]=1;study_c[study]++;sample_c[sample]++;print $2"\t"$3"\t"$4"\t"$5;sel++}' "${SP}" > "${OUT_DIR}/_sp_sel.tsv" || true
    awk -F'\t' -v OFS='\t' -v K="${TARGET}" -v MPS="${MAX_PER_STUDY_KNOWN}" -v MPA="${MAX_PER_SAMPLE}" 'BEGIN{sel=0}{run=$2;study=$3;sample=$4;if(sel>=K)next;if(seen[run])next;if(study_c[study]>=MPS)next;if(sample_c[sample]>=MPA)next;seen[run]=1;study_c[study]++;sample_c[sample]++;print $2"\t"$3"\t"$4"\t"$5;sel++}' "${BK}" > "${OUT_DIR}/_bk_sel.tsv" || true
    printf "run_accession\tstudy_accession\tsample_accession\tvalue\n" > "${KNOWN_OUT}"
    cat "${OUT_DIR}/_sc_sel.tsv" "${OUT_DIR}/_sp_sel.tsv" "${OUT_DIR}/_bk_sel.tsv" >> "${KNOWN_OUT}" || true
    CUR=$(awk 'NR>1' "${KNOWN_OUT}" | wc -l | tr -d " ")
    if (( CUR < K_KNOWN )); then
      SHORT=$((K_KNOWN-CUR))
      awk -F'\t' -v OFS='\t' 'NR>1{print $1"\t"$2"\t"$3}' "${KNOWN_OUT}" > "${OUT_DIR}/_sel_idx.tsv"
      awk -F'\t' 'NR==FNR{a[$1]=1;next}!a[$2]{print $0}' "${OUT_DIR}/_sel_idx.tsv" "${KNOWN_SORTED_TSV}" | awk -F'\t' -v OFS='\t' -v K="${SHORT}" -v MPS="${MAX_PER_STUDY_KNOWN}" -v MPA="${MAX_PER_SAMPLE}" 'BEGIN{sel=0}{run=$2;study=$3;sample=$4;if(sel>=K)next;if(study_c[study]>=MPS)next;if(sample_c[sample]>=MPA)next;study_c[study]++;sample_c[sample]++;print $2"\t"$3"\t"$4"\t"$5;sel++}' >> "${KNOWN_OUT}" || true
    fi
  elif [[ "${CAT}" == "localization" ]]; then
    CO="${OUT_DIR}/known_coords.tsv"; TX="${OUT_DIR}/known_text.tsv"
    awk -F'\t' '($5~ /^[ \t]*-?[0-9]+\.[0-9]+,-?[0-9]+\.[0-9]+[ \t]*$/){print $0}' "${KNOWN_SORTED_TSV}" > "${CO}" || true
    awk -F'\t' '($5!~/^[ \t]*-?[0-9]+\.[0-9]+,-?[0-9]+\.[0-9]+[ \t]*$/ && $5!=""){print $0}' "${KNOWN_SORTED_TSV}" > "${TX}" || true
    HALF=$((K_KNOWN/2))
    awk -F'\t' -v OFS='\t' -v K="${HALF}" -v MPS="${MAX_PER_STUDY_KNOWN}" -v MPA="${MAX_PER_SAMPLE}" 'BEGIN{sel=0}{run=$2;study=$3;sample=$4;if(sel>=K)next;if(seen[run])next;if(study_c[study]>=MPS)next;if(sample_c[sample]>=MPA)next;seen[run]=1;study_c[study]++;sample_c[sample]++;print $2"\t"$3"\t"$4"\t"$5;sel++}' "${CO}" > "${OUT_DIR}/_co_sel.tsv" || true
    awk -F'\t' -v OFS='\t' -v K="${HALF}" -v MPS="${MAX_PER_STUDY_KNOWN}" -v MPA="${MAX_PER_SAMPLE}" 'BEGIN{sel=0}{run=$2;study=$3;sample=$4;if(sel>=K)next;if(seen[run])next;if(study_c[study]>=MPS)next;if(sample_c[sample]>=MPA)next;seen[run]=1;study_c[study]++;sample_c[sample]++;print $2"\t"$3"\t"$4"\t"$5;sel++}' "${TX}" > "${OUT_DIR}/_tx_sel.tsv" || true
    printf "run_accession\tstudy_accession\tsample_accession\tvalue\n" > "${KNOWN_OUT}"
    cat "${OUT_DIR}/_co_sel.tsv" "${OUT_DIR}/_tx_sel.tsv" >> "${KNOWN_OUT}" || true
    CUR=$(awk 'NR>1' "${KNOWN_OUT}" | wc -l | tr -d " ")
    if (( CUR < K_KNOWN )); then
      SHORT=$((K_KNOWN-CUR))
      awk -F'\t' -v OFS='\t' 'NR>1{print $1"\t"$2"\t"$3}' "${KNOWN_OUT}" > "${OUT_DIR}/_sel_idx.tsv"
      awk -F'\t' 'NR==FNR{a[$1]=1;next}!a[$2]{print $0}' "${OUT_DIR}/_sel_idx.tsv" "${KNOWN_SORTED_TSV}" | awk -F'\t' -v OFS='\t' -v K="${SHORT}" -v MPS="${MAX_PER_STUDY_KNOWN}" -v MPA="${MAX_PER_SAMPLE}" 'BEGIN{sel=0}{run=$2;study=$3;sample=$4;if(sel>=K)next;if(study_c[study]>=MPS)next;if(sample_c[sample]>=MPA)next;study_c[study]++;sample_c[sample]++;print $2"\t"$3"\t"$4"\t"$5;sel++}' >> "${KNOWN_OUT}" || true
    fi
  else
    awk -F'\t' -v OFS='\t' -v K="${K_KNOWN}" -v MPS="${MAX_PER_STUDY_KNOWN}" -v MPA="${MAX_PER_SAMPLE}" -v MPV="${MAX_PER_VALUE}" '
      BEGIN{sel=0}
      {
        run=$2;study=$3;sample=$4;val=$5
        if(sel>=K)next
        if(seen[run])next
        if(study_c[study]>=MPS)next
        if(sample_c[sample]>=MPA)next
        if(val!="" && MPV>0 && val_c[val]>=MPV)next
        seen[run]=1;study_c[study]++;sample_c[sample]++;if(val!="")val_c[val]++
        out[++sel]=$2"\t"$3"\t"$4"\t"$5
      }
      END{
        print "run_accession\tstudy_accession\tsample_accession\tvalue"
        for(i=1;i<=sel;i++)print out[i]
      }
    ' "${KNOWN_SORTED_TSV}" > "${KNOWN_OUT}" || true
  fi

  awk -F'\t' -v OFS='\t' -v K="${K_UNKNOWN}" -v MPS="${MAX_PER_STUDY_UNKNOWN}" -v MPA="${MAX_PER_SAMPLE}" '
    BEGIN{sel=0; print "run_accession\tstudy_accession\tsample_accession\tvalue"}
    {
      run=$2;study=$3;sample=$4
      if(sel>=K)next
      if(seen[run])next
      if(study_c[study]>=MPS)next
      if(sample_c[sample]>=MPA)next
      seen[run]=1;study_c[study]++;sample_c[sample]++
      print $2"\t"$3"\t"$4"\t"; sel++
    }
  ' "${UNKNOWN_SORTED_TSV}" > "${UNKNOWN_OUT}" || true

  awk -F'\t' 'NR>1{print $1}' "${KNOWN_OUT}" > "${OUT_DIR}/known_ids.txt"
  awk -F'\t' 'NR>1{print $1}' "${UNKNOWN_OUT}" > "${OUT_DIR}/unknown_ids.txt"
  paste -d '\n' "${OUT_DIR}/known_ids.txt" "${OUT_DIR}/unknown_ids.txt" > "${FINAL_RUNS}"

  {
    echo "category: ${CAT}"
    echo "known_total: $(awk 'NR>1' "${KNOWN_OUT}" | wc -l | tr -d " ")"
    echo "unknown_total: $(awk 'NR>1' "${UNKNOWN_OUT}" | wc -l | tr -d " ")"
    echo "[known value counts]"
    awk -F'\t' 'NR>1{c[$4]++}END{for(k in c)printf "%s\t%d\n",k,c[k]}' "${KNOWN_OUT}" | sort -k2,2nr
    echo "[unknown count]"
    awk 'END{print NR-1}' "${UNKNOWN_OUT}"
  } > "${REPORT}"

  XML_DIR_KN="${OUT_DIR}/xml_known"
  XML_DIR_UN="${OUT_DIR}/xml_unknown"
  mkdir -p "${XML_DIR_KN}" "${XML_DIR_UN}"

  if [ -s "${OUT_DIR}/known_ids.txt" ]; then
    N=0
    TOT=$(wc -l < "${OUT_DIR}/known_ids.txt" | tr -d ' ')
    while IFS=$'\n' read -r RUN; do
      OUT_XML="${XML_DIR_KN}/${RUN}_metadata.xml"
      if [ ! -s "${OUT_XML}" ]; then
        /usr/bin/curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=sra&id=${RUN}&retmode=text" -o "${OUT_XML}"
        /usr/bin/sleep 0.34
      fi
      N=$((N+1))
      if (( N % 100 == 0 )); then
        echo "${CAT} known XML ${N}/${TOT}"
      fi
    done < "${OUT_DIR}/known_ids.txt"
  fi

  if [ -s "${OUT_DIR}/unknown_ids.txt" ]; then
    N=0
    TOT=$(wc -l < "${OUT_DIR}/unknown_ids.txt" | tr -d ' ')
    while IFS=$'\n' read -r RUN; do
      OUT_XML="${XML_DIR_UN}/${RUN}_metadata.xml"
      if [ ! -s "${OUT_XML}" ]; then
        /usr/bin/curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=sra&id=${RUN}&retmode=text" -o "${OUT_XML}"
        /usr/bin/sleep 0.34
      fi
      N=$((N+1))
      if (( N % 20 == 0 )); then
        echo "${CAT} unknown XML ${N}/${TOT}"
      fi
    done < "${OUT_DIR}/unknown_ids.txt"
  fi
done
