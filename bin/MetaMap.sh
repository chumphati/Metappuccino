#!/bin/bash
export PATH="/usr/bin:/bin:/usr/sbin:/sbin:/usr/local/bin:$PATH"

##FUNCTIONS
#wait for a job to be completed to launch the next one
wait_for_flag_file() {
    local flag_file=$1
    while [ ! -f "$flag_file" ]; do
        sleep 60
    done
}

##STEP 1: GET AND CLEAN METADATA
#get metadata from sra ncbi if asked from a sra list
if [ ! -f "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/STEP1_1.flag" ]; then
  qsub -q common /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/bin/STEP1/download_metadata.sh
fi
#clean metadata output table for xml config
wait_for_flag_file "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/STEP1_1.flag"
if [ ! -f "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/STEP1_2.flag" ]; then
  qsub -q common /store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/bin/STEP1/clean_metadata.sh
fi

##STEP 2: FILL MISSING METADATA
#get basic information for each run in output final table

#fill the missing information in the output final table with LLM
#biology info
#donor information
#merge in final table

##STEP 3: ASSOCIATE TERMS WITH CODE
#association uberon/dot with ref table

#fill the unknown match with LLM
