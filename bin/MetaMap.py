########################################################################################################################
#IMPORT LIB
import os
import subprocess
import argparse
import time

########################################################################################################################
#FUNCTIONS
#wait for a job to be completed to launch the next one
def wait_for_flag_file(flag_path):
    while not os.path.isfile(flag_path):
        time.sleep(10)

########################################################################################################################
#MAIN FUNCTION
def main():
    #arg parse and help description
    parser = argparse.ArgumentParser(description="Automates metadata extraction and completion based on LLMs.")
    parser.add_argument("--requirements", action="store_true", help="Install requirements.txt")
    parser.add_argument("--getmetadata", action="store_true", help="Download and clean metadata from NCBI [Input: List of run accessions]")
    parser.add_argument("--fillmetadata", action="store_true", help="Fill metadata with LLMs [Input: CLEAN_METADATA_SRA.txt]")
    args = parser.parse_args()

    #scripts to execute and flags
    step1_flag = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/STEP1_1.flag"
    step2_flag = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/STEP1_2.flag"
    step3_flag = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/logs/STEP2_1.flag"
    install_requirements = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/bin/STEP1/install_requirements.sh"
    download_script = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/bin/STEP1/download_metadata.sh"
    clean_script = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/bin/STEP1/clean_metadata.sh"
    get_stable_metadata = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/bin/STEP2/get_stable_metadata.sh"

    ##INSTALL REQUIREMENTS
    if args.requirements:
        subprocess.run(["qsub", "-q", "common", install_requirements])

    ##STEP 1: GET AND CLEAN METADATA
    if args.getmetadata:
        #get metadata from sra ncbi if asked from a sra list
        if not os.path.isfile(step1_flag):
            subprocess.run(["qsub", "-q", "common", download_script])
        wait_for_flag_file(step1_flag)

        #clean metadata output table for xml config
        if not os.path.isfile(step2_flag):
            subprocess.run(["qsub", "-q", "common", clean_script])

    ##STEP 2: FILL MISSING METADATA
    if args.fillmetadata:
        #get basic information for each run in output final table
        if not os.path.isfile(step3_flag):
            subprocess.run(["qsub", "-q", "common", get_stable_metadata])
        wait_for_flag_file(step3_flag)

        #fill the missing information in the output final table with LLM
        #biology info
        #donor information
        #merge in final table

    ##STEP 3: ASSOCIATE TERMS WITH CODE
    #association uberon/dot with ref table

    #fill the unknown match with LLM

if __name__ == "__main__":
    main()
