########################################################################################################################
#IMPORT LIB
import os
import subprocess
import argparse
import time
import sys
import shutil

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
    parser.add_argument("--metamap_dir", type=str, required=True,
                        help="Path to the MetaMap directory")
    parser.add_argument("--env_requirement", type=str, required=True,
                        help="Path to the venv build with requirement.txt")
    parser.add_argument("--requirements", action="store_true",
                        help="Install requirements.txt and CUDA configuration for GPU. Warning: It is assumed that a venv was created and the path correctly given in --env_requirement. In the case a CUDA configuration is needed, please specify cuda path in --cuda_path if it is different from /usr/local/cuda.")
    parser.add_argument("--cuda", type=str, default="/usr/local/cuda",
                        help="Path to CUDA installation if different from '/usr/local/cuda'")
    parser.add_argument("--getmetadata", action="store_true",
                        help="Download and clean metadata from NCBI [Input: List of run accessions]")
    parser.add_argument("--fillmetadata", action="store_true",
                        help="Fill metadata with LLMs [Input: Cleaned sra file]")
    parser.add_argument("--associateinformation", action="store_true",
                        help="Associate medical codes with LLMs answers and clean them [Input: LLMs answers]")
    parser.add_argument("--completestudy", action="store_true",
                        help="Fill metadata with study information if needed [Input: Cleaned sra file]")
    args = parser.parse_args()

    #scripts to execute and flags
    metamap_dir = args.metamap_dir
    env_dir = args.env_requirement
    cuda_path = args.cuda
    tmp_dir = os.path.join(metamap_dir, "results", "tmp")

    step1_flag = os.path.join(tmp_dir, "STEP1_1.flag")
    step2_flag = os.path.join(tmp_dir, "STEP1_2.flag")
    step3_flag = os.path.join(tmp_dir, "STEP2_1.flag")
    step4_flag = os.path.join(tmp_dir, "STEP2_2.flag")
    step5_flag = os.path.join(tmp_dir, "STEP2_3.flag")
    step6_flag = os.path.join(tmp_dir, "STEP3.flag")
    step7_flag = os.path.join(tmp_dir, "STEP4_1.flag")
    step8_flag = os.path.join(tmp_dir, "STEP4_2.flag")

    install_requirements = os.path.join(metamap_dir, "bin", "STEP1", "install_requirements.sh")
    download_script = os.path.join(metamap_dir, "bin", "STEP1", "download_metadata.sh")
    clean_script = os.path.join(metamap_dir, "bin", "STEP1", "clean_metadata.sh")
    get_stable_metadata = os.path.join(metamap_dir, "bin", "STEP2", "get_stable_metadata.sh")
    llm_specific_biology_information = os.path.join(metamap_dir, "bin", "STEP2", "llm_specific_biology_information.sh")
    split_col_cleanmetadata = os.path.join(metamap_dir, "bin", "STEP2", "split_col_cleanmetadata.sh")
    associate_information = os.path.join(metamap_dir, "bin", "STEP3", "associate_codes_clean.sh")
    llm_study_information = os.path.join(metamap_dir, "bin", "STEP4", "llm_study_information.sh")
    process_study_llm = os.path.join(metamap_dir, "bin", "STEP4", "sort_entropy.sh")

    ##INSTALL REQUIREMENTS
    try:
        if not shutil.which("qsub"):
            print("❌ Error: 'qsub' command not found", file=sys.stderr)
            sys.exit(1)

        if args.requirements:
            subprocess.run(["qsub", "-q", "common", "-v", "METAMAP="+metamap_dir+","+"ENV_REQUIREMENT="+env_dir+","+"PATH_CUDA="+cuda_path, install_requirements], check=True)
            print("✔ Installation requirements completed!")

        ##STEP 1: GET AND CLEAN METADATA
        if args.getmetadata:
            #get metadata from sra ncbi if asked from a sra list
            if not os.path.isfile(step1_flag):
                # print("qsub", "-q", "common", "-v", "METAMAP="+metamap_dir+","+"ENV_REQUIREMENT="+env_dir,  download_script)
                subprocess.run(["qsub", "-q", "common", "-v", "METAMAP="+metamap_dir+","+"ENV_REQUIREMENT="+env_dir,  download_script], check=True)
            wait_for_flag_file(step1_flag)
            print("✔ Metadata download completed!")

            #clean metadata output table for xml config
            if not os.path.isfile(step2_flag):
                subprocess.run(["qsub", "-q", "common", "-v", "METAMAP="+metamap_dir, clean_script], check=True)
            wait_for_flag_file(step2_flag)
            print("✔ Metadata cleaned!")

        ##STEP 2: FILL MISSING METADATA
        if args.fillmetadata:
            #get basic information for each run in output final table
            if not os.path.isfile(step3_flag):
                subprocess.run(["qsub", "-q", "common", "-v", "METAMAP="+metamap_dir+","+"ENV_REQUIREMENT="+env_dir, get_stable_metadata], check=True)
            wait_for_flag_file(step3_flag)
            print("✔ Initial data retrieval directly from databases completed!")

            #split specific, study and donor analysis
            if not os.path.isfile(step4_flag):
                subprocess.run(["qsub", "-q", "common", "-v", "OUTPUT_DIR="+metamap_dir+","+"ENV_REQUIREMENT="+env_dir, split_col_cleanmetadata], check=True)
            wait_for_flag_file(step4_flag)
            print("✔ Initial data retrieval directly from databases completed!")

            #fill the missing information in the output final table with LLM
            #biology info
            if not os.path.isfile(step5_flag):
                subprocess.run(["qsub", "-q", "alphafold", "-v", "METAMAP="+metamap_dir+","+"ENV_REQUIREMENT="+env_dir, llm_specific_biology_information], check=True)
            wait_for_flag_file(step5_flag)
            print("✔ Specific run information filled by LLM model successfully!")

            #donor information
            #merge in final table

        ##STEP 3: ASSOCIATE TERMS WITH CODE
        if args.associateinformation:
            #association uberon/dot with ref table and clean
            if not os.path.isfile(step6_flag):
                subprocess.run(["qsub", "-q", "common", "-v", "METAMAP="+metamap_dir+","+"ENV_REQUIREMENT="+env_dir, associate_information], check=True)
            wait_for_flag_file(step6_flag)
            print("✔ Code association and cleaning LLM answers successfully completed!")

        ##STEP 4: STUDY COMPLETION
        if args.completestudy:
            # fill the missing information in the output final table with study info
            if not os.path.isfile(step7_flag):
                subprocess.run(
                    ["qsub", "-q", "alphafold", "-v", "METAMAP=" + metamap_dir + "," + "ENV_REQUIREMENT=" + env_dir,
                     llm_study_information], check=True)
            wait_for_flag_file(step7_flag)
            print("✔ Study information for projects filled by LLM model successfully!")

            # process study info via entropy
            if not os.path.isfile(step8_flag):
                subprocess.run(
                    ["qsub", "-q", "common", "-v", "METAMAP=" + metamap_dir + "," + "ENV_REQUIREMENT=" + env_dir,
                     process_study_llm], check=True)
            wait_for_flag_file(step8_flag)
            print("✔ Study information information processed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error in subprocess: {e.cmd} returned non-zero exit status {e.returncode}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()