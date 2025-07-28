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
    parser.add_argument("--metappuccino_dir", type=str, required=True,
                        help="Path to the Metappuccino directory")
    parser.add_argument("--res_dir", type=str, required=True,
                        help="Path to the results directory")
    parser.add_argument("--env_requirement", type=str, required=True,
                        help="Path to the venv build with requirement.txt")
    parser.add_argument("--model", type=str, required=True,
                        help="LLM model used for inference. Default = mistral 7B ft.")
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
    args = parser.parse_args()

    #scripts to execute and flags
    metappuccino_dir = args.metappuccino_dir
    res_dir = args.res_dir
    env_dir = args.env_requirement
    cuda_path = args.cuda
    model_path = args.model
    tmp_dir = os.path.join(metappuccino_dir+"/"+res_dir, "tmp")

    step1_flag = os.path.join(tmp_dir, "STEP1_1.flag")
    step2_flag = os.path.join(tmp_dir, "STEP2_1.flag")
    step3_flag = os.path.join(tmp_dir, "STEP2_2.flag")
    step4_flag = os.path.join(tmp_dir, "STEP3_1.flag")
    step5_flag = os.path.join(tmp_dir, "STEP3_2.flag")

    install_requirements = os.path.join(metappuccino_dir, "bin", "INSTALL_DOWNLOAD", "install_requirements.sh")
    download_metadata = os.path.join(metappuccino_dir, "bin", "INSTALL_DOWNLOAD", "download_metadata.sh")
    extract_preprocess = os.path.join(metappuccino_dir, "bin", "PRE_PROCESSING", "extract_preprocess.sh")
    summary_context = os.path.join(metappuccino_dir, "bin", "PRE_PROCESSING", "summary_context.sh")
    llm_metadata_inference = os.path.join(metappuccino_dir, "bin", "LLM_INFERENCE", "llm_metadata_inference.sh")
    reload_model = os.path.join(metappuccino_dir, "bin", "LLM_INFERENCE", "reload_model.sh")

    ##INSTALL REQUIREMENTS
    try:
        if not shutil.which("sbatch") and not shutil.which("qsub"):
            print("❌ Error: 'sbatch' or 'qsub' command not found", file=sys.stderr)
            sys.exit(1)

        if args.requirements:
            if shutil.which("qsub"):
                subprocess.run(["qsub", "-q", "alphafold", "-v", "METAPPUCCINO="+metappuccino_dir+","+"RES="+res_dir+","+"ENV_REQUIREMENT="+env_dir+","+"PATH_CUDA="+cuda_path, install_requirements], check=True)
                print("✔ Installation requirements completed!")
            elif shutil.which("sbatch"):
                subprocess.run(["sbatch", "--export=METAPPUCCINO="+metappuccino_dir+","+"RES="+res_dir+","+"ENV_REQUIREMENT="+env_dir+","+"PATH_CUDA="+cuda_path, install_requirements], check=True)
                print("✔ Installation requirements completed!")

        ##STEP 1: GET AND CLEAN METADATA
        if args.getmetadata:
            #get metadata from sra ncbi if asked from a sra list
            if not os.path.isfile(step1_flag):
                if shutil.which("qsub"):
                    subprocess.run(["qsub", "-q", "alphafold", "-v", "METAPPUCCINO="+metappuccino_dir+","+"RES="+res_dir+","+"ENV_REQUIREMENT="+env_dir,  download_metadata], check=True)
                elif shutil.which("sbatch"):
                    subprocess.run(["sbatch", "--export=METAPPUCCINO="+metappuccino_dir+","+"RES="+res_dir+","+"ENV_REQUIREMENT="+env_dir,  download_metadata], check=True)
            wait_for_flag_file(step1_flag)
            print("✔ Metadata download completed!")

            #clean metadata output table for xml config
            if not os.path.isfile(step2_flag):
                if shutil.which("qsub"):
                    subprocess.run(["qsub", "-q", "alphafold", "-v", "METAPPUCCINO="+metappuccino_dir+"RES="+res_dir+","+"ENV_REQUIREMENT="+env_dir, extract_preprocess], check=True)
                elif shutil.which("sbatch"):
                    subprocess.run(["sbatch", "--export=METAPPUCCINO="+metappuccino_dir+","+"RES="+res_dir+","+"ENV_REQUIREMENT="+env_dir, extract_preprocess], check=True)
            wait_for_flag_file(step2_flag)
            print("✔ Preprocessing completed successfully!")

            if not os.path.isfile(step3_flag):
                if shutil.which("qsub"):
                    subprocess.run(["qsub", "-q", "alphafold", "-v", "METAPPUCCINO="+metappuccino_dir+"RES="+res_dir+","+"ENV_REQUIREMENT="+env_dir, summary_context], check=True)
                elif shutil.which("sbatch"):
                    subprocess.run(["sbatch", "--export=METAPPUCCINO="+metappuccino_dir+","+"RES="+res_dir+","+"ENV_REQUIREMENT="+env_dir, summary_context], check=True)
            wait_for_flag_file(step3_flag)
            print("✔ Summary completed successfully!")

        ##STEP 2: FILL MISSING METADATA
        if args.fillmetadata:
            #fill the missing information in the output final table with LLM
            if not os.path.isfile(step4_flag):
                if shutil.which("qsub"):
                    subprocess.run(["qsub", "-q", "alphafold", "-v", "METAPPUCCINO="+metappuccino_dir+","+"RES="+res_dir+","+"ENV_REQUIREMENT="+env_dir+","+"MODEL="+model_path, llm_metadata_inference], check=True)
                elif shutil.which("sbatch"):
                    subprocess.run(["sbatch", "--export=METAPPUCCINO="+metappuccino_dir+","+"RES="+res_dir+","+"ENV_REQUIREMENT="+env_dir+","+"MODEL="+model_path, llm_metadata_inference], check=True)
            wait_for_flag_file(step4_flag)
            print("✔ LLM inference completed successfully!")

            #reload context
            if not os.path.isfile(step5_flag):
                if shutil.which("qsub"):
                    subprocess.run(["qsub", "-q", "alphafold", "-v", "METAPPUCCINO="+metappuccino_dir+","+"RES="+res_dir+","+"ENV_REQUIREMENT="+env_dir+","+"MODEL="+model_path, reload_model], check=True)
                elif shutil.which("sbatch"):
                    subprocess.run(["sbatch", "--export=METAPPUCCINO="+metappuccino_dir+","+"RES="+res_dir+","+"ENV_REQUIREMENT="+env_dir+","+"MODEL="+model_path, reload_model], check=True)
            wait_for_flag_file(step5_flag)
            print("✔ Context reloaded successfully!")

        ##STEP 3: ASSOCIATE TERMS WITH CODE
        # if args.associateinformation:
        #     #association uberon/dot with ref table and clean
        #     if not os.path.isfile(step6_flag):
        #         # subprocess.run(["qsub", "-q", "alphafold", "-v", "METAPPUCCINO="+metappuccino_dir+","+"RES="+res_dir+","+"ENV_REQUIREMENT="+env_dir, associate_information], check=True)
        #         subprocess.run(["sbatch", "--export=METAPPUCCINO="+metappuccino_dir+","+"RES="+res_dir+","+"ENV_REQUIREMENT="+env_dir, associate_information], check=True)
        #     wait_for_flag_file(step6_flag)
        #     print("✔ Code association and cleaning LLM answers successfully completed!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error in subprocess: {e.cmd} returned non-zero exit status {e.returncode}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()