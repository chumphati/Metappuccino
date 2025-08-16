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
    parser = argparse.ArgumentParser(description="Automates metadata extraction and completion based on LLMs.")
    parser.add_argument("--metappuccino_dir", type=str, required=True,
                        help="Path to the Metappuccino directory")
    parser.add_argument("--res_dir", type=str, required=True,
                        help="Path to the results directory")
    parser.add_argument("--env_requirement", type=str, required=True,
                        help="Path to the venv build with requirement.txt")
    parser.add_argument("--partition", type=str, default="", required=True,
                        help="Partition to request (required).")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to LLM model used for inference. mistral 7B ft is to download on hugging face.")
    parser.add_argument("--logan_path", type=str, default="",
                        help="Path to logan complementary information. Warning: 'sample_acc' must be to run accessions column. Default = mistral 7B ft.")
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
    parser.add_argument("--visualisation", action="store_true",
                        help="Build graphs to visualise the inferred metadata [Input: Metappuccino metadata output csv file]")
    parser.add_argument("--verbose", action="store_true",
                        help="Verbose output")
    parser.add_argument("--tmp_keep", action="store_true",
                        help="Keep final temporary file. Default = deleted.")
    parser.add_argument("--iteration_limit", type=int, default=1, help="Number of attempts to restart inference if less than 30% of categories have been predicted or if the JSON is malformed.")
    parser.add_argument("--local", action="store_true",
                        help="Run steps locally (sequentially) instead of submitting to PBS/Slurm.")
    parser.add_argument("--node", type=str, default="", help="Specific node name to request (optional).")
    parser.add_argument("--gpus", type=int, default=1, help="GPUs to use/request for LLM inference (passed to scheduler and as N_GPUS).")
    parser.add_argument("--cpus", type=int, default=30, help="CPUs to request from scheduler.")
    parser.add_argument("--mem", type=str, default="50gb", help="Memory to request from scheduler for LLM steps (e.g., '80gb').")
    parser.add_argument("--per_gpu_jobs", action="store_true",
                        help="Submit one scheduler job per GPU (creates N jobs with SHARD env).")
    args = parser.parse_args()

    metappuccino_dir = args.metappuccino_dir
    res_dir = args.res_dir
    tmp_keep = args.tmp_keep
    env_dir = args.env_requirement
    cuda_path = args.cuda
    model_path = args.model
    logan_path = args.logan_path
    iteration_limit = args.iteration_limit
    verbose = args.verbose
    node_req_in = args.node.strip()
    sched_gpus = args.gpus
    cpus_req = args.cpus
    mem_req = args.mem
    partition_req = args.partition.strip()
    queue_req = args.partition.strip()
    tmp_dir = os.path.join(metappuccino_dir, res_dir, "tmp")

    if node_req_in and node_req_in.isdigit():
        node_req = f"node{node_req_in}"
    else:
        node_req = node_req_in

    step1_flag = os.path.join(tmp_dir, "STEP1_1.flag")
    step2_0_flag = os.path.join(tmp_dir, "STEP2_0.flag")
    step2_flag = os.path.join(tmp_dir, "STEP2_1.flag")
    step3_flag = os.path.join(tmp_dir, "STEP2_2.flag")
    step4_flag = os.path.join(tmp_dir, "STEP3_1.flag")
    step5_flag = os.path.join(tmp_dir, "STEP3_2.flag")
    step6_flag = os.path.join(tmp_dir, "STEP4_1.flag")
    step7_flag = os.path.join(tmp_dir, "STEP4_2.flag")
    step8_flag = os.path.join(tmp_dir, "STEP4_3.flag")

    install_requirements = os.path.join(metappuccino_dir, "bin", "INSTALL_DOWNLOAD", "install_requirements.sh")
    download_metadata = os.path.join(metappuccino_dir, "bin", "INSTALL_DOWNLOAD", "download_metadata.sh")
    clean_metadata = os.path.join(metappuccino_dir, "bin", "PRE_PROCESSING", "clean_metadata.sh")
    extract_preprocess = os.path.join(metappuccino_dir, "bin", "PRE_PROCESSING", "extract_preprocess.sh")
    summary_context = os.path.join(metappuccino_dir, "bin", "PRE_PROCESSING", "summary_context.sh")
    llm_metadata_inference = os.path.join(metappuccino_dir, "bin", "LLM_INFERENCE", "llm_metadata_inference.sh")
    reload_model = os.path.join(metappuccino_dir, "bin", "LLM_INFERENCE", "reload_model.sh")
    normalize_final = os.path.join(metappuccino_dir, "bin", "NORMALISE_OUTS", "normalize_final.sh")
    visualisation = os.path.join(metappuccino_dir, "bin", "NORMALISE_OUTS", "visualisation.sh")

    # verbose helpers
    verbose_env = "TRUE" if verbose else "FALSE"
    vprint = print if verbose else (lambda *a, **k: None)

    ##INSTALL REQUIREMENTS
    try:
        if not args.local:
            if not shutil.which("sbatch") and not shutil.which("qsub"):
                print("Error: 'sbatch' or 'qsub' command not found", file=sys.stderr)
                sys.exit(1)

        if args.requirements:
            if args.local:
                subprocess.run(["bash", install_requirements,
                                metappuccino_dir, res_dir, env_dir, cuda_path],
                               check=True)
                vprint("✔ Installation requirements completed!")
            else:
                if shutil.which("qsub"):
                    cmd = ["qsub"]
                    if queue_req:
                        cmd += ["-q", queue_req]
                    pbs_l = "select=1"
                    if cpus_req:
                        pbs_l += f":ncpus={cpus_req}"
                    if mem_req:
                        pbs_l += f":mem={mem_req}"
                    cmd += ["-l", pbs_l,
                            "-v", f"METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},PATH_CUDA={cuda_path}", install_requirements]
                    vprint("Submitting:", " ".join(cmd))
                    subprocess.run(cmd, check=True)
                elif shutil.which("sbatch"):
                    sbatch_opts = []
                    if partition_req:
                        sbatch_opts += ["--partition", partition_req]
                    if cpus_req:
                        sbatch_opts += [f"--cpus-per-task={cpus_req}"]
                    if mem_req:
                        sbatch_opts += [f"--mem={mem_req}"]
                    cmd = ["sbatch", *sbatch_opts,
                           f"--export=METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},PATH_CUDA={cuda_path}", install_requirements]
                    vprint("Submitting:", " ".join(cmd))
                    subprocess.run(cmd, check=True)

        ##STEP 1: GET AND CLEAN METADATA
        if args.getmetadata:
            if not os.path.isfile(step1_flag):
                if args.local:
                    subprocess.run(["bash", download_metadata,
                                    metappuccino_dir, res_dir, env_dir, verbose_env],
                                   check=True)
                else:
                    if shutil.which("qsub"):
                        cmd = ["qsub"]
                        if queue_req:
                            cmd += ["-q", queue_req]
                        pbs_l = "select=1"
                        if cpus_req:
                            pbs_l += f":ncpus={cpus_req}"
                        if mem_req:
                            pbs_l += f":mem={mem_req}"
                        cmd += ["-l", pbs_l,
                                "-v", f"METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},VERBOSE={verbose_env}", download_metadata]
                        vprint("Submitting:", " ".join(cmd))
                        subprocess.run(cmd, check=True)
                    elif shutil.which("sbatch"):
                        sbatch_opts = []
                        if partition_req:
                            sbatch_opts += ["--partition", partition_req]
                        if cpus_req:
                            sbatch_opts += [f"--cpus-per-task={cpus_req}"]
                        if mem_req:
                            sbatch_opts += [f"--mem={mem_req}"]
                        cmd = ["sbatch", *sbatch_opts,
                               f"--export=METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},VERBOSE={verbose_env}", download_metadata]
                        vprint("Submitting:", " ".join(cmd))
                        subprocess.run(cmd, check=True)
            wait_for_flag_file(step1_flag)
            vprint("✔ Metadata download completed!")

            if not os.path.isfile(step2_0_flag):
                if args.local:
                    subprocess.run(["bash", clean_metadata,
                                    metappuccino_dir, res_dir],
                                   check=True)
                else:
                    if shutil.which("qsub"):
                        cmd = ["qsub"]
                        if queue_req:
                            cmd += ["-q", queue_req]
                        pbs_l = "select=1"
                        if cpus_req:
                            pbs_l += f":ncpus={cpus_req}"
                        if mem_req:
                            pbs_l += f":mem={mem_req}"
                        cmd += ["-l", pbs_l,
                                "-v", f"METAPPUCCINO={metappuccino_dir},RES={res_dir}", clean_metadata]
                        vprint("Submitting:", " ".join(cmd))
                        subprocess.run(cmd, check=True)
                    elif shutil.which("sbatch"):
                        sbatch_opts = []
                        if partition_req:
                            sbatch_opts += ["--partition", partition_req]
                        if cpus_req:
                            sbatch_opts += [f"--cpus-per-task={cpus_req}"]
                        if mem_req:
                            sbatch_opts += [f"--mem={mem_req}"]
                        cmd = ["sbatch", *sbatch_opts,
                               f"--export=METAPPUCCINO={metappuccino_dir},RES={res_dir}", clean_metadata]
                        vprint("Submitting:", " ".join(cmd))
                        subprocess.run(cmd, check=True)
            wait_for_flag_file(step2_0_flag)
            vprint("✔ Metadata cleaned!")

            if not os.path.isfile(step2_flag):
                if args.local:
                    subprocess.run(["bash", extract_preprocess,
                                    metappuccino_dir, res_dir, env_dir, logan_path, verbose_env],
                                   check=True)
                else:
                    if shutil.which("qsub"):
                        cmd = ["qsub"]
                        if queue_req:
                            cmd += ["-q", queue_req]
                        pbs_l = "select=1"
                        if cpus_req:
                            pbs_l += f":ncpus={cpus_req}"
                        if mem_req:
                            pbs_l += f":mem={mem_req}"
                        cmd += ["-l", pbs_l,
                                "-v", f"METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},LOGAN_PATH={logan_path},VERBOSE={verbose_env}", extract_preprocess]
                        vprint("Submitting:", " ".join(cmd))
                        subprocess.run(cmd, check=True)
                    elif shutil.which("sbatch"):
                        sbatch_opts = []
                        if partition_req:
                            sbatch_opts += ["--partition", partition_req]
                        if cpus_req:
                            sbatch_opts += [f"--cpus-per-task={cpus_req}"]
                        if mem_req:
                            sbatch_opts += [f"--mem={mem_req}"]
                        cmd = ["sbatch", *sbatch_opts,
                               f"--export=METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},LOGAN_PATH={logan_path},VERBOSE={verbose_env}", extract_preprocess]
                        vprint("Submitting:", " ".join(cmd))
                        subprocess.run(cmd, check=True)
            wait_for_flag_file(step2_flag)
            vprint("✔ Preprocessing completed successfully!")

            if not os.path.isfile(step3_flag):
                if args.local:
                    subprocess.run(["bash", summary_context,
                                    metappuccino_dir, res_dir, env_dir, verbose_env],
                                   check=True)
                else:
                    if shutil.which("qsub"):
                        cmd = ["qsub"]
                        if queue_req:
                            cmd += ["-q", queue_req]
                        pbs_l = "select=1"
                        if cpus_req:
                            pbs_l += f":ncpus={cpus_req}"
                        if mem_req:
                            pbs_l += f":mem={mem_req}"
                        cmd += ["-l", pbs_l,
                                "-v", f"METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},VERBOSE={verbose_env}", summary_context]
                        vprint("Submitting:", " ".join(cmd))
                        subprocess.run(cmd, check=True)
                    elif shutil.which("sbatch"):
                        sbatch_opts = []
                        if partition_req:
                            sbatch_opts += ["--partition", partition_req]
                        if cpus_req:
                            sbatch_opts += [f"--cpus-per-task={cpus_req}"]
                        if mem_req:
                            sbatch_opts += [f"--mem={mem_req}"]
                        cmd = ["sbatch", *sbatch_opts,
                               f"--export=METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},VERBOSE={verbose_env}", summary_context]
                        vprint("Submitting:", " ".join(cmd))
                        subprocess.run(cmd, check=True)
            wait_for_flag_file(step3_flag)
            vprint("✔ Summary completed successfully!")

        ##STEP 2: FILL MISSING METADATA
        if args.fillmetadata:
            if not os.path.isfile(step4_flag):
                if args.local:
                    subprocess.run(["bash", llm_metadata_inference,
                                    metappuccino_dir, res_dir, env_dir, model_path, verbose_env, str(sched_gpus)],
                                   check=True)
                else:
                    if args.per_gpu_jobs and (sched_gpus and sched_gpus > 1):
                        if shutil.which("qsub"):
                            for i in range(sched_gpus):
                                pbs_l = "select=1"
                                if node_req:
                                    pbs_l += f":host={node_req}"
                                if cpus_req:
                                    pbs_l += f":ncpus={cpus_req}"
                                if mem_req:
                                    pbs_l += f":mem={mem_req}"
                                pbs_l += f":ngpus=1"
                                cmd = ["qsub"]
                                if queue_req:
                                    cmd += ["-q", queue_req]
                                cmd += ["-l", pbs_l,
                                        "-v", f"METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},MODEL={model_path},VERBOSE={verbose_env},N_GPUS=1,SHARD_TOTAL={sched_gpus},SHARD_ID={i}",
                                        llm_metadata_inference]
                                vprint("Submitting:", " ".join(cmd))
                                subprocess.run(cmd, check=True)
                        elif shutil.which("sbatch"):
                            for i in range(sched_gpus):
                                sbatch_opts = []
                                if partition_req:
                                    sbatch_opts += ["--partition", partition_req]
                                sbatch_opts += ["--gres=gpu:1"]
                                if node_req:
                                    sbatch_opts += [f"--nodelist={node_req}"]
                                if cpus_req:
                                    sbatch_opts += [f"--cpus-per-task={cpus_req}"]
                                if mem_req:
                                    sbatch_opts += [f"--mem={mem_req}"]
                                cmd = ["sbatch", *sbatch_opts,
                                       f"--export=METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},MODEL={model_path},VERBOSE={verbose_env},N_GPUS=1,SHARD_TOTAL={sched_gpus},SHARD_ID={i}",
                                       llm_metadata_inference]
                                vprint("Submitting:", " ".join(cmd))
                                subprocess.run(cmd, check=True)
                        else:
                            raise RuntimeError("No scheduler detected.")
                    else:
                        if shutil.which("qsub"):
                            pbs_l = "select=1"
                            if node_req:
                                pbs_l += f":host={node_req}"
                            if cpus_req:
                                pbs_l += f":ncpus={cpus_req}"
                            if mem_req:
                                pbs_l += f":mem={mem_req}"
                            if sched_gpus and sched_gpus > 0:
                                pbs_l += f":ngpus={sched_gpus}"
                            cmd = ["qsub"]
                            if queue_req:
                                cmd += ["-q", queue_req]
                            cmd += ["-l", pbs_l,
                                    "-v", f"METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},MODEL={model_path},VERBOSE={verbose_env},N_GPUS={sched_gpus}",
                                    llm_metadata_inference]
                            vprint("Submitting:", " ".join(cmd))
                            subprocess.run(cmd, check=True)
                        elif shutil.which("sbatch"):
                            sbatch_opts = []
                            if partition_req:
                                sbatch_opts += ["--partition", partition_req]
                            if sched_gpus and sched_gpus > 0:
                                sbatch_opts += [f"--gres=gpu:{sched_gpus}"]
                            if node_req:
                                sbatch_opts += [f"--nodelist={node_req}"]
                            if cpus_req:
                                sbatch_opts += [f"--cpus-per-task={cpus_req}"]
                            if mem_req:
                                sbatch_opts += [f"--mem={mem_req}"]
                            cmd = ["sbatch", *sbatch_opts,
                                   f"--export=METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},MODEL={model_path},VERBOSE={verbose_env},N_GPUS={sched_gpus}",
                                   llm_metadata_inference]
                            vprint("Submitting:", " ".join(cmd))
                            subprocess.run(cmd, check=True)
            wait_for_flag_file(step4_flag)
            vprint("✔ LLM inference completed successfully!")

            if not os.path.isfile(step5_flag):
                if args.local:
                    subprocess.run(["bash", reload_model,
                                    metappuccino_dir, res_dir, env_dir, model_path, str(iteration_limit), verbose_env, str(sched_gpus)],
                                   check=True)
                else:
                    if args.per_gpu_jobs and (sched_gpus and sched_gpus > 1):
                        if shutil.which("qsub"):
                            for i in range(sched_gpus):
                                pbs_l = "select=1"
                                if node_req:
                                    pbs_l += f":host={node_req}"
                                if cpus_req:
                                    pbs_l += f":ncpus={cpus_req}"
                                if mem_req:
                                    pbs_l += f":mem={mem_req}"
                                pbs_l += f":ngpus=1"
                                cmd = ["qsub"]
                                if queue_req:
                                    cmd += ["-q", queue_req]
                                cmd += ["-l", pbs_l,
                                        "-v", f"METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},MODEL={model_path},ITERATION_LIMIT={iteration_limit},VERBOSE={verbose_env},N_GPUS=1,SHARD_TOTAL={sched_gpus},SHARD_ID={i}",
                                        reload_model]
                                vprint("Submitting:", " ".join(cmd))
                                subprocess.run(cmd, check=True)
                        elif shutil.which("sbatch"):
                            for i in range(sched_gpus):
                                sbatch_opts = []
                                if partition_req:
                                    sbatch_opts += ["--partition", partition_req]
                                sbatch_opts += ["--gres=gpu:1"]
                                if node_req:
                                    sbatch_opts += [f"--nodelist={node_req}"]
                                if cpus_req:
                                    sbatch_opts += [f"--cpus-per-task={cpus_req}"]
                                if mem_req:
                                    sbatch_opts += [f"--mem={mem_req}"]
                                cmd = ["sbatch", *sbatch_opts,
                                       f"--export=METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},MODEL={model_path},ITERATION_LIMIT={iteration_limit},VERBOSE={verbose_env},N_GPUS=1,SHARD_TOTAL={sched_gpus},SHARD_ID={i}",
                                       reload_model]
                                vprint("Submitting:", " ".join(cmd))
                                subprocess.run(cmd, check=True)
                        else:
                            raise RuntimeError("No scheduler detected.")
                    else:
                        if shutil.which("qsub"):
                            pbs_l = "select=1"
                            if node_req:
                                pbs_l += f":host={node_req}"
                            if cpus_req:
                                pbs_l += f":ncpus={cpus_req}"
                            if mem_req:
                                pbs_l += f":mem={mem_req}"
                            if sched_gpus and sched_gpus > 0:
                                pbs_l += f":ngpus={sched_gpus}"
                            cmd = ["qsub"]
                            if queue_req:
                                cmd += ["-q", queue_req]
                            cmd += ["-l", pbs_l,
                                    "-v", f"METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},MODEL={model_path},ITERATION_LIMIT={iteration_limit},VERBOSE={verbose_env},N_GPUS={sched_gpus}",
                                    reload_model]
                            vprint("Submitting:", " ".join(cmd))
                            subprocess.run(cmd, check=True)
                        elif shutil.which("sbatch"):
                            sbatch_opts = []
                            if partition_req:
                                sbatch_opts += ["--partition", partition_req]
                            if sched_gpus and sched_gpus > 0:
                                sbatch_opts += [f"--gres=gpu:{sched_gpus}"]
                            if node_req:
                                sbatch_opts += [f"--nodelist={node_req}"]
                            if cpus_req:
                                sbatch_opts += [f"--cpus-per-task={cpus_req}"]
                            if mem_req:
                                sbatch_opts += [f"--mem={mem_req}"]
                            cmd = ["sbatch", *sbatch_opts,
                                   f"--export=METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},MODEL={model_path},ITERATION_LIMIT={iteration_limit},VERBOSE={verbose_env},N_GPUS={sched_gpus}",
                                   reload_model]
                            vprint("Submitting:", " ".join(cmd))
                            subprocess.run(cmd, check=True)
            wait_for_flag_file(step5_flag)
            vprint("✔ Context reloaded successfully!")

        ##STEP 3: ASSOCIATE TERMS WITH CODE
        if args.associateinformation:
            if not os.path.isfile(step6_flag):
                if args.local:
                    subprocess.run(["bash", normalize_final,
                                    metappuccino_dir, res_dir, env_dir, verbose_env],
                                   check=True)
                else:
                    if shutil.which("qsub"):
                        cmd = ["qsub"]
                        if queue_req:
                            cmd += ["-q", queue_req]
                        pbs_l = "select=1"
                        if cpus_req:
                            pbs_l += f":ncpus={cpus_req}"
                        if mem_req:
                            pbs_l += f":mem={mem_req}"
                        cmd += ["-l", pbs_l,
                                "-v",
                                f"METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},VERBOSE={verbose_env}",
                                normalize_final]
                        vprint("Submitting:", " ".join(cmd))
                        subprocess.run(cmd, check=True)
                    elif shutil.which("sbatch"):
                        sbatch_opts = []
                        if partition_req:
                            sbatch_opts += ["--partition", partition_req]
                        if cpus_req:
                            sbatch_opts += [f"--cpus-per-task={cpus_req}"]
                        if mem_req:
                            sbatch_opts += [f"--mem={mem_req}"]
                        cmd = ["sbatch", *sbatch_opts,
                               f"--export=METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},VERBOSE={verbose_env}",
                               normalize_final]
                        vprint("Submitting:", " ".join(cmd))
                        subprocess.run(cmd, check=True)
            wait_for_flag_file(step6_flag)
            vprint("✔ Code association and cleaning LLM answers successfully completed!")

        if args.visualisation:
            if not os.path.isfile(step7_flag):
                if args.local:
                    subprocess.run(["bash", visualisation,
                                    metappuccino_dir, res_dir, env_dir, verbose_env],
                                   check=True)
                else:
                    if shutil.which("qsub"):
                        cmd = ["qsub"]
                        if queue_req:
                            cmd += ["-q", queue_req]
                        pbs_l = "select=1"
                        if cpus_req:
                            pbs_l += f":ncpus={cpus_req}"
                        if mem_req:
                            pbs_l += f":mem={mem_req}"
                        cmd += ["-l", pbs_l,
                                "-v",
                                f"METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},VERBOSE={verbose_env}",
                                visualisation]
                        vprint("Submitting:", " ".join(cmd))
                        subprocess.run(cmd, check=True)
                    elif shutil.which("sbatch"):
                        sbatch_opts = []
                        if partition_req:
                            sbatch_opts += ["--partition", partition_req]
                        if cpus_req:
                            sbatch_opts += [f"--cpus-per-task={cpus_req}"]
                        if mem_req:
                            sbatch_opts += [f"--mem={mem_req}"]
                        cmd = ["sbatch", *sbatch_opts,
                               f"--export=METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},VERBOSE={verbose_env}",
                               visualisation]
                        vprint("Submitting:", " ".join(cmd))
                        subprocess.run(cmd, check=True)
            wait_for_flag_file(step7_flag)
            vprint("✔ Graphs build successfully!")

        if not tmp_keep:
            tmp_dir = os.path.join(metappuccino_dir, res_dir, "tmp")
            if os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir)
                vprint(f"✔ Temporary files deleted successfully!")
            else:
                vprint(f"Temporary directory '{tmp_dir}' does not exist or was already deleted.")

    except subprocess.CalledProcessError as e:
        print(f"Error in subprocess: {e.cmd} returned non-zero exit status {e.returncode}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
