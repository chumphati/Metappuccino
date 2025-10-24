########################################################################################################################
#IMPORT LIB
import os
import subprocess
import argparse
import time
import sys
import shutil
import importlib.resources as pkg_resources
import re

########################################################################################################################
#FUNCTIONS
#wait for a job to be completed to launch the next one
def wait_for_flag_file(flag_path):
    while not os.path.isfile(flag_path):
        time.sleep(10)

#get automatic link to tool
def get_metappuccino_dir():
    return str(pkg_resources.files("metappuccino"))

#auto path
def resolve_path(rel_path: str, metappuccino_dir: str = None):
    if metappuccino_dir:
        return os.path.join(metappuccino_dir, rel_path)
    else:
        return str(pkg_resources.files("metappuccino") / rel_path)

def _slurm_job_active(job_id: str) -> bool:
    try:
        r = subprocess.run(["squeue", "-j", str(job_id), "-h"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        return r.returncode == 0 and bool(r.stdout.strip())
    except Exception:
        return False

def _pbs_job_active(job_id: str) -> bool:
    try:
        r = subprocess.run(["qstat", str(job_id)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True)
        return r.returncode == 0
    except Exception:
        return False

def wait_for_flag_or_job_end(flag_path: str, job_handles=(), poll_seconds: int = 10, grace_seconds: int = 30):
    while True:
        if os.path.isfile(flag_path):
            return
        if job_handles:
            any_active = False
            for sched, jid in job_handles:
                if sched == "slurm":
                    if _slurm_job_active(jid):
                        any_active = True
                elif sched == "pbs":
                    if _pbs_job_active(jid):
                        any_active = True
            if not any_active:
                t0 = time.time()
                while time.time() - t0 < grace_seconds:
                    if os.path.isfile(flag_path):
                        return
                    time.sleep(1)
                raise RuntimeError(f"Sub-job(s) finished but flag not found: {flag_path}")
        time.sleep(poll_seconds)

def _parse_qsub_jobid(stdout: str) -> str:
    line = stdout.strip().splitlines()[-1].strip()
    return line.split()[0]

def _parse_sbatch_jobid(stdout: str) -> str:
    m = re.search(r"Submitted batch job (\d+)", stdout)
    if not m:
        raise RuntimeError(f"Unable to parse sbatch output: {stdout!r}")
    return m.group(1)

def submit_job(cmd_list, scheduler: str, env):
    res = subprocess.run(cmd_list, check=True, capture_output=True, text=True, env=env)
    if scheduler == "pbs":
        jid = _parse_qsub_jobid(res.stdout)
        return ("pbs", jid)
    elif scheduler == "slurm":
        jid = _parse_sbatch_jobid(res.stdout)
        return ("slurm", jid)
    else:
        raise ValueError("scheduler must be 'pbs' or 'slurm'")

def ensure_flag_after_local(flag_path: str, grace_seconds: int = 30):
    t0 = time.time()
    while time.time() - t0 < grace_seconds:
        if os.path.isfile(flag_path):
            return
        time.sleep(1)
    raise RuntimeError(f"Local step finished but flag not found: {flag_path}")

########################################################################################################################
#MAIN FUNCTION
def main():
    parser = argparse.ArgumentParser(description="Automates metadata extraction and completion based on LLM.")
    parser.add_argument("--sample_input", type=str, required=True,
                        help="Path to the sample input file. Must be a '.csv' or '.txt' file, with one run accession number per ligne.")
    parser.add_argument("--res_dir", type=str, required=True,
                        help="Path to the results directory")
    parser.add_argument("--env_requirement", type=str, required=True,
                        help="Path to the venv builded with requirement.txt")
    parser.add_argument("--model", type=str,
                        help="Path to LLM model used for inference. A specific model trained for the task can be downloaded.")
    parser.add_argument("--gguf", action="store_true",
                        help="LLM  inference based on a gguf model (!= Metappuccino fine-tuned model).")
    parser.add_argument("--metappuccino_dir", type=str,
                        help="Path to the Metappuccino directory (Use in case of installation from source).")
    parser.add_argument("--working_dir", type=str, default="",
                        help="Absolute path to the working directory on the compute node—preferably a local scratch location—with sufficient writable space. Examples: $SLURM_TMPDIR, $TMPDIR, /scratchlocal/$USER/$PBS_JOBID. Contents are temporary and may be cleaned up at job end.")
    parser.add_argument("--logan_path", type=str, default="",
                        help="Path to logan complementary information. Warning: 'sample_acc' must be to run accessions column. Default = mistral 7B ft.")
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
    parser.add_argument("--iteration_limit", type=int, default=0, help="Number of attempts to restart inference if less than 30%% of categories have been predicted or if the JSON is malformed.")
    parser.add_argument("--local", action="store_true",
                        help="Run steps locally (sequentially) instead of submitting on a scheduler.")
    parser.add_argument("--node", type=str, default="", help="Specific node name to request (if working on a scheduler).")
    parser.add_argument("--partition", type=str, default="",
                        help="Partition to request (if working on a scheduler).")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs to request for LLM inference.")
    parser.add_argument("--cpus", type=int, default=30, help="Number of CPUs to request.")
    parser.add_argument("--mem", type=str, default="50gb", help="Memory to request.")
    parser.add_argument("--ncbi_email", type=str, default=None, help="Contact email for NCBI E-utilities")
    parser.add_argument("--ncbi_api_key", type=str, default=None, help="NCBI E-utilities API key")
    parser.add_argument("--per_gpu_jobs", action="store_true",
                        help="Submit one scheduler job per GPU (if several GPUs are available and user wants to divide the LLM inference across all of them).")
    args = parser.parse_args()

    metappuccino_dir = args.metappuccino_dir or get_metappuccino_dir()
    env = os.environ.copy()
    env["METAPPUCCINO"] = metappuccino_dir

    sample_input = args.sample_input
    res_dir = args.res_dir
    tmp_keep = args.tmp_keep
    env_dir = args.env_requirement
    model_path = args.model
    working_dir = args.working_dir
    logan_path = args.logan_path
    iteration_limit = args.iteration_limit
    verbose = args.verbose
    node_req_in = args.node.strip()
    sched_gpus = args.gpus
    cpus_req = args.cpus
    mem_req = args.mem
    ncbi_email = args.ncbi_email
    ncbi_api_key = args.ncbi_api_key
    partition_req = args.partition.strip()
    queue_req = args.partition.strip()
    tmp_dir = os.path.join(res_dir, "tmp")

    if node_req_in and node_req_in.isdigit():
        node_req = f"node{node_req_in}"
    else:
        node_req = node_req_in

    step1_0_flag = os.path.join(tmp_dir, "STEP1_0.flag")
    step1_flag = os.path.join(tmp_dir, "STEP1_1.flag")
    step2_0_flag = os.path.join(tmp_dir, "STEP2_0.flag")
    step2_flag = os.path.join(tmp_dir, "STEP2_1.flag")
    step3_flag = os.path.join(tmp_dir, "STEP2_2.flag")
    step4_flag = os.path.join(tmp_dir, "STEP3_1.flag")
    step5_flag = os.path.join(tmp_dir, "STEP3_2.flag")
    step6_flag = os.path.join(tmp_dir, "STEP4_1.flag")
    step7_flag = os.path.join(tmp_dir, "STEP4_2.flag")
    step8_flag = os.path.join(tmp_dir, "STEP4_3.flag")

    init = str(resolve_path("bin/INSTALL_DOWNLOAD/init.sh", args.metappuccino_dir))
    download_metadata = str(resolve_path("bin/INSTALL_DOWNLOAD/download_metadata.sh", args.metappuccino_dir))
    clean_metadata = str(resolve_path("bin/PRE_PROCESSING/clean_metadata.sh", args.metappuccino_dir))
    extract_preprocess = str(resolve_path("bin/PRE_PROCESSING/extract_preprocess.sh", args.metappuccino_dir))
    summary_context = str(resolve_path("bin/PRE_PROCESSING/summary_context.sh", args.metappuccino_dir))
    normalize_final = str(resolve_path("bin/NORMALISE_OUTS/normalize_final.sh", args.metappuccino_dir))
    visualisation = str(resolve_path("bin/NORMALISE_OUTS/visualisation.sh", args.metappuccino_dir))

    if args.gguf:
        llm_metadata_inference = str(resolve_path("bin/LLM_INFERENCE/llm_metadata_inference.sh", args.metappuccino_dir))
        reload_model = str(resolve_path("bin/LLM_INFERENCE/reload_model.sh", args.metappuccino_dir))
    else:
        llm_metadata_inference = str(resolve_path("bin/LLM_INFERENCE/llm_MI_per_category.sh", args.metappuccino_dir))
        reload_model = str(resolve_path("bin/LLM_INFERENCE/reload_MI_per_category.sh", args.metappuccino_dir))

    verbose_env = "TRUE" if verbose else "FALSE"
    vprint = print if verbose else (lambda *a, **k: None)

    ##DOWNLOAD/CLEAN
    try:
        if not args.local:
            if not shutil.which("sbatch") and not shutil.which("qsub"):
                print("Error: 'sbatch' or 'qsub' command not found", file=sys.stderr)
                sys.exit(1)

        ##STEP 1: GET AND CLEAN METADATA
        if not os.path.isfile(step1_0_flag):
            if args.local:
                subprocess.run(["bash", init,
                                metappuccino_dir, res_dir, working_dir, env_dir],
                               check=True, env=env)
                ensure_flag_after_local(step1_0_flag)
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
                            "-v", f"METAPPUCCINO={metappuccino_dir},RES={res_dir},NODE_WORK_PATH={working_dir},ENV_REQUIREMENT={env_dir}", init]
                    vprint("Submitting:", " ".join(cmd))
                    handle = submit_job(cmd, "pbs", env)
                    wait_for_flag_or_job_end(step1_0_flag, [handle])
                elif shutil.which("sbatch"):
                    sbatch_opts = []
                    if partition_req:
                        sbatch_opts += ["--partition", partition_req]
                    if cpus_req:
                        sbatch_opts += [f"--cpus-per-task={cpus_req}"]
                    if mem_req:
                        sbatch_opts += [f"--mem={mem_req}"]
                    cmd = ["sbatch", *sbatch_opts,
                           f"--export=METAPPUCCINO={metappuccino_dir},RES={res_dir},NODE_WORK_PATH={working_dir},ENV_REQUIREMENT={env_dir}", init]
                    vprint("Submitting:", " ".join(cmd))
                    handle = submit_job(cmd, "slurm", env)
                    wait_for_flag_or_job_end(step1_0_flag, [handle])
        wait_for_flag_file(step1_0_flag)
        vprint("✔ Setup initialisation done!", file=sys.stdout)

        if args.getmetadata:
            if not os.path.isfile(step1_flag):
                if args.local:
                    subprocess.run(["bash", download_metadata,
                                    metappuccino_dir, res_dir, env_dir, verbose_env, working_dir, sample_input, cpus_req, ncbi_api_key, ncbi_email],
                                   check=True, env=env)
                    ensure_flag_after_local(step1_flag)
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
                                "-v", f"METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},VERBOSE={verbose_env},NODE_WORK_PATH={working_dir},RUNS_INPUTS={sample_input},N_CPUS={cpus_req},NCBI_API_KEYS={ncbi_api_key},NCBI_EMAIL={ncbi_email}", download_metadata]
                        vprint("Submitting:", " ".join(cmd))
                        handle = submit_job(cmd, "pbs", env)
                        wait_for_flag_or_job_end(step1_flag, [handle])
                    elif shutil.which("sbatch"):
                        sbatch_opts = []
                        if partition_req:
                            sbatch_opts += ["--partition", partition_req]
                        if cpus_req:
                            sbatch_opts += [f"--cpus-per-task={cpus_req}"]
                        if mem_req:
                            sbatch_opts += [f"--mem={mem_req}"]
                        cmd = ["sbatch", *sbatch_opts,
                               f"--export=METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},VERBOSE={verbose_env},NODE_WORK_PATH={working_dir},RUNS_INPUTS={sample_input},N_CPUS={cpus_req},NCBI_API_KEYS={ncbi_api_key},NCBI_EMAIL={ncbi_email}", download_metadata]
                        vprint("Submitting:", " ".join(cmd))
                        handle = submit_job(cmd, "slurm", env)
                        wait_for_flag_or_job_end(step1_flag, [handle])
            wait_for_flag_file(step1_flag)
            vprint("✔ Metadata download completed!", file=sys.stdout)

            if not os.path.isfile(step2_0_flag):
                if args.local:
                    subprocess.run(["bash", clean_metadata,
                                    metappuccino_dir, res_dir, working_dir, env_dir],
                                   check=True, env=env)
                    ensure_flag_after_local(step2_0_flag)
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
                                "-v", f"METAPPUCCINO={metappuccino_dir},RES={res_dir},NODE_WORK_PATH={working_dir},ENV_REQUIREMENT={env_dir}", clean_metadata]
                        vprint("Submitting:", " ".join(cmd))
                        handle = submit_job(cmd, "pbs", env)
                        wait_for_flag_or_job_end(step2_0_flag, [handle])
                    elif shutil.which("sbatch"):
                        sbatch_opts = []
                        if partition_req:
                            sbatch_opts += ["--partition", partition_req]
                        if cpus_req:
                            sbatch_opts += [f"--cpus-per-task={cpus_req}"]
                        if mem_req:
                            sbatch_opts += [f"--mem={mem_req}"]
                        cmd = ["sbatch", *sbatch_opts,
                               f"--export=METAPPUCCINO={metappuccino_dir},RES={res_dir},NODE_WORK_PATH={working_dir},ENV_REQUIREMENT={env_dir}", clean_metadata]
                        vprint("Submitting:", " ".join(cmd))
                        handle = submit_job(cmd, "slurm", env)
                        wait_for_flag_or_job_end(step2_0_flag, [handle])
            wait_for_flag_file(step2_0_flag)
            vprint("✔ Metadata cleaned!", file=sys.stdout)

            if not os.path.isfile(step2_flag):
                if args.local:
                    subprocess.run(["bash", extract_preprocess,
                                    metappuccino_dir, res_dir, env_dir, logan_path, verbose_env, working_dir],
                                   check=True, env=env)
                    ensure_flag_after_local(step2_flag)
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
                                "-v", f"METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},LOGAN_PATH={logan_path},VERBOSE={verbose_env},NODE_WORK_PATH={working_dir}", extract_preprocess]
                        vprint("Submitting:", " ".join(cmd))
                        handle = submit_job(cmd, "pbs", env)
                        wait_for_flag_or_job_end(step2_flag, [handle])
                    elif shutil.which("sbatch"):
                        sbatch_opts = []
                        if partition_req:
                            sbatch_opts += ["--partition", partition_req]
                        if cpus_req:
                            sbatch_opts += [f"--cpus-per-task={cpus_req}"]
                        if mem_req:
                            sbatch_opts += [f"--mem={mem_req}"]
                        cmd = ["sbatch", *sbatch_opts,
                               f"--export=METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},LOGAN_PATH={logan_path},VERBOSE={verbose_env},NODE_WORK_PATH={working_dir}", extract_preprocess]
                        vprint("Submitting:", " ".join(cmd))
                        handle = submit_job(cmd, "slurm", env)
                        wait_for_flag_or_job_end(step2_flag, [handle])
            wait_for_flag_file(step2_flag)
            vprint("✔ Preprocessing completed successfully!", file=sys.stdout)

            if not os.path.isfile(step3_flag):
                if args.local:
                    subprocess.run(["bash", summary_context,
                                    metappuccino_dir, res_dir, env_dir, verbose_env,working_dir],
                                   check=True, env=env)
                    ensure_flag_after_local(step3_flag)
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
                                "-v", f"METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},VERBOSE={verbose_env},NODE_WORK_PATH={working_dir}", summary_context]
                        vprint("Submitting:", " ".join(cmd))
                        handle = submit_job(cmd, "pbs", env)
                        wait_for_flag_or_job_end(step3_flag, [handle])
                    elif shutil.which("sbatch"):
                        sbatch_opts = []
                        if partition_req:
                            sbatch_opts += ["--partition", partition_req]
                        if cpus_req:
                            sbatch_opts += [f"--cpus-per-task={cpus_req}"]
                        if mem_req:
                            sbatch_opts += [f"--mem={mem_req}"]
                        cmd = ["sbatch", *sbatch_opts,
                               f"--export=METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},VERBOSE={verbose_env},NODE_WORK_PATH={working_dir}", summary_context]
                        vprint("Submitting:", " ".join(cmd))
                        handle = submit_job(cmd, "slurm", env)
                        wait_for_flag_or_job_end(step3_flag, [handle])
            wait_for_flag_file(step3_flag)
            vprint("✔ Summary completed successfully!", file=sys.stdout)

        ##STEP 2: FILL MISSING METADATA
        if args.fillmetadata:
            if not os.path.isfile(step4_flag):
                if args.local:
                    subprocess.run(["bash", llm_metadata_inference,
                                    metappuccino_dir, res_dir, env_dir, model_path, verbose_env, str(sched_gpus), working_dir],
                                   check=True, env=env)
                    ensure_flag_after_local(step4_flag)
                else:
                    if args.per_gpu_jobs and (sched_gpus and sched_gpus > 1):
                        if shutil.which("qsub"):
                            handles = []
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
                                        "-v", f"METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},MODEL={model_path},VERBOSE={verbose_env},N_GPUS=1,SHARD_TOTAL={sched_gpus},SHARD_ID={i},NODE_WORK_PATH={working_dir}",
                                        llm_metadata_inference]
                                vprint("Submitting:", " ".join(cmd))
                                handles.append(submit_job(cmd, "pbs", env))
                            wait_for_flag_or_job_end(step4_flag, handles)
                        elif shutil.which("sbatch"):
                            handles = []
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
                                       f"--export=METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},MODEL={model_path},VERBOSE={verbose_env},N_GPUS=1,SHARD_TOTAL={sched_gpus},SHARD_ID={i},NODE_WORK_PATH={working_dir}",
                                       llm_metadata_inference]
                                vprint("Submitting:", " ".join(cmd))
                                handles.append(submit_job(cmd, "slurm", env))
                            wait_for_flag_or_job_end(step4_flag, handles)
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
                                    "-v", f"METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},MODEL={model_path},VERBOSE={verbose_env},N_GPUS={sched_gpus},NODE_WORK_PATH={working_dir}",
                                    llm_metadata_inference]
                            vprint("Submitting:", " ".join(cmd))
                            handle = submit_job(cmd, "pbs", env)
                            wait_for_flag_or_job_end(step4_flag, [handle])
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
                                   f"--export=METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},MODEL={model_path},VERBOSE={verbose_env},N_GPUS={sched_gpus},NODE_WORK_PATH={working_dir}",
                                   llm_metadata_inference]
                            vprint("Submitting:", " ".join(cmd))
                            handle = submit_job(cmd, "slurm", env)
                            wait_for_flag_or_job_end(step4_flag, [handle])
            wait_for_flag_file(step4_flag)
            vprint("✔ LLM inference completed successfully!", file=sys.stdout)

            if not os.path.isfile(step5_flag):
                if args.local:
                    subprocess.run(["bash", reload_model,
                                    metappuccino_dir, res_dir, env_dir, model_path, str(iteration_limit), verbose_env, str(sched_gpus),working_dir],
                                   check=True, env=env)
                    ensure_flag_after_local(step5_flag)
                else:
                    if args.per_gpu_jobs and (sched_gpus and sched_gpus > 1):
                        if shutil.which("qsub"):
                            handles = []
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
                                        "-v", f"METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},MODEL={model_path},ITERATION_LIMIT={iteration_limit},VERBOSE={verbose_env},N_GPUS=1,SHARD_TOTAL={sched_gpus},SHARD_ID={i},NODE_WORK_PATH={working_dir}",
                                        reload_model]
                                vprint("Submitting:", " ".join(cmd))
                                handles.append(submit_job(cmd, "pbs", env))
                            wait_for_flag_or_job_end(step5_flag, handles)
                        elif shutil.which("sbatch"):
                            handles = []
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
                                       f"--export=METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},MODEL={model_path},ITERATION_LIMIT={iteration_limit},VERBOSE={verbose_env},N_GPUS=1,SHARD_TOTAL={sched_gpus},SHARD_ID={i},NODE_WORK_PATH={working_dir}",
                                       reload_model]
                                vprint("Submitting:", " ".join(cmd))
                                handles.append(submit_job(cmd, "slurm", env))
                            wait_for_flag_or_job_end(step5_flag, handles)
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
                                    "-v", f"METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},MODEL={model_path},ITERATION_LIMIT={iteration_limit},VERBOSE={verbose_env},N_GPUS={sched_gpus},NODE_WORK_PATH={working_dir}",
                                    reload_model]
                            vprint("Submitting:", " ".join(cmd))
                            handle = submit_job(cmd, "pbs", env)
                            wait_for_flag_or_job_end(step5_flag, [handle])
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
                                   f"--export=METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},MODEL={model_path},ITERATION_LIMIT={iteration_limit},VERBOSE={verbose_env},N_GPUS={sched_gpus},NODE_WORK_PATH={working_dir}",
                                   reload_model]
                            vprint("Submitting:", " ".join(cmd))
                            handle = submit_job(cmd, "slurm", env)
                            wait_for_flag_or_job_end(step5_flag, [handle])
            wait_for_flag_file(step5_flag)
            vprint("✔ Context reloaded successfully!", file=sys.stdout)

        ##STEP 3: ASSOCIATE TERMS WITH CODE
        if args.associateinformation:
            if not os.path.isfile(step6_flag):
                if args.local:
                    subprocess.run(["bash", normalize_final,
                                    metappuccino_dir, res_dir, env_dir, verbose_env,working_dir],
                                   check=True, env=env)
                    ensure_flag_after_local(step6_flag)
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
                                f"METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},VERBOSE={verbose_env},NODE_WORK_PATH={working_dir}",
                                normalize_final]
                        vprint("Submitting:", " ".join(cmd))
                        handle = submit_job(cmd, "pbs", env)
                        wait_for_flag_or_job_end(step6_flag, [handle])
                    elif shutil.which("sbatch"):
                        sbatch_opts = []
                        if partition_req:
                            sbatch_opts += ["--partition", partition_req]
                        if cpus_req:
                            sbatch_opts += [f"--cpus-per-task={cpus_req}"]
                        if mem_req:
                            sbatch_opts += [f"--mem={mem_req}"]
                        cmd = ["sbatch", *sbatch_opts,
                               f"--export=METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},VERBOSE={verbose_env},NODE_WORK_PATH={working_dir}",
                               normalize_final]
                        vprint("Submitting:", " ".join(cmd))
                        handle = submit_job(cmd, "slurm", env)
                        wait_for_flag_or_job_end(step6_flag, [handle])
            wait_for_flag_file(step6_flag)
            vprint("✔ Code association and cleaning LLM answers successfully completed!", file=sys.stdout)

        if args.visualisation:
            if not os.path.isfile(step7_flag):
                if args.local:
                    subprocess.run(["bash", visualisation,
                                    metappuccino_dir, res_dir, env_dir, verbose_env,working_dir],
                                   check=True, env=env)
                    ensure_flag_after_local(step7_flag)
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
                                f"METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},VERBOSE={verbose_env},NODE_WORK_PATH={working_dir}",
                                visualisation]
                        vprint("Submitting:", " ".join(cmd))
                        handle = submit_job(cmd, "pbs", env)
                        wait_for_flag_or_job_end(step7_flag, [handle])
                    elif shutil.which("sbatch"):
                        sbatch_opts = []
                        if partition_req:
                            sbatch_opts += ["--partition", partition_req]
                        if cpus_req:
                            sbatch_opts += [f"--cpus-per-task={cpus_req}"]
                        if mem_req:
                            sbatch_opts += [f"--mem={mem_req}"]
                        cmd = ["sbatch", *sbatch_opts,
                               f"--export=METAPPUCCINO={metappuccino_dir},RES={res_dir},ENV_REQUIREMENT={env_dir},VERBOSE={verbose_env},NODE_WORK_PATH={working_dir}",
                               visualisation]
                        vprint("Submitting:", " ".join(cmd))
                        handle = submit_job(cmd, "slurm", env)
                        wait_for_flag_or_job_end(step7_flag, [handle])
            wait_for_flag_file(step7_flag)
            vprint("✔ Graphs build successfully!", file=sys.stdout)

        if not tmp_keep:
            tmp_dir = os.path.join(res_dir, "tmp")
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
