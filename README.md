# Metappuccino

Metappuccino is a tool that **completes and normalizes SRA metadata** thanks to a large language model. From a simple list of run accessions, it processes from the download/cleanup to the completion, normalization, and visualization of some type of information. It can run **locally** or submit work to **PBS** or **Slurm**s.

---

## Table of contents

* [Features](#features)
* [Installation](#installation)
  * [Download the LLM Model](#download-the-LLM-model)
  * [Metappuccino installation](#metappuccino-installation)
  * [GPU setup](#gpu-setup)
  * [Additional setup](#additional-setup)
* [Minimal usage](#minimal-usage)
* [Scheduler submission](#scheduler-submission-no-script-file)

  * [PBS (qsub)](#pbs-qsub)
  * [Slurm (sbatch)](#slurm-sbatch)
* [Arguments](#arguments)
* [Inputs & outputs](#inputs--outputs)
* [Typical pipeline steps](#typical-pipeline-steps)
* [Tips & troubleshooting](#tips--troubleshooting)

---

## Features

* Pulls and cleans NCBI/ENA metadata for a list of run accessions.
* LLM-based inference for the following information: .
* Normalizes cell line, organs, disease with Cellosaurus, Disease Ontology and Uberon.
* Produces final tables with the metadata + visual summaries.
* Runs **locally** or via **PBS/Slurm** (auto-detect), with fine-grained step flags.
* Supports **per-GPU sharding** and multi-GPU inference.
* Works with HF models and/or GGUF backends (e.g., llama.cpp). Default model = specific model fine-tuned for the task.

---

## Installation

To install Metappuccino, three steps have to be completed: fetching the LLM model of your choice for inference, installing the tool and its dependencies, and configuring the GPU if you intend to use it.

### Download the LLM Model

#### MetappuccinoLLModel: Metappuccino's LLM
A specific model based on Mistral 7B has been fine-tuned to achieve better performance than open source models (by Sept. 2025). 
It can be used in Metappuccino by downloading it and providing the path to it in the settings when launching (`--model`).

```bash

```

#### Open-source GGUF model
If you don't want to use MetappuccinoLLModel, you can choose to upload a public model such as GPT, Llama or Deepseek, or your own model. To do so, it must be in GGUF format and will be launched via the llama-cpp backend. 
>Warning: This mode requires you to put the `--gguf` flag in Metappuccino, and requires a specific GPU configuration (see the [Additional setup](#additional-setup) section).

*Example of downloading a gguf model from Hugging Face:*
```python
from huggingface_hub import snapshot_download

HF_TOKEN = <HF_TOKEN> #key you have to create on hugging face

snapshot_download(
    repo_id="<REPO_URL>",
    local_dir="<OUT_DIR_URL>",
    use_auth_token=HF_TOKEN,
    resume_download=True,
    max_workers=4
)

```

### Metappuccino installation
*Please note: it is safer to install the tool from the release.*

#### Install from wheel (*RECOMMENDED*)
*pip* must be available on your machine.

```bash
wget https://github.com/chumphati/Metappuccino/releases/download/<VERSION>/metappuccino-VERSION-py3-none-any.whl
python3 -m venv .venv && source .venv/bin/activate #create a python virtual environment
pip install metappuccino-<VERSION>-py3-none-any.whl
metappuccino --help
```

>Warning: if the model used is different from MetappuccinoLLModel, an additional installation in the same python environment will be required (see section [Additional setup](#additional-setup)).

#### Install from source
```bash
#SSH
git clone git@github.com:chumphati/Metappuccino.git
#HTTPS
git clone https://github.com/chumphati/Metappuccino.git
cd metappuccino
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

> **Python**: 3.10+ recommended (3.9 works only if your packaged wheel declares it).

### GPU setup

A wheel **does not** install GPU drivers or CUDA. To enable GPU natively:

1. **Host prerequisites**

   * NVIDIA **driver** installed.
   * (Optional) CUDA Toolkit if you need to build GPU backends.

2. **Install a CUDA-matching PyTorch build** (example: CUDA 12.1):

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch==<version+cu121>" "torchvision==<version+cu121>"
```

3. Metappuccino's model requires at least **40 GB of VRAM** to be fully loaded on GPU.

### Additional setup

**Optional: In case of use of GGUF external model**

llama-cpp-python installation with CUDA (on the node where cuda is already installed):
```bash
export CMAKE_ARGS="-DGGML_CUDA=on -DCUDA_PATH=<CUDA_PATH> \
  -DCUDAToolkit_ROOT=<CUDA_PATH> \
  -DCUDAToolkit_INCLUDE_DIR=<CUDA_PATH>/include \
  -DCUDAToolkit_LIBRARY_DIR=<CUDA_PATH>/lib64"
export CUDACXX=<CUDA_PATH>/bin/nvcc
pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
```

or directly from a precompiled CUDA wheel
```bash
#if CUDA 12.1
pip install --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 llama-cpp-python
#Warning: this build will unable CPU inference mode
```

This must be manually done. llama-cpp is not installed in Metappuccino's wheel.

>If you don't have a GPU available and you want to use an external model, you can only run the last line of the first block without exports and the model will be automatically loaded on CPU. Please note, however, that this will increase the calculation time considerably.

---

## Minimal usage: local run

>Metappuccino required a HTTP/HTTPS internet connexion to fetch the data.

### If wheel installation

```bash
metappuccino \
  --res_dir "/abs/path/results" \
  --sample_input "/abs/path/runs.txt" \
  --env_requirement "/abs/path/.venv" \
  --model "/abs/path/MetappuccinoLLModel" \
  --iteration_limit 1 --gpus 0 --cpus 4 --mem "8gb" \
  --getmetadata --fillmetadata --associateinformation --visualisation \
  --local --verbose
```

### If source installation

```bash
python3 path_to_Metappuccino/bin/Metappuccino.py \
  --metappuccino_dir "path_to_Metappuccino" \
  --res_dir "/abs/path/results" \
  --sample_input "/abs/path/runs.txt" \
  --env_requirement "/abs/path/.venv" \
  --model "/abs/path/MetappuccinoLLModel" \
  --iteration_limit 1 --gpus 0 --cpus 4 --mem "8gb" \
  --getmetadata --fillmetadata --associateinformation --visualisation \
  --local --verbose
```

> `--metappuccino_dir` must point to the repository root that contains `bin/Metappuccino.py`.

---

## Scheduler submission

### PBS (qsub)

*e.g.:*

```bash
qsub -N metappuccino -q <queue> \
  -l select=1:ncpus=1:mem=8gb -l walltime=100:00:00 <<'SH'
#!/bin/bash
source <ENV_PATH>/bin/activate
metappuccino \
  --res_dir "/abs/path/results" \
  --sample_input "/abs/path/runs.txt" \
  --env_requirement "/abs/path/Metappuccino/.venv" \
  --working_dir "/scratchlocal/$USER" \
  [ --metappuccino_dir "/abs/path/Metappuccino" \ ]
  --model "/abs/path/model.gguf" \
  --partition "<queue>" \
  --gpus 1 --cpus 20 --mem "50gb" --per_gpu_jobs \
  --iteration_limit 3 \
  --getmetadata --fillmetadata --associateinformation --visualisation \
  --verbose
deactivate
SH
```

### Slurm (sbatch)

*e.g.:*

```bash
sbatch -J metappuccino -p <partition> --time=100:00:00 \
  --nodes=1 --cpus-per-task=1 --mem=8G \
  --wrap 'bash -lc "
    source <ENV_PATH>/bin/activate
    metappuccino \
      --res_dir "/abs/path/results" \
      --sample_input "/abs/path/runs.txt" \
      --env_requirement "/abs/path/Metappuccino/.venv" \
      --working_dir "/scratchlocal/$USER" \
      [ --metappuccino_dir "/abs/path/Metappuccino" \ ]
      --model "/abs/path/model.gguf" \
      --partition "<queue>" \
      --gpus 1 --cpus 20 --mem "50gb" --per_gpu_jobs \
      --iteration_limit 3 \
      --getmetadata --fillmetadata --associateinformation --visualisation \
      --verbose
    deactivate"
  '
```

> Omit `--nodelist/--node` to let the scheduler pick a node automatically.
> Add path Metappuccino directory with `--metappuccino_dir` if it has been installed from source.

---

## Arguments

| Argument                 | Type / Default                                                                    | Description                                                                                                                                                                                                    |
|--------------------------|-----------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--sample_input`         | str, **required**                                                                 | `.txt`/`.csv` with one **run accession number** per line.                                                                                                                                                      |
| `--res_dir`              | str, **required**                                                                 | Output/results directory.                                                                                                                                                                                      |
| `--env_requirement`      | str, **required**                                                                 | Path to Python **venv** (activated inside jobs).                                                                                                                                                               |
| `--model [--gguf]`       | str, **required**                                                                 | Path to model (e.g., `.../model.gguf` or HF model path used by your scripts). If gguf model used, add `--gguf`                                                                                                 |
| `--getmetadata`          | flag, **required, see [Typical pipeline steps](#typical-pipeline-steps) section** | Run the metadata preprocessing: download, clean, and summarize if needed.                                                                                                                                      |
| `--fillmetadata`         | flag, **required, see [Typical pipeline steps](#typical-pipeline-steps) section** | Run LLM inference based on `--model` for missing metadata.                                                                                                                                                     |
| `--associateinformation` | flag, **required, see [Typical pipeline steps](#typical-pipeline-steps) section** | Normalize terms with Cellosaurus, Disease Ontology and Uberon and clean outputs.                                                                                                                               |
| `--visualisation`        | flag, **required, see [Typical pipeline steps](#typical-pipeline-steps) section** | Build figures from curated metadata (model confidence, summaries, etc).                                                                                                                                        |
| `--partition`            | str (**required if scheduler submission**)                                        | Queue/partition **if a scheduler is used** (PBS: queue via `-q`, Slurm: `--partition`).                                                                                                                        |
| `--working_dir`          | str (**required if scheduler submission**)                                        | Scratch dir on the compute node (fast local disk preferred).                                                                                                                                                   |
| `--node`                 | str, default `""` (**required if scheduler submission**)                          | Node name to send the jobs on **if a scheduler is used**. Must have a GPU available on it if GPU mode active.                                                                                                  |
| `--metappuccino_dir`     | str (**required if installation from source**)                                    | Absolute path to the Metappuccino repository in case of source installation (must contain `bin/Metappuccino.py`).                                                                                              |
| `--gpus`                 | int, default `1`, min `0`                                                         | Number of GPUs to use/request. With `--per_gpu_jobs`, spawns per-GPU shards. If 0, LLM inference will be conducted on CPUS.                                                                                    |
| `--cpus`                 | int, default `30`, min `8`                                                        | Number of CPUs to request.                                                                                                                                                                                     |
| `--mem`                  | str, default `"50gb"`                                                             | Memory request (e.g., `50gb`, `80G`).                                                                                                                                                                          |
| `--per_gpu_jobs`         | flag                                                                              | Submit one job per GPU. Evenly split the inference list across the number of GPUs specified by --gpus to run in parallel. **Warning: if the model doesn’t fit on a single GPU, disable this parallelization.** |
| `--iteration_limit`      | int, default `0`                                                                  | Max restarts of LLM inference if malformed JSON or <30% categories predicted.                                                                                                                                  |
| `--logan_path`           | str, default `""`                                                                 | Additional metadata extracted from Logan search (`ID` column expected).                                                                                                                                        |
| `--tmp_keep`             | flag                                                                              | Keep temporary files.                                                                                                                                                                                          |
| `--local`                | flag                                                                              | Force local execution (no scheduler).                                                                                                                                                                          |
| `--verbose`              | flag                                                                              | Verbose logging.                                                                                                                                                                                               |

---

## Inputs & outputs

**Input (`--sample_input`)**

* `.txt` or `.csv` file with one run accession per line (or a headered `run_accession` column).

**Outputs (under `--res_dir`)**

* `logs/` — all logs.
* `ORIGINAL_METADATA/` — fetched metadata.
* `COMPLETED_INFERENCE/` — per-run LLM JSONs.
* `tmp/` — working files if `--tmp_keep`.
* Final normalized tables/visualizations.

---

## Typical pipeline steps

1. **Get & clean metadata** (`--getmetadata`)
   Initialize, download raw metadata, clean, build run-level summaries.

2. **LLM inference** (`--fillmetadata`)
   Complete fields (organ, disease, treatment, …).

3. **Normalize & code** (`--associateinformation`)
   Map free text to controlled terms/codes; resolve inconsistencies.

4. **Visualization** (`--visualisation`)
   Produce graphs/figures from curated metadata and model confidence.

> Provide all four flags to run the full pipeline in one go, or run step-by-step.
> If the analyses are interrupted, simply restart the same command and the analysis will resume from the cut. 
> It is possible to run only specific steps, or to execute them one by one. However, running an intermediate step requires ensuring that the corresponding intermediate files and the latest output are placed in the appropriate folders, as they would be during the normal execution of the pipeline. So, this option is mainly limited to resuming an analysis or stopping it before completion.

---

## Tips & troubleshooting

* **No scheduler available?** Add `--local`. Be sure to have CPU and GPU requirements available.
* **Jobs stick in queue?** Remove `--node`, confirm `--partition`/queue, and reduce `--gpus`.
* **GPU not used?** Verify host NVIDIA driver; ensure CUDA-matching Torch wheels and CUDA-compiled backends (e.g., `llama-cpp-python` with `-DGGML_CUDA=on`).
* **Shared FS**: `--res_dir` and `--env_requirement` should be visible from compute nodes.
* **Scratch**: use fast local disks for `--working_dir` (e.g., `/scratchlocal/$USER` or `/tmp/$USER`).
* **Absolute paths**: always pass absolute paths for cluster jobs.
* **Logs**: check `--res_dir/logs` and per-step logs (LLM: `llm_inference.out/err`, `llm_log_SB*.txt`).

---

**Happy brewing ☕ — Metappuccino is ready to serve your metadata!**
