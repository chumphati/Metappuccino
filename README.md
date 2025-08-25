# Metappuccino

Metappuccino is a tool that **completes and normalizes SRA metadata** thanks to a large language model. From a simple list of run accessions, it processes from the download/cleanup to the completion, normalization, and visualization of some type of information. It can run **locally** or submit work to **PBS** or **Slurm**s.

---

## Table of contents

* [Features](#features)
* [Installation](#installation)
  * [Download the LLM Model](#download-the-LLM-model)
  * [Clone the repository](#clone-the-repository)
  * [Install from wheel](#install-from-wheel)
  * [Run with Docker / GHCR](#run-with-docker--ghcr)
  * [GPU setup without Docker](#gpu-setup-without-docker)
* [Minimal usage](#minimal-usage)
* [Scheduler submission (no script file)](#scheduler-submission-no-script-file)

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

### Download the LLM Model

### Clone the repository
```bash
#SSH
git clone git@github.com:chumphati/Metappuccino.git
#HTTPS
https://github.com/chumphati/Metappuccino.git

cd metappuccino
```

### Install from wheel

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install metappuccino-<VERSION>-py3-none-any.whl
metappuccino --help
```

> **Python**: 3.10+ recommended (3.9 works only if your packaged wheel declares it).

### Run with Docker / GHCR

**Docker**

```bash
```

**Apptainer/Singularity:**

```bash

```

### GPU setup without Docker

A wheel **does not** install GPU drivers or CUDA. To enable GPU natively:

1. **Host prerequisites**

   * NVIDIA **driver** installed.
   * (Optional) CUDA Toolkit if you need to build GPU backends.

2. **Install a CUDA-matching PyTorch build** (example: CUDA 12.1):

```bash
pip install --index-url https://download.pytorch.org/whl/cu121 \
  "torch==<version+cu121>" "torchvision==<version+cu121>"
```

3. **llama-cpp-python** with CUDA (if you use it):

```bash
export CMAKE_ARGS="-DGGML_CUDA=on -DCUDA_PATH=/usr/local/cuda \
  -DCUDAToolkit_ROOT=/usr/local/cuda \
  -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda/include \
  -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda/lib64"
export CUDACXX=/usr/local/cuda/bin/nvcc
pip install --upgrade --force-reinstall --no-cache-dir llama-cpp-python
```
Caution: replace the CUDA paths with the ones of your machine.

---

## Minimal usage

Run locally (single machine):

```bash
metappuccino \
  --metappuccino_dir "/abs/path/Metappuccino" \
  --res_dir "/abs/path/results" \
  --sample_input "/abs/path/runs.txt" \
  --env_requirement "/abs/path/.venv" \
  --working_dir "/abs/path/work" \
  --model "/abs/path/model.gguf" \
  --iteration_limit 1 --gpus 0 --cpus 4 --mem "8gb" \
  --getmetadata --fillmetadata --associateinformation --visualisation \
  --local --verbose
```

> `--metappuccino_dir` must point to the repository root that contains `bin/Metappuccino.py`.

---

## Scheduler submission

### PBS (qsub)

Heredoc one-liner (no extra files):

```bash
qsub -N metappuccino -q <queue> \
  -l select=1:ncpus=20:mem=50gb:ngpus=1 -l walltime=10000:00:00 <<'SH'
#!/bin/bash
source /abs/path/Metappuccino/.venv/bin/activate
cd /abs/path/Metappuccino
metappuccino \
  --metappuccino_dir "/abs/path/Metappuccino" \
  --res_dir "/abs/path/results" \
  --sample_input "/abs/path/runs.txt" \
  --env_requirement "/abs/path/Metappuccino/.venv" \
  --working_dir "/scratchlocal/$USER" \
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

`--wrap` one-liner:

```bash
sbatch -J metappuccino -p <partition> --time=10000:00:00 \
  --nodes=1 --cpus-per-task=20 --mem=50G --gres=gpu:1 \
  --wrap 'bash -lc "
    source /abs/path/Metappuccino/.venv/bin/activate
    cd /abs/path/Metappuccino
    metappuccino \
      --metappuccino_dir \"/abs/path/Metappuccino\" \
      --res_dir \"/abs/path/results\" \
      --sample_input \"/abs/path/runs.txt\" \
      --env_requirement \"/abs/path/Metappuccino/.venv\" \
      --working_dir \"/scratchlocal/$USER\" \
      --model \"/abs/path/model.gguf\" \
      --partition <partition> \
      --gpus 1 --cpus 20 --mem 50gb --per_gpu_jobs \
      --iteration_limit 3 \
      --getmetadata --fillmetadata --associateinformation --visualisation \
      --verbose
    deactivate"
  '
```

> Omit `--nodelist/--node` to let the scheduler pick a node automatically.

---

## Arguments

| Argument                 | Type / Default                   | Description                                                                   |
| ------------------------ | -------------------------------- | ----------------------------------------------------------------------------- |
| `--metappuccino_dir`     | str, **required**                | Absolute path to the Metappuccino repo (must contain `bin/Metappuccino.py`).  |
| `--sample_input`         | str, **required**                | `.txt`/`.csv` with one **run accession** per line (or headered column).       |
| `--res_dir`              | str, **required**                | Output/results directory (shared FS recommended on clusters).                 |
| `--env_requirement`      | str, **required**                | Path to Python **venv** with runtime deps (activated inside jobs).            |
| `--working_dir`          | str, **required**                | Scratch dir on the compute node (fast local disk preferred).                  |
| `--model`                | str, **required**                | Path to model (e.g., `.../model.gguf` or HF model path used by your scripts). |
| `--partition`            | str                              | Queue/partition (PBS: queue via `-q`, Slurm: `--partition`).                  |
| `--node`                 | str, default `""`                | Specific node name (optional).                                                |
| `--gpus`                 | int, default `1`                 | GPUs to use/request. With `--per_gpu_jobs`, spawns per-GPU shards.            |
| `--cpus`                 | int, default `30`                | CPUs to request.                                                              |
| `--mem`                  | str, default `"50gb"`            | Memory request (e.g., `50gb`, `80G`).                                         |
| `--per_gpu_jobs`         | flag                             | Submit one job per GPU (sharded inference).                                   |
| `--iteration_limit`      | int, default `1`                 | Max restarts if malformed JSON or <30% categories predicted.                  |
| `--logan_path`           | str, default `""`                | Optional auxiliary info file (`sample_acc` column expected).                  |
| `--cuda`                 | str, default `"/usr/local/cuda"` | CUDA path for building CUDA backends if needed.                               |
| `--requirements`         | flag                             | Install requirements/CUDA on node (requires proper privileges).               |
| `--getmetadata`          | flag                             | Run init + download + clean + summarization.                                  |
| `--fillmetadata`         | flag                             | Run LLM inference for missing metadata.                                       |
| `--associateinformation` | flag                             | Map terms to codes and clean outputs.                                         |
| `--visualisation`        | flag                             | Build figures from curated metadata.                                          |
| `--tmp_keep`             | flag                             | Keep temporary files.                                                         |
| `--local`                | flag                             | Force local execution (no scheduler).                                         |
| `--verbose`              | flag                             | Verbose logging.                                                              |

---

## Inputs & outputs

**Input (`--sample_input`)**

* `.txt` or `.csv` file with one run accession per line (or a headered `run_accession` column).

**Outputs (under `--res_dir`)**

* `logs/` — all logs.
* `ORIGINAL_METADATA/` — fetched metadata.
* `COMPLETED_INFERENCE/` — per-run LLM JSONs.
* `tmp/` — working files, flags (`STEP*.flag`), curated DB, shards, etc.
* Final normalized CSVs/visualizations in step-specific subfolders.

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
