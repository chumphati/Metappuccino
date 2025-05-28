# MetaMap

MetaMap automates the extraction, reconstruction, and enrichment of metadata from public RNA-seq datasets using instruction-tuned large language models (LLMs). It combines sample-level and study-level inference, entropy-based filtering, and ontology mapping to produce high-quality structured metadata.
<br>
It is compatible with the HPC schedulers SLURM and PBS.

## Requirements
### System Requirements
- CPU: At least 8 cores
- GPU: 1× or multiple (≥50GB VRAM)
- CUDA: Version ≥11.7 recommended

### Python packages

    pip install -r requirements.txt

[Llama.cpp](https://github.com/ggerganov/llama.cpp) is used for LLM inference. The downloading of llama-cpp-python python wrapper is included in the requirement file.

### CUDA support for llama-cpp-python

    echo "Installing llama-cpp-python with CUDA support." >> "/scratchlocal/$USER/$PBS_JOBID/llm_log_SB.txt"
    export CMAKE_ARGS="-DGGML_CUDA=on -DCUDA_PATH=$PATH_CUDA -DCUDAToolkit_ROOT=$PATH_CUDA -DCUDAToolkit_INCLUDE_DIR=$PATH_CUDA/include -DCUDAToolkit_LIBRARY_DIR=$PATH_CUDA/lib64"
    export CUDACXX=$PATH_CUDA/bin/nvcc
    python3 -m pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

## Installation
1. Clone the repository:


    git clone git@github.com:chumphati/MetaMap.git

2. Download the MetaMap fine-tuned LLM from HuggingFace:


    huggingface-cli download your-username/metamap-mistral7B-ft --local-dir ./models/MetaMap_Mistral7B_FT

## Quick Start
### Local Execution

    nohup python3 bin/MetaMap.py \
      --metamap_dir ./MetaMap \
      --res_dir results_dir \
      --env_requirement ./venv \
      --mode local
      --model ./models/MetaMap_LLM/llama-model.gguf \
      --getmetadata --fillmetadata --associateinformation --completestudy \
      > results/logsMetaMap.log 2>&1 &


Warning: At least one GPU with enough RAM must be available.

### HPC Cluster Execution
1. PBS


    qsub PATH_TO_CLONED_REPO/MetaMap/bin/MetaMap.sh


2. SLURM


    sbatch PATH_TO_CLONED_REPO/MetaMap/bin/MetaMap.sh


## Arguments
Run:

    python3 PATH_TO_CLONED_REPO/MetaMap/bin/MetaMap.py --help

| Argument               | Description                                                        |
|------------------------|--------------------------------------------------------------------|
| `--metamap_dir`        | Path to the MetaMap base directory                                 |
| `--res_dir`            | Directory for storing output results                               |
| `--env_requirement`    | Path to the Python virtual environment with installed requirements |
| `--model`              | Path to the LLM model file or directory                            |
| `--requirements`       | Install Python requirements and CUDA support                      |
| `--cuda`               | Path to CUDA installation (default: `/usr/local/cuda`)             |
| `--getmetadata`        | Download and clean metadata from SRA                               |
| `--fillmetadata`       | Predict missing metadata fields with LLMs                          |
| `--associateinformation` | Match LLM predictions to ontology terms                         |
| `--completestudy`      | Complete metadata using study-level context                        |


## Output Structure

- **results_dir/**
  - `logs/` &mdash; Logs of each processing step  
  - `tmp/` &mdash; Temporary files and flags  
  - `METADATA/` &mdash; Per-run metadata outputs  
  - `SPECIFIC_RUN_ANALYSIS/` &mdash; Final LLM inference  
    - `INFO_BIO_LLM/` &mdash; Sample inference  
    - `INFO_STUDY_LLM/` &mdash; Study inference  
    - `final_llm_sample_analysis.csv` &mdash; Final complete prediction  


## License

Apache License2.0