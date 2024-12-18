# MetaMap

Automates metadata extraction and completion based on LLMs.

# Requirements
Install in python environment:

    pip install -r requirements.txt

Use of [Llama.cpp](https://github.com/ggerganov/llama.cpp) to launch te used LLMs. The repository must be cloned, and its path specified in the arguments when launching MetaMap.

    git clone git@github.com:ggerganov/llama.cpp.git

The dependant models for the tool are:
- Llama 70B

Those models must be stored in the folder models.

# Installation
Clone the repository:

    git clone git@github.com:chumphati/MetaMap.git

# Documentation

# Quick start
#### Running MetaMap

    nohup python3 bin/MetaMap.py > results/logsMetaMap.log 2>&1 &

#### Main argument input