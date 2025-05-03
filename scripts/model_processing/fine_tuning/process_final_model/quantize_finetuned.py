import subprocess
import os

#QUANTIFY MODEL IN 4 BITS GGUF FOR LLAMA.CPP

pruned_model_path = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/COMPLETE_FT_15K_FIRST/mistral7B_full_finetuned'
quantized_model_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/models/Mistral-7B-Instruct-v0.3-FT-15k-alldata.gguf'
# quantize4bits_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/models/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf'

llama_cpp_conversion_script = ('/store/EQUIPES/SSFA/MEMBERS/fiona.hak/llama.cpp/convert_hf_to_gguf.py')

conversion_command = [
    'python', llama_cpp_conversion_script,
    pruned_model_path,
    '--outfile', quantized_model_file,
    '--outtype', 'auto'
]

# quantize4bits = [
#     '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/llama.cpp/build/bin/llama-quantize',
#     quantized_model_file,
#     quantize4bits_file,
#     'Q4_K_M'
# ]


subprocess.run(conversion_command, check=True)
# subprocess.run(quantize4bits, check=True)
