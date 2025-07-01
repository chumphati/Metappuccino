import subprocess
import os

#QUANTIFY MODEL IN 4 BITS GGUF FOR LLAMA.CPP

quantized_model_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/4bits_quantified/Mistral7B-Instruct-ft-v1500tt.gguf'
quantize4bits_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/4bits_quantified/Mistral7B-Instruct-ft-v1500tt-Q4_K_M.gguf'

llama_cpp_conversion_script = ('/store/EQUIPES/SSFA/MEMBERS/fiona.hak/llama.cpp/convert_hf_to_gguf.py')

quantize4bits = [
    '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/llama.cpp/build/bin/llama-quantize',
    quantized_model_file,
    quantize4bits_file,
    'Q4_K_M'
]

subprocess.run(quantize4bits, check=True)
