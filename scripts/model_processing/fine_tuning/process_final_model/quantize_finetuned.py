import subprocess
import os

#QUANTIFY MODEL IN 4 BITS GGUF FOR LLAMA.CPP

quantized_model_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/gguf/Llama-3.1-70B-Instruct-original.gguf'
quantize4bits_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/4bits_quantified/Llama-3.1-70B-Instruct-Q8_0.gguf'

llama_cpp_conversion_script = ('/store/EQUIPES/SSFA/MEMBERS/fiona.hak/llama.cpp/convert_hf_to_gguf.py')

quantize4bits = [
    '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/llama.cpp/build/bin/llama-quantize',
    quantized_model_file,
    quantize4bits_file,
    'Q8_0'
]

subprocess.run(quantize4bits, check=True)
