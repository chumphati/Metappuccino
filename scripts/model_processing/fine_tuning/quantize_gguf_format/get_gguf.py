import subprocess
import os

#GET GGUF FILE

pruned_model_path = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/Qwen3-30B-A3B-Instruct-2507'
quantized_model_file = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/Metappuccino/models/Qwen3-30B-A3B-Instruct-2507.gguf'

llama_cpp_conversion_script = ('/store/EQUIPES/SSFA/MEMBERS/fiona.hak/llama.cpp/convert_hf_to_gguf.py')

conversion_command = [
    'python', llama_cpp_conversion_script,
    pruned_model_path,
    '--outfile', quantized_model_file,
    '--outtype', 'auto'
]

subprocess.run(conversion_command, check=True)
