##########################################################################################
# IMPORT
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.utils.prune as prune
import pickle
import os
import argparse

##########################################################################################
# PARAMETERS

parser = argparse.ArgumentParser(description="Process metadata with LLM")
parser.add_argument("--base_path", type=str, required=True, help="Base path to MetaMap")
args = parser.parse_args()
base_path = args.base_path

model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
device = "cuda" if torch.cuda.is_available() else "cpu"
activation_threshold = 1e-3
pruning_amount_attention = 0.15 #15% weak weights (Q/K/V)
pruning_amount_mlp = 0.20 #20% weak weights (MLP)
activation_save_path = os.path.join(base_path, "activations.pkl")
model_save_path = os.path.join(base_path, "llama-3-pruned")
prompt = "Ton prompt ici"

##########################################################################################
# ORIGINAL MODEL

# Charger depuis un cache centralisé sans re-télécharger chaque fois
cache_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/PRUNING_MODEL/hf_cache"
os.makedirs(cache_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", cache_dir=cache_dir)
model.eval()

##########################################################################################
# FOLLOW ACTIVATIONS
activation_dict = {}


def save_activations(layer_name):
    def hook(module, input, output):
        activation_dict[layer_name] = output.detach().cpu()
    return hook


for i, layer in enumerate(model.model.layers):
    layer.self_attn.q_proj.register_forward_hook(save_activations(f"layer_{i}_q_proj"))
    layer.self_attn.k_proj.register_forward_hook(save_activations(f"layer_{i}_k_proj"))
    layer.self_attn.v_proj.register_forward_hook(save_activations(f"layer_{i}_v_proj"))
    layer.mlp.gate_proj.register_forward_hook(save_activations(f"layer_{i}_fc_gate"))
    layer.mlp.up_proj.register_forward_hook(save_activations(f"layer_{i}_fc_up"))
    layer.mlp.down_proj.register_forward_hook(save_activations(f"layer_{i}_fc_down"))

with torch.no_grad():
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model(**inputs)

#save
with open(activation_save_path, "wb") as file:
    pickle.dump(activation_dict, file)

##########################################################################################
# IDENTIFY RARE MODULES FOR SPECIFIC PROMPT

neurons_to_prune = {}
for layer_name, activation in activation_dict.items():
    mean_activation = torch.mean(torch.abs(activation)).item()
    if mean_activation < activation_threshold:
        neurons_to_prune[layer_name] = mean_activation
        print(f"{layer_name} marqué pour pruning (activation faible : {mean_activation:.6f})")

##########################################################################################
# COMBINE PRUNING

model_cpu = model.to("cpu")

for i, layer in enumerate(model.model.layers):

    # Magnitude Pruning Attention (Q/K/V)
    for proj in ["q_proj", "k_proj", "v_proj"]:
        proj_layer = getattr(layer.self_attn, proj)
        prune.l1_unstructured(proj_layer, 'weight', amount=pruning_amount_attention)
        prune.remove(proj_layer, 'weight')

    # Magnitude Pruning MLP
    for fc in ["gate_proj", "down_proj", "up_proj"]:
        fc_layer = getattr(layer.mlp, fc)
        prune.l1_unstructured(fc_layer, 'weight', amount=pruning_amount_mlp)
        prune.remove(fc_layer, 'weight')

    # Pruning non activated modules
    for module_name in ["q_proj", "k_proj", "v_proj", "gate_proj", "down_proj", "up_proj"]:
        key = f"layer_{i}_{module_name}"
        if key in neurons_to_prune:
            target_layer = getattr(layer.self_attn if "proj" in module_name else layer.mlp, module_name)
            prune.l1_unstructured(target_layer, 'weight', amount=0.9)
            prune.remove(target_layer, 'weight')



#save final model
model.save_pretrained(model_save_path)
