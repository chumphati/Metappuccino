import torch
from transformers import AutoModelForCausalLM

model_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/PRUNING_MODEL/llama-3-pruned"
model = AutoModelForCausalLM.from_pretrained(model_path)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Nb param pruned model : {total_params:,}")
print(f"Trainable params : {trainable_params:,}")
