import torch
from transformers import AutoModelForCausalLM


def compare_models(original_path, pruned_path):
    model_original = AutoModelForCausalLM.from_pretrained(original_path)
    model_pruned = AutoModelForCausalLM.from_pretrained(pruned_path)

    for (name_orig, param_orig), (name_pruned, param_pruned) in zip(model_original.named_parameters(), model_pruned.named_parameters()):
        if name_orig != name_pruned:
            print(f"Name mismatch parameters: {name_orig} vs {name_pruned}")
            continue

        zeros_original = torch.sum(param_orig == 0).item()
        zeros_pruned = torch.sum(param_pruned == 0).item()
        total_params = param_orig.numel()

        print(f"Layer: {name_orig}")
        print(f"Nb 0 in original: {zeros_original} ({zeros_original/total_params*100:.2f}%)")
        print(f"Nb 0 in pruned: {zeros_pruned} ({zeros_pruned/total_params*100:.2f}%)")
        if zeros_pruned > zeros_original:
            print("Pruning success.\n")
        else:
            print("No change with pruning.\n")


original_model_path = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
pruned_model_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/PRUNING_MODEL/llama-3-pruned"

compare_models(original_model_path, pruned_model_path)
