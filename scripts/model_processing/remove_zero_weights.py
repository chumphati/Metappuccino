import torch
from transformers import AutoModelForCausalLM

pruned_model_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/PRUNING_MODEL/llama-3-pruned"
compact_model_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/PRUNING_MODEL/llama-3-pruned-compact"

model = AutoModelForCausalLM.from_pretrained(pruned_model_path, torch_dtype=torch.float16)

new_layers = {}
with torch.no_grad():
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and hasattr(module, "weight"):
            weight = module.weight

            if torch.all(weight == 0):
                continue

            nonzero_rows = (weight.abs().sum(dim=1) != 0).nonzero(as_tuple=True)[0]
            nonzero_cols = (weight.abs().sum(dim=0) != 0).nonzero(as_tuple=True)[0]

            if nonzero_rows.numel() > 0 and nonzero_cols.numel() > 0:
                new_weight = weight[nonzero_rows][:, nonzero_cols].clone()
                in_features = nonzero_cols.numel()
                out_features = nonzero_rows.numel()

                if in_features > 0 and out_features > 0:
                    new_layer = torch.nn.Linear(in_features, out_features, bias=False, dtype=weight.dtype)
                    new_layer.weight = torch.nn.Parameter(new_weight)
                    new_layers[name] = new_layer


def replace_module(model, module_name, new_module):
    parts = module_name.split('.')
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


for name, new_layer in new_layers.items():
    replace_module(model, name, new_layer)

model.save_pretrained(compact_model_path)
total_params = sum(p.numel() for p in model.parameters())
print(f"Nb params pruned: {total_params:,}")