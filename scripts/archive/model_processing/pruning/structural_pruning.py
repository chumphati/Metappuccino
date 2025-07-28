##########################################################################################
#IMPORT
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import pickle
import os
import argparse
import logging
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

##########################################################################################
#PARAMETERS
parser = argparse.ArgumentParser(description="Process metadata with LLM")
parser.add_argument("--base_path", type=str, required=True, help="Base path to MetaMap")
args = parser.parse_args()
base_path = args.base_path

model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
device = "cuda" if torch.cuda.is_available() else "cpu"
activation_save_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/store/first_pruned_results/PRUNING_MODEL/activations_gpu.pkl"
model_save_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/PRUNING_MODEL/llama-3-pruned-struct"
prompt = ("Run accession: [number]. Metadata to analyze: [information]. For each row in the metadata line "
          "(the first line contains the column names), extract and format the following information concisely. "
          "For each missing category, provide a single answer without redundancy. Each category **MUST** have one distinct "
          "and explicit answer, even if inferred. **Do not leave any category empty.** Do not repeat information already provided "
          "in previous categories. Remove redundant text. Tissue type – The tissue type from which the sample originates (e.g., liver, lung, brain). "
          "If not specified, deduce from context in the two last columns. Cell line – Specify the cell line, or state 'Primary tissue' if the sample is "
          "from a primary tissue and not a cell line. Cell type – The type of cell in the sample (e.g., neuron, fibroblast, CD8 T cell, CD4 T cell, monocyte NK cell, "
          "mast cell, melanocyte, dendritic cell, etc...). If not provided, deduce based on the tissue type and state the inference. Use the Cell Ontology terms "
          "terminology. UBERON organ and code – Provide me the organ concerned by this study, in the UBERON GTEX terminology for the tissue type (e.g., "
          "UBERON:000XXXX + name of the organ). If not specified, deduce from context, or search one related to the tissue. Disease Ontology Term – Return the "
          "Disease Ontology term corresponding to the disease associated with the sample in the format DOID:XXXXX + Disease Name. If the sample is explicitly "
          "described as 'normal' or 'healthy', do not infer any disease. In this case, do not search for disease-related information in the context. If the sample "
          "is not explicitly labeled as 'normal' or 'healthy' or 'no disease', infer the disease from the context only if it is directly related to the sample "
          "(e.g., sample title, description, or metadata fields directly describing the sample). In case of cancer, something adjacent means that it's healthy. "
          "Non-disease conditions (e.g., pregnancy, aging, lifestyle factors) should be placed in the Donor information output column instead of the Disease "
          "Ontology Term field. Treatment - Determine from the context and the disease estimated with treatment could be possible for the pathology (eg: Nivolumab, "
          "Ipilimumab, vemurafenib, etc...). If no treatment available, try to find with your knowledge a path to create a new treatment or a gene to target for example. "
          "Treatment Time - Based on the given context, determine the treatment time category by searching in which state the treatment is on the given sample. Only two "
          "answers are possible: Assign 'Pre-treatment' if the context indicates that the sample or data was collected before the start of treatment. Or assign "
          "'On-treatment' if the context suggests that the sample or data was collected while the patient was undergoing treatment. If no clear indication is found, return 'nan'. "
          "Response - Search on the context, on protocols if any kind of resistance to the disease or the reverse is notified. Answer within those categories: "
          "'Progressive Disease', 'Stable Disease', 'Recist criteria'. If no such information found or can't be deduced from context, answer nan. Phenotype - Based on "
          "the given context, determine if the phenotype classification is 'parental' (Refers to the original, untreated cell line or population, which has not been exposed "
          "to selective pressure (such as drug treatment). Typically represents the baseline phenotype.) or 'persistant' (Refers to cells or populations that have survived treatment and "
          "exhibit drug persistence or resistance, often through adaptive mechanisms rather than genetic mutations.). Library selection fixed - Based on the given context, determine the "
          "library selection fixed category by searching for specific keywords or synonyms that match one of the five strict categories: 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or 'other'. "
          "Assign 'polyA' if the context contains any of the following terms or similar meaning: 'PolyA', 'poly.A', 'oligo.dT', 'oligodT', 'truseq.mrna', 'truseq.stranded.mrna', "
          "'truseq.standard.mrna', 'smarter.mRNA', 'stranded.mRNA'. Assign 'inverse rRNA' if the context mentions depletion of ribosomal RNA with any of these terms or similar meaning: "
          "'ribominus', 'ribodep', 'ribozero', 'ribo.zero', 'riboerase', 'ribogone', 'ribocop', 'ribo-dep', 'ribo-mi', 'ribo minus', 'depleted ribosom', 'remove ribosom', "
          "'TruSeq.Stranded.Total', 'TruSeq.Total', 'SMARTer.Stranded.Total', 'SMARTer.Total'. Assign 'hybrid selection' if the context refers to hybrid capture or exon selection using any "
          "of these terms or similar meaning: 'Hybrid.Selection', 'Exon.capture', 'Exome.capture', 'RNA.Exome', 'geoMX'. Assign 'small RNA' if the context refers to small RNA isolation with keywords "
          "such as 'TruSeq.Small', 'size.fraction' or similar meaning. Assign 'other' if none of the above terms are found. Return only the exact category name: 'polyA', 'inverse rRNA', "
          "'hybrid selection', 'small RNA', or 'other', with no additional text. Library source - Based on the given context, determine the library source category by searching for specific keywords "
          "that match one of the two strict categories: 'single-cell' or 'bulk'. Assign 'single-cell' if the context contains any of the following terms: 'TRANSCRIPTOMIC SINGLE CELL', 'chromium', "
          "'10x', 'single.cell' or similar meaning. Assign 'bulk' if none of the above terms are found. Return only the exact category name: 'single-cell' or 'bulk', with no additional text. "
          "Donor information - All information on the host that can be deduced from the context (eg., age, sex, blood analysis, any personal information). It can be principally found in the two "
          "last columns. If any information is missing in the metadata, provide an informed estimate when possible (e.g., based on general knowledge or known standards of the platform). Don't duplicate the answer. "
          "I want only one answer per category. Strict output format (no additional text or special characters, no duplicated answers) I wait from you: "
          "Cell line: [single unique answer] Cell type: [single unique answer] UBERON organ and code: [single unique answer] Disease Ontology Term: [single unique answer] Treatment: [single unique answer] "
          "Treatment Time: [single unique answer] Response: [single unique answer] Phenotype: [single unique answer] Library selection fixed: [single unique answer] Library source: [single unique answer] "
          "Donor information: [single unique answer] Here is the strict output:")

logging.basicConfig(level=logging.INFO,
                    filename='/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/PRUNING_MODEL/structural_pruning.log',
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s')

torch.set_num_threads(40)
logging.info("Multithreading configured with cpu.")
cache_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/store/first_pruned_results/PRUNING_MODEL/hf_cache"
os.makedirs(cache_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto",
                                             cache_dir=cache_dir)
model.eval()
logging.info("Model and tokenizer loaded.")
model_cpu = model.to("cpu")

activation_threshold = 0.005
new_hidden_size = 6912
current_hidden_size = model_cpu.model.layers[0].self_attn.q_proj.weight.shape[0]
neurons_to_remove = current_hidden_size - new_hidden_size
pruning_adjustment = neurons_to_remove / current_hidden_size if current_hidden_size > new_hidden_size else 0.0
default_pruning_amount_attention = 0.15
pruning_amount_attention = pruning_adjustment if pruning_adjustment > 0 else default_pruning_amount_attention
pruning_amount_mlp = 0.20  # 20% for MLP layers

##########################################################################################
#ACTIVATIONS
activation_dict = {}
if os.path.exists(activation_save_path):
    with open(activation_save_path, "rb") as file:
        logging.info("Activation file found.")
        activation_dict = pickle.load(file)
else:
    def save_activations(layer_name):
        def hook(module, input, output):
            activation_dict[layer_name] = output.detach().cpu()
            logging.info(f"Activation recorded for {layer_name}")
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

    with open(activation_save_path, "wb") as file:
        pickle.dump(activation_dict, file)

##########################################################################################
# IDENTIFY RARE MODULES FOR SPECIFIC PROMPT
neurons_to_prune = {}
for layer_name, activation in activation_dict.items():
    mean_activation = torch.mean(torch.abs(activation)).item()
    if mean_activation < activation_threshold:
        neurons_to_prune[layer_name] = mean_activation
        logging.info(
            f"Activation distribution = mean: {torch.tensor(list(neurons_to_prune.values())).mean()}, min: {torch.tensor(list(neurons_to_prune.values())).min()}")
        logging.info(f"{layer_name} marked for pruning (weak activation : {mean_activation:.6f})")

##########################################################################################
#FUNCTIONS


def reduce_input_dim_linear_layer(layer, target_in):
    """
    Reduce the input dimension of a linear layer by slicing its weight matrix if necessary.
    The output dimension (rows) remains unchanged.
    """
    old_weight = layer.weight.data
    old_bias = layer.bias.data if layer.bias is not None else None
    out_features, in_features = old_weight.shape
    if in_features > target_in:
        new_weight = old_weight[:, :target_in]
    else:
        new_weight = old_weight
        target_in = in_features
    new_layer = nn.Linear(target_in, out_features, bias=(old_bias is not None))
    new_layer.weight.data = new_weight
    if old_bias is not None:
        new_layer.bias.data = old_bias
    return new_layer


def apply_structured_pruning(module, amount, n, dim):
    prune.ln_structured(module, 'weight', amount=amount, n=n, dim=dim)
    with torch.no_grad():
        mask = module.weight_mask
        module.weight.data = module.weight * mask
    prune.remove(module, 'weight')


def compress_linear_layer(layer, dim=0):
    """
    Physically remove rows (dim=0) or columns (dim=1) that are entirely zero.
    Return (new_layer, keep_indices).
    """
    W = layer.weight.data
    B = layer.bias.data if layer.bias is not None else None
    W_np = W.cpu().numpy()
    if dim == 0:
        row_sum = np.sum(np.abs(W_np), axis=1)
        keep_rows = np.where(row_sum != 0)[0]
        new_out = len(keep_rows)
        in_features = W_np.shape[1]
        new_layer = nn.Linear(in_features, new_out, bias=(B is not None))
        W_reduced = W_np[keep_rows, :]
        new_layer.weight.data = torch.from_numpy(W_reduced).to(W.device, dtype=W.dtype)
        if B is not None:
            new_layer.bias.data = B[keep_rows].to(W.device, dtype=W.dtype)
        return new_layer, keep_rows.tolist()
    else:
        col_sum = np.sum(np.abs(W_np), axis=0)
        keep_cols = np.where(col_sum != 0)[0]
        out_features = W_np.shape[0]
        new_in = len(keep_cols)
        new_layer = nn.Linear(new_in, out_features, bias=(B is not None))
        W_reduced = W_np[:, keep_cols]
        new_layer.weight.data = torch.from_numpy(W_reduced).to(W.device, dtype=W.dtype)
        if B is not None:
            new_layer.bias.data = B.to(W.device, dtype=W.dtype)
        return new_layer, keep_cols.tolist()


def compress_structured_layer(layer, dim=0):
    """
    Compress the pruned layer physically by removing zero rows/columns.
    """
    new_layer, _ = compress_linear_layer(layer, dim=dim)
    return new_layer


def flatten_and_slice_2d(layer, target_out, target_in):
    """Ensure layer.weight is a 2D tensor with correct dimensions."""
    w = layer.weight.data
    if w.dim() == 4 and w.shape[2] == 1 and w.shape[3] == 1:
        w = w.view(w.shape[0], w.shape[1])
    out_dim, in_dim = w.shape
    if out_dim >= target_out and in_dim >= target_in:
        w = w[:target_out, :target_in]
    layer.weight.data = w


def expand_linear_layer(layer, target_out, target_in):
    """
    Expand a linear layer whose weight is of shape [old_out, in_features] to a new layer of shape
    [target_out, in_features] by copying the existing weights into the top rows and padding the rest with zeros.
    Ideally, target_in should equal in_features; if not, we log a warning and use the actual in_features.
    """
    old_weight = layer.weight.data
    old_bias = layer.bias.data if layer.bias is not None else None
    old_out, in_features = old_weight.shape
    if in_features != target_in:
        import logging
        logging.warning(
            f"Input dimension mismatch during expansion: layer has in_features {in_features} but target_in was {target_in}. Using target_in={target_in} after reduction.")
        layer = reduce_input_dim_linear_layer(layer, target_in)
        old_weight = layer.weight.data
        old_out, in_features = old_weight.shape
    new_layer = nn.Linear(target_in, target_out, bias=(old_bias is not None))
    new_weight = torch.zeros((target_out, target_in), dtype=old_weight.dtype, device=old_weight.device)
    new_weight[:old_out, :] = old_weight
    new_layer.weight.data = new_weight
    if old_bias is not None:
        new_bias = torch.zeros((target_out,), dtype=old_bias.dtype, device=old_bias.device)
        new_bias[:old_out] = old_bias
        new_layer.bias.data = new_bias
    return new_layer

##########################################################################################
#MAIN


for i, layer in enumerate(model_cpu.model.layers):
    logging.info(f"Starting pruning for layer {i + 1}/{len(model_cpu.model.layers)}")

    #Magnitude Pruning for Attention (Q/K/V)
    for proj in ["q_proj", "k_proj", "v_proj"]:
        proj_layer = getattr(layer.self_attn, proj)
        current_dim = proj_layer.weight.shape[0]
        if proj == "q_proj":
            if current_dim > new_hidden_size:
                neurons_to_remove_proj = current_dim - new_hidden_size
                pruning_adjustment_proj = neurons_to_remove_proj / current_dim
                amount = pruning_adjustment_proj if pruning_adjustment_proj > 0 else default_pruning_amount_attention
                logging.info(
                    f"Pruning {proj} in layer {i}: current_dim={current_dim}, target={new_hidden_size}, amount={amount}")
                apply_structured_pruning(proj_layer, amount, n=2, dim=0)
                new_proj_layer = compress_structured_layer(proj_layer, dim=0)
                setattr(layer.self_attn, proj, new_proj_layer)
            else:
                logging.info(
                    f"Skipping pruning for {proj} in layer {i} (current_dim={current_dim} <= target {new_hidden_size})")
        else:
            logging.info(f"Skipping structured pruning for {proj} in layer {i} (preserving dimension {current_dim})")

    #Magnitude Pruning for MLP layers
    for fc in ["gate_proj", "down_proj", "up_proj"]:
        fc_layer = getattr(layer.mlp, fc)
        current_dim = fc_layer.weight.shape[0]
        if current_dim > new_hidden_size:
            neurons_to_remove_fc = current_dim - new_hidden_size
            pruning_adjustment_fc = neurons_to_remove_fc / current_dim
            amount = pruning_adjustment_fc if pruning_adjustment_fc > 0 else pruning_amount_mlp
            logging.info(
                f"Pruning {fc} in layer {i}: current_dim={current_dim}, target={new_hidden_size}, amount={amount}")
            apply_structured_pruning(fc_layer, amount, n=2, dim=0)
            new_fc_layer = compress_structured_layer(fc_layer, dim=0)
            setattr(layer.mlp, fc, new_fc_layer)
        else:
            logging.info(
                f"Skipping pruning for {fc} in layer {i} (current_dim={current_dim} <= target {new_hidden_size})")

    #Pruning non-activated modules
    for module_name in ["q_proj", "k_proj", "v_proj", "gate_proj", "down_proj", "up_proj"]:
        key = f"layer_{i}_{module_name}"
        if key in neurons_to_prune:
            target_layer = getattr(layer.self_attn if "proj" in module_name else layer.mlp, module_name)
            apply_structured_pruning(target_layer, 0.9, n=2, dim=0)
            new_target_layer = compress_structured_layer(target_layer, dim=0)
            if "proj" in module_name:
                setattr(layer.self_attn, module_name, new_target_layer)
            else:
                setattr(layer.mlp, module_name, new_target_layer)
    logging.info(f"Completed pruning for layer {i + 1}")

logging.info("Parameters after pruning and compression:")
for name, param in model_cpu.named_parameters():
    logging.info(f"{name}: {param.numel()} params after pruning")
total_params = sum(p.numel() for p in model_cpu.parameters())
logging.info(f"Total number of parameters after pruning: {total_params}")

#weights in attention projections and layer norms
for i, layer in enumerate(model_cpu.model.layers):
    with torch.no_grad():
        flatten_and_slice_2d(layer.self_attn.q_proj, new_hidden_size, new_hidden_size)
        for proj in ["k_proj", "v_proj"]:
            proj_layer = getattr(layer.self_attn, proj)

        old_norm_weight = layer.input_layernorm.weight.data
        new_norm = type(layer.input_layernorm)(new_hidden_size, eps=model_cpu.config.rms_norm_eps)
        new_norm.weight = nn.Parameter(old_norm_weight[:new_hidden_size].clone())
        layer.input_layernorm = new_norm

        old_norm_weight = layer.post_attention_layernorm.weight.data
        new_norm = type(layer.post_attention_layernorm)(new_hidden_size, eps=model_cpu.config.rms_norm_eps)
        new_norm.weight = nn.Parameter(old_norm_weight[:new_hidden_size].clone())
        layer.post_attention_layernorm = new_norm

##########################################################################################
# AVE FINAL MODEL

#configuration with new hidden size
model.config.hidden_size = new_hidden_size
model.config.intermediate_size = new_hidden_size
model.config.num_attention_heads = new_hidden_size // model.config.head_dim

#update embeddings and output layers
logging.info(f"embed_tokens.weight : {model_cpu.model.embed_tokens.weight.shape}")
with torch.no_grad():
    old_embedding = model_cpu.model.embed_tokens.weight.data
    model_cpu.model.embed_tokens = nn.Embedding(old_embedding.shape[0], new_hidden_size)
    model_cpu.model.embed_tokens.weight.data = old_embedding[:, :new_hidden_size]

    #Final layer norm
    old_output_norm_weight = model_cpu.model.norm.weight.data
    model_cpu.model.norm = type(model_cpu.model.norm)(new_hidden_size, eps=model_cpu.config.rms_norm_eps)
    model_cpu.model.norm.weight.data = old_output_norm_weight[:new_hidden_size]

    #lm_head
    old_lm_head_weight = model_cpu.lm_head.weight.data
    model_cpu.lm_head = nn.Linear(new_hidden_size, model_cpu.config.vocab_size, bias=False)
    model_cpu.lm_head.weight.data = old_lm_head_weight[:, :new_hidden_size]

logging.info(f"All embeddings and output layers resized to new hidden_size {new_hidden_size}.")

#save
model_cpu.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
