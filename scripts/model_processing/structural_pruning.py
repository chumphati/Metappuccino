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
pruning_amount_attention = 0.15  #15% weak col (Q/K/V)
pruning_amount_mlp = 0.20  #20% weak col (MLP)
activation_save_path = os.path.join(base_path, "activations.pkl")
model_save_path = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/PRUNING_MODEL/llama-3-pruned-struct"
prompt = "Run accession: [number]. Metadata to analyze: [information]. For each row in the metadata line (the first line contains the column names), extract and format the following information concisely. For each missing category, provide a single answer without redundancy. Each category **MUST** have one distinct and explicit answer, even if inferred. **Do not leave any category empty.** Do not repeat information already provided in previous categories. Remove redundant text. Tissue type – The tissue type from which the sample originates (e.g., liver, lung, brain). If not specified, deduce from context in the two last columns. Cell line – Specify the cell line, or state 'Primary tissue' if the sample is from a primary tissue and not a cell line. Cell type – The type of cell in the sample (e.g., neuron, fibroblast, CD8 T cell, CD4 T cell, monocyte NK cell, mast cell, melanocyte, dendritic cell, etc...). If not provided, deduce based on the tissue type and state the inference. Use thee Cell Ontology terms terminology. UBERON organ and code – Provide me the organ concerned by this study, in the UBERON GTEX terminology for the tissue type (e.g., UBERON:000XXXX + name of the organ). If not specified, deduce from context, or search one related to the tissue. Disease Ontology Term – Return the Disease Ontology term corresponding to the disease associated with the sample in the format DOID:XXXXX + Disease Name. If the sample is explicitly described as 'normal' or 'healthy', do not infer any disease. In this case, do not search for disease-related information in the context. If the sample is not explicitly labeled as 'normal' or 'healthy' or 'no disease', infer the disease from the context only if it is directly related to the sample (e.g., sample title, description, or metadata fields directly describing the sample). In case of cancer, something adjacent means that it's healthy. Non-disease conditions (e.g., pregnancy, aging, lifestyle factors) should be placed in the Donor information output column instead of the Disease Ontology Term field. Treatment - Determine from the context and the desease estimated with treatment could be possible for the pathology (eg: Nivolumab, Ipilimumab, vemurafenib, etc...). If no treatment avaible, try to find with your knowledge a path to create a new treatment or a gene to target for example. Treatment Time - Based on the given context, determine the treatment time category by searching in which state the tratment is on the given sample. Only two answer are possible: Assign 'Pre-treatment' if the context indicates that the sample or data was collected before the start of treatment. Or assign 'On-treatment' if the context suggests that the sample or data was collected while the patient was undergoing treatment. If no clear indication is found, return 'nan'. Response - Search on the context, on protocols if any kind of resistance to the disease or the reverse is notified. Answer within those categories: 'Progressive Disease', 'Stable Disease', 'Recist criteria'. If no such information founded or can't be deducted from context, answer nan. Phenotype - Based on the given context, determine if the phenotype classification is 'parental' (Refers to the original, untreated cell line or population, which has not been exposed to selective pressure (such as drug treatment). Typically represents the baseline phenotype.) or 'persistant' (Refers to cells or populations that have survived treatment and exhibit drug persistence or resistance, often through adaptive mechanisms rather than genetic mutations.). Library selection fixed - Based on the given context, determine the library selection fixed category by searching for specific keywords or synonyms that match one of the five strict categories: 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or 'other'. Assign 'polyA' if the context contains any of the following terms or similar meaning: 'PolyA', 'poly.A', 'oligo.dT', 'oligodT', 'truseq.mrna', 'truseq.stranded.mrna', 'truseq.standard.mrna', 'smarter.mRNA', 'stranded.mRNA'. Assign 'inverse rRNA' if the context mentions depletion of ribosomal RNA with any of these terms or similar meaning: 'ribominus', 'ribodep', 'ribozero', 'ribo.zero', 'riboerase', 'ribogone', 'ribocop', 'ribo-dep', 'ribo-mi', 'ribo minus', 'depleted ribosom', 'remove ribosom', 'TruSeq.Stranded.Total', 'TruSeq.Total', 'SMARTer.Stranded.Total', 'SMARTer.Total'. Assign 'hybrid selection' if the context refers to hybrid capture or exon selection using any of these terms or similar meaning: 'Hybrid.Selection', 'Exon.capture', 'Exome.capture', 'RNA.Exome', 'geoMX'. Assign 'small RNA' if the context refers to small RNA isolation with keywords such as 'TruSeq.Small', 'size.fraction' or similar meaning. Assign 'other' if none of the above terms are found. Return only the exact category name: 'polyA', 'inverse rRNA', 'hybrid selection', 'small RNA', or 'other', with no additional text. Library source - Based on the given context, determine the library source category by searching for specific keywords that match one of the two strict categories: 'single-cell' or 'bulk'. Assign 'single-cell' if the context contains any of the following terms: 'TRANSCRIPTOMIC SINGLE CELL', 'chromium', '10x', 'single.cell' or similar meaning. Assign 'bulk' if none of the above terms are found. Return only the exact category name: 'single-cell' or 'bulk', with no additional text. Donor information - All information on the host that can be deduce of the context (eg., age, sex, blood analysis, any personnal information). It can be principally founded in the two last columns. If any information is missing in the metadata, provide an informed estimate when possible (e.g., based on general knowledge or known standards of the platform). Don't double the answer. I want only one answer per category. Strict output format (no additional text or special characters, no duplicated answers) I wait from you: Cell line: [single unique answer] Cell type: [single unique answer] UBERON organ and code: [single unique answer] Disease Ontology Term: [single unique answer] Treatment: [single unique answer] Treatment Time: [single unique answer] Response: [single unique answer] Phenotype: [single unique answer] Library selection fixed: [single unique answer] Library source: [single unique answer] Donor information: [single unique answer] Here is the strict output:"

##########################################################################################
# ORIGINAL MODEL

cache_dir = "/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/PRUNING_MODEL/hf_cache"
os.makedirs(cache_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto", cache_dir=cache_dir)
model.eval()

##########################################################################################
# FOLLOW ACTIVATIONS

activation_dict = {}

if os.path.exists(activation_save_path):
    with open(activation_save_path, "rb") as file:
        activation_dict = pickle.load(file)
else:
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

    with open(activation_save_path, "wb") as file:
        pickle.dump(activation_dict, file)

##########################################################################################
# IDENTIFY RARE MODULES FOR SPECIFIC PROMPT

neurons_to_prune = {}
for layer_name, activation in activation_dict.items():
    mean_activation = torch.mean(torch.abs(activation)).item()
    if mean_activation < activation_threshold:
        neurons_to_prune[layer_name] = mean_activation
        print(f"{layer_name} marqued for pruning (weak activation : {mean_activation:.6f})")

##########################################################################################
# COMBINE PRUNING

model_cpu = model.to("cpu")

for i, layer in enumerate(model_cpu.model.layers):

    # Magnitude Pruning Attention (Q/K/V)
    for proj in ["q_proj", "k_proj", "v_proj"]:
        proj_layer = getattr(layer.self_attn, proj)
        prune.ln_structured(proj_layer, 'weight', amount=pruning_amount_attention, n=2, dim=0)
        prune.remove(proj_layer, 'weight')

    # Magnitude Pruning MLP
    for fc in ["gate_proj", "down_proj", "up_proj"]:
        fc_layer = getattr(layer.mlp, fc)
        prune.ln_structured(fc_layer, 'weight', amount=pruning_amount_mlp, n=2, dim=0)
        prune.remove(fc_layer, 'weight')

    # Pruning non activated modules
    for module_name in ["q_proj", "k_proj", "v_proj", "gate_proj", "down_proj", "up_proj"]:
        key = f"layer_{i}_{module_name}"
        if key in neurons_to_prune:
            target_layer = getattr(layer.self_attn if "proj" in module_name else layer.mlp, module_name)
            prune.ln_structured(target_layer, 'weight', amount=0.9, n=2, dim=0)
            prune.remove(target_layer, 'weight')

##########################################################################################
# SAVE FINAL MODEL

model_cpu.save_pretrained(model_save_path)
