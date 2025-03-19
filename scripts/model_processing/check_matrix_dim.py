import torch
from transformers import AutoModelForCausalLM


def verify_pruned_model(model):
    config = model.config
    hidden_size = config.hidden_size
    # Essayer d'obtenir head_dim depuis config, sinon calculer à partir de num_attention_heads
    head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)
    print(
        f"Configuration du modèle: hidden_size = {hidden_size}, head_dim = {head_dim}, num_attention_heads = {config.num_attention_heads}\n", flush=True)

    # Vérification de la couche d'embedding
    embed_shape = model.model.embed_tokens.weight.shape
    print(f"Shape de l'embedding: {embed_shape}", flush=True)
    if embed_shape[1] != hidden_size:
        print(f"  -> Mismatch dans l'embedding: {embed_shape[1]} != {hidden_size}", flush=True)

    # Parcourir les couches du Transformer
    for i, layer in enumerate(model.model.layers):
        print(f"\n=== Couche {i} ===", flush=True)
        # Projections d'attention
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            proj_layer = getattr(layer.self_attn, proj_name, None)
            if proj_layer is not None:
                shape = proj_layer.weight.shape
                print(f"{proj_name:10s} weight shape: {shape}", flush=True)
                # Exemple de vérification : si q_proj et k_proj et v_proj sont attendus en [hidden_size, hidden_size]
                # Vous pouvez adapter la vérification selon votre logique de pruning.
                if shape[1] != hidden_size:
                    print(f"  -> Mismatch sur {proj_name}: dimension 1 = {shape[1]} != {hidden_size}", flush=True)

        # Couches MLP
        for mlp_name in ['gate_proj', 'up_proj', 'down_proj']:
            mlp_layer = getattr(layer.mlp, mlp_name, None)
            if mlp_layer is not None:
                shape = mlp_layer.weight.shape
                print(f"{mlp_name:10s} weight shape: {shape}", flush=True)
                if shape[0] != hidden_size:
                    print(f"  -> Mismatch sur {mlp_name}: dimension 0 = {shape[0]} != {hidden_size}", flush=True)

        # Couches de normalisation
        for norm_name in ['input_layernorm', 'post_attention_layernorm']:
            norm_layer = getattr(layer, norm_name, None)
            if norm_layer is not None:
                shape = norm_layer.weight.shape
                print(f"{norm_name:30s} weight shape: {shape}", flush=True)
                if shape[0] != hidden_size:
                    print(f"  -> Mismatch sur {norm_name}: {shape[0]} != {hidden_size}", flush=True)

    # Couche finale de normalisation (si présente)
    if hasattr(model.model, 'norm'):
        final_norm_shape = model.model.norm.weight.shape
        print(f"\nFinal norm weight shape: {final_norm_shape}", flush=True)
        if final_norm_shape[0] != hidden_size:
            print(f"  -> Mismatch sur la couche finale norm: {final_norm_shape[0]} != {hidden_size}", flush=True)

    # Tête LM
    lm_head_shape = model.lm_head.weight.shape
    print(f"\nLM head weight shape: {lm_head_shape}", flush=True)
    if lm_head_shape[1] != hidden_size:
        print(f"  -> Mismatch sur la tête LM: dimension 1 = {lm_head_shape[1]} != {hidden_size}", flush=True)


def main():
    pruned_model_path = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/PRUNING_MODEL/llama-3-pruned-struct'
    print(f"Chargement du modèle pruné depuis : {pruned_model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        pruned_model_path,
        torch_dtype=torch.float16,
        ignore_mismatched_sizes=True
    )
    model.eval()
    verify_pruned_model(model)


if __name__ == '__main__':
    main()
