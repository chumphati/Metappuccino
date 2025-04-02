import torch
from transformers import AutoModelForCausalLM


#VERIFY DIMENSIONS MATRIX IN LAYERS
def verify_pruned_model(model):
    config = model.config
    hidden_size = config.hidden_size
    head_dim = getattr(config, "head_dim", hidden_size // config.num_attention_heads)
    print(f"Model config: hidden_size = {hidden_size}, head_dim = {head_dim}, num_attention_heads = {config.num_attention_heads}\n", flush=True)

    #embeddings
    embed_shape = model.model.embed_tokens.weight.shape
    print(f"embedding shape: {embed_shape}", flush=True)
    if embed_shape[1] != hidden_size:
        print(f"  -> Mismatch in embedding: {embed_shape[1]} != {hidden_size}", flush=True)

    #transformer layers
    for i, layer in enumerate(model.model.layers):
        print(f"\nLayer {i}", flush=True)
        #attention layers
        for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            proj_layer = getattr(layer.self_attn, proj_name, None)
            if proj_layer is not None:
                shape = proj_layer.weight.shape
                print(f"{proj_name:10s} weight shape: {shape}", flush=True)
                if shape[1] != hidden_size:
                    print(f"Mismatch {proj_name}: dimension 1 = {shape[1]} != {hidden_size}", flush=True)

        #MLP layers
        for mlp_name in ['gate_proj', 'up_proj', 'down_proj']:
            mlp_layer = getattr(layer.mlp, mlp_name, None)
            if mlp_layer is not None:
                shape = mlp_layer.weight.shape
                print(f"{mlp_name:10s} weight shape: {shape}", flush=True)
                if shape[0] != hidden_size:
                    print(f"Mismatch {mlp_name}: dimension 0 = {shape[0]} != {hidden_size}", flush=True)

        #norm layers
        for norm_name in ['input_layernorm', 'post_attention_layernorm']:
            norm_layer = getattr(layer, norm_name, None)
            if norm_layer is not None:
                shape = norm_layer.weight.shape
                print(f"{norm_name:30s} weight shape: {shape}", flush=True)
                if shape[0] != hidden_size:
                    print(f"Mismatch {norm_name}: {shape[0]} != {hidden_size}", flush=True)

    #final nirm layer
    if hasattr(model.model, 'norm'):
        final_norm_shape = model.model.norm.weight.shape
        print(f"\nFinal norm weight shape: {final_norm_shape}", flush=True)
        if final_norm_shape[0] != hidden_size:
            print(f"Mismatch {final_norm_shape[0]} != {hidden_size}", flush=True)

    #LM head
    lm_head_shape = model.lm_head.weight.shape
    print(f"\nLM head weight shape: {lm_head_shape}", flush=True)
    if lm_head_shape[1] != hidden_size:
        print(f"Mismatch dimension 1 = {lm_head_shape[1]} != {hidden_size}", flush=True)


def main():
    pruned_model_path = '/store/EQUIPES/SSFA/MEMBERS/fiona.hak/MetaMap/results/PRUNING_MODEL/llama-3-pruned-struct'
    print(f"Charge model from {pruned_model_path}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        pruned_model_path,
        torch_dtype=torch.float16,
        ignore_mismatched_sizes=True
    )
    model.eval()
    verify_pruned_model(model)


if __name__ == '__main__':
    main()
