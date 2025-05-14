from huggingface_hub import snapshot_download

HF_TOKEN = "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"

snapshot_download(
    repo_id="meta-llama/Llama-3.1-70B-Instruct",
    local_dir="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/Llama-3.1-70B-Instruct",
    use_auth_token=HF_TOKEN,
    resume_download=True,
    max_workers=4
)
