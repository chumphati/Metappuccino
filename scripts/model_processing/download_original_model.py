from huggingface_hub import snapshot_download

HF_TOKEN = "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"

snapshot_download(
    repo_id="google/gemma-3-270m-it",
    local_dir="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/gemma-3-270m-it",
    use_auth_token=HF_TOKEN,
    resume_download=True,
    max_workers=4
)
