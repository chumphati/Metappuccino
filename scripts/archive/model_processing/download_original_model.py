from huggingface_hub import snapshot_download

HF_TOKEN = "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"

snapshot_download(
    repo_id="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
    local_dir="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/DeepSeek-Coder-V2-Lite-Instruct",
    use_auth_token=HF_TOKEN,
    resume_download=True,
    max_workers=4
)
