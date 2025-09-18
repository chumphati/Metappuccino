from huggingface_hub import snapshot_download

HF_TOKEN = ""

snapshot_download(
    repo_id="Qwen/Qwen3-30B-A3B-Instruct-2507",
    local_dir="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/Qwen3-30B-A3B-Instruct-2507",
    use_auth_token=HF_TOKEN,
    resume_download=True,
    max_workers=4
)
