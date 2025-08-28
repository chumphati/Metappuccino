from huggingface_hub import snapshot_download

HF_TOKEN = ""

snapshot_download(
    repo_id="MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF",
    local_dir="/store/EQUIPES/SSFA/MEMBERS/fiona.hak/models/Meta-Llama-3-70B-Instruct-GGUF",
    use_auth_token=HF_TOKEN,
    resume_download=True,
    max_workers=4
)
