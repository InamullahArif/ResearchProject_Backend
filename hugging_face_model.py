# hf_download_all.py
from huggingface_hub import snapshot_download

REPO_ID = "vaibhav07112004/fraud-detection-models"

local_dir = snapshot_download(
    repo_id=REPO_ID,
    local_dir="hf_models",
    local_dir_use_symlinks=False
)

print("Downloaded to:", local_dir)